"""Integration contracts for the HCCL world-to-attribution transaction seam."""

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
from alberta_framework.core.hccl_causal_attribution import (
    HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER,
    HCCLActionLayer,
    HCCLActionReceipt,
    HCCLSignalContrast,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCL_WORLD_ATTRIBUTION_ADAPTER_STATUS,
    HCCLWorldAttributionAdapter,
    HCCLWorldAttributionAdapterConfig,
    HCCLWorldAttributionAdapterResult,
    HCCLWorldAttributionAdapterState,
    load_hccl_world_attribution_checkpoint,
    measure_hccl_world_attribution_state_nbytes,
    run_hccl_world_attribution_scan,
    save_hccl_world_attribution_checkpoint,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    HCCLCausalCoreConfig,
)

pytestmark = pytest.mark.integration


def _adapter() -> HCCLWorldAttributionAdapter:
    return HCCLWorldAttributionAdapter(
        HCCLWorldAttributionAdapterConfig(
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
        )
    )


def _smoke_adapter() -> HCCLWorldAttributionAdapter:
    return HCCLWorldAttributionAdapter(
        HCCLWorldAttributionAdapterConfig(
            proposal_owner_digest=(
                0x10203040,
                0x50607080,
                0x90A0B0C0,
                0xD0E0F001,
                0x12345678,
                0x9ABCDEF0,
                0x0F1E2D3C,
                0x4B5A6978,
            ),
            world_config=HCCLCausalCoreConfig.mechanics_smoke(),
        )
    )


def _identity_rows(offset: int) -> jax.Array:
    return jnp.asarray(
        [
            [offset + 1, offset + 2, offset + 3, offset + 4],
            [offset + 5, offset + 6, offset + 7, offset + 8],
        ],
        dtype=jnp.uint32,
    )


def _receipts(
    adapter: HCCLWorldAttributionAdapter,
    state: HCCLWorldAttributionAdapterState,
    event: Any,
    *,
    actions: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
        (0, 0),
        (1, 0),
        (1, 1),
    ),
) -> tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt]:
    masks = jnp.ones((2, 2), dtype=jnp.bool_)
    bound = tuple(
        adapter.bind_action_receipt(
            state,
            event,
            layer=layer,
            actions_before_mask=jnp.asarray(layer_actions, dtype=jnp.int32),
            actions_after_mask=jnp.asarray(layer_actions, dtype=jnp.int32),
            hard_action_masks=masks,
            action_receipt_identity_words=_identity_rows(100 + 40 * int(layer)),
        )
        for layer, layer_actions in zip(
            (HCCLActionLayer.BASE, HCCLActionLayer.MEMORY, HCCLActionLayer.PLANNER),
            actions,
            strict=True,
        )
    )
    return cast(tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt], bound)


def _stage(
    adapter: HCCLWorldAttributionAdapter,
    state: HCCLWorldAttributionAdapterState,
    event: Any,
    receipts: tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
    *,
    gate: bool = True,
) -> HCCLWorldAttributionAdapterResult:
    return adapter.stage(
        state,
        event,
        *receipts,
        downstream_candidate_valid=jnp.asarray(gate, dtype=jnp.bool_),
    )


def _row(tree: object, index: int) -> object:
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _stack(left: object, right: object) -> object:
    return jax.tree.map(lambda x, y: jnp.stack((x, y)), left, right)


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            dtype, jax.dtypes.prng_key
        ):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _assert_contrast(
    actual: HCCLSignalContrast,
    positive: Any,
    negative: Any,
) -> None:
    chex.assert_trees_all_equal(
        actual,
        HCCLSignalContrast(  # type: ignore[call-arg]
            task_score=positive.task_score - negative.task_score,
            net_reward=positive.net_reward - negative.net_reward,
            safety_cost=positive.safety_cost - negative.safety_cost,
            message_charge=positive.message_charge - negative.message_charge,
        ),
    )


def test_strict_non_authoritative_config_and_exact_state_ownership_are_public() -> None:
    adapter = _adapter()
    payload = adapter.to_config()
    assert payload["mechanism_status"] == HCCL_WORLD_ATTRIBUTION_ADAPTER_STATUS
    assert payload["mechanism_status"] == "l0-development-world-attribution-transaction-only"
    assert payload["proposal_order"] == list(HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER)
    assert payload["proposal_calls_per_valid_transaction"] == 8
    assert payload["committed_world_successors_per_transaction"] == 1
    assert payload["world_state_owners"] == 1
    assert payload["attribution_state_owners"] == 1
    assert payload["composite_link_owners"] == 1
    assert payload["composite_jit_supported"] is False
    assert payload["prebound_scan_execution"] == "host-eager-python-loop"
    for name in (
        "agent_implementation_present",
        "agent_component_provenance_bound",
        "action_layer_provenance_bound",
        "schedule_execution_authorized",
        "artifact_authorized",
        "output_writes_authorized",
        "threshold_authorized",
        "seed_authority",
        "evidence_authorized",
        "promotion_authorized",
    ):
        assert payload[name] is False
    assert payload["synthetic_world_bound_source_placeholders"] is True
    assert HCCLWorldAttributionAdapter.from_config(payload).to_config() == payload
    assert alberta_framework.HCCLWorldAttributionAdapter is HCCLWorldAttributionAdapter
    assert core_api.HCCLWorldAttributionAdapterConfig is HCCLWorldAttributionAdapterConfig

    state = adapter.init(jr.key(3))
    assert tuple(field.name for field in dataclasses.fields(cast(Any, state))) == (
        "world_state",
        "attribution_state",
        "composite_link_words",
    )
    assert state.world_state.step_words.dtype == jnp.uint32
    assert state.world_state.step_words.shape == (2,)
    assert state.attribution_state.transaction_words.dtype == jnp.uint32
    assert state.attribution_state.transaction_words.shape == (2,)
    chex.assert_trees_all_equal(
        state.world_state.step_words,
        state.attribution_state.transaction_words,
    )
    assert state.composite_link_words.shape == (4,)
    assert state.composite_link_words.dtype == jnp.uint32
    assert tuple(int(word) for word in state.composite_link_words) == (
        1042938720,
        1314705779,
        447420755,
        998477259,
    )
    assert bool(adapter.state_valid(state))

    malformed = dict(payload)
    malformed["evidence_authorized"] = True
    with pytest.raises(ValueError, match="config|authorized|unsupported"):
        HCCLWorldAttributionAdapter.from_config(malformed)

    for name, equal_but_wrong_type in (
        ("agent_implementation_present", 0),
        ("proposal_calls_per_valid_transaction", 8.0),
        ("synthetic_world_bound_source_placeholders", 1),
    ):
        malformed = dict(payload)
        malformed[name] = equal_but_wrong_type
        with pytest.raises(ValueError, match="unsupported"):
            HCCLWorldAttributionAdapter.from_config(malformed)

    with pytest.raises(TypeError, match="exact dict"):
        HCCLWorldAttributionAdapter.from_config(cast(Any, tuple(payload.items())))


def test_smoke_world_profile_is_bound_through_adapter_config_state_and_resources() -> None:
    smoke = _smoke_adapter()
    payload = smoke.to_config()
    world_payload = cast(dict[str, object], payload["world_config"])
    attribution_payload = cast(dict[str, object], payload["attribution_config"])
    assert world_payload["schedule_profile"] == HCCL_CAUSAL_CORE_SMOKE_PROFILE
    assert world_payload["maximum_committed_transitions"] == 420
    assert attribution_payload["max_transactions"] == 420
    assert HCCLWorldAttributionAdapter.from_config(payload).to_config() == payload

    canonical = _adapter()
    canonical_state = canonical.init(jr.key(37))
    smoke_state = smoke.init(jr.key(37))
    assert not bool(canonical.state_valid(smoke_state))
    assert not bool(smoke.state_valid(canonical_state))
    assert not bool(
        jnp.all(canonical_state.composite_link_words == smoke_state.composite_link_words)
    )
    assert smoke.resource_budget(smoke_state).maximum_committed_transactions == 420

    malformed = dict(payload)
    malformed_attribution = dict(attribution_payload)
    malformed_attribution["max_transactions"] = 8998
    malformed["attribution_config"] = malformed_attribution
    with pytest.raises(ValueError, match="unsupported"):
        HCCLWorldAttributionAdapter.from_config(malformed)


def test_exact_receipt_binding_fixed_eight_world_proposals_and_pp_only_adoption() -> None:
    adapter = _adapter()
    initial = adapter.init(jr.key(5))
    event = adapter.world.prepare_event(initial.world_state)
    base, memory, planner = _receipts(adapter, initial, event)

    chex.assert_trees_all_equal(
        base.source.source_identity_words,
        initial.world_state.content_tag_words,
    )
    chex.assert_trees_all_equal(base.source.source_transition_words, initial.world_state.step_words)
    chex.assert_trees_all_equal(
        base.source.decision_words,
        initial.attribution_state.decision_words,
    )
    chex.assert_trees_all_equal(base.exogenous_identity_words, event.content_tag_words)
    chex.assert_trees_all_equal(base.action_receipt_identity_words, _identity_rows(100))
    chex.assert_trees_all_equal(memory.action_receipt_identity_words, _identity_rows(140))
    chex.assert_trees_all_equal(planner.action_receipt_identity_words, _identity_rows(180))

    result = _stage(adapter, initial, event, (base, memory, planner))
    assert bool(result.update_applied)
    assert bool(result.world_source_clock_bound)
    assert bool(result.event_receipt_identity_bound)
    assert bool(result.action_receipt_identities_bound)
    assert bool(result.equal_action_world_payloads_bit_exact)
    assert bool(result.causal_core_signal_contract_valid)
    assert bool(result.world_duplicate_mm_bit_exact)
    assert bool(result.attribution.duplicate_mm_bit_exact)
    assert int(result.work.world_proposal_calls) == 8
    assert int(result.work.attribution_proposal_calls) == 8
    assert int(result.work.designated_counterfactual_world_slots) == 7
    assert int(result.work.discarded_world_proposal_calls) == 7
    assert int(result.work.committed_pp_world_successors) == 1
    np.testing.assert_array_equal(
        np.asarray(result.attribution.proposals.vertex.actions),
        [[1, 0], [0, 0], [1, 0], [0, 0], [1, 1], [1, 1], [1, 0], [1, 0]],
    )
    chex.assert_trees_all_equal(_row(result.world_proposals, 0), _row(result.world_proposals, 7))
    actions = np.asarray(result.attribution.proposals.vertex.actions)
    for index in range(8):
        for previous in range(index):
            if np.array_equal(actions[index], actions[previous]):
                chex.assert_trees_all_equal(
                    _row(result.world_proposals, index),
                    _row(result.world_proposals, previous),
                )
    chex.assert_trees_all_equal(
        result.state.world_state,
        cast(Any, _row(result.world_proposals, 4)).candidate_state,
    )
    chex.assert_trees_all_equal(
        result.state.attribution_state.last_committed_pp,
        _row(result.attribution.proposals, 4),
    )
    chex.assert_trees_all_equal(result.pre_transaction_words, jnp.asarray([0, 0], jnp.uint32))
    chex.assert_trees_all_equal(result.post_transaction_words, jnp.asarray([0, 1], jnp.uint32))
    chex.assert_trees_all_equal(
        result.state.world_state.step_words,
        result.state.attribution_state.transaction_words,
    )


def test_world_signals_feed_all_typed_contrasts_without_channel_collapse() -> None:
    adapter = _adapter()
    state = adapter.init(jr.key(7))
    event = adapter.world.prepare_event(state.world_state)
    result = _stage(adapter, state, event, _receipts(adapter, state, event))
    mm = cast(Any, _row(result.world_proposals, 0)).signals
    bb = cast(Any, _row(result.world_proposals, 3)).signals
    pp = cast(Any, _row(result.world_proposals, 4)).signals
    _assert_contrast(result.attribution.contrasts.memory_total, mm, bb)
    _assert_contrast(result.attribution.contrasts.planner_total, pp, mm)
    _assert_contrast(result.attribution.contrasts.pp_minus_bb, pp, bb)
    assert bool(result.attribution.typed_signals_valid)
    assert bool(result.attribution.telescoping_valid)
    assert bool(result.causal_core_signal_contract_valid)
    assert not bool(jnp.any(result.world_proposals.signals.message_charge))
    assert not bool(jnp.any(result.world_proposals.signals.safety_cost))
    chex.assert_trees_all_equal(
        result.world_proposals.signals.net_reward,
        jnp.broadcast_to(result.world_proposals.signals.task_score[:, None], (8, 2)),
    )
    for contrast in (
        result.attribution.contrasts.memory_total,
        result.attribution.contrasts.memory_interaction,
        result.attribution.contrasts.planner_total,
        result.attribution.contrasts.planner_interaction,
        result.attribution.contrasts.pp_minus_bb,
    ):
        assert contrast.task_score.shape == ()
        assert contrast.net_reward.shape == (2,)
        assert contrast.safety_cost.shape == (2,)
        assert contrast.message_charge.shape == (2,)


def test_stale_tampered_cross_world_cross_event_and_failed_proposals_roll_back() -> None:
    adapter = _adapter()
    initial = adapter.init(jr.key(11))
    event = adapter.world.prepare_event(initial.world_state)
    receipts = _receipts(adapter, initial, event)
    direct = _stage(adapter, initial, event, receipts)
    assert bool(direct.update_applied)

    stale = _stage(adapter, direct.state, event, receipts)
    assert not bool(stale.update_applied)
    chex.assert_trees_all_equal(stale.state, direct.state)

    tampered_memory = dataclasses.replace(  # type: ignore[type-var]
        receipts[1],
        action_receipt_identity_words=receipts[1].action_receipt_identity_words.at[0, 0].add(1),
    )
    tampered = _stage(adapter, initial, event, (receipts[0], tampered_memory, receipts[2]))
    assert not bool(tampered.action_receipt_identities_bound)
    assert not bool(tampered.update_applied)
    chex.assert_trees_all_equal(tampered.state, initial)

    other = adapter.init(jr.key(13))
    other_event = adapter.world.prepare_event(other.world_state)
    other_receipts = _receipts(adapter, other, other_event)
    crossed_world = _stage(adapter, initial, other_event, other_receipts)
    assert not bool(crossed_world.world_source_clock_bound)
    assert not bool(crossed_world.update_applied)
    chex.assert_trees_all_equal(crossed_world.state, initial)

    crossed_event = _stage(adapter, initial, other_event, receipts)
    assert not bool(crossed_event.event_receipt_identity_bound)
    chex.assert_trees_all_equal(crossed_event.state, initial)

    invalid_base = dataclasses.replace(  # type: ignore[type-var]
        receipts[0],
        actions_after_mask=jnp.asarray([2, 0], dtype=jnp.int32),
    )
    failed = _stage(adapter, initial, event, (invalid_base, receipts[1], receipts[2]))
    assert not bool(failed.all_world_proposals_valid)
    assert not bool(failed.update_applied)
    chex.assert_trees_all_equal(failed.state, initial)


def test_shared_hard_mask_is_required_across_all_three_action_layers() -> None:
    adapter = _adapter()
    initial = adapter.init(jr.key(15))
    event = adapter.world.prepare_event(initial.world_state)
    base, memory, planner = _receipts(adapter, initial, event)
    divergent_mask = jnp.asarray(
        [[True, True], [True, False]],
        dtype=jnp.bool_,
    )
    rebound_memory = adapter.bind_action_receipt(
        initial,
        event,
        layer=HCCLActionLayer.MEMORY,
        actions_before_mask=memory.actions_before_mask,
        actions_after_mask=memory.actions_after_mask,
        hard_action_masks=divergent_mask,
        action_receipt_identity_words=memory.action_receipt_identity_words,
    )
    result = _stage(
        adapter,
        initial,
        event,
        (base, rebound_memory, planner),
    )

    assert not bool(result.action_receipt_identities_bound)
    assert not bool(result.attribution.action_receipts_valid)
    assert int(result.work.world_proposal_calls) == 8
    assert int(result.work.attribution_proposal_calls) == 0
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, initial)


def test_composite_link_rejects_same_clock_states_crossed_between_lives() -> None:
    adapter = _adapter()
    first = adapter.init(jr.key(101))
    second = adapter.init(jr.key(103))
    first_event = adapter.world.prepare_event(first.world_state)
    second_event = adapter.world.prepare_event(second.world_state)
    first_advanced = _stage(
        adapter,
        first,
        first_event,
        _receipts(adapter, first, first_event),
    ).state
    second_advanced = _stage(
        adapter,
        second,
        second_event,
        _receipts(adapter, second, second_event),
    ).state
    chex.assert_trees_all_equal(
        first_advanced.world_state.step_words,
        second_advanced.world_state.step_words,
    )
    assert not bool(
        jnp.all(
            first_advanced.attribution_state.last_attribution_tag_words
            == second_advanced.attribution_state.last_attribution_tag_words
        )
    )

    crossed = dataclasses.replace(  # type: ignore[type-var]
        first_advanced,
        attribution_state=second_advanced.attribution_state,
    )
    assert not bool(adapter.state_valid(crossed))
    with pytest.raises(ValueError, match="invalid"):
        save_hccl_world_attribution_checkpoint(adapter, crossed)


def test_downstream_rejection_is_bit_exact_and_same_receipt_retry_matches_direct() -> None:
    adapter = _adapter()
    initial = adapter.init(jr.key(17))
    event = adapter.world.prepare_event(initial.world_state)
    receipts = _receipts(adapter, initial, event)
    rejected = _stage(adapter, initial, event, receipts, gate=False)
    assert not bool(rejected.update_applied)
    assert int(rejected.work.world_proposal_calls) == 8
    assert int(rejected.work.discarded_world_proposal_calls) == 8
    assert int(rejected.work.committed_pp_world_successors) == 0
    chex.assert_trees_all_equal(rejected.state, initial)
    chex.assert_trees_all_equal(adapter.world.prepare_event(rejected.state.world_state), event)

    retry = _stage(adapter, rejected.state, event, receipts)
    direct = _stage(adapter, initial, event, receipts)
    chex.assert_trees_all_equal(retry, direct)


def test_eager_and_prebound_host_scan_are_atomic_with_explicit_jit_boundary() -> None:
    adapter = _adapter()
    initial = adapter.init(jr.key(19))
    event_0 = adapter.world.prepare_event(initial.world_state)
    receipts_0 = _receipts(adapter, initial, event_0)
    eager_0 = _stage(adapter, initial, event_0, receipts_0)
    with pytest.raises(TypeError, match="host/eager only"):
        jax.jit(
            lambda state, event, base, memory, planner: adapter.stage(
                state,
                event,
                base,
                memory,
                planner,
                downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
            )
        )(initial, event_0, *receipts_0)

    event_1 = adapter.world.prepare_event(eager_0.state.world_state)
    receipts_1 = _receipts(adapter, eager_0.state, event_1)
    events = cast(Any, _stack(event_0, event_1))
    base = cast(Any, _stack(receipts_0[0], receipts_1[0]))
    memory = cast(Any, _stack(receipts_0[1], receipts_1[1]))
    planner = cast(Any, _stack(receipts_0[2], receipts_1[2]))
    gates = jnp.ones((2,), dtype=jnp.bool_)
    scan_eager = run_hccl_world_attribution_scan(
        adapter,
        initial,
        events,
        base,
        memory,
        planner,
        gates,
    )
    with pytest.raises(TypeError, match="host/eager only"):
        jax.jit(run_hccl_world_attribution_scan, static_argnums=(0,))(
            adapter,
            initial,
            events,
            base,
            memory,
            planner,
            gates,
        )
    chex.assert_trees_all_equal(
        scan_eager.post_transaction_words,
        jnp.asarray([[0, 1], [0, 2]], dtype=jnp.uint32),
    )
    np.testing.assert_array_equal(np.asarray(scan_eager.update_applied), [True, True])


def test_strict_resources_and_in_memory_checkpoint_detect_tampering() -> None:
    adapter = _adapter()
    initial = adapter.init(jr.key(23))
    event = adapter.world.prepare_event(initial.world_state)
    result = _stage(adapter, initial, event, _receipts(adapter, initial, event))
    budget = adapter.resource_budget(result.state)
    assert budget.total_persistent_state_nbytes == measure_hccl_world_attribution_state_nbytes(
        result.state
    )
    assert budget.world_state_owners == 1
    assert budget.attribution_state_owners == 1
    assert budget.composite_link_nbytes == 16
    assert budget.total_persistent_state_nbytes == (
        budget.world_state_nbytes
        + budget.attribution_state_nbytes
        + budget.composite_link_nbytes
    )
    assert budget.max_world_proposal_calls_per_transaction == 8
    assert budget.max_committed_world_successors_per_transaction == 1
    assert budget.output_write_calls == 0

    checkpoint = save_hccl_world_attribution_checkpoint(adapter, result.state)
    restored_adapter, restored_state = load_hccl_world_attribution_checkpoint(checkpoint)
    assert restored_adapter.to_config() == adapter.to_config()
    chex.assert_trees_all_equal(restored_state, result.state)

    tampered = dataclasses.replace(
        checkpoint,
        state=cast(Any, checkpoint.state).replace(
            world_state=cast(Any, checkpoint.state.world_state).replace(
                positions=checkpoint.state.world_state.positions.at[0].add(jnp.float32(0.1))
            )
        ),
    )
    with pytest.raises(ValueError, match="checkpoint"):
        load_hccl_world_attribution_checkpoint(tampered)

    wrong_fixed_type = dataclasses.replace(cast(Any, checkpoint), output_writes_authorized=0)
    with pytest.raises(ValueError, match="output_writes_authorized"):
        load_hccl_world_attribution_checkpoint(cast(Any, wrong_fixed_type))

    wrong_state_bytes_type = dataclasses.replace(
        cast(Any, checkpoint),
        state_nbytes=float(checkpoint.state_nbytes),
    )
    with pytest.raises(TypeError, match="state_nbytes"):
        load_hccl_world_attribution_checkpoint(cast(Any, wrong_state_bytes_type))

    wrong_config_type = dict(checkpoint.config)
    wrong_config_type["schedule_execution_authorized"] = 0
    with pytest.raises(ValueError, match="unsupported"):
        load_hccl_world_attribution_checkpoint(
            dataclasses.replace(checkpoint, config=wrong_config_type)
        )

    wrong_budget_type = dict(checkpoint.resource_budget)
    wrong_budget_type["max_world_proposal_calls_per_transaction"] = 8.0
    with pytest.raises(ValueError, match="resource budget"):
        load_hccl_world_attribution_checkpoint(
            dataclasses.replace(checkpoint, resource_budget=wrong_budget_type)
        )
