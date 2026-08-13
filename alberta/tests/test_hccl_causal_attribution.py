"""Contracts for the mechanism-only HCCL adjacent-cube attribution rung."""

from __future__ import annotations

import dataclasses
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import alberta_framework
import alberta_framework.core.hccl_causal_attribution as attribution_module
from alberta_framework.core.checkpoints import load_checkpoint as load_generic_checkpoint
from alberta_framework.core.checkpoints import load_checkpoint_metadata
from alberta_framework.core.hccl_causal_attribution import (
    HCCL_CAUSAL_ATTRIBUTION_EVIDENCE_LEVEL,
    HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER,
    HCCLActionLayer,
    HCCLActionReceipt,
    HCCLCausalAttributionConfig,
    HCCLCausalAttributionKernel,
    HCCLCausalAttributionState,
    HCCLCausalSourceReceipt,
    HCCLExogenousReceipt,
    HCCLJointActionVertex,
    HCCLProposalCallback,
    HCCLTransitionProposal,
    HCCLTypedSignals,
    load_hccl_causal_attribution_checkpoint,
    measure_hccl_causal_attribution_state_nbytes,
    run_hccl_causal_attribution_scan,
    save_hccl_causal_attribution_checkpoint,
)

pytestmark = pytest.mark.integration


def _kernel(*, max_transactions: int = 2**64 - 1) -> HCCLCausalAttributionKernel:
    return HCCLCausalAttributionKernel(
        HCCLCausalAttributionConfig(
            source_dim=3,
            exogenous_dim=2,
            transition_dim=3,
            n_actions=2,
            proposal_owner_digest=tuple(range(8)),
            max_transactions=max_transactions,
            max_abs_source=100.0,
            max_abs_exogenous=100.0,
            max_abs_transition=100.0,
            max_abs_signal=100.0,
        )
    )


def _source(
    kernel: HCCLCausalAttributionKernel,
    state: HCCLCausalAttributionState,
) -> HCCLCausalSourceReceipt:
    return kernel.bind_source(
        state,
        source_vector=jnp.asarray([0.25, -0.5, 0.75], dtype=jnp.float32),
        source_identity_words=jnp.asarray([101, 103, 107, 109], dtype=jnp.uint32),
        agent_identity_words=jnp.asarray(
            [[11, 13, 17, 19], [23, 29, 31, 37]],
            dtype=jnp.uint32,
        ),
        raw_observation_identity_words=jnp.asarray(
            [[41, 43, 47, 53], [59, 61, 67, 71]],
            dtype=jnp.uint32,
        ),
        fast_state_words=jnp.asarray([[0, 5], [0, 7]], dtype=jnp.uint32),
        slow_context_birth_words=jnp.asarray(
            [[73, 79, 83, 89], [97, 101, 103, 107]],
            dtype=jnp.uint32,
        ),
        feature_birth_words=jnp.asarray(
            [[109, 113, 127, 131], [137, 139, 149, 151]],
            dtype=jnp.uint32,
        ),
        memory_generation_words=jnp.asarray([[0, 11], [0, 13]], dtype=jnp.uint32),
        planner_model_words=jnp.asarray([[0, 17], [0, 19]], dtype=jnp.uint32),
        hard_mask_generation_words=jnp.asarray([[0, 23], [0, 29]], dtype=jnp.uint32),
        rng_receipt_identity_words=jnp.asarray(
            [[157, 163, 167, 173], [179, 181, 191, 193]],
            dtype=jnp.uint32,
        ),
    )


def _bound_inputs(
    kernel: HCCLCausalAttributionKernel,
    state: HCCLCausalAttributionState,
    *,
    actions: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
        (0, 0),
        (1, 0),
        (1, 1),
    ),
) -> tuple[
    HCCLCausalSourceReceipt,
    HCCLExogenousReceipt,
    tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
]:
    source = _source(kernel, state)
    exogenous = kernel.bind_exogenous(
        source,
        exogenous_identity_words=jnp.asarray([197, 199, 211, 223], dtype=jnp.uint32),
        exogenous_source_words=jnp.asarray([0, 31], dtype=jnp.uint32),
        payload=jnp.asarray([0.5, -0.25], dtype=jnp.float32),
    )
    masks = jnp.ones((2, 2), dtype=jnp.bool_)
    receipts = tuple(
        kernel.bind_action_receipt(
            source,
            exogenous,
            layer=layer,
            actions_before_mask=jnp.asarray(layer_actions, dtype=jnp.int32),
            actions_after_mask=jnp.asarray(layer_actions, dtype=jnp.int32),
            hard_action_masks=masks,
            action_receipt_identity_words=jnp.asarray(
                [
                    [227 + 20 * int(layer), 229, 233, 239],
                    [241 + 20 * int(layer), 251, 257, 263],
                ],
                dtype=jnp.uint32,
            ),
        )
        for layer, layer_actions in zip(
            (HCCLActionLayer.BASE, HCCLActionLayer.MEMORY, HCCLActionLayer.PLANNER),
            actions,
            strict=True,
        )
    )
    return (
        source,
        exogenous,
        cast(
            tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
            receipts,
        ),
    )


def _signals(vertex_actions: jax.Array, exogenous_payload: jax.Array) -> HCCLTypedSignals:
    task = (
        vertex_actions[0].astype(jnp.float32)
        + jnp.float32(2.0) * vertex_actions[1].astype(jnp.float32)
        + exogenous_payload[0]
    )
    safety = jnp.asarray(
        [
            jnp.float32(0.25) * vertex_actions[0],
            jnp.float32(0.25) * vertex_actions[1],
        ],
        dtype=jnp.float32,
    )
    message = jnp.asarray([0.25, 0.5], dtype=jnp.float32)
    return HCCLTypedSignals(  # type: ignore[call-arg]
        task_score=task,
        net_reward=task - safety - message,
        safety_cost=safety,
        message_charge=message,
    )


def _proposal_callback(
    kernel: HCCLCausalAttributionKernel,
    calls: list[int] | None = None,
) -> HCCLProposalCallback:
    def callback(
        source: HCCLCausalSourceReceipt,
        exogenous: HCCLExogenousReceipt,
        vertex: HCCLJointActionVertex,
        slot: jax.Array,
    ) -> HCCLTransitionProposal:
        if calls is not None:
            calls.append(int(slot))
        candidate = source.source_vector + jnp.asarray(
            [
                vertex.actions[0].astype(jnp.float32),
                vertex.actions[1].astype(jnp.float32),
                exogenous.payload[1],
            ],
            dtype=jnp.float32,
        )
        return kernel.bind_proposal(
            source,
            exogenous,
            vertex,
            candidate_transition=candidate,
            signals=_signals(vertex.actions, exogenous.payload),
            accepted=jnp.asarray(True, dtype=jnp.bool_),
        )

    return callback


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            dtype, jax.dtypes.prng_key
        ):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def test_config_is_strict_public_and_has_no_execution_or_claim_authority() -> None:
    kernel = _kernel()
    payload = kernel.to_config()
    assert payload["evidence_level"] == HCCL_CAUSAL_ATTRIBUTION_EVIDENCE_LEVEL == "L0"
    assert payload["mechanism_status"] == "mechanism-only"
    assert payload["hccl_execution_authorized"] is False
    assert payload["environment_implemented"] is False
    assert payload["noise_and_seed_semantics_pinned"] is False
    assert payload["artifact_or_claim_authority"] is False
    assert payload["proposal_order"] == list(HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER)
    assert "host-preflight-cannot-suppress-traced-callback-staging" in payload["limitations"]
    assert (
        "proposal-callback-purity-and-side-effect-freedom-are-caller-obligations"
        in payload["limitations"]
    )
    assert (
        "generic-kernel-does-not-enforce-equal-payloads-for-equal-effective-actions"
        in payload["limitations"]
    )
    assert HCCLCausalAttributionKernel.from_config(payload).to_config() == payload
    assert alberta_framework.HCCLCausalAttributionKernel is HCCLCausalAttributionKernel

    malformed = dict(payload)
    malformed["hccl_execution_authorized"] = True
    with pytest.raises(ValueError):
        HCCLCausalAttributionKernel.from_config(malformed)


@pytest.mark.parametrize(
    ("field", "equal_but_wrong_type"),
    (
        ("hccl_execution_authorized", 0),
        ("proposal_calls_per_valid_transaction", 8.0),
        ("max_unique_effective_joint_actions", 4.0),
        ("max_abs_source", 100),
    ),
)
def test_config_rejects_python_equal_values_with_noncanonical_types(
    field: str,
    equal_but_wrong_type: object,
) -> None:
    payload = _kernel().to_config()
    payload[field] = equal_but_wrong_type

    with pytest.raises((TypeError, ValueError), match="exact float|noncanonical"):
        HCCLCausalAttributionKernel.from_config(payload)


def _reseal_persisted_state(
    kernel: HCCLCausalAttributionKernel,
    state: HCCLCausalAttributionState,
    *,
    proposal: HCCLTransitionProposal | None = None,
    contrasts: Any | None = None,
) -> HCCLCausalAttributionState:
    selected_proposal = state.last_committed_pp if proposal is None else proposal
    selected_proposal = selected_proposal.replace(  # type: ignore[attr-defined]
        content_tag_words=kernel._proposal_tag(selected_proposal)  # noqa: SLF001
    )
    selected_contrasts = state.last_contrasts if contrasts is None else contrasts
    return cast(
        HCCLCausalAttributionState,
        state.replace(  # type: ignore[attr-defined]
            last_committed_pp=selected_proposal,
            last_contrasts=selected_contrasts,
            last_attribution_tag_words=attribution_module._attribution_tag(  # noqa: SLF001
                kernel._owner,  # noqa: SLF001
                state.transaction_words,
                selected_proposal,
                selected_contrasts,
            ),
        ),
    )


def test_fixed_eight_call_order_duplicate_mm_and_pp_only_commit() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)
    calls: list[int] = []
    result = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        _proposal_callback(kernel, calls),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert calls == list(range(8))
    assert bool(result.update_applied)
    assert bool(result.duplicate_mm_bit_exact)
    assert int(result.work.proposal_calls) == 8
    assert int(result.work.designated_counterfactual_slots) == 7
    assert int(result.work.discarded_proposal_calls) == 7
    assert int(result.work.committed_pp_calls) == 1
    assert int(result.unique_joint_action_vertices) == 3
    assert int(result.unique_joint_action_receipt_vertices) == 7
    assert int(result.committed_slot) == 4
    chex.assert_trees_all_equal(
        result.state.last_committed_pp,
        jax.tree.map(lambda leaf: leaf[4], result.proposals),
    )
    chex.assert_trees_all_equal(
        result.state.transaction_words,
        jnp.asarray([0, 1], dtype=jnp.uint32),
    )


def test_retagged_impossible_persisted_pp_and_message_contrasts_are_rejected(
    tmp_path: Path,
) -> None:
    kernel = _kernel()
    initial = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, initial)
    committed = kernel.stage(
        initial,
        source,
        exogenous,
        base,
        memory,
        planner,
        _proposal_callback(kernel),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    ).state
    assert bool(kernel.state_valid(committed))
    pp = committed.last_committed_pp

    impossible_predecessor = pp.replace(  # type: ignore[attr-defined]
        source_decision_words=committed.transaction_words,
        source_transition_words=committed.transaction_words,
    )
    wrong_decision = pp.replace(  # type: ignore[attr-defined]
        source_decision_words=pp.source_decision_words.at[1].add(jnp.uint32(7))
    )
    wrong_layers = pp.replace(  # type: ignore[attr-defined]
        vertex=pp.vertex.replace(  # type: ignore[attr-defined]
            layer_codes=jnp.asarray((0, 0), dtype=jnp.int32)
        )
    )
    out_of_range_action = pp.replace(  # type: ignore[attr-defined]
        vertex=pp.vertex.replace(  # type: ignore[attr-defined]
            actions=pp.vertex.actions.at[0].set(jnp.int32(2))
        )
    )
    duplicate_receipt_identity = pp.replace(  # type: ignore[attr-defined]
        vertex=pp.vertex.replace(  # type: ignore[attr-defined]
            action_receipt_identity_words=pp.vertex.action_receipt_identity_words.at[1].set(
                pp.vertex.action_receipt_identity_words[0]
            )
        )
    )
    invalid_states = tuple(
        _reseal_persisted_state(kernel, committed, proposal=item)
        for item in (
            impossible_predecessor,
            wrong_decision,
            wrong_layers,
            out_of_range_action,
            duplicate_receipt_identity,
        )
    )
    for invalid in invalid_states:
        assert not bool(kernel.state_valid(invalid))

    one_message = jnp.asarray((1.0, 0.0), dtype=jnp.float32)
    message_contrasts = committed.last_contrasts.replace(  # type: ignore[attr-defined]
        memory_total=committed.last_contrasts.memory_total.replace(  # type: ignore[attr-defined]
            message_charge=one_message
        ),
        pp_minus_bb=committed.last_contrasts.pp_minus_bb.replace(  # type: ignore[attr-defined]
            message_charge=one_message
        ),
        telescoping_sum=committed.last_contrasts.telescoping_sum.replace(  # type: ignore[attr-defined]
            message_charge=one_message
        ),
    )
    assert bool(attribution_module._telescoping_roundoff_valid(message_contrasts))  # noqa: SLF001
    invalid_messages = _reseal_persisted_state(
        kernel,
        committed,
        contrasts=message_contrasts,
    )
    assert not bool(kernel.state_valid(invalid_messages))

    with pytest.raises(ValueError, match="invalid"):
        save_hccl_causal_attribution_checkpoint(
            kernel,
            invalid_states[0],
            tmp_path / "invalid-retagged-state",
        )


def test_typed_signal_contrasts_and_telescoping_are_exact() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)
    result = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        _proposal_callback(kernel),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    contrasts = result.contrasts
    assert float(contrasts.memory_total.task_score) == pytest.approx(1.0)
    assert float(contrasts.memory_interaction.task_score) == pytest.approx(0.0)
    assert float(contrasts.planner_total.task_score) == pytest.approx(2.0)
    assert float(contrasts.planner_interaction.task_score) == pytest.approx(0.0)
    assert float(contrasts.pp_minus_bb.task_score) == pytest.approx(3.0)
    chex.assert_trees_all_equal(
        contrasts.pp_minus_bb,
        contrasts.telescoping_sum,
    )
    chex.assert_trees_all_equal(
        contrasts.telescoping_residual,
        jax.tree.map(jnp.zeros_like, contrasts.telescoping_residual),
    )
    assert bool(result.telescoping_valid)
    assert bool(result.typed_signals_valid)


def test_nonassociative_float32_telescoping_uses_a_bounded_roundoff_audit() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)
    tasks = jnp.asarray(
        (
            27.392338,
            0.0,
            0.0,
            -46.042656,
            -91.8053,
            0.0,
            0.0,
            27.392338,
        ),
        dtype=jnp.float32,
    )

    def callback(
        source_row: HCCLCausalSourceReceipt,
        exogenous_row: HCCLExogenousReceipt,
        vertex: HCCLJointActionVertex,
        slot: jax.Array,
    ) -> HCCLTransitionProposal:
        task = tasks[slot]
        zeros = jnp.zeros((2,), dtype=jnp.float32)
        signals = HCCLTypedSignals(  # type: ignore[call-arg]
            task_score=task,
            net_reward=jnp.broadcast_to(task, (2,)),
            safety_cost=zeros,
            message_charge=zeros,
        )
        return kernel.bind_proposal(
            source_row,
            exogenous_row,
            vertex,
            candidate_transition=source_row.source_vector,
            signals=signals,
            accepted=jnp.asarray(True, dtype=jnp.bool_),
        )

    result = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        callback,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )

    assert float(result.contrasts.telescoping_residual.task_score) != 0.0
    assert bool(result.telescoping_valid)
    assert bool(result.update_applied)
    assert bool(kernel.state_valid(result.state))


def test_telescoping_roundoff_envelope_has_finite_inside_and_outside_boundaries() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)
    ordinary = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        _proposal_callback(kernel),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    ).contrasts

    def with_task(value: Any, task: float) -> Any:
        return value.replace(task_score=jnp.asarray(task, dtype=jnp.float32))

    shared = {
        "memory_total": with_task(ordinary.memory_total, 1.0e6),
        "planner_total": with_task(ordinary.planner_total, -1.0e6),
        "telescoping_sum": with_task(ordinary.telescoping_sum, 0.0),
    }
    inside = ordinary.replace(  # type: ignore[attr-defined]
        **shared,
        pp_minus_bb=with_task(ordinary.pp_minus_bb, 0.5),
        telescoping_residual=with_task(ordinary.telescoping_residual, 0.5),
    )
    outside = ordinary.replace(  # type: ignore[attr-defined]
        **shared,
        pp_minus_bb=with_task(ordinary.pp_minus_bb, 1.0),
        telescoping_residual=with_task(ordinary.telescoping_residual, 1.0),
    )
    nonfinite = ordinary.replace(  # type: ignore[attr-defined]
        **shared,
        pp_minus_bb=with_task(ordinary.pp_minus_bb, float("inf")),
        telescoping_residual=with_task(ordinary.telescoping_residual, float("inf")),
    )

    assert bool(attribution_module._telescoping_roundoff_valid(inside))  # noqa: SLF001
    assert not bool(attribution_module._telescoping_roundoff_valid(outside))  # noqa: SLF001
    assert not bool(attribution_module._telescoping_roundoff_valid(nonfinite))  # noqa: SLF001


def test_exogenous_receipt_is_bound_to_exact_source_content() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, receipts = _bound_inputs(kernel, state)
    crossed_source = kernel.bind_source(
        state,
        source_vector=source.source_vector.at[0].add(jnp.float32(0.125)),
        source_identity_words=source.source_identity_words,
        agent_identity_words=source.agent_identity_words,
        raw_observation_identity_words=source.raw_observation_identity_words,
        fast_state_words=source.fast_state_words,
        slow_context_birth_words=source.slow_context_birth_words,
        feature_birth_words=source.feature_birth_words,
        memory_generation_words=source.memory_generation_words,
        planner_model_words=source.planner_model_words,
        hard_mask_generation_words=source.hard_mask_generation_words,
        rng_receipt_identity_words=source.rng_receipt_identity_words,
    )
    crossed_receipts = tuple(
        kernel.bind_action_receipt(
            crossed_source,
            exogenous,
            layer=layer,
            actions_before_mask=receipt.actions_before_mask,
            actions_after_mask=receipt.actions_after_mask,
            hard_action_masks=receipt.hard_action_masks,
            action_receipt_identity_words=receipt.action_receipt_identity_words,
        )
        for layer, receipt in zip(
            (HCCLActionLayer.BASE, HCCLActionLayer.MEMORY, HCCLActionLayer.PLANNER),
            receipts,
            strict=True,
        )
    )
    crossed_base, crossed_memory, crossed_planner = cast(
        tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
        crossed_receipts,
    )
    calls: list[int] = []
    result = kernel.stage(
        state,
        crossed_source,
        exogenous,
        crossed_base,
        crossed_memory,
        crossed_planner,
        _proposal_callback(kernel, calls),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )

    assert calls == []
    assert not bool(result.exogenous_receipt_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_all_action_layers_require_one_bit_exact_hard_mask() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)
    divergent_mask = jnp.asarray(
        [[True, True], [True, False]],
        dtype=jnp.bool_,
    )
    rebound_memory = kernel.bind_action_receipt(
        source,
        exogenous,
        layer=HCCLActionLayer.MEMORY,
        actions_before_mask=memory.actions_before_mask,
        actions_after_mask=memory.actions_after_mask,
        hard_action_masks=divergent_mask,
        action_receipt_identity_words=memory.action_receipt_identity_words,
    )
    calls: list[int] = []
    result = kernel.stage(
        state,
        source,
        exogenous,
        base,
        rebound_memory,
        planner,
        _proposal_callback(kernel, calls),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )

    assert calls == []
    assert not bool(result.action_receipts_valid)
    assert not bool(result.update_applied)


def test_fixed_message_delivery_cannot_vary_across_counterfactual_actions() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)

    def callback(
        source_row: HCCLCausalSourceReceipt,
        exogenous_row: HCCLExogenousReceipt,
        vertex: HCCLJointActionVertex,
        slot: jax.Array,
    ) -> HCCLTransitionProposal:
        ordinary = _signals(vertex.actions, exogenous_row.payload)
        message = ordinary.message_charge.at[0].add(
            jnp.where(slot == 4, jnp.float32(0.125), jnp.float32(0.0))
        )
        signals = HCCLTypedSignals(  # type: ignore[call-arg]
            task_score=ordinary.task_score,
            net_reward=ordinary.task_score - ordinary.safety_cost - message,
            safety_cost=ordinary.safety_cost,
            message_charge=message,
        )
        return kernel.bind_proposal(
            source_row,
            exogenous_row,
            vertex,
            candidate_transition=source_row.source_vector,
            signals=signals,
            accepted=jnp.asarray(True, dtype=jnp.bool_),
        )

    result = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        callback,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )

    assert bool(result.all_child_proposals_valid)
    assert not bool(result.typed_signals_valid)
    assert int(result.work.discarded_proposal_calls) == 8
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_invalid_stale_or_identity_aliased_receipts_invoke_zero_callbacks() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)
    tampered = dataclasses.replace(  # type: ignore[type-var]
        memory,
        actions_after_mask=jnp.asarray([0, 1], dtype=jnp.int32),
    )
    calls: list[int] = []
    rejected = kernel.stage(
        state,
        source,
        exogenous,
        base,
        tampered,
        planner,
        _proposal_callback(kernel, calls),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert calls == []
    assert bool(rejected.host_preflight_performed)
    assert not bool(rejected.preflight_valid)
    assert int(rejected.work.proposal_calls) == 0
    chex.assert_trees_all_equal(rejected.state, state)

    aliased_memory = kernel.bind_action_receipt(
        source,
        exogenous,
        layer=HCCLActionLayer.MEMORY,
        actions_before_mask=memory.actions_before_mask,
        actions_after_mask=memory.actions_after_mask,
        hard_action_masks=memory.hard_action_masks,
        action_receipt_identity_words=base.action_receipt_identity_words,
    )
    rejected = kernel.stage(
        state,
        source,
        exogenous,
        base,
        aliased_memory,
        planner,
        _proposal_callback(kernel, calls),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert calls == []
    assert not bool(rejected.receipt_identities_distinct)

    stale_source = dataclasses.replace(  # type: ignore[type-var]
        source,
        decision_words=jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    rejected = kernel.stage(
        state,
        stale_source,
        exogenous,
        base,
        memory,
        planner,
        _proposal_callback(kernel, calls),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert calls == []
    chex.assert_trees_all_equal(rejected.state, state)


def test_child_or_post_staging_candidate_failure_rolls_back_after_all_calls() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)
    calls: list[int] = []
    valid_callback = _proposal_callback(kernel, calls)

    def divergent_mm(
        source_row: HCCLCausalSourceReceipt,
        exogenous_row: HCCLExogenousReceipt,
        vertex: HCCLJointActionVertex,
        slot: jax.Array,
    ) -> HCCLTransitionProposal:
        proposal = valid_callback(source_row, exogenous_row, vertex, slot)
        return cast(
            HCCLTransitionProposal,
            jax.lax.cond(
                slot == 7,
                lambda item: cast(Any, item).replace(
                    candidate_transition=item.candidate_transition.at[0].add(jnp.float32(1.0))
                ),
                lambda item: item,
                proposal,
            ),
        )

    rejected = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        divergent_mm,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert calls == list(range(8))
    assert not bool(rejected.duplicate_mm_bit_exact)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, state)

    calls.clear()
    rejected = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        _proposal_callback(kernel, calls),
        downstream_candidate_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert calls == list(range(8))
    assert bool(rejected.all_child_proposals_valid)
    assert not bool(rejected.downstream_candidate_valid)
    chex.assert_trees_all_equal(rejected.state, state)


def test_action_equality_never_substitutes_for_receipt_identity() -> None:
    kernel = _kernel()
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(
        kernel,
        state,
        actions=((1, 1), (1, 1), (1, 1)),
    )
    result = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        _proposal_callback(kernel),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(result.update_applied)
    assert int(result.unique_joint_action_vertices) == 1
    assert int(result.unique_joint_action_receipt_vertices) == 7
    assert int(result.work.proposal_calls) == 8


def test_exact_capacity_resources_and_checkpoint(tmp_path: Path) -> None:
    kernel = _kernel(max_transactions=1)
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)
    first = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        _proposal_callback(kernel),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(first.update_applied)
    budget = kernel.resource_budget(first.state)
    assert budget.total_state_nbytes == measure_hccl_causal_attribution_state_nbytes(first.state)
    assert budget.max_proposal_calls_per_transaction == 8
    assert budget.designated_counterfactual_slots_per_transaction == 7
    assert budget.max_discarded_proposal_calls_per_transaction == 8
    assert budget.max_unique_effective_joint_actions == 4
    assert budget.max_unique_joint_action_receipt_vertices == 7

    checkpoint = tmp_path / "hccl-attribution"
    save_hccl_causal_attribution_checkpoint(kernel, first.state, checkpoint)
    restored_kernel, restored_state = load_hccl_causal_attribution_checkpoint(checkpoint)
    assert restored_kernel.to_config() == kernel.to_config()
    chex.assert_trees_all_equal(restored_state, first.state)

    next_source, next_exogenous, (next_base, next_memory, next_planner) = _bound_inputs(
        kernel,
        first.state,
    )
    calls: list[int] = []
    exhausted = kernel.stage(
        first.state,
        next_source,
        next_exogenous,
        next_base,
        next_memory,
        next_planner,
        _proposal_callback(kernel, calls),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert calls == []
    assert not bool(exhausted.lifetime_capacity_available)
    chex.assert_trees_all_equal(exhausted.state, first.state)


def test_checkpoint_rejects_equal_valued_wrong_types_and_second_read_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _kernel(max_transactions=2)
    state = kernel.init()
    source, exogenous, (base, memory, planner) = _bound_inputs(kernel, state)
    committed = kernel.stage(
        state,
        source,
        exogenous,
        base,
        memory,
        planner,
        _proposal_callback(kernel),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    ).state
    checkpoint = tmp_path / "strict-hccl-attribution"
    save_hccl_causal_attribution_checkpoint(kernel, committed, checkpoint)
    metadata = load_checkpoint_metadata(checkpoint)

    wrong_fixed = deepcopy(metadata)
    wrong_fixed["hccl_execution_authorized"] = 0
    with monkeypatch.context() as scoped:
        scoped.setattr(
            attribution_module,
            "load_checkpoint_metadata",
            lambda _path: wrong_fixed,
        )
        with pytest.raises(ValueError, match="hccl_execution_authorized"):
            load_hccl_causal_attribution_checkpoint(checkpoint)

    wrong_config = deepcopy(metadata)
    wrong_config["kernel_config"]["max_unique_effective_joint_actions"] = 4.0
    wrong_config["config_sha256"] = attribution_module._canonical_digest(  # noqa: SLF001
        wrong_config["kernel_config"]
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            attribution_module,
            "load_checkpoint_metadata",
            lambda _path: wrong_config,
        )
        with pytest.raises(ValueError, match="noncanonical"):
            load_hccl_causal_attribution_checkpoint(checkpoint)

    wrong_resource = deepcopy(metadata)
    wrong_resource["resource_budget"]["max_committed_calls_per_transaction"] = True
    with monkeypatch.context() as scoped:
        scoped.setattr(
            attribution_module,
            "load_checkpoint_metadata",
            lambda _path: wrong_resource,
        )
        with pytest.raises(ValueError, match="resource budget"):
            load_hccl_causal_attribution_checkpoint(checkpoint)

    original_load = load_generic_checkpoint

    def load_with_equal_valued_type_drift(template: Any, path: Any) -> tuple[Any, Any]:
        restored, second_metadata = original_load(template, path)
        changed = deepcopy(second_metadata)
        changed["artifact_or_claim_authority"] = 0
        return restored, changed

    with monkeypatch.context() as scoped:
        scoped.setattr(
            attribution_module,
            "load_checkpoint",
            load_with_equal_valued_type_drift,
        )
        with pytest.raises(ValueError, match="metadata changed"):
            load_hccl_causal_attribution_checkpoint(checkpoint)


def test_eager_jit_valid_kernel_and_scan_parity() -> None:
    kernel = _kernel()
    initial = kernel.init()
    source_0, exogenous_0, (base_0, memory_0, planner_0) = _bound_inputs(kernel, initial)
    callback = _proposal_callback(kernel)
    eager = kernel.stage(
        initial,
        source_0,
        exogenous_0,
        base_0,
        memory_0,
        planner_0,
        callback,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    compiled = jax.jit(
        lambda source_state, source, exogenous, base, memory, planner: kernel.stage(
            source_state,
            source,
            exogenous,
            base,
            memory,
            planner,
            callback,
            downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
        )
    )(
        initial,
        source_0,
        exogenous_0,
        base_0,
        memory_0,
        planner_0,
    )
    eager_without_host_preflight = dataclasses.replace(  # type: ignore[type-var]
        eager,
        host_preflight_performed=jnp.asarray(False, dtype=jnp.bool_),
    )
    chex.assert_trees_all_close(
        _materialize_keys(eager_without_host_preflight),
        _materialize_keys(compiled),
        rtol=1e-6,
        atol=1e-7,
    )
    assert not bool(compiled.host_preflight_performed)

    source_1, exogenous_1, (base_1, memory_1, planner_1) = _bound_inputs(kernel, eager.state)
    batched_source = jax.tree.map(lambda left, right: jnp.stack((left, right)), source_0, source_1)
    batched_exogenous = jax.tree.map(
        lambda left, right: jnp.stack((left, right)),
        exogenous_0,
        exogenous_1,
    )
    batched_base, batched_memory, batched_planner = tuple(
        jax.tree.map(lambda left, right: jnp.stack((left, right)), first, second)
        for first, second in zip(
            (base_0, memory_0, planner_0),
            (base_1, memory_1, planner_1),
            strict=True,
        )
    )
    valid = jnp.ones((2,), dtype=jnp.bool_)
    scan_eager = run_hccl_causal_attribution_scan(
        kernel,
        initial,
        batched_source,
        batched_exogenous,
        batched_base,
        batched_memory,
        batched_planner,
        valid,
        callback,
    )
    scan_compiled = jax.jit(
        lambda state, sources, exogenous, base, memory, planner, gates: (
            run_hccl_causal_attribution_scan(
                kernel,
                state,
                sources,
                exogenous,
                base,
                memory,
                planner,
                gates,
                callback,
            )
        )
    )(
        initial,
        batched_source,
        batched_exogenous,
        batched_base,
        batched_memory,
        batched_planner,
        valid,
    )
    chex.assert_trees_all_close(scan_eager, scan_compiled, rtol=1e-6, atol=1e-7)
    chex.assert_trees_all_equal(
        scan_eager.transaction_words,
        jnp.asarray([[0, 1], [0, 2]], dtype=jnp.uint32),
    )
