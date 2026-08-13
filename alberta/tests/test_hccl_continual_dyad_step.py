# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Owned one-event orchestration for the production HCCL continual dyad."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.hccl_continual_dyad_factory import (
    HCCLContinualDyadFactory,
    HCCLContinualDyadFactoryConfig,
)
from alberta_framework.core.hccl_continual_dyad_runner import (
    _ProductionLifeEventExecutor,
)
from alberta_framework.core.hccl_continual_dyad_transaction import (
    HCCLContinualDyadResult,
    HCCLContinualDyadState,
    HCCLContinualDyadTransaction,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_MASKS = jnp.ones((2, 2), dtype=jnp.bool_)


@dataclasses.dataclass(frozen=True, slots=True)
class _Rig:
    transaction: HCCLContinualDyadTransaction
    state: HCCLContinualDyadState


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> Iterator[None]:
    with jax.disable_jit():
        yield


@pytest.fixture(scope="module")
def rig() -> _Rig:
    with jax.disable_jit():
        initialized = HCCLContinualDyadFactory(
            HCCLContinualDyadFactoryConfig.mechanics_smoke()
        ).init(jr.key(1101, impl="threefry2x32"))
    return _Rig(transaction=initialized.transaction, state=initialized.state)


@pytest.fixture(scope="module")
def first_step(rig: _Rig) -> HCCLContinualDyadResult:
    with jax.disable_jit():
        return rig.transaction.step(rig.state, _MASKS)


@pytest.fixture(scope="module")
def second_step(
    rig: _Rig,
    first_step: HCCLContinualDyadResult,
) -> HCCLContinualDyadResult:
    with jax.disable_jit():
        return rig.transaction.step(first_step.state, _MASKS)


def _assert_tree_exact(actual: object, expected: object) -> None:
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        if isinstance(actual_leaf, (jax.Array, np.ndarray)) or isinstance(
            expected_leaf, (jax.Array, np.ndarray)
        ):
            actual_jax = jnp.asarray(actual_leaf)
            expected_jax = jnp.asarray(expected_leaf)
            if jax.dtypes.issubdtype(actual_jax.dtype, jax.dtypes.prng_key):
                assert jax.dtypes.issubdtype(expected_jax.dtype, jax.dtypes.prng_key)
                assert str(jr.key_impl(actual_jax)) == str(jr.key_impl(expected_jax))
                actual_jax = jr.key_data(actual_jax)
                expected_jax = jr.key_data(expected_jax)
            actual_array = np.asarray(actual_jax)
            expected_array = np.asarray(expected_jax)
            assert actual_array.shape == expected_array.shape
            assert actual_array.dtype == expected_array.dtype
            assert actual_array.tobytes(order="C") == expected_array.tobytes(order="C")
        else:
            assert type(actual_leaf) is type(expected_leaf)
            assert actual_leaf == expected_leaf


def _manual_step(
    transaction: HCCLContinualDyadTransaction,
    state: HCCLContinualDyadState,
) -> HCCLContinualDyadResult:
    event = transaction.prepare_event(state)
    binding = transaction.bind_current_actions(state, event)
    inputs = transaction.causal_core_memory_event_inputs(state, event)
    prepared = transaction.prepare_transaction(
        state,
        event,
        binding,
        inputs[0],
        inputs[1],
        _MASKS,
    )
    receipt = transaction.integrity_receipt(prepared)
    return transaction.adopt_prepared_transaction(state, prepared, receipt)


def _assert_canonical_memory_inputs(result: HCCLContinualDyadResult, step: int) -> None:
    prepared_agents = (result.prepared.agent_0, result.prepared.agent_1)
    for row, prepared_agent in enumerate(prepared_agents):
        event_input = prepared_agent.memory_preparation.event_input
        assert not bool(event_input.query_uncertainty_available)
        assert not bool(event_input.entry_uncertainty_available)
        assert bool(event_input.entry_safety_cost_available)
        assert int(event_input.provenance_id) == 2 * step + row
        assert int(event_input.source_id) == row
        np.testing.assert_array_equal(
            event_input.entry_reliability,
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        for zero in (
            event_input.query_uncertainty,
            event_input.entry_uncertainty,
            event_input.entry_safety_cost,
        ):
            assert np.asarray(zero).dtype == np.dtype(np.float32)
            assert int(np.asarray(zero).view(np.uint32)) == 0


def test_step_owns_every_intermediate_and_matches_the_manual_six_stage_chain(
    rig: _Rig,
    first_step: HCCLContinualDyadResult,
) -> None:
    parameters = inspect.signature(rig.transaction.step).parameters
    assert tuple(parameters) == ("state", "next_hard_action_masks")
    for forbidden in (
        "event",
        "binding",
        "agent_0_event_input",
        "agent_1_event_input",
        "prepared",
        "receipt",
        "candidate_evidence",
        "partner_policy_fusion_input",
        "partner_policy_fusion_feedback",
        "extended_action_masks",
    ):
        assert forbidden not in parameters

    manual = _manual_step(rig.transaction, rig.state)

    _assert_tree_exact(first_step, manual)
    _assert_tree_exact(first_step.prepared.source_state, rig.state)
    np.testing.assert_array_equal(
        first_step.receipt.prepared_content_tag_words,
        first_step.prepared.content_tag_words,
    )
    assert bool(first_step.update_applied)
    assert not bool(first_step.complete_source_returned)
    assert bool(rig.transaction.state_valid(first_step.state))
    np.testing.assert_array_equal(
        first_step.state.hccl_state.world_state.step_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    _assert_canonical_memory_inputs(first_step, 0)
    # The inner preparation truthfully records a supplied event: the public
    # step, rather than the inner transaction kernel, issued that receipt.
    assert int(first_step.prepared.work.supplied_event_receipts) == 1
    assert int(first_step.prepared.work.event_receipt_preparations) == 0


def test_two_owned_steps_advance_one_shared_clock_and_canonical_provenance(
    rig: _Rig,
    first_step: HCCLContinualDyadResult,
    second_step: HCCLContinualDyadResult,
) -> None:
    assert bool(first_step.update_applied)
    assert bool(second_step.update_applied)
    assert bool(rig.transaction.state_valid(second_step.state))
    _assert_canonical_memory_inputs(first_step, 0)
    _assert_canonical_memory_inputs(second_step, 1)
    np.testing.assert_array_equal(
        second_step.prepared.hccl_result.pre_transaction_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    np.testing.assert_array_equal(
        second_step.state.hccl_state.world_state.step_words,
        jnp.asarray((0, 2), dtype=jnp.uint32),
    )
    for child in (second_step.agent_0_adoption, second_step.agent_1_adoption):
        assert child is not None
        assert bool(child.diagnostics.transaction_applied)
    np.testing.assert_array_equal(
        second_step.action_stack_owners_committed,
        (True, True),
    )
    np.testing.assert_array_equal(second_step.context_owners_committed, (True, True))
    np.testing.assert_array_equal(second_step.lineage_owners_committed, (True, True))


def test_typed_child_veto_returns_source_and_untouched_owned_retry_commits(
    rig: _Rig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = rig.transaction
    original_adoption = transaction.agent_1.adopt_finalized_transition

    def reject_after_real_reconstruction(*args: Any, **kwargs: Any) -> object:
        child = original_adoption(*args, **kwargs)
        return child.replace(
            state=args[0],
            diagnostics=child.diagnostics.replace(
                transaction_applied=jnp.asarray(False, dtype=jnp.bool_),
                complete_source_returned=jnp.asarray(True, dtype=jnp.bool_),
                rejected=jnp.asarray(True, dtype=jnp.bool_),
            ),
        )

    with monkeypatch.context() as local:
        local.setattr(
            transaction.agent_1,
            "adopt_finalized_transition",
            reject_after_real_reconstruction,
        )
        refused = transaction.step(rig.state, _MASKS)

    _assert_tree_exact(refused.state, rig.state)
    assert not bool(refused.update_applied)
    assert bool(refused.complete_source_returned)
    np.testing.assert_array_equal(refused.child_adoptions_valid, (True, False))

    retry = transaction.step(rig.state, _MASKS)
    assert bool(retry.update_applied)
    assert not bool(retry.complete_source_returned)
    _assert_tree_exact(refused.prepared.event, retry.prepared.event)
    _assert_tree_exact(refused.prepared.binding, retry.prepared.binding)
    _assert_canonical_memory_inputs(retry, 0)


def test_step_fails_closed_before_mutation_on_malformed_masks_or_source(
    rig: _Rig,
) -> None:
    with pytest.raises((TypeError, ValueError), match="mask|shape"):
        rig.transaction.step(rig.state, jnp.ones((2, 3), dtype=jnp.bool_))

    tampered = rig.state.replace(content_token=jnp.zeros_like(rig.state.content_token))
    with pytest.raises(ValueError, match="valid continual-dyad state|valid"):
        rig.transaction.step(tampered, _MASKS)
    _assert_tree_exact(rig.state, rig.state)


def test_production_life_executor_delegates_once_to_the_owned_step(
    rig: _Rig,
    first_step: HCCLContinualDyadResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counted_step(
        self: HCCLContinualDyadTransaction,
        state: HCCLContinualDyadState,
        masks: jax.Array,
    ) -> HCCLContinualDyadResult:
        nonlocal calls
        calls += 1
        assert self is rig.transaction
        _assert_tree_exact(state, rig.state)
        np.testing.assert_array_equal(masks, _MASKS)
        return first_step

    with monkeypatch.context() as local:
        local.setattr(HCCLContinualDyadTransaction, "step", counted_step)
        executor = _ProductionLifeEventExecutor(
            rig.transaction,
            rig.state,
            total_steps=420,
        )
        event = executor.execute_event(0)

    assert calls == 1
    assert event.committed is True
    np.testing.assert_array_equal(event.pre_step_words, (0, 0))
    np.testing.assert_array_equal(event.post_step_words, (0, 1))
    final_state = executor.final_state
    assert type(final_state) is HCCLContinualDyadState
    np.testing.assert_array_equal(
        final_state.hccl_state.world_state.step_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
