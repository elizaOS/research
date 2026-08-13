"""Cheap contracts for the private primitive-HCCL operational runner path."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import pytest

import alberta_framework.core.hccl_continual_dyad_operational_runner as operational
from alberta_framework.core.hccl_causal_attribution import (
    HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


def test_operational_schemas_and_private_dependencies_are_explicit() -> None:
    assert operational.HCCL_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA.endswith(
        "operational-work.v1"
    )
    assert operational.HCCL_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA.endswith(
        "operational-transcript.v1"
    )
    assert operational.HCCL_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA.endswith(
        "operational-result.v1"
    )
    assert operational.HCCL_CONTINUAL_DYAD_OPERATIONAL_EVIDENCE_LEVEL == "L0"
    assert "audit-work-equivalence-is-explicitly-not-targeted" in (
        operational.HCCL_CONTINUAL_DYAD_OPERATIONAL_LIMITATIONS
    )
    assert set(operational.HCCL_CONTINUAL_DYAD_OPERATIONAL_PRIVATE_DEPENDENCIES) == {
        "HCCLContinualDyadTransaction._causal_core_memory_event_inputs",
        "HCCLContinualDyadTransaction._composed_observation",
        "HCCLContinualDyadTransaction._hard_action_masks",
        "HCCLContinualDyadTransaction._horde_targets",
        "HCCLContinualDyadTransaction._make_binding",
        "HCCLContinualDyadTransaction._memory_feedback",
        "HCCLContinualDyadTransaction._planner_binding_words",
        "HCCLContinualDyadTransaction._prototype",
        "HCCLContinualDyadTransaction._seal_state",
        "HCCLContinualDyadTransaction._state_contract",
        "HCCLContinualDyadTransaction._transition",
    }


def test_operational_work_is_distinct_strict_and_truthful() -> None:
    work = operational._make_operational_work(
        world_proposal_calls=8,
        attribution_proposal_calls=8,
        runner_checkpoint_state_validations=1,
    )
    assert type(work) is operational.HCCLContinualDyadOperationalWork
    assert work.world_event_preparations == 1
    assert work.action_binding_constructions == 1
    assert work.action_receipt_bindings == 3
    assert work.memory_metadata_derivations == 1
    assert work.memory_metadata_records == 2
    assert work.context_preparations == (1, 1)
    assert work.hccl_stage_calls == 1
    assert work.world_proposal_calls == 8
    assert work.attribution_proposal_calls == 8
    assert work.action_stack_memory_preparations == (1, 1)
    assert work.planner_completed_transition_calls == 1
    assert work.final_action_bindings == (1, 1)
    assert work.persistent_state_seals == 1
    assert work.runner_checkpoint_state_validations == 1
    assert work.through_memory_transaction_seals == 0
    assert work.prepared_transaction_seals == 0
    assert work.outer_preparation_receipts == 0
    assert work.child_integrity_receipts == (0, 0)
    assert work.child_adoption_calls == (0, 0)
    assert work.outer_prepared_reconstructions == 0
    assert work.child_finalization_reconstructions == (0, 0)
    assert work.audit_work_equivalence_targeted is False
    assert work.persistent_state_and_transcript_bit_equivalence_targeted is True
    payload = work.to_config()
    assert payload["schema"] == operational.HCCL_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA
    assert payload["context_steps"] == (1, 1)

    with pytest.raises(ValueError, match="world proposal calls"):
        operational._make_operational_work(
            world_proposal_calls=7,
            attribution_proposal_calls=8,
            runner_checkpoint_state_validations=0,
        )
    with pytest.raises(TypeError, match="checkpoint"):
        operational._make_operational_work(
            world_proposal_calls=8,
            attribution_proposal_calls=8,
            runner_checkpoint_state_validations=cast(int, True),
        )


def test_transcript_and_result_schemas_are_sufficient_but_not_audit_results() -> None:
    assert tuple(
        field.name
        for field in dataclasses.fields(
            operational.HCCLContinualDyadOperationalTranscript
        )
    ) == (
        "schema",
        "event",
        "binding",
        "pp_proposal",
        "pre_transaction_words",
        "post_transaction_words",
    )
    assert tuple(
        field.name
        for field in dataclasses.fields(
            operational.HCCLContinualDyadOperationalEventResult
        )
    ) == (
        "schema",
        "state",
        "transcript",
        "work",
        "update_applied",
    )


def test_local_exact_tree_comparison_handles_scalar_bit_patterns() -> None:
    negative_zero = {"scalar": jnp.asarray(-0.0, dtype=jnp.float32)}
    same_negative_zero = {"scalar": jnp.asarray(-0.0, dtype=jnp.float32)}
    positive_zero = {"scalar": jnp.asarray(0.0, dtype=jnp.float32)}
    assert operational._tree_exact_equal(negative_zero, same_negative_zero)
    assert not operational._tree_exact_equal(negative_zero, positive_zero)


def test_local_exact_tree_comparison_normalizes_typed_key_raw_bits() -> None:
    key = jr.key(17)
    same_key = jr.wrap_key_data(jr.key_data(key))
    different_key = jr.key(18)
    assert operational._tree_exact_equal({"key": key}, {"key": same_key})
    assert not operational._tree_exact_equal({"key": key}, {"key": different_key})


def test_operational_pp_slot_is_derived_from_the_owned_proposal_order() -> None:
    assert HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER[operational._PP_SLOT] == "PP-planner"
    module_source = inspect.getsource(operational)
    assert (
        '_PP_SLOT = HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER.index("PP-planner")'
        in module_source
    )


def _call_names(function: Any) -> list[str]:
    tree = ast.parse(inspect.getsource(function))
    return [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]


def test_operational_kernel_has_one_syntactic_learning_donor_site_and_no_audit_api() -> None:
    calls = _call_names(operational._execute_operational_event)
    forbidden = {
        "transaction.step",
        "transaction.prepare_event",
        "transaction.bind_current_actions",
        "transaction.causal_core_memory_event_inputs",
        "transaction.prepare_through_memory",
        "transaction.complete_with_factorized_planner",
        "transaction.prepare_transaction",
        "transaction.integrity_receipt",
        "transaction.adopt_prepared_transaction",
        "transaction.state_valid",
    }
    assert forbidden.isdisjoint(calls)
    for expected in (
        "transaction.hccl.world.prepare_event",
        "transaction._make_binding",
        "transaction._causal_core_memory_event_inputs",
        "transaction.context.prepare",
        "transaction.hccl.stage",
        "derive_hccl_memory_credit_estimands",
        "transaction._horde_targets",
        "transaction.context.step",
        "transaction._composed_observation",
        "transaction._memory_feedback",
        "transaction._transition",
        "adapters[index].prepare_memory_transition",
        "transaction.planner.completed_transition",
        "adapters[index].bind_final_action",
        "transaction._seal_state",
    ):
        assert calls.count(expected) == 1, expected


def _raw_result(state: object) -> operational.HCCLContinualDyadOperationalEventResult:
    result = object.__new__(operational.HCCLContinualDyadOperationalEventResult)
    object.__setattr__(
        result,
        "schema",
        operational.HCCL_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA,
    )
    object.__setattr__(result, "state", state)
    object.__setattr__(result, "transcript", object())
    object.__setattr__(result, "work", object())
    object.__setattr__(result, "update_applied", True)
    return result


def _raw_executor(
    *,
    transaction: object,
    state: object,
    checkpoint_interval: int | None,
) -> operational._HCCLContinualDyadOperationalExecutor:
    executor = object.__new__(operational._HCCLContinualDyadOperationalExecutor)
    object.__setattr__(executor, "_transaction", transaction)
    object.__setattr__(executor, "_state", state)
    object.__setattr__(executor, "_absolute_step", 0)
    object.__setattr__(executor, "_maximum_transitions", 10)
    object.__setattr__(executor, "_checkpoint_interval", checkpoint_interval)
    return executor


def test_operational_executor_does_not_publish_when_kernel_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = object()
    executor = _raw_executor(
        transaction=object(),
        state=source,
        checkpoint_interval=None,
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("donor rejected")

    monkeypatch.setattr(operational, "_execute_operational_event", fail)
    with pytest.raises(operational.HCCLContinualDyadOperationalError, match="donor rejected"):
        executor.step(cast(Any, object()))
    assert executor.state is source
    assert executor.absolute_step == 0


def test_operational_executor_checks_periodic_candidate_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RejectingTransaction:
        @staticmethod
        def state_valid(_state: object) -> bool:
            return False

    source = object()
    candidate = object()
    executor = _raw_executor(
        transaction=_RejectingTransaction(),
        state=source,
        checkpoint_interval=1,
    )
    monkeypatch.setattr(
        operational,
        "_execute_operational_event",
        lambda *_args, **_kwargs: _raw_result(candidate),
    )
    with pytest.raises(
        operational.HCCLContinualDyadOperationalError,
        match="checkpoint",
    ):
        executor.step(cast(Any, object()))
    assert executor.state is source
    assert executor.absolute_step == 0


def test_operational_executor_checks_final_candidate_without_periodic_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RejectingTransaction:
        @staticmethod
        def state_valid(_state: object) -> bool:
            return False

    source = object()
    candidate = object()
    executor = _raw_executor(
        transaction=_RejectingTransaction(),
        state=source,
        checkpoint_interval=None,
    )
    object.__setattr__(executor, "_absolute_step", 9)
    monkeypatch.setattr(
        operational,
        "_execute_operational_event",
        lambda *_args, **_kwargs: _raw_result(candidate),
    )
    with pytest.raises(
        operational.HCCLContinualDyadOperationalError,
        match="final-checkpoint",
    ):
        executor.step(cast(Any, object()))
    assert executor.state is source
    assert executor.absolute_step == 9
