# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Narrow raw-to-final STOMP owner trace and lifecycle-finalization contracts."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from test_stomp_option_lifecycle_borrowed_state import (
    _external_transition,
    _lifecycle,
    _started,
)

from alberta_framework.core import stomp_owner_finalization as finalization_module
from alberta_framework.core.stomp_owner_finalization import (
    STOMP_OWNER_STAGE_DYNA,
    STOMP_OWNER_STAGE_FEATURE_ROUTE,
    STOMP_OWNER_STAGE_MEMORY_DISPATCH,
    STOMP_OWNER_STAGE_OPTION_SEARCH,
    STOMP_OWNER_STAGE_PARTNER_DISPATCH,
    make_stomp_owner_finalization_trace,
    make_stomp_owner_stage_receipt,
    stomp_owner_finalization_trace_valid,
    stomp_typed_tree_digest,
)


def _raw_adoption_and_trace():
    api = _lifecycle()
    source = _started(api)
    metadata, raw, declaration, reward, next_observation = _external_transition(
        api,
        source,
    )
    adopted = api.adopt_external_stomp_update(
        metadata,
        source.stomp_state,
        raw,
        declaration,
        env_reward=reward,
        next_observation=next_observation,
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert bool(adopted.metadata_advanced)
    owner = raw.state
    digest = stomp_typed_tree_digest(owner)
    zero = jnp.asarray(0, dtype=jnp.int32)
    false = jnp.asarray(False, dtype=jnp.bool_)
    true = jnp.asarray(True, dtype=jnp.bool_)
    stages = tuple(
        make_stomp_owner_stage_receipt(
            owner,
            owner,
            stage_kind=kind,
            configured=false,
            evaluated=false,
            stomp_update_evaluations=zero,
            learner_updates_applied=zero,
            source_digest=digest,
            destination_digest=digest,
            classified_delta_valid=true,
        )
        for kind in (
            STOMP_OWNER_STAGE_OPTION_SEARCH,
            STOMP_OWNER_STAGE_FEATURE_ROUTE,
            STOMP_OWNER_STAGE_DYNA,
            STOMP_OWNER_STAGE_MEMORY_DISPATCH,
            STOMP_OWNER_STAGE_PARTNER_DISPATCH,
        )
    )
    trace = make_stomp_owner_finalization_trace(
        owner,
        stages,  # type: ignore[arg-type]
        owner,
        real_control_stomp_evaluations=jnp.asarray(1, dtype=jnp.int32),
        imagined_stomp_evaluations=zero,
        option_search_learner_updates=zero,
        raw_digest=digest,
        final_digest=digest,
    )
    return api, adopted, trace


def test_narrow_trace_jit_scan_and_metadata_only_finalization() -> None:
    api, adopted, trace = _raw_adoption_and_trace()

    def scan_trace(owner_trace):
        def step(carry, _):
            return carry, stomp_owner_finalization_trace_valid(carry)

        return jax.lax.scan(step, owner_trace, jnp.arange(2, dtype=jnp.int32))

    final_trace, accepted = jax.jit(scan_trace)(trace)
    chex.assert_trees_all_equal(final_trace, trace)
    assert bool(jnp.all(accepted))

    finalized = api.finalize_external_stomp_owner(adopted.state, trace)
    assert bool(finalized.raw_metadata_valid)
    assert bool(finalized.raw_owner_binding_matches)
    assert bool(finalized.final_owner_state_valid)
    assert bool(finalized.stage_trace_valid)
    assert bool(finalized.audit_state_preserved)
    assert bool(finalized.lifecycle_identity_preserved)
    assert bool(finalized.metadata_finalized)
    assert not bool(finalized.derivation_recomputed)
    assert not bool(finalized.caller_authenticated)
    assert int(finalized.state.revision) == int(adopted.state.revision)
    chex.assert_trees_all_equal(
        finalized.state.audit_state,
        adopted.state.audit_state,
    )
    attached = api.attach_borrowed_stomp(finalized.state, trace.final_state)
    assert bool(attached.transaction_applied)


def _reseal_stage(receipt):
    return receipt.replace(
        receipt_checksum=stomp_typed_tree_digest(
            finalization_module._stage_receipt_payload(receipt)
        )
    )


def _reseal_trace(trace):
    return trace.replace(
        trace_checksum=stomp_typed_tree_digest(
            finalization_module._trace_payload(trace)
        )
    )


def test_reordered_dropped_and_coherently_forged_stage_chains_fail_closed() -> None:
    api, adopted, trace = _raw_adoption_and_trace()

    reordered = trace.replace(
        stages=(trace.stages[1], trace.stages[0], *trace.stages[2:])
    )
    reordered = _reseal_trace(reordered)
    assert not bool(stomp_owner_finalization_trace_valid(reordered))
    assert not bool(
        api.finalize_external_stomp_owner(
            adopted.state,
            reordered,
        ).metadata_finalized
    )

    dropped = trace.replace(stages=trace.stages[:-1])
    try:
        stomp_owner_finalization_trace_valid(dropped)
    except ValueError as error:
        assert "exactly five" in str(error)
    else:  # pragma: no cover - static contract must raise
        raise AssertionError("dropped trace did not fail its static contract")

    forged_first = trace.stages[0].replace(
        destination_digest=trace.stages[0].destination_digest.at[0].add(
            jnp.uint32(1)
        )
    )
    forged_first = _reseal_stage(forged_first)
    forged = trace.replace(stages=(forged_first, *trace.stages[1:]))
    forged = _reseal_trace(forged)
    assert not bool(stomp_owner_finalization_trace_valid(forged))
    rejected = api.finalize_external_stomp_owner(adopted.state, forged)
    assert not bool(rejected.metadata_finalized)
    chex.assert_trees_all_equal(rejected.state, adopted.state)
