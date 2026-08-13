# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""V2 atomic retirement/replacement feasibility contracts."""

from __future__ import annotations

import dataclasses

import chex
import jax.numpy as jnp
import numpy as np
import pytest
from test_authorized_option_replacement import (
    _context,
    _transition,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _controller_type() -> type[object]:
    from alberta_framework.core.authorized_option_atomic_swap import (
        AuthorizedOptionAtomicSwapController,
    )

    return AuthorizedOptionAtomicSwapController


def test_atomic_swap_never_persists_cold_slot_without_fresh_source_bound_semantic() -> None:
    controller_type = _controller_type()
    context = _context(install_fresh_alternate=False)
    controller = controller_type(context.controller)
    arm_inputs, observation, live = _transition(
        context.retired_state.scheduler_state,
        context.next_step,
    )
    prepared = controller.prepare(
        context.pre_retirement_state,
        context.retirement_handoff,
        context.retirement_authority,
        context.phase_one_key,
        context.phase_two_key,
        arm_inputs,
        observation,
        live,
    )

    assert bool(prepared.diagnostics.source_state_valid)
    assert bool(prepared.diagnostics.transient_retirement_applied)
    assert not bool(prepared.diagnostics.fresh_candidate_available)
    assert not bool(prepared.diagnostics.atomic_swap_ready)
    assert not bool(jnp.any(prepared.replacement_prepared.changed_slots))

    authority = controller.authority_receipt(
        prepared,
        context.installation_authority,
        swap_authorized=True,
    )
    result = controller.commit(context.pre_retirement_state, prepared, authority)
    assert not bool(result.transaction_applied)
    assert not bool(result.cold_state_persisted)
    assert not bool(result.replacement_applied)
    chex.assert_trees_all_equal(result.state, context.pre_retirement_state)


def test_fresh_atomic_swap_is_all_installed_to_all_installed_and_fail_closed() -> None:
    controller_type = _controller_type()
    context = _context()
    controller = controller_type(context.controller)
    arm_inputs, observation, live = _transition(
        context.retired_state.scheduler_state,
        context.next_step,
    )
    prepared = controller.prepare(
        context.pre_retirement_state,
        context.retirement_handoff,
        context.retirement_authority,
        context.phase_one_key,
        context.phase_two_key,
        arm_inputs,
        observation,
        live,
    )
    assert bool(prepared.diagnostics.source_state_valid)
    assert bool(prepared.diagnostics.transient_retirement_applied)
    assert bool(prepared.diagnostics.fresh_candidate_available)
    assert bool(prepared.diagnostics.atomic_swap_ready)
    np.testing.assert_array_equal(
        prepared.replacement_prepared.changed_slots,
        [True, False, False, False],
    )

    authority = controller.authority_receipt(
        prepared,
        context.installation_authority,
        swap_authorized=True,
    )
    result = controller.commit(context.pre_retirement_state, prepared, authority)
    assert bool(result.prepared_integrity_valid)
    assert bool(result.preparation_derivation_valid)
    assert bool(result.authority_valid)
    assert bool(result.transaction_applied)
    assert bool(result.retirement_applied)
    assert bool(result.replacement_applied)
    assert not bool(result.cold_state_persisted)
    assert bool(jnp.all(context.pre_retirement_state.installed_slot_mask))
    assert bool(jnp.all(result.state.installed_slot_mask))
    np.testing.assert_array_equal(result.reset_slots, [True, False, False, False])
    np.testing.assert_array_equal(result.preserved_slots, [False, True, True, True])
    np.testing.assert_array_equal(
        result.state.scheduler_state.installation_state.installed_semantic_digests[1:],
        context.pre_retirement_state.scheduler_state.installation_state.installed_semantic_digests[
            1:
        ],
    )

    replay = controller.commit(result.state, prepared, authority)
    assert not bool(replay.transaction_applied)
    assert not bool(replay.cold_state_persisted)
    chex.assert_trees_all_equal(replay.state, result.state)

    declined_authority = controller.authority_receipt(
        prepared,
        context.installation_authority,
        swap_authorized=False,
    )
    declined = controller.commit(
        context.pre_retirement_state,
        prepared,
        declined_authority,
    )
    assert not bool(declined.transaction_applied)
    assert not bool(declined.cold_state_persisted)
    chex.assert_trees_all_equal(declined.state, context.pre_retirement_state)

    tampered = dataclasses.replace(
        prepared,
        prepared_checksum=prepared.prepared_checksum + jnp.uint32(1),
    )
    rejected = controller.commit(context.pre_retirement_state, tampered, authority)
    assert not bool(rejected.transaction_applied)
    assert not bool(rejected.cold_state_persisted)
    chex.assert_trees_all_equal(rejected.state, context.pre_retirement_state)
