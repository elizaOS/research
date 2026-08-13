# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""End-to-end contracts for bounded repeated authorized option cycles."""

from __future__ import annotations

import copy
import dataclasses
import functools
from collections.abc import Iterator
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_authorized_option_replacement import (
    _context as _one_shot_context,
)
from test_authorized_option_replacement import (
    _supported_scheduler_state,
)
from test_authorized_option_retirement import _receipt as _retirement_receipt
from test_cumulant_option_scheduler import _receipt as _installation_receipt
from test_cumulant_option_scheduler import _transition

from alberta_framework.core.repeated_option_lifecycle import (
    REPEATED_OPTION_LIFECYCLE_ASSESSMENT,
    REPEATED_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA,
    REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY,
    RepeatedOptionLifecycle,
    RepeatedOptionLifecycleCommitResult,
    RepeatedOptionLifecycleConfig,
    RepeatedOptionLifecyclePrepared,
    RepeatedOptionLifecycleState,
    RepeatedOptionReplacementAuthorityReceipt,
    RepeatedOptionRetirementAuthorityReceipt,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_test() -> Iterator[None]:
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]


class _Initial(NamedTuple):
    lifecycle: RepeatedOptionLifecycle
    state: RepeatedOptionLifecycleState
    one_shot: Any


class _Cycle(NamedTuple):
    before_retirement: RepeatedOptionLifecycleState
    after_retirement: RepeatedOptionLifecycleState
    retirement_receipt: RepeatedOptionRetirementAuthorityReceipt
    prepared: RepeatedOptionLifecyclePrepared
    replacement_receipt: RepeatedOptionReplacementAuthorityReceipt
    commit: RepeatedOptionLifecycleCommitResult
    cycle_key: jax.Array
    phase_one_key: jax.Array
    phase_two_key: jax.Array


@functools.cache
def _initial() -> _Initial:
    one_shot = _one_shot_context(max_installations=8)
    lifecycle = RepeatedOptionLifecycle(
        one_shot.controller,
        RepeatedOptionLifecycleConfig(max_cycles=2),
    )
    state = lifecycle.init(one_shot.pre_retirement_state)
    assert bool(lifecycle.state_valid(state))
    return _Initial(lifecycle, state, one_shot)


def _retire(
    lifecycle: RepeatedOptionLifecycle,
    state: RepeatedOptionLifecycleState,
    handoff: Any,
    *,
    cycle_seed: int,
    phase_one_seed: int,
    phase_two_seed: int,
    authority_revision: int,
) -> tuple[
    RepeatedOptionLifecycleState,
    RepeatedOptionRetirementAuthorityReceipt,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    cycle_key = jr.key(cycle_seed, impl="threefry2x32")
    phase_one = jr.key(phase_one_seed, impl="threefry2x32")
    phase_two = jr.key(phase_two_seed, impl="threefry2x32")
    projected = lifecycle.replacement._as_retirement_state(state.cycle_state)
    child_receipt = _retirement_receipt(
        projected,
        handoff,
        phase_one,
        phase_two,
        revision=authority_revision,
    )
    receipt = lifecycle.retirement_authority_receipt(
        state,
        child_receipt,
        cycle_key,
    )
    result = lifecycle.retire(
        state,
        handoff,
        receipt,
        cycle_key,
        phase_one,
        phase_two,
    )
    assert bool(result.diagnostics.wrapper_transaction_applied)
    assert bool(result.retirement.transaction_applied)
    assert bool(lifecycle.state_valid(result.state))
    return result.state, receipt, cycle_key, phase_one, phase_two


def _prepare(
    lifecycle: RepeatedOptionLifecycle,
    state: RepeatedOptionLifecycleState,
    step: int,
) -> tuple[RepeatedOptionLifecyclePrepared, Any]:
    arm_inputs, observation, live = _transition(state.cycle_state.scheduler_state, step)
    arm = lifecycle.arm(state, arm_inputs)
    prepared = lifecycle.prepare(state, arm, observation, live)
    return prepared, live


def _replacement_receipt(
    lifecycle: RepeatedOptionLifecycle,
    state: RepeatedOptionLifecycleState,
    prepared: RepeatedOptionLifecyclePrepared,
    cycle_key: jax.Array,
    *,
    authorized: bool,
) -> RepeatedOptionReplacementAuthorityReceipt:
    authority = _installation_receipt(
        state.cycle_state.scheduler_state,
        authorized=authorized,
    )
    return lifecycle.replacement_authority_receipt(
        state,
        prepared,
        authority,
        cycle_key,
        replacement_authorized=authorized,
    )


def _supported_wrapper_state(
    lifecycle: RepeatedOptionLifecycle,
    state: RepeatedOptionLifecycleState,
) -> RepeatedOptionLifecycleState:
    """Test-only audit fixture update; production observations use public control."""

    scheduler = lifecycle.replacement.scheduler
    supported_scheduler = _supported_scheduler_state(
        scheduler,
        state.cycle_state.scheduler_state,
    )
    child = lifecycle.replacement._with_checksum(
        dataclasses.replace(
            state.cycle_state,
            scheduler_state=supported_scheduler,
            canonical_scheduler_checksum=supported_scheduler.binding_checksum,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    supported = lifecycle._with_checksum(
        dataclasses.replace(
            state,
            cycle_state=child,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    assert bool(lifecycle.state_valid(supported))
    return supported


def _install_test_alternate_for_next_proposal(
    lifecycle: RepeatedOptionLifecycle,
    state: RepeatedOptionLifecycleState,
) -> RepeatedOptionLifecycleState:
    """Arrange a changed next bundle without altering production lifecycle code."""

    controller = lifecycle.replacement
    scheduler = controller.scheduler
    source = state.cycle_state.scheduler_state
    step = int(source.step_words[1])
    arm_inputs, observation, live = _transition(source, step)
    denied = scheduler.observe(
        source,
        scheduler.arm(source, arm_inputs),
        observation,
        live,
        _installation_receipt(source, authorized=False),
    )
    assert bool(denied.transaction_applied)
    assert bool(denied.discovery.discovered.ready)
    preview_step = int(denied.state.step_words[1])
    next_arm_inputs, next_observation, next_live = _transition(
        denied.state,
        preview_step,
    )
    preview = scheduler.observe(
        denied.state,
        scheduler.arm(denied.state, next_arm_inputs),
        next_observation,
        next_live,
        _installation_receipt(denied.state, authorized=False),
    )
    assert bool(preview.discovery.discovered.ready)
    discovered = denied.discovery.discovered
    assert bool(discovered.ready)
    future_index = int(preview.discovery.discovered.selected_candidate_indices[0])
    current_index = int(source.installation_state.installed_bundle.selected_candidate_indices[0])
    alternate_index = next(
        index for index in (0, 1, 2) if index not in {future_index, current_index}
    )
    descriptors = discovered.selected_descriptors.at[0].set(
        jnp.asarray(
            scheduler.discovery.config.candidate_descriptors[alternate_index],
            dtype=jnp.int32,
        )
    )
    indices = discovered.selected_candidate_indices.at[0].set(alternate_index)
    cumulants, available = scheduler.installation._compute_live_tail(
        descriptors,
        source.installation_state.last_raw_features,
        source.installation_state.last_raw_available,
        live,
    )
    assert bool(available.all())
    alternate = dataclasses.replace(
        discovered,
        selected_candidate_indices=indices,
        selected_descriptors=descriptors,
        selected_cumulants=cumulants,
        binding_digest=jnp.zeros((2,), dtype=jnp.uint32),
    )
    alternate = dataclasses.replace(
        alternate,
        binding_digest=scheduler.discovery._bundle_checksum(alternate),
    )
    installed = scheduler.installation.install(
        source.installation_state,
        alternate,
        jr.key(880, impl="threefry2x32"),
        inputs=live,
    )
    assert bool(installed.applied)
    updated_scheduler = scheduler._with_checksum(
        dataclasses.replace(
            denied.state,
            installation_state=installed.state,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    updated_child = controller.init(
        updated_scheduler,
        retirement_authority_issuer_digest=(
            state.cycle_state.expected_retirement_authority_issuer_digest
        ),
        controller_owner_digest=state.cycle_state.controller_owner_digest,
    )
    updated = lifecycle._with_checksum(
        dataclasses.replace(
            state,
            cycle_state=updated_child,
            revision=state.revision + jnp.int32(1),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    assert bool(lifecycle.state_valid(updated))
    return updated


def _handoff(lifecycle: RepeatedOptionLifecycle, state: RepeatedOptionLifecycleState) -> Any:
    scheduler = lifecycle.replacement.scheduler
    child = state.cycle_state.scheduler_state
    handoff = scheduler._retirement_handoff(
        child.discovery_state,
        child.installation_state,
        child.step_words,
        available=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(handoff.available)
    assert bool(handoff.proposed_retirement_mask[0])
    return handoff


def _complete_replacement(
    lifecycle: RepeatedOptionLifecycle,
    state: RepeatedOptionLifecycleState,
    cycle_key: jax.Array,
    *,
    first_step: int,
) -> tuple[
    RepeatedOptionLifecyclePrepared,
    RepeatedOptionReplacementAuthorityReceipt,
    RepeatedOptionLifecycleCommitResult,
]:
    current = state
    for step in range(first_step, first_step + 5):
        prepared, _ = _prepare(lifecycle, current, step)
        ready = bool(prepared.replacement_prepared.diagnostics.candidate_ready_for_authority)
        receipt = _replacement_receipt(
            lifecycle,
            current,
            prepared,
            cycle_key,
            authorized=ready,
        )
        result = lifecycle.commit(current, prepared, receipt, cycle_key)
        if bool(result.diagnostics.cycle_completed):
            return prepared, receipt, result
        assert bool(result.diagnostics.ordinary_advance_adopted)
        current = result.state
    raise AssertionError("a fresh one-slot replacement was not ready within the fixed test bound")


@functools.cache
def _two_cycles() -> tuple[_Initial, _Cycle, _Cycle]:
    initial = _initial()
    lifecycle = initial.lifecycle
    one_shot = initial.one_shot

    retired_one, retirement_receipt_one, cycle_key_one, phase_one, phase_two = _retire(
        lifecycle,
        initial.state,
        one_shot.retirement_handoff,
        cycle_seed=901,
        phase_one_seed=701,
        phase_two_seed=702,
        authority_revision=1,
    )
    prepared_one, _ = _prepare(lifecycle, retired_one, one_shot.next_step)
    assert bool(prepared_one.replacement_prepared.diagnostics.candidate_ready_for_authority)
    replacement_receipt_one = _replacement_receipt(
        lifecycle,
        retired_one,
        prepared_one,
        cycle_key_one,
        authorized=True,
    )
    commit_one = lifecycle.commit(
        retired_one,
        prepared_one,
        replacement_receipt_one,
        cycle_key_one,
    )
    assert bool(commit_one.diagnostics.cycle_completed)
    cycle_one = _Cycle(
        initial.state,
        retired_one,
        retirement_receipt_one,
        prepared_one,
        replacement_receipt_one,
        commit_one,
        cycle_key_one,
        phase_one,
        phase_two,
    )

    before_two = _install_test_alternate_for_next_proposal(lifecycle, commit_one.state)
    before_two = _supported_wrapper_state(lifecycle, before_two)
    handoff_two = _handoff(lifecycle, before_two)
    retired_two, retirement_receipt_two, cycle_key_two, phase_three, phase_four = _retire(
        lifecycle,
        before_two,
        handoff_two,
        cycle_seed=902,
        phase_one_seed=703,
        phase_two_seed=704,
        authority_revision=2,
    )
    next_step = int(retired_two.cycle_state.scheduler_state.step_words[1])
    prepared_two, replacement_receipt_two, commit_two = _complete_replacement(
        lifecycle,
        retired_two,
        cycle_key_two,
        first_step=next_step,
    )
    cycle_two = _Cycle(
        before_two,
        retired_two,
        retirement_receipt_two,
        prepared_two,
        replacement_receipt_two,
        commit_two,
        cycle_key_two,
        phase_three,
        phase_four,
    )
    return initial, cycle_one, cycle_two


def test_two_successful_cycles_reuse_exactly_one_owner_and_exhaust_at_cap() -> None:
    initial, first, second = _two_cycles()
    lifecycle = initial.lifecycle
    final = second.commit.state

    assert int(first.commit.state.completed_cycles) == 1
    assert int(final.completed_cycles) == 2
    assert int(final.total_retirements) == 2
    assert int(final.total_replacements) == 2
    assert bool(final.unavailable)
    assert int(final.error) == REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY
    assert not bool(final.cycle_key_active)
    assert bool(final.cycle_state.installed_slot_mask.all())
    assert bool(lifecycle.state_valid(final))
    np.testing.assert_array_equal(
        final.cycle_key_history,
        jnp.stack((jr.key_data(first.cycle_key), jr.key_data(second.cycle_key))),
    )

    budget = lifecycle.resource_budget(final)
    assert budget.persistent_lifecycle_owner_count == 1
    assert budget.duplicated_scheduler_state_nbytes == 0
    assert budget.duplicated_installation_state_nbytes == 0
    assert budget.pending_proposal_slots == 0
    assert budget.persisted_receipt_count == 0
    assert budget.completed_cycles == budget.max_cycles == 2
    assert budget.remaining_cycles == 0
    assert budget.total_retirements == budget.total_replacements == 2
    assert budget.wrapper_metadata_nbytes == (
        budget.persistent_state_nbytes - budget.child_state_nbytes
    )
    assert budget.assessment == REPEATED_OPTION_LIFECYCLE_ASSESSMENT
    assert not budget.output_writes
    assert not budget.evidence_authority
    assert not budget.scientific_promotion_allowed


def test_decline_then_retry_requires_new_preparation_and_receipt() -> None:
    initial = _initial()
    lifecycle = initial.lifecycle
    one_shot = initial.one_shot
    retired, _, cycle_key, _, _ = _retire(
        lifecycle,
        initial.state,
        one_shot.retirement_handoff,
        cycle_seed=921,
        phase_one_seed=721,
        phase_two_seed=722,
        authority_revision=1,
    )
    prepared, _ = _prepare(lifecycle, retired, one_shot.next_step)
    declined_receipt = _replacement_receipt(
        lifecycle,
        retired,
        prepared,
        cycle_key,
        authorized=False,
    )
    declined = lifecycle.commit(retired, prepared, declined_receipt, cycle_key)
    assert bool(declined.diagnostics.ordinary_advance_adopted)
    assert not bool(declined.diagnostics.cycle_completed)
    assert bool(declined.state.cycle_key_active)
    assert int(declined.state.completed_cycles) == 0
    assert bool(declined.state.cycle_state.scheduler_state.retry_due)

    replay = lifecycle.commit(declined.state, prepared, declined_receipt, cycle_key)
    assert replay.replacement is None
    assert not bool(replay.diagnostics.prepared_binding_valid)
    chex.assert_trees_all_equal(replay.state, declined.state)

    next_step = int(declined.state.cycle_state.scheduler_state.step_words[1])
    fresh_prepared, fresh_receipt, accepted = _complete_replacement(
        lifecycle,
        declined.state,
        cycle_key,
        first_step=next_step,
    )
    assert bool(accepted.diagnostics.cycle_completed)
    assert not bool(jnp.array_equal(fresh_prepared.prepared_checksum, prepared.prepared_checksum))
    assert not bool(
        jnp.array_equal(
            fresh_receipt.source_state_checksum,
            declined_receipt.source_state_checksum,
        )
    )


def test_stale_cross_cycle_receipts_and_reused_cycle_keys_are_fail_closed() -> None:
    initial, first, second = _two_cycles()
    lifecycle = initial.lifecycle

    stale_retirement = lifecycle.retire(
        second.before_retirement,
        _handoff(lifecycle, second.before_retirement),
        first.retirement_receipt,
        first.cycle_key,
        first.phase_one_key,
        first.phase_two_key,
    )
    assert not bool(stale_retirement.diagnostics.receipt_binding_valid)
    assert not bool(stale_retirement.diagnostics.wrapper_transaction_applied)
    chex.assert_trees_all_equal(stale_retirement.state, second.before_retirement)

    stale_replacement = lifecycle.commit(
        second.after_retirement,
        first.prepared,
        first.replacement_receipt,
        first.cycle_key,
    )
    assert stale_replacement.replacement is None
    assert not bool(stale_replacement.diagnostics.prepared_binding_valid)
    assert not bool(stale_replacement.diagnostics.receipt_binding_valid)
    chex.assert_trees_all_equal(stale_replacement.state, second.after_retirement)

    handoff_two = _handoff(lifecycle, second.before_retirement)
    projected = lifecycle.replacement._as_retirement_state(second.before_retirement.cycle_state)
    p1 = jr.key(731, impl="threefry2x32")
    p2 = jr.key(732, impl="threefry2x32")
    nested = _retirement_receipt(projected, handoff_two, p1, p2, revision=2)
    with pytest.raises(ValueError, match="stale, cross-cycle"):
        lifecycle.retirement_authority_receipt(
            second.before_retirement,
            nested,
            first.cycle_key,
        )


def test_exhaustion_is_exact_noop_and_state_validation_is_jit_scan_deterministic() -> None:
    initial, first, second = _two_cycles()
    lifecycle = initial.lifecycle
    final = second.commit.state

    refused = lifecycle.retire(
        final,
        _handoff(lifecycle, _supported_wrapper_state(lifecycle, final)),
        second.retirement_receipt,
        second.cycle_key,
        second.phase_one_key,
        second.phase_two_key,
    )
    assert not bool(refused.diagnostics.cycle_capacity_available)
    assert not bool(refused.diagnostics.wrapper_transaction_applied)
    chex.assert_trees_all_equal(refused.state, final)
    with pytest.raises(ValueError, match="capacity-exhausted"):
        lifecycle.retirement_authority_receipt(
            final,
            second.retirement_receipt.retirement_authority,
            jr.key(999, impl="threefry2x32"),
        )

    eager = lifecycle.state_valid(initial.state)
    compiled = jax.jit(lifecycle.state_valid)(initial.state)
    _, scanned = jax.lax.scan(
        lambda carry, _: (carry, lifecycle.state_valid(carry)),
        initial.state,
        xs=None,
        length=3,
    )
    assert bool(eager)
    np.testing.assert_array_equal(compiled, eager)
    np.testing.assert_array_equal(scanned, jnp.ones((3,), dtype=jnp.bool_))


def test_public_retirement_is_eager_jit_and_scan_replay_safe() -> None:
    initial = _initial()
    lifecycle = initial.lifecycle
    one_shot = initial.one_shot
    cycle_key = jr.key(941, impl="threefry2x32")
    phase_one = jr.key(741, impl="threefry2x32")
    phase_two = jr.key(742, impl="threefry2x32")
    projected = lifecycle.replacement._as_retirement_state(initial.state.cycle_state)
    child_receipt = _retirement_receipt(
        projected,
        one_shot.retirement_handoff,
        phase_one,
        phase_two,
        revision=1,
    )
    receipt = lifecycle.retirement_authority_receipt(
        initial.state,
        child_receipt,
        cycle_key,
    )
    eager = lifecycle.retire(
        initial.state,
        one_shot.retirement_handoff,
        receipt,
        cycle_key,
        phase_one,
        phase_two,
    )
    compiled = jax.jit(lifecycle.retire)(
        initial.state,
        one_shot.retirement_handoff,
        receipt,
        cycle_key,
        phase_one,
        phase_two,
    )
    chex.assert_trees_all_equal(compiled, eager)

    def scan_step(
        state: RepeatedOptionLifecycleState,
        _: jax.Array,
    ) -> tuple[RepeatedOptionLifecycleState, jax.Array]:
        result = lifecycle.retire(
            state,
            one_shot.retirement_handoff,
            receipt,
            cycle_key,
            phase_one,
            phase_two,
        )
        return result.state, result.diagnostics.wrapper_transaction_applied

    final, applied = jax.lax.scan(
        scan_step,
        initial.state,
        jnp.arange(2, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(applied, (True, False))
    chex.assert_trees_all_equal(final, compiled.state)


def test_checkpoint_roundtrip_and_corruption_rejection_are_exact() -> None:
    initial, _, second = _two_cycles()
    lifecycle = initial.lifecycle
    state = second.commit.state
    installation = state.cycle_state.scheduler_state.installation_state
    payload = lifecycle.checkpoint_payload(state)
    assert payload["schema_version"] == REPEATED_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA
    assert payload["persistent_lifecycle_owner_count"] == 1
    assert payload["proposal_persisted"] is False
    assert payload["receipt_persisted"] is False

    kwargs = {
        "expected_semantic_generation": installation.installed_bundle.semantic_generation,
        "expected_source_digest": installation.installed_bundle.source_digest,
        "expected_consumer_source_digest": installation.consumer_source_digest,
        "expected_consumer_representation_digest": (installation.consumer_representation_digest),
        "expected_lifecycle_id": installation.lifecycle_id,
        "expected_installation_authority_issuer_digest": (
            state.cycle_state.scheduler_state.expected_authority_issuer_digest
        ),
        "expected_retirement_authority_issuer_digest": (
            state.cycle_state.expected_retirement_authority_issuer_digest
        ),
        "expected_controller_owner_digest": state.cycle_state.controller_owner_digest,
        "expected_descriptor_generation": state.cycle_state.descriptor_generation,
        "expected_descriptor_digest": state.cycle_state.descriptor_digest,
        "expected_installed_bundle": installation.installed_bundle,
        "expected_completed_cycles": state.completed_cycles,
        "expected_revision": state.revision,
    }
    restored = lifecycle.restore_checkpoint(payload, **kwargs)
    chex.assert_trees_all_equal(restored, state)

    corrupt = copy.deepcopy(payload)
    fields = corrupt["controller_fields"]
    assert isinstance(fields, dict)
    revision = fields["revision"]
    assert isinstance(revision, dict)
    encoded = revision["bytes_hex"]
    assert isinstance(encoded, str)
    revision["bytes_hex"] = ("01" if encoded[:2] != "01" else "02") + encoded[2:]
    with pytest.raises(ValueError, match="digest differs|invalid, stale, or rebound"):
        lifecycle.restore_checkpoint(corrupt, **kwargs)

    with pytest.raises(ValueError, match="invalid, stale, or rebound"):
        lifecycle.restore_checkpoint(
            payload,
            **(kwargs | {"expected_completed_cycles": jnp.asarray(1, dtype=jnp.int32)}),
        )


def test_config_roundtrip_and_fail_closed_schema() -> None:
    config = RepeatedOptionLifecycleConfig(max_cycles=3)
    assert RepeatedOptionLifecycleConfig.from_config(config.to_config()) == config
    with pytest.raises(ValueError, match="positive exact Python int32"):
        RepeatedOptionLifecycleConfig(max_cycles=0)
    tampered = config.to_config()
    tampered["persistent_owner_count"] = 2
    with pytest.raises(ValueError, match="persistent_owner_count"):
        RepeatedOptionLifecycleConfig.from_config(tampered)
