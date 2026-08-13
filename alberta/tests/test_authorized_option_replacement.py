# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Adversarial contracts for one-cold-slot authorized replacement."""

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
from test_authorized_option_retirement import _receipt as _retirement_receipt
from test_cumulant_option_scheduler import (
    CONSUMER_SOURCE,
    GENERATION,
    LIFECYCLE_ID,
    REPRESENTATION,
    SOURCE,
    _discovery_config,
    _force_extended_action,
    _init,
    _transition,
)
from test_cumulant_option_scheduler import (
    ISSUER as INSTALLATION_ISSUER,
)
from test_cumulant_option_scheduler import (
    _receipt as _installation_receipt,
)
from test_cumulant_option_scheduler import (
    _scheduler as _base_scheduler,
)

from alberta_framework.core.authorized_option_replacement import (
    AUTHORIZED_OPTION_REPLACEMENT_ASSESSMENT,
    AuthorizedOptionReplacementConfig,
    AuthorizedOptionReplacementController,
    AuthorizedOptionReplacementPrepared,
    AuthorizedOptionReplacementState,
)
from alberta_framework.core.authorized_option_retirement import (
    AuthorizedOptionRetirementConfig,
    AuthorizedOptionRetirementController,
)
from alberta_framework.core.cumulant_option_installation import (
    CUMULANT_OPTION_INSTALLER_ERROR_CAPACITY,
    CumulantOptionInstallation,
    CumulantOptionInstallationConfig,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionScheduler,
    CumulantOptionSchedulerConfig,
    CumulantOptionSchedulerState,
)
from alberta_framework.core.cumulant_subtask_discovery import (
    CUMULANT_SOURCE_CONTROLLABLE_EVENT,
    CumulantSubtaskDiscovery,
    CumulantSubtaskProposalBundle,
)
from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleAudit,
    option_semantic_digest,
)

RETIREMENT_ISSUER = option_semantic_digest({"authority": "replacement-retirement-only"})
CONTROLLER_OWNER = option_semantic_digest({"owner": "replacement-controller"})

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_test() -> Iterator[None]:
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]


class _Context(NamedTuple):
    controller: AuthorizedOptionReplacementController
    pre_retirement_state: AuthorizedOptionReplacementState
    retired_state: AuthorizedOptionReplacementState
    prepared: AuthorizedOptionReplacementPrepared
    installation_authority: Any
    retirement_handoff: Any
    retirement_authority: Any
    phase_one_key: jax.Array
    phase_two_key: jax.Array
    next_step: int


def _scheduler(
    *,
    max_installations: int = 4,
    option_planning_backups_per_step: int = 1,
    reserved_observation_suffix: int = 0,
) -> CumulantOptionScheduler:
    """Scheduler with three interchangeable descriptors in the first family."""

    base = _base_scheduler()
    base_installation = base.installation
    first = _discovery_config().controllable_event_descriptors[0]
    discovery = CumulantSubtaskDiscovery(
        dataclasses.replace(
            _discovery_config(),
            controllable_event_descriptors=(
                first,
                (CUMULANT_SOURCE_CONTROLLABLE_EVENT, 0, -1, 11),
                (CUMULANT_SOURCE_CONTROLLABLE_EVENT, 0, 1, 12),
            ),
        )
    )
    template = dataclasses.replace(
        base_installation.stomp_agent.config,
        subtask_specs=(),
        observation_dim=(
            discovery.config.raw_feature_dim + reserved_observation_suffix
        ),
        option_planning_backups_per_step=option_planning_backups_per_step,
    )
    base_audit = base_installation.lifecycle.audit.config
    audit = OptionLifecycleAudit(
        dataclasses.replace(
            base_audit,
            outcome_dim=base_audit.outcome_dim + reserved_observation_suffix,
            signature_scales=(
                base_audit.signature_scales
                + (1.0,) * reserved_observation_suffix
            ),
        )
    )
    installation = CumulantOptionInstallation(
        discovery,
        template,
        audit,
        base_installation.lifecycle.config,
        CumulantOptionInstallationConfig(
            polarized_cumulant_threshold=(base_installation.config.polarized_cumulant_threshold),
            max_option_steps=base_installation.config.max_option_steps,
            max_installations=max_installations,
        ),
    )
    return CumulantOptionScheduler(
        installation,
        CumulantOptionSchedulerConfig(
            proposal_period=1,
            maintenance_period=2,
            max_steps=16,
            max_install_attempts=8,
            max_retry_streak=3,
            max_maintenance_handoffs=4,
        ),
    )


def _authorized_step(
    scheduler: CumulantOptionScheduler,
    state: CumulantOptionSchedulerState,
    step: int,
) -> Any:
    arm_inputs, observation, live = _transition(state, step)
    return scheduler.observe(
        state,
        scheduler.arm(state, arm_inputs),
        observation,
        live,
        _installation_receipt(state, authorized=True),
    )


def _supported_scheduler_state(
    scheduler: CumulantOptionScheduler,
    state: CumulantOptionSchedulerState,
) -> CumulantOptionSchedulerState:
    """Give the real audit enough frozen observations to retire slot zero."""

    installation = state.installation_state
    audit = scheduler.installation.lifecycle.audit
    audit_state = installation.lifecycle_state.audit_state
    n = audit.config.n_options
    signature_dim = audit.config.signature_dim
    signatures = jnp.zeros((n, 1, signature_dim), dtype=jnp.float32)
    signatures = signatures.at[:, 0, 0].set(jnp.arange(n, dtype=jnp.float32) * 20.0)
    audit_state = dataclasses.replace(
        audit_state,
        revision=jnp.asarray(1, dtype=jnp.int32),
        observation_count=jnp.asarray(1, dtype=jnp.int32),
        has_last_transition=jnp.asarray(True, dtype=jnp.bool_),
        last_transition_id=jnp.asarray([0, 1], dtype=jnp.uint32),
        initiation_opportunities=jnp.full((n, 1), 4, dtype=jnp.int32),
        initiation_starts=jnp.full((n, 1), 4, dtype=jnp.int32),
        execution_starts=jnp.full((n,), 4, dtype=jnp.int32),
        natural_completions=jnp.full((n,), 2, dtype=jnp.int32),
        censor_only_endings=jnp.full((n,), 2, dtype=jnp.int32),
        completion_moment_counts=jnp.full((n,), 2, dtype=jnp.int32),
        model_error_counts=jnp.full((n,), 2, dtype=jnp.int32),
        model_squared_error_sums=jnp.full(
            (n, signature_dim),
            8.0e12,
            dtype=jnp.float32,
        ),
        context_signature_counts=jnp.full((n, 1), 2, dtype=jnp.int32),
        context_signature_sums=signatures,
        comparison_treatment_counts=jnp.full((n, 1), 2, dtype=jnp.int32),
        comparison_primitive_counts=jnp.full((n, 1), 2, dtype=jnp.int32),
        comparison_treatment_ipw_masses=jnp.full((n, 1), 2.0, dtype=jnp.float32),
        comparison_primitive_ipw_masses=jnp.full((n, 1), 2.0, dtype=jnp.float32),
    )
    audit_state = audit._with_checksum(audit_state)
    lifecycle_api = scheduler.installation.lifecycle.with_external_semantic_digests(
        installation.installed_semantic_digests
    )
    lifecycle = lifecycle_api._with_checksum(
        dataclasses.replace(installation.lifecycle_state, audit_state=audit_state)
    )
    installation = scheduler.installation._with_checksum(
        dataclasses.replace(installation, lifecycle_state=lifecycle)
    )
    supported = scheduler._with_checksum(
        dataclasses.replace(state, installation_state=installation)
    )
    assert bool(scheduler.state_valid(supported))
    return supported


def _alternate_bundle(
    scheduler: CumulantOptionScheduler,
    source: CumulantOptionSchedulerState,
    discovered: CumulantSubtaskProposalBundle,
    live: Any,
    *,
    forbidden_index: int,
) -> CumulantSubtaskProposalBundle:
    """Construct another valid first-family proposal for source setup only."""

    family_zero = (0, 1, 2)
    alternate = next(index for index in family_zero if index != forbidden_index)
    descriptors = discovered.selected_descriptors.at[0].set(
        jnp.asarray(
            scheduler.discovery.config.candidate_descriptors[alternate],
            dtype=jnp.int32,
        )
    )
    indices = discovered.selected_candidate_indices.at[0].set(alternate)
    cumulants, available = scheduler.installation._compute_live_tail(
        descriptors,
        source.installation_state.last_raw_features,
        source.installation_state.last_raw_available,
        live,
    )
    assert bool(available.all())
    bundle = dataclasses.replace(
        discovered,
        selected_candidate_indices=indices,
        selected_descriptors=descriptors,
        selected_cumulants=cumulants,
        binding_digest=jnp.zeros((2,), dtype=jnp.uint32),
    )
    return dataclasses.replace(
        bundle,
        binding_digest=scheduler.discovery._bundle_checksum(bundle),
    )


@functools.cache
def _context(
    *,
    max_installations: int = 4,
    option_planning_backups_per_step: int = 1,
    install_fresh_alternate: bool = True,
    reserved_observation_suffix: int = 0,
) -> _Context:
    scheduler = _scheduler(
        max_installations=max_installations,
        option_planning_backups_per_step=option_planning_backups_per_step,
        reserved_observation_suffix=reserved_observation_suffix,
    )
    state = _init(scheduler, seed=17)
    installed_result = None
    step = 0
    while step < 8:
        installed_result = _authorized_step(scheduler, state, step)
        assert bool(installed_result.transaction_applied)
        state = installed_result.state
        if bool(installed_result.installation_applied):
            break
        step += 1
    assert installed_result is not None and bool(installed_result.installation_applied)

    # Consume one ordinary transition, then inspect the deterministic proposal
    # at the following transition.  Install a different valid descriptor now,
    # leaving the next real discovery observation as an exact one-slot change.
    step += 1
    arm_inputs, observation, live = _transition(state, step)
    denied = scheduler.observe(
        state,
        scheduler.arm(state, arm_inputs),
        observation,
        live,
        _installation_receipt(state, authorized=False),
    )
    assert bool(denied.transaction_applied)
    assert bool(denied.discovery.discovered.ready)
    step += 1
    next_arm_inputs, next_observation, next_live = _transition(denied.state, step)
    preview = scheduler.observe(
        denied.state,
        scheduler.arm(denied.state, next_arm_inputs),
        next_observation,
        next_live,
        _installation_receipt(denied.state, authorized=False),
    )
    assert bool(preview.discovery.discovered.ready)
    future_index = int(preview.discovery.discovered.selected_candidate_indices[0])
    installation_source = state
    state = denied.state
    if install_fresh_alternate:
        alternate = _alternate_bundle(
            scheduler,
            installation_source,
            denied.discovery.discovered,
            live,
            forbidden_index=future_index,
        )
        installed_alternate = scheduler.installation.install(
            installation_source.installation_state,
            alternate,
            jr.key(700, impl="threefry2x32"),
            inputs=live,
        )
        assert bool(installed_alternate.applied)
        state = scheduler._with_checksum(
            dataclasses.replace(
                denied.state,
                installation_state=installed_alternate.state,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
    state = _supported_scheduler_state(scheduler, state)

    retirement = AuthorizedOptionRetirementController(
        scheduler.installation,
        AuthorizedOptionRetirementConfig(
            minimum_context_support=2,
            maximum_completion_reliability=0.5,
            minimum_normalized_model_rmse=1.0,
            maximum_planning_uses=0,
            max_retirements=1,
        ),
    )
    controller = AuthorizedOptionReplacementController(scheduler, retirement)
    pre_retirement = controller.init(
        state,
        retirement_authority_issuer_digest=RETIREMENT_ISSUER,
        controller_owner_digest=CONTROLLER_OWNER,
    )
    assert not bool(jnp.array_equal(INSTALLATION_ISSUER, RETIREMENT_ISSUER))
    handoff = scheduler._retirement_handoff(
        state.discovery_state,
        state.installation_state,
        state.step_words,
        available=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert int(handoff.proposed_retirement_slots[0]) == 0
    assert bool(handoff.proposed_retirement_mask[0])
    phase_one = jr.key(701, impl="threefry2x32")
    phase_two = jr.key(702, impl="threefry2x32")
    projected = controller._as_retirement_state(pre_retirement)
    retirement_authority = _retirement_receipt(
        projected,
        handoff,
        phase_one,
        phase_two,
    )
    retired = controller.retire(
        pre_retirement,
        handoff,
        retirement_authority,
        phase_one,
        phase_two,
    )
    assert bool(retired.transaction_applied)
    assert int(jnp.sum(~retired.state.installed_slot_mask)) == 1

    arm_inputs, observation, live = _transition(retired.state.scheduler_state, step)
    arm = controller.arm(retired.state, arm_inputs)
    prepared = controller.prepare(retired.state, arm, observation, live)
    assert bool(prepared.diagnostics.transaction_valid)
    if install_fresh_alternate:
        assert bool(prepared.diagnostics.candidate_ready_for_authority)
        np.testing.assert_array_equal(prepared.changed_slots, [True, False, False, False])
    else:
        assert not bool(prepared.diagnostics.candidate_ready_for_authority)
        np.testing.assert_array_equal(prepared.changed_slots, [False, False, False, False])
    authority = _installation_receipt(
        retired.state.scheduler_state,
        authorized=True,
    )
    return _Context(
        controller,
        pre_retirement,
        retired.state,
        prepared,
        authority,
        handoff,
        retirement_authority,
        phase_one,
        phase_two,
        step,
    )


@pytest.mark.integration
def test_distinct_authorities_retire_then_replace_exactly_one_slot() -> None:
    context = _context()
    receipt = context.controller.authority_receipt(
        context.prepared,
        context.installation_authority,
        replacement_authorized=True,
    )
    before = context.retired_state.scheduler_state.installation_state
    result = context.controller.commit(context.retired_state, context.prepared, receipt)
    assert bool(result.diagnostics.preparation_derivation_valid)
    assert bool(result.diagnostics.replacement_applied)
    np.testing.assert_array_equal(result.reset_slots, [True, False, False, False])
    np.testing.assert_array_equal(result.preserved_slots, [False, True, True, True])
    assert bool(result.state.installed_slot_mask.all())
    assert not bool(result.cold_mask_active)
    assert not bool(result.diagnostics.proposal_persisted)
    assert not bool(result.diagnostics.candidate_materialization_persisted_on_decline)
    np.testing.assert_array_equal(
        result.state.scheduler_state.installation_state.installed_semantic_digests[1:],
        before.installed_semantic_digests[1:],
    )
    chex.assert_trees_all_equal(
        context.controller.release_scheduler_state(result.state),
        result.state.scheduler_state,
    )


@pytest.mark.integration
def test_decline_commits_only_ordinary_incumbent_advance_and_retry() -> None:
    context = _context()
    receipt = context.controller.authority_receipt(
        context.prepared,
        context.installation_authority,
        replacement_authorized=False,
    )
    result = context.controller.commit(context.retired_state, context.prepared, receipt)
    assert bool(result.diagnostics.ordinary_advance_applied)
    assert not bool(result.diagnostics.replacement_attempted)
    assert not bool(result.diagnostics.replacement_applied)
    chex.assert_trees_all_equal(result.state, context.prepared.fallback_state)
    assert bool(result.state.scheduler_state.retry_due)
    assert bool(result.cold_mask_active)
    np.testing.assert_array_equal(
        result.state.scheduler_state.installation_state.installed_semantic_digests,
        context.retired_state.scheduler_state.installation_state.installed_semantic_digests,
    )
    np.testing.assert_array_equal(
        jr.key_data(result.state.scheduler_state.installation_rng_key),
        jr.key_data(context.retired_state.scheduler_state.installation_rng_key),
    )
    np.testing.assert_array_equal(
        result.state.scheduler_state.install_attempt_words,
        context.retired_state.scheduler_state.install_attempt_words,
    )
    assert (
        int(result.state.scheduler_state.step_words[1])
        == int(context.retired_state.scheduler_state.step_words[1]) + 1
    )

    replay = context.controller.commit(result.state, context.prepared, receipt)
    assert not bool(replay.diagnostics.ordinary_advance_applied)
    chex.assert_trees_all_equal(replay.state, result.state)


@pytest.mark.integration
def test_stale_mirror_detects_accidental_scheduler_lifecycle_splice() -> None:
    context = _context()
    assert bool(
        context.controller.scheduler.state_valid(context.prepared.fallback_state.scheduler_state)
    )
    assert bool(context.controller.state_valid(context.retired_state))
    spliced = context.controller._with_checksum(
        dataclasses.replace(
            context.retired_state,
            scheduler_state=context.prepared.fallback_state.scheduler_state,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    assert not bool(context.controller.state_valid(spliced))
    assert not bool(
        jnp.array_equal(
            spliced.canonical_scheduler_checksum,
            spliced.scheduler_state.binding_checksum,
        )
    )

    # These are unkeyed corruption checks, not caller authentication.  A caller
    # with private constructor/checksum access can reseal all redundant fields;
    # the public API invariant is instead that init/retire/control never accept
    # a separately evolved retirement installation subtree.
    resealed = context.controller._with_checksum(
        dataclasses.replace(
            spliced,
            canonical_scheduler_checksum=spliced.scheduler_state.binding_checksum,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    assert bool(context.controller.state_valid(resealed))

    retired_again = context.controller.retire(
        spliced,
        context.retirement_handoff,
        context.retirement_authority,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert not bool(retired_again.transaction_applied)
    chex.assert_trees_all_equal(retired_again.state, spliced)
    receipt = context.controller.authority_receipt(
        context.prepared,
        context.installation_authority,
        replacement_authorized=True,
    )
    committed = context.controller.commit(spliced, context.prepared, receipt)
    assert not bool(committed.diagnostics.destination_state_valid)
    assert not bool(committed.diagnostics.replacement_applied)
    chex.assert_trees_all_equal(committed.state, spliced)


@pytest.mark.integration
def test_non_single_cold_phase_and_second_retirement_are_exact_noops() -> None:
    context = _context()
    two_cold = context.controller._with_checksum(
        dataclasses.replace(
            context.retired_state,
            installed_slot_mask=(context.retired_state.installed_slot_mask.at[1].set(False)),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    assert not bool(context.controller.state_valid(two_cold))
    receipt = context.controller.authority_receipt(
        context.prepared,
        context.installation_authority,
        replacement_authorized=True,
    )
    rejected = context.controller.commit(two_cold, context.prepared, receipt)
    assert not bool(rejected.diagnostics.destination_state_valid)
    assert not bool(rejected.diagnostics.replacement_applied)
    chex.assert_trees_all_equal(rejected.state, two_cold)

    second = context.controller.retire(
        context.retired_state,
        context.retirement_handoff,
        context.retirement_authority,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert not bool(second.phase_valid)
    assert not bool(second.transaction_applied)
    chex.assert_trees_all_equal(second.state, context.retired_state)


@pytest.mark.integration
def test_invalid_authority_matrix_advances_once_without_candidate_leak() -> None:
    context = _context()
    base = context.controller.authority_receipt(
        context.prepared,
        context.installation_authority,
        replacement_authorized=True,
    )
    nested = base.installation_authority
    invalid_receipts = {
        "replacement_bit": dataclasses.replace(
            base,
            replacement_authorized=jnp.asarray(False, dtype=jnp.bool_),
        ),
        "installation_go": dataclasses.replace(
            base,
            installation_authority=dataclasses.replace(
                nested,
                go_no_go_authorized=jnp.asarray(False, dtype=jnp.bool_),
            ),
        ),
        "installation_safety": dataclasses.replace(
            base,
            installation_authority=dataclasses.replace(
                nested,
                safety_boundary_authorized=jnp.asarray(False, dtype=jnp.bool_),
            ),
        ),
        "installation_issuer": dataclasses.replace(
            base,
            installation_authority=dataclasses.replace(
                nested,
                issuer_digest=RETIREMENT_ISSUER,
            ),
        ),
        "controller_owner": dataclasses.replace(
            base,
            controller_owner_digest=base.controller_owner_digest + jnp.uint32(1),
        ),
        "source_scheduler": dataclasses.replace(
            base,
            source_scheduler_checksum=(base.source_scheduler_checksum + jnp.uint32(1)),
        ),
        "source_controller_revision": dataclasses.replace(
            base,
            source_controller_revision=base.source_controller_revision + jnp.int32(1),
        ),
        "descriptor_generation": dataclasses.replace(
            base,
            source_descriptor_generation=(base.source_descriptor_generation + jnp.int32(1)),
        ),
        "descriptor_digest": dataclasses.replace(
            base,
            source_descriptor_digest=base.source_descriptor_digest + jnp.uint32(1),
        ),
        "slot": dataclasses.replace(
            base,
            replacement_slot=jnp.asarray(1, dtype=jnp.int32),
        ),
        "reset_mask": dataclasses.replace(
            base,
            expected_reset_slots=jnp.roll(base.expected_reset_slots, 1),
        ),
        "candidate_binding": dataclasses.replace(
            base,
            candidate_binding_digest=base.candidate_binding_digest + jnp.uint32(1),
        ),
        "candidate_semantics": dataclasses.replace(
            base,
            candidate_semantic_digests=(
                base.candidate_semantic_digests.at[0, 0].add(jnp.uint32(1))
            ),
        ),
        "candidate_transition": dataclasses.replace(
            base,
            candidate_transition_id=base.candidate_transition_id.at[1].add(jnp.uint32(1)),
        ),
        "candidate_observation": dataclasses.replace(
            base,
            candidate_state_observation_count=(
                base.candidate_state_observation_count + jnp.int32(1)
            ),
        ),
        "prepared_checksum": dataclasses.replace(
            base,
            prepared_checksum=base.prepared_checksum + jnp.uint32(1),
        ),
        "authority_replay": dataclasses.replace(
            base,
            installation_authority=dataclasses.replace(
                nested,
                authority_revision_words=(
                    context.retired_state.scheduler_state.last_authority_revision_words
                ),
            ),
        ),
    }
    for name, receipt in invalid_receipts.items():
        assert not bool(
            context.controller._authority_valid(
                context.retired_state,
                context.prepared,
                receipt,
            )
        ), name

    # Commit one representative from each independent veto class.  Every
    # other field above is exact-checked by the same authority predicate.
    for name in (
        "replacement_bit",
        "installation_issuer",
        "controller_owner",
        "candidate_binding",
    ):
        receipt = invalid_receipts[name]
        result = context.controller.commit(
            context.retired_state,
            context.prepared,
            receipt,
        )
        assert bool(result.diagnostics.ordinary_advance_applied), name
        assert not bool(result.diagnostics.authority_valid), name
        assert not bool(result.diagnostics.replacement_applied), name
        chex.assert_trees_all_equal(result.state, context.prepared.fallback_state)
        chex.assert_trees_all_equal(
            result.materialization,
            context.prepared.scheduler_result.materialization,
        )
        np.testing.assert_array_equal(
            jr.key_data(result.state.scheduler_state.installation_rng_key),
            jr.key_data(context.retired_state.scheduler_state.installation_rng_key),
        )


@pytest.mark.integration
def test_replayed_replacement_receipt_against_new_destination_is_noop() -> None:
    context = _context()
    receipt = context.controller.authority_receipt(
        context.prepared,
        context.installation_authority,
        replacement_authorized=True,
    )
    applied = context.controller.commit(
        context.retired_state,
        context.prepared,
        receipt,
    )
    assert bool(applied.diagnostics.replacement_applied)
    replay = context.controller.commit(applied.state, context.prepared, receipt)
    assert not bool(replay.diagnostics.destination_matches_source)
    assert not bool(replay.diagnostics.ordinary_advance_applied)
    assert not bool(replay.diagnostics.replacement_applied)
    chex.assert_trees_all_equal(replay.state, applied.state)


@pytest.mark.integration
def test_masked_start_update_reproject_into_one_canonical_scheduler() -> None:
    context = _context()
    declined_receipt = context.controller.authority_receipt(
        context.prepared,
        context.installation_authority,
        replacement_authorized=False,
    )
    declined = context.controller.commit(
        context.retired_state,
        context.prepared,
        declined_receipt,
    )
    # Force the base head to request live option 1.  Cold option 0 remains
    # ineligible under the retirement mask throughout start and update.
    forced_scheduler = _force_extended_action(
        context.controller.scheduler,
        declined.state.scheduler_state,
        3,
    )
    forced = context.controller._with_checksum(
        dataclasses.replace(
            declined.state,
            scheduler_state=forced_scheduler,
            canonical_scheduler_checksum=forced_scheduler.binding_checksum,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    assert bool(context.controller.state_valid(forced))
    mask_before = context.controller.extended_action_mask(forced)
    np.testing.assert_array_equal(
        mask_before,
        [True, True, False, True, True, True],
    )
    step = context.next_step + 1
    arm_inputs, observation, live = _transition(forced.scheduler_state, step)
    ready_prepared = context.controller.prepare(
        forced,
        context.controller.arm(forced, arm_inputs),
        observation,
        live,
    )
    ready_receipt = context.controller.authority_receipt(
        ready_prepared,
        _installation_receipt(forced.scheduler_state, authorized=True),
        replacement_authorized=False,
    )
    ready = context.controller.commit(forced, ready_prepared, ready_receipt)
    assert bool(ready.diagnostics.ordinary_advance_applied)
    started = context.controller.start(ready.state, ready.materialization)
    assert started.applied and started.retirement is not None
    assert (
        int(
            started.state.scheduler_state.installation_state.lifecycle_state.stomp_state.executing_option
        )
        == 1
    )
    np.testing.assert_array_equal(started.state.installed_slot_mask, forced.installed_slot_mask)
    np.testing.assert_array_equal(
        started.state.canonical_scheduler_checksum,
        started.state.scheduler_state.binding_checksum,
    )
    chex.assert_trees_all_equal(
        context.controller._as_retirement_state(started.state).installation_state,
        started.state.scheduler_state.installation_state,
    )

    step += 1
    arm_inputs, observation, live = _transition(started.state.scheduler_state, step)
    prepared = context.controller.prepare(
        started.state,
        context.controller.arm(started.state, arm_inputs),
        observation,
        live,
    )
    assert bool(prepared.diagnostics.transaction_valid)
    assert not bool(prepared.diagnostics.quiescent)
    receipt = context.controller.authority_receipt(
        prepared,
        _installation_receipt(started.state.scheduler_state, authorized=True),
        replacement_authorized=True,
    )
    advanced = context.controller.commit(started.state, prepared, receipt)
    assert bool(advanced.diagnostics.ordinary_advance_applied)
    assert not bool(advanced.diagnostics.replacement_applied)
    updated = context.controller.update(
        advanced.state,
        advanced.materialization,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        execution_boundary=True,
        enable_planning=False,
    )
    assert updated.applied and updated.retirement is not None
    np.testing.assert_array_equal(
        updated.state.installed_slot_mask,
        context.retired_state.installed_slot_mask,
    )
    np.testing.assert_array_equal(
        updated.state.canonical_scheduler_checksum,
        updated.state.scheduler_state.binding_checksum,
    )
    chex.assert_trees_all_equal(
        context.controller._as_retirement_state(updated.state).installation_state,
        updated.state.scheduler_state.installation_state,
    )


@pytest.mark.integration
def test_installer_capacity_and_freshness_are_noncompensating_vetoes() -> None:
    context = _context()
    source = context.retired_state
    installation = source.scheduler_state.installation_state
    capacity_installation = context.controller.scheduler.installation._with_checksum(
        dataclasses.replace(
            installation,
            installation_count=jnp.asarray(
                context.controller.scheduler.installation.config.max_installations,
                dtype=jnp.int32,
            ),
            installer_unavailable=jnp.asarray(True, dtype=jnp.bool_),
            installer_error=jnp.asarray(
                CUMULANT_OPTION_INSTALLER_ERROR_CAPACITY,
                dtype=jnp.int32,
            ),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    capacity_scheduler = context.controller.scheduler._with_checksum(
        dataclasses.replace(
            source.scheduler_state,
            installation_state=capacity_installation,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    at_capacity = context.controller._with_checksum(
        dataclasses.replace(
            source,
            scheduler_state=capacity_scheduler,
            canonical_scheduler_checksum=capacity_scheduler.binding_checksum,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    assert bool(context.controller.state_valid(at_capacity))
    arm_inputs, observation, live = _transition(
        at_capacity.scheduler_state,
        context.next_step,
    )
    prepared = context.controller.prepare(
        at_capacity,
        context.controller.arm(at_capacity, arm_inputs),
        observation,
        live,
    )
    assert not bool(prepared.diagnostics.installer_capacity_available)
    assert not bool(prepared.diagnostics.candidate_ready_for_authority)
    receipt = context.controller.authority_receipt(
        prepared,
        _installation_receipt(at_capacity.scheduler_state, authorized=True),
        replacement_authorized=True,
    )
    result = context.controller.commit(at_capacity, prepared, receipt)
    assert bool(result.diagnostics.ordinary_advance_applied)
    assert not bool(result.diagnostics.replacement_applied)
    assert bool(jnp.any(~result.state.installed_slot_mask))

    stale_source = context.prepared.fallback_state
    assert bool(context.controller.state_valid(stale_source))
    arm_inputs, observation, live = _transition(
        stale_source.scheduler_state,
        context.next_step,
    )
    stale = context.controller.prepare(
        stale_source,
        context.controller.arm(stale_source, arm_inputs),
        observation,
        live,
    )
    assert not bool(stale.diagnostics.ordinary_scheduler_transaction_valid)
    assert not bool(stale.diagnostics.fresh_transition)
    assert not bool(stale.diagnostics.transaction_valid)


@pytest.mark.integration
def test_forged_bundle_checksum_and_fresh_wrapper_receipt_are_rejected() -> None:
    context = _context()
    prepared = context.prepared
    authentic = prepared.scheduler_result.discovery.discovered
    forged_index = next(
        index
        for index in (0, 1, 2)
        if index
        not in {
            int(authentic.selected_candidate_indices[0]),
            int(
                context.retired_state.scheduler_state.installation_state.installed_bundle.selected_candidate_indices[
                    0
                ]
            ),
        }
    )
    descriptors = authentic.selected_descriptors.at[0].set(
        jnp.asarray(
            context.controller.scheduler.discovery.config.candidate_descriptors[forged_index],
            dtype=jnp.int32,
        )
    )
    indices = authentic.selected_candidate_indices.at[0].set(forged_index)
    source_installation = context.retired_state.scheduler_state.installation_state
    cumulants, available = context.controller.scheduler.installation._compute_live_tail(
        descriptors,
        source_installation.last_raw_features,
        source_installation.last_raw_available,
        prepared.live_inputs,
    )
    assert bool(available.all())
    forged_bundle = dataclasses.replace(
        authentic,
        selected_candidate_indices=indices,
        selected_descriptors=descriptors,
        selected_cumulants=cumulants,
        binding_digest=jnp.zeros((2,), dtype=jnp.uint32),
    )
    forged_bundle = dataclasses.replace(
        forged_bundle,
        binding_digest=(context.controller.scheduler.discovery._bundle_checksum(forged_bundle)),
    )
    forged_discovery = dataclasses.replace(
        prepared.scheduler_result.discovery,
        discovered=forged_bundle,
    )
    forged_scheduler_result = dataclasses.replace(
        prepared.scheduler_result,
        discovery=forged_discovery,
    )
    semantics = context.controller.scheduler.installation.semantic_digests_for_bundle(forged_bundle)
    changed = jnp.any(
        semantics != source_installation.installed_semantic_digests,
        axis=1,
    )
    forged = dataclasses.replace(
        prepared,
        scheduler_result=forged_scheduler_result,
        candidate_semantic_digests=semantics,
        changed_slots=changed,
        prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
    )
    forged = context.controller._with_prepared_checksum(forged)
    fresh_receipt = context.controller.authority_receipt(
        forged,
        context.installation_authority,
        replacement_authorized=True,
    )
    result = context.controller.commit(context.retired_state, forged, fresh_receipt)
    assert bool(result.diagnostics.prepared_integrity_valid)
    assert not bool(result.diagnostics.preparation_derivation_valid)
    assert not bool(result.diagnostics.ordinary_advance_applied)
    assert not bool(result.diagnostics.replacement_applied)
    chex.assert_trees_all_equal(result.state, context.retired_state)


@pytest.mark.integration
def test_checkpoint_resource_and_atomic_adoption_parity() -> None:
    context = _context()
    receipt = context.controller.authority_receipt(
        context.prepared,
        context.installation_authority,
        replacement_authorized=True,
    )
    committed = context.controller.commit(
        context.retired_state,
        context.prepared,
        receipt,
    )
    assert bool(committed.diagnostics.replacement_applied)
    for ordinary, replacement in ((True, True), (True, False), (False, False)):
        eager = context.controller._atomic_adoption_kernel(
            context.retired_state,
            context.prepared.fallback_state,
            committed.state,
            jnp.asarray(ordinary, dtype=jnp.bool_),
            jnp.asarray(replacement, dtype=jnp.bool_),
        )
        compiled = context.controller._compiled_atomic_adoption_kernel(
            context.retired_state,
            context.prepared.fallback_state,
            committed.state,
            jnp.asarray(ordinary, dtype=jnp.bool_),
            jnp.asarray(replacement, dtype=jnp.bool_),
        )
        chex.assert_trees_all_equal(eager, compiled)

    payload = context.controller.checkpoint_payload(context.retired_state)
    assert payload["proposal_persisted"] is False
    controller_fields = payload["controller_fields"]
    assert isinstance(controller_fields, dict)
    assert all("prepared" not in name for name in controller_fields)
    assert all("candidate" not in name for name in controller_fields)
    installation = context.retired_state.scheduler_state.installation_state
    restore_kwargs = {
        "expected_semantic_generation": GENERATION,
        "expected_source_digest": SOURCE,
        "expected_consumer_source_digest": CONSUMER_SOURCE,
        "expected_consumer_representation_digest": REPRESENTATION,
        "expected_lifecycle_id": LIFECYCLE_ID,
        "expected_installation_authority_issuer_digest": INSTALLATION_ISSUER,
        "expected_retirement_authority_issuer_digest": RETIREMENT_ISSUER,
        "expected_controller_owner_digest": CONTROLLER_OWNER,
        "expected_descriptor_generation": context.retired_state.descriptor_generation,
        "expected_descriptor_digest": context.retired_state.descriptor_digest,
        "expected_installed_bundle": installation.installed_bundle,
    }
    restored = context.controller.restore_checkpoint(
        copy.deepcopy(payload),
        **restore_kwargs,
    )
    chex.assert_trees_all_equal(restored, context.retired_state)

    external_tampers = {
        "expected_semantic_generation": GENERATION + 1,
        "expected_source_digest": SOURCE + jnp.uint32(1),
        "expected_consumer_source_digest": CONSUMER_SOURCE + jnp.uint32(1),
        "expected_consumer_representation_digest": REPRESENTATION + jnp.uint32(1),
        "expected_lifecycle_id": LIFECYCLE_ID + jnp.uint32(1),
        "expected_installation_authority_issuer_digest": (INSTALLATION_ISSUER + jnp.uint32(1)),
        "expected_retirement_authority_issuer_digest": (RETIREMENT_ISSUER + jnp.uint32(1)),
        "expected_controller_owner_digest": CONTROLLER_OWNER + jnp.uint32(1),
        "expected_descriptor_generation": (
            context.retired_state.descriptor_generation + jnp.int32(1)
        ),
        "expected_descriptor_digest": (context.retired_state.descriptor_digest + jnp.uint32(1)),
    }
    for name, tampered_value in external_tampers.items():
        kwargs = dict(restore_kwargs)
        kwargs[name] = tampered_value
        with pytest.raises(ValueError, match="invalid|stale|rebound"):
            context.controller.restore_checkpoint(
                copy.deepcopy(payload),
                **kwargs,
            )

    config_tamper = copy.deepcopy(payload)
    replacement_config = config_tamper["config"]
    assert isinstance(replacement_config, dict)
    replacement_child = replacement_config["replacement"]
    assert isinstance(replacement_child, dict)
    replacement_child["go_no_go_authority"] = True
    with pytest.raises(ValueError, match="config"):
        context.controller.restore_checkpoint(
            config_tamper,
            **restore_kwargs,
        )

    field_tamper = copy.deepcopy(payload)
    fields = field_tamper["controller_fields"]
    assert isinstance(fields, dict)
    encoded_owner = fields["controller_owner_digest"]
    assert isinstance(encoded_owner, dict)
    encoded_owner["bytes_hex"] = "00" * 32
    with pytest.raises(ValueError, match="digest|invalid|rebound"):
        context.controller.restore_checkpoint(
            field_tamper,
            **restore_kwargs,
        )
    wrong = copy.deepcopy(payload)
    wrong["unexpected"] = False
    with pytest.raises(ValueError, match="keys differ"):
        context.controller.restore_checkpoint(
            wrong,
            **restore_kwargs,
        )

    budget = context.controller.resource_budget(
        context.retired_state,
        context.prepared,
    )
    assert budget.assessment == AUTHORIZED_OPTION_REPLACEMENT_ASSESSMENT
    assert budget.duplicated_installation_state_nbytes == 0
    assert budget.pending_proposal_slots == 0
    assert budget.commit_preparation_recomputations == 1
    assert budget.host_prepare and budget.host_commit
    assert not budget.jit_commit and budget.jit_atomic_adoption_kernel
    assert not budget.evidence_authority
    assert not budget.safety_authority


@pytest.mark.unit
def test_config_is_exactly_one_and_init_does_not_accept_retired_splice() -> None:
    with pytest.raises(ValueError, match="max_replacements"):
        AuthorizedOptionReplacementConfig(max_replacements=2)
    config = AuthorizedOptionReplacementConfig()
    assert AuthorizedOptionReplacementConfig.from_config(config.to_config()) == config
    for field in (
        "evidence_authority",
        "promotion_authority",
        "safety_authority",
        "go_no_go_authority",
        "retirement_authority",
        "discovery_authority",
        "dispatch_authority",
        "autonomous_curation_authority",
        "scientific_promotion_allowed",
    ):
        payload = config.to_config()
        payload[field] = True
        with pytest.raises(ValueError, match=field):
            AuthorizedOptionReplacementConfig.from_config(payload)
    scheduler = _scheduler()
    state = _init(scheduler, seed=44)
    retirement = AuthorizedOptionRetirementController(scheduler.installation)
    controller = AuthorizedOptionReplacementController(scheduler, retirement)
    with pytest.raises(TypeError):
        controller.init(
            state,
            retirement_authority_issuer_digest=RETIREMENT_ISSUER,
            controller_owner_digest=CONTROLLER_OWNER,
            retired_state=retirement,
        )
