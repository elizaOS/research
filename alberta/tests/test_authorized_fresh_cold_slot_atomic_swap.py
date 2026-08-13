# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Red-first contracts for authorized v2 filtered-cohort atomic adoption."""

from __future__ import annotations

import copy
import dataclasses
import functools
import importlib
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_authorized_option_replacement import _context, _transition

from alberta_framework.core.fresh_cold_slot_cumulant_cohort import (
    FreshColdSlotCumulantCohortFilter,
    FreshColdSlotCumulantCohortFilterConfig,
    FreshColdSlotCumulantCohortSource,
)
from alberta_framework.core.option_lifecycle_audit import option_semantic_digest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

OUTER_ISSUER = option_semantic_digest({"authority": "fresh-cold-slot-atomic-v2-test"})
OUTER_OWNER = option_semantic_digest({"owner": "fresh-cold-slot-atomic-v2-test"})


def _api() -> tuple[type[Any], type[Any]]:
    module = importlib.import_module(
        "alberta_framework.core.authorized_fresh_cold_slot_atomic_swap"
    )
    return (
        module.AuthorizedFreshColdSlotAtomicSwapController,
        module.AuthorizedFreshColdSlotAtomicSwapConfig,
    )


def _cohort_filter(context: Any) -> FreshColdSlotCumulantCohortFilter:
    installation = context.controller.scheduler.installation
    return FreshColdSlotCumulantCohortFilter(
        installation,
        FreshColdSlotCumulantCohortFilterConfig.from_installation(installation),
    )


def _fresh_prepared(
    context: Any,
    cohort_filter: FreshColdSlotCumulantCohortFilter,
    arm_inputs: Any,
    live: Any,
) -> Any:
    retired_installation = context.retired_state.scheduler_state.installation_state
    source = FreshColdSlotCumulantCohortSource(
        discovery_result=context.prepared.scheduler_result.discovery,
        installed_bundle=retired_installation.installed_bundle,
        installed_semantic_digests=retired_installation.installed_semantic_digests,
        installed_slot_mask=context.retired_state.installed_slot_mask,
        previous_raw_features=arm_inputs.current_raw_features,
        previous_raw_available=arm_inputs.current_raw_available,
        live_inputs=live,
    )
    return cohort_filter.prepare(source)


@functools.cache
def _fixture() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    controller_type, config_type = _api()
    context = _context()
    arm_inputs, observation, live = _transition(
        context.retired_state.scheduler_state,
        context.next_step,
    )
    cohort_filter = _cohort_filter(context)
    fresh = _fresh_prepared(context, cohort_filter, arm_inputs, live)
    assert bool(fresh.diagnostics.candidate_ready)
    controller = controller_type(context.controller, cohort_filter, config_type())
    state = controller.init(
        context.pre_retirement_state,
        authority_issuer_digest=OUTER_ISSUER,
        controller_owner_digest=OUTER_OWNER,
    )
    install_key = jr.key(801, impl="threefry2x32")
    successor_key = jr.key(802, impl="threefry2x32")
    prepared = controller.prepare(
        state,
        context.retirement_handoff,
        context.retirement_authority,
        context.phase_one_key,
        context.phase_two_key,
        arm_inputs,
        observation,
        live,
        fresh,
        install_key,
        successor_key,
    )
    return (
        controller,
        state,
        prepared,
        context,
        cohort_filter,
        fresh,
        install_key,
        successor_key,
    )


def _authority(controller: Any, prepared: Any, context: Any, **changes: Any) -> Any:
    values: dict[str, Any] = {
        "authority_issuer_digest": OUTER_ISSUER,
        "authority_revision_words": jnp.asarray([0, 1], dtype=jnp.uint32),
        "swap_authorized": True,
        "outer_veto_passed": True,
    }
    values.update(changes)
    return controller.authority_receipt(
        prepared,
        context.installation_authority,
        **values,
    )


def test_v2_config_state_and_resource_contract_are_explicit_and_nonauthoritative() -> None:
    controller, state, prepared, _context_api, _filter, _fresh, _key, _next = _fixture()
    config = controller.config
    assert config.to_config()["schema_version"].endswith(".v2")
    assert type(config).from_config(config.to_config()) == config
    wrong = config.to_config()
    wrong["go_no_go_authority"] = True
    with pytest.raises(ValueError, match="go_no_go_authority"):
        type(config).from_config(wrong)
    assert bool(controller.state_valid(state))
    assert prepared.schema_version.endswith(".v2")
    assert controller.state_schema.endswith(".v2")
    budget = controller.resource_budget(state, prepared)
    assert budget.persistent_state_nbytes > 0
    assert budget.prepared_state_nbytes > budget.persistent_state_nbytes
    assert budget.caller_keys_per_preparation == 4
    assert budget.wrapper_rng_split_calls_per_commit == 0
    assert budget.wrapper_generated_root_keys_per_commit == 0
    assert budget.child_rng_uses_supplied_caller_keys_only
    assert budget.max_adopted_installations_per_commit == 1
    assert budget.prepare_retirement_derivations == 1
    assert budget.prepare_retirement_rebind_evaluations == 2
    assert budget.prepare_scheduler_observations == 3
    assert budget.prepare_filter_derivations == 2
    assert budget.prepare_candidate_installation_evaluations == 1
    assert budget.commit_preparation_recomputations == 1
    assert budget.commit_retirement_derivations == 1
    assert budget.commit_retirement_rebind_evaluations == 2
    assert budget.commit_lower_preparation_recomputations == 2
    assert budget.commit_scheduler_observations == 6
    assert budget.commit_filter_derivations == 2
    assert budget.commit_candidate_installation_evaluations == 3
    assert budget.max_transient_cold_destinations == 1
    assert budget.persistent_cold_destinations == 0
    assert not budget.go_no_go_authority
    assert not budget.safety_authority
    assert not budget.evidence_authority
    assert not budget.promotion_authority
    assert not budget.delight_available
    assert budget.actor_backward_calls == 0


def test_fresh_filtered_swap_is_all_installed_to_all_installed_and_replay_safe() -> None:
    controller, state, prepared, context, _filter, fresh, install_key, successor_key = (
        _fixture()
    )
    before = copy.deepcopy(state)
    assert bool(prepared.diagnostics.source_state_valid)
    assert bool(prepared.diagnostics.source_all_slots_installed)
    assert bool(prepared.diagnostics.transient_retirement_applied)
    assert bool(prepared.diagnostics.filter_source_exact)
    assert bool(prepared.diagnostics.fresh_preparation_exact)
    assert bool(prepared.diagnostics.fresh_cohort_ready)
    assert bool(prepared.diagnostics.atomic_swap_ready)
    lower = prepared.external_adoption_prepared.scheduler_prepared.diagnostics
    assert bool(lower.retirement_scheduler_relation_valid)
    assert bool(lower.all_installed_source_mask_exact)
    assert bool(lower.one_cold_retired_destination_mask_exact)
    assert bool(lower.retirement_target_exact)
    assert bool(lower.retirement_authority_revision_bound)
    np.testing.assert_array_equal(fresh.changed_slots, [True, False, False, False])

    authority = _authority(controller, prepared, context)
    result = controller.commit(state, prepared, authority)
    assert bool(result.prepared_integrity_valid)
    assert bool(result.preparation_derivation_valid)
    assert bool(result.authority_valid)
    assert bool(result.transaction_applied)
    assert bool(result.retirement_applied)
    assert bool(result.replacement_applied)
    assert not bool(result.cold_state_persisted)
    assert bool(jnp.all(state.replacement_state.installed_slot_mask))
    assert bool(jnp.all(result.state.replacement_state.installed_slot_mask))
    np.testing.assert_array_equal(result.reset_slots, [True, False, False, False])
    np.testing.assert_array_equal(result.preserved_slots, [False, True, True, True])
    np.testing.assert_array_equal(
        result.state.replacement_state.scheduler_state.installation_state
        .installed_semantic_digests[1:],
        state.replacement_state.scheduler_state.installation_state
        .installed_semantic_digests[1:],
    )
    np.testing.assert_array_equal(
        jr.key_data(result.installation_key_consumed),
        jr.key_data(install_key),
    )
    np.testing.assert_array_equal(
        jr.key_data(
            result.state.replacement_state.scheduler_state.installation_rng_key
        ),
        jr.key_data(successor_key),
    )
    chex.assert_trees_all_equal(state, before)

    replay = controller.commit(result.state, prepared, authority)
    assert not bool(replay.transaction_applied)
    assert not bool(replay.cold_state_persisted)
    chex.assert_trees_all_equal(replay.state, result.state)


def test_no_fresh_decline_outer_veto_and_tamper_are_exact_source_noops() -> None:
    controller, state, prepared, context, cohort_filter, fresh, _key, _next = _fixture()
    authority = _authority(controller, prepared, context)

    declined = controller.commit(
        state,
        prepared,
        _authority(controller, prepared, context, swap_authorized=False),
    )
    vetoed = controller.commit(
        state,
        prepared,
        _authority(controller, prepared, context, outer_veto_passed=False),
    )
    tampered = controller.commit(
        state,
        dataclasses.replace(
            prepared,
            prepared_checksum=prepared.prepared_checksum + jnp.uint32(1),
        ),
        authority,
    )
    stale_authority = controller.commit(
        state,
        prepared,
        dataclasses.replace(
            authority,
            authority_revision_words=state.last_authority_revision_words,
        ),
    )
    wrong_identity = controller.commit(
        state,
        prepared,
        dataclasses.replace(
            authority,
            scheduler_identity_digest=authority.scheduler_identity_digest
            + jnp.uint32(1),
        ),
    )

    no_fresh_source = dataclasses.replace(
        fresh.source,
        installed_slot_mask=jnp.ones_like(fresh.source.installed_slot_mask),
    )
    no_fresh = cohort_filter.prepare(no_fresh_source)
    assert not bool(no_fresh.diagnostics.candidate_ready)
    no_fresh_prepared = controller.prepare(
        state,
        context.retirement_handoff,
        context.retirement_authority,
        context.phase_one_key,
        context.phase_two_key,
        prepared.arm_inputs,
        prepared.observation,
        prepared.live_inputs,
        no_fresh,
        prepared.installation_key,
        prepared.successor_scheduler_key,
    )
    assert not bool(no_fresh_prepared.diagnostics.fresh_preparation_exact)
    assert not bool(no_fresh_prepared.diagnostics.atomic_swap_ready)
    no_fresh_result = controller.commit(
        state,
        no_fresh_prepared,
        _authority(controller, no_fresh_prepared, context),
    )

    for name, result in {
        "decline": declined,
        "outer_veto": vetoed,
        "tamper": tampered,
        "stale_authority": stale_authority,
        "wrong_identity": wrong_identity,
        "no_fresh": no_fresh_result,
    }.items():
        assert not bool(result.transaction_applied), name
        assert not bool(result.retirement_applied), name
        assert not bool(result.replacement_applied), name
        assert not bool(result.cold_state_persisted), name
        chex.assert_trees_all_equal(result.state, state)


def test_install_and_successor_key_substitution_are_checksum_valid_exact_noops() -> None:
    controller, state, prepared, context, _filter, _fresh, _key, _next = _fixture()
    authority = _authority(controller, prepared, context)
    substitutions = {
        "installation_key": controller._with_prepared_checksum(
            dataclasses.replace(
                prepared,
                installation_key=jr.key(811, impl="threefry2x32"),
            )
        ),
        "successor_scheduler_key": controller._with_prepared_checksum(
            dataclasses.replace(
                prepared,
                successor_scheduler_key=jr.key(812, impl="threefry2x32"),
            )
        ),
    }
    for name, substituted in substitutions.items():
        result = controller.commit(state, substituted, authority)
        assert bool(result.prepared_integrity_valid), name
        assert not bool(result.preparation_derivation_valid), name
        assert not bool(result.transaction_applied), name
        assert not bool(result.cold_state_persisted), name
        chex.assert_trees_all_equal(result.state, state)

    authority_substitutions = {
        "installation_key_data": dataclasses.replace(
            authority,
            installation_key_data=jr.key_data(jr.key(813, impl="threefry2x32")),
        ),
        "successor_scheduler_key_data": dataclasses.replace(
            authority,
            successor_scheduler_key_data=jr.key_data(jr.key(814, impl="threefry2x32")),
        ),
    }
    for name, substituted in authority_substitutions.items():
        result = controller.commit(state, prepared, substituted)
        assert not bool(result.authority_valid), name
        assert not bool(result.transaction_applied), name
        assert not bool(result.cold_state_persisted), name
        chex.assert_trees_all_equal(result.state, state)


def test_checksum_valid_bundle_and_target_tamper_fail_full_rederivation() -> None:
    controller, state, prepared, context, cohort_filter, fresh, _key, _next = _fixture()
    authority = _authority(controller, prepared, context)

    edited_bundle = dataclasses.replace(
        fresh.filtered_bundle,
        selected_scores=fresh.filtered_bundle.selected_scores.at[0].add(
            jnp.asarray(0.25, dtype=jnp.float32)
        ),
        binding_digest=jnp.zeros((2,), dtype=jnp.uint32),
    )
    edited_bundle = dataclasses.replace(
        edited_bundle,
        binding_digest=cohort_filter._bundle_checksum(edited_bundle),
    )
    edited_fresh = cohort_filter._with_prepared_checksum(
        dataclasses.replace(fresh, filtered_bundle=edited_bundle)
    )
    bundle_tamper = controller._with_prepared_checksum(
        dataclasses.replace(prepared, supplied_fresh_prepared=edited_fresh)
    )

    foreign_target = jnp.roll(
        fresh.target_mask,
        shift=1,
    )
    edited_target_fresh = cohort_filter._with_prepared_checksum(
        dataclasses.replace(fresh, target_mask=foreign_target)
    )
    target_tamper = controller._with_prepared_checksum(
        dataclasses.replace(
            prepared,
            supplied_fresh_prepared=edited_target_fresh,
        )
    )

    for name, tampered in {
        "checksum_valid_bundle": bundle_tamper,
        "checksum_valid_target": target_tamper,
    }.items():
        result = controller.commit(state, tampered, authority)
        assert bool(result.prepared_integrity_valid), name
        assert not bool(result.preparation_derivation_valid), name
        assert not bool(result.transaction_applied), name
        assert not bool(result.cold_state_persisted), name
        chex.assert_trees_all_equal(result.state, state)


def test_foreign_controller_filter_and_config_identities_fail_closed() -> None:
    controller, state, prepared, context, _filter, _fresh, _key, _next = _fixture()
    authority = _authority(controller, prepared, context)
    foreign_digest = option_semantic_digest({"identity": "foreign-v2-filter-config"})

    foreign_state = controller._with_state_checksum(
        dataclasses.replace(state, filter_identity_digest=foreign_digest)
    )
    assert not bool(controller.state_valid(foreign_state))
    foreign_preparation = controller.prepare(
        foreign_state,
        prepared.retirement_handoff,
        prepared.retirement_authority,
        prepared.phase_one_key,
        prepared.phase_two_key,
        prepared.arm_inputs,
        prepared.observation,
        prepared.live_inputs,
        prepared.supplied_fresh_prepared,
        prepared.installation_key,
        prepared.successor_scheduler_key,
    )
    assert not bool(foreign_preparation.diagnostics.identities_exact)
    assert not bool(foreign_preparation.diagnostics.atomic_swap_ready)

    prepared_identity_tamper = controller._with_prepared_checksum(
        dataclasses.replace(prepared, controller_identity_digest=foreign_digest)
    )
    prepared_result = controller.commit(state, prepared_identity_tamper, authority)
    assert bool(prepared_result.prepared_integrity_valid)
    assert not bool(prepared_result.preparation_derivation_valid)
    assert not bool(prepared_result.transaction_applied)
    chex.assert_trees_all_equal(prepared_result.state, state)

    for name, receipt in {
        "foreign_filter": dataclasses.replace(
            authority,
            filter_identity_digest=foreign_digest,
        ),
        "foreign_controller": dataclasses.replace(
            authority,
            controller_identity_digest=foreign_digest,
        ),
        "foreign_replacement_config": dataclasses.replace(
            authority,
            replacement_identity_digest=foreign_digest,
        ),
    }.items():
        result = controller.commit(state, prepared, receipt)
        assert not bool(result.authority_valid), name
        assert not bool(result.transaction_applied), name
        assert not bool(result.cold_state_persisted), name
        chex.assert_trees_all_equal(result.state, state)


def test_v1_config_checkpoint_and_dataclass_schemas_remain_unchanged() -> None:
    controller, state, prepared, _context_api, _filter, _fresh, _key, _next = _fixture()
    replacement = controller.replacement
    source = state.replacement_state
    config_before = copy.deepcopy(replacement.to_config())
    checkpoint_before = replacement.checkpoint_payload(source)

    controller.resource_budget(state, prepared)

    assert replacement.to_config() == config_before
    chex.assert_trees_all_equal(
        replacement.checkpoint_payload(source),
        checkpoint_before,
    )
    assert tuple(field.name for field in dataclasses.fields(type(source))) == (
        "scheduler_state",
        "canonical_scheduler_checksum",
        "installed_slot_mask",
        "descriptor_generation",
        "descriptor_digest",
        "expected_retirement_authority_issuer_digest",
        "controller_owner_digest",
        "controller_revision",
        "retirement_words",
        "last_retirement_authority_revision_words",
        "last_retirement_scheduler_step_words",
        "retirement_unavailable",
        "retirement_error",
        "replacement_words",
        "last_replacement_authority_revision_words",
        "binding_checksum",
    )
    assert tuple(
        field.name for field in dataclasses.fields(type(source.scheduler_state))
    ) == (
        "discovery_state",
        "installation_state",
        "installation_rng_key",
        "expected_authority_issuer_digest",
        "step_words",
        "proposal_observation_words",
        "install_attempt_words",
        "install_applied_words",
        "maintenance_handoff_words",
        "control_update_words",
        "last_authority_revision_words",
        "retry_streak",
        "retry_due",
        "schedule_unavailable",
        "schedule_error",
        "binding_checksum",
    )
