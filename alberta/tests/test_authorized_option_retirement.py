# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Adversarial contracts for authority-gated option retirement."""

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
from test_cumulant_option_installation import (
    _bundle as _installation_bundle,
)
from test_cumulant_option_installation import (
    _composition as _installation_composition,
)
from test_cumulant_option_installation import (
    _init as _installation_init,
)
from test_cumulant_option_installation import (
    _inputs as _installation_inputs,
)

from alberta_framework.core.authorized_option_retirement import (
    AUTHORIZED_OPTION_RETIREMENT_ASSESSMENT,
    AuthorizedOptionRetirementConfig,
    AuthorizedOptionRetirementController,
    AuthorizedOptionRetirementState,
    OptionRetirementAuthorityReceipt,
)
from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionLiveInputs,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionRetirementHandoff,
)
from alberta_framework.core.option_lifecycle_audit import option_semantic_digest
from alberta_framework.core.stomp_option_lifecycle import STOMPOptionLifecycle

ISSUER = option_semantic_digest({"authority": "authorized-retirement-test"})

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _clear_jax_caches_after_test() -> Iterator[None]:
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]


class _Context(NamedTuple):
    controller: AuthorizedOptionRetirementController
    state: AuthorizedOptionRetirementState
    handoff: CumulantOptionRetirementHandoff
    receipt: OptionRetirementAuthorityReceipt
    phase_one_key: jax.Array
    phase_two_key: jax.Array


@functools.cache
def _base_installation() -> tuple[Any, Any]:
    installation = _installation_composition()
    state = _installation_init(installation, seed=0)
    prior = _installation_inputs(
        1,
        raw=(0.1, 0.2),
        event=0.0,
        atom=0.0,
        bottleneck=0.0,
    )
    cold = installation.materialize_cold(state, prior)
    current = _installation_inputs(
        2,
        raw=(0.3, 0.4),
        event=0.25,
        atom=0.35,
        bottleneck=0.45,
    )
    bundle = _installation_bundle(
        installation,
        current,
        prior.raw_features,
    )
    installed = installation.install(cold.state, bundle, jr.key(8), inputs=current)
    assert bool(installed.applied)
    return installation, installed.state


def _supported_installation(
    *,
    positive_margin: bool = False,
    missing_support: bool = False,
    active_option: bool = False,
) -> tuple[AuthorizedOptionRetirementController, AuthorizedOptionRetirementState]:
    installation_api, installation = _base_installation()
    audit = installation_api.lifecycle.audit
    audit_state = installation.lifecycle_state.audit_state
    n = audit.config.n_options
    signature_dim = audit.config.signature_dim
    context_signatures = jnp.zeros((n, 1, signature_dim), dtype=jnp.float32)
    context_signatures = context_signatures.at[:, 0, 0].set(
        jnp.arange(n, dtype=jnp.float32) * 20.0
    )
    treatment_counts = jnp.full((n, 1), 2, dtype=jnp.int32)
    primitive_counts = jnp.full((n, 1), 2, dtype=jnp.int32)
    if missing_support:
        treatment_counts = treatment_counts.at[0, 0].set(1)
    treatment_rewards = jnp.zeros((n, 1), dtype=jnp.float32)
    if positive_margin:
        treatment_rewards = treatment_rewards.at[0, 0].set(2.0)
    active_changes: dict[str, jax.Array] = {}
    if active_option:
        active_changes = {
            "active_option": jnp.asarray(0, dtype=jnp.int32),
            "active_context": jnp.asarray(0, dtype=jnp.int32),
            "active_generation": audit_state.semantic_generations[0],
            "active_steps": jnp.asarray(1, dtype=jnp.int32),
            "active_discount": jnp.asarray(1.0, dtype=jnp.float32),
        }
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
        context_signature_sums=context_signatures,
        comparison_treatment_counts=treatment_counts,
        comparison_primitive_counts=primitive_counts,
        comparison_treatment_ipw_reward_sums=treatment_rewards,
        comparison_treatment_ipw_masses=jnp.full((n, 1), 2.0, dtype=jnp.float32),
        comparison_primitive_ipw_masses=jnp.full((n, 1), 2.0, dtype=jnp.float32),
        **active_changes,
    )
    audit_state = audit._with_checksum(audit_state)
    assert bool(audit.state_valid(audit_state))
    lifecycle = installation_api.lifecycle.with_external_semantic_digests(
        installation.installed_semantic_digests
    )
    lifecycle_state = lifecycle._with_checksum(
        dataclasses.replace(
            installation.lifecycle_state,
            audit_state=audit_state,
        )
    )
    installation = installation_api._with_checksum(
        dataclasses.replace(
            installation,
            lifecycle_state=lifecycle_state,
        )
    )
    assert bool(installation_api.state_valid(installation))
    controller = AuthorizedOptionRetirementController(
        installation_api,
        AuthorizedOptionRetirementConfig(
            minimum_context_support=2,
            maximum_completion_reliability=0.5,
            minimum_normalized_model_rmse=1.0,
            maximum_planning_uses=0,
            max_retirements=2,
        ),
    )
    state = controller.init(
        installation,
        authority_issuer_digest=ISSUER,
        controller_owner_digest=option_semantic_digest(
            {"owner": "authorized-option-retirement-test"}
        ),
    )
    return controller, state


def _handoff(
    controller: AuthorizedOptionRetirementController,
    state: AuthorizedOptionRetirementState,
    *,
    scheduler_step: int = 10,
) -> CumulantOptionRetirementHandoff:
    installation = state.installation_state
    lifecycle = installation.lifecycle_state
    audit = lifecycle.audit_state
    report = controller.installation.lifecycle.audit.maintenance_report(audit)
    return CumulantOptionRetirementHandoff(
        report=report,
        available=jnp.asarray(True, dtype=jnp.bool_),
        scheduler_step_words=jnp.asarray([0, scheduler_step], dtype=jnp.uint32),
        discovery_semantic_generation=state.descriptor_generation,
        discovery_source_digest=installation.installed_bundle.source_digest,
        discovery_canonical_digest=installation.installed_bundle.canonical_digest,
        last_transition_id=installation.last_materialization_transition_id,
        consumer_source_digest=installation.consumer_source_digest,
        consumer_representation_digest=installation.consumer_representation_digest,
        lifecycle_id=installation.lifecycle_id,
        installation_revision=installation.revision,
        lifecycle_revision=lifecycle.revision,
        audit_revision=audit.revision,
        option_semantic_digests=installation.installed_semantic_digests,
        option_semantic_generations=audit.semantic_generations,
        proposed_retirement_slots=report.proposed_replacement_slots,
        proposed_retirement_mask=report.proposed_replacement_mask,
        retirement_authority=jnp.asarray(False, dtype=jnp.bool_),
        go_no_go_authority=jnp.asarray(False, dtype=jnp.bool_),
        safety_authority=jnp.asarray(False, dtype=jnp.bool_),
    )


def _receipt(
    state: AuthorizedOptionRetirementState,
    handoff: CumulantOptionRetirementHandoff,
    phase_one_key: jax.Array,
    phase_two_key: jax.Array,
    *,
    revision: int = 1,
) -> OptionRetirementAuthorityReceipt:
    installation = state.installation_state
    lifecycle = installation.lifecycle_state
    audit = lifecycle.audit_state
    return OptionRetirementAuthorityReceipt(
        retirement_authorized=jnp.asarray(True, dtype=jnp.bool_),
        go_no_go_authorized=jnp.asarray(True, dtype=jnp.bool_),
        safety_boundary_authorized=jnp.asarray(True, dtype=jnp.bool_),
        issuer_digest=state.expected_authority_issuer_digest,
        controller_owner_digest=state.controller_owner_digest,
        authority_revision_words=jnp.asarray([0, revision], dtype=jnp.uint32),
        valid_from_scheduler_step_words=jnp.asarray([0, 1], dtype=jnp.uint32),
        valid_through_scheduler_step_words=jnp.asarray([0, 100], dtype=jnp.uint32),
        scheduler_step_words=handoff.scheduler_step_words,
        descriptor_generation=state.descriptor_generation,
        descriptor_digest=state.descriptor_digest,
        discovery_source_digest=installation.installed_bundle.source_digest,
        discovery_canonical_digest=installation.installed_bundle.canonical_digest,
        consumer_source_digest=installation.consumer_source_digest,
        consumer_representation_digest=installation.consumer_representation_digest,
        lifecycle_id=installation.lifecycle_id,
        installation_revision=installation.revision,
        lifecycle_revision=lifecycle.revision,
        audit_revision=audit.revision,
        controller_revision=state.controller_revision,
        option_semantic_digests=installation.installed_semantic_digests,
        option_semantic_generations=audit.semantic_generations,
        retirement_slots=handoff.proposed_retirement_slots,
        retirement_mask=handoff.proposed_retirement_mask,
        phase_one_key_data=jr.key_data(phase_one_key),
        phase_two_key_data=jr.key_data(phase_two_key),
    )


def _context(
    *,
    positive_margin: bool = False,
    missing_support: bool = False,
    active_option: bool = False,
) -> _Context:
    controller, state = _supported_installation(
        positive_margin=positive_margin,
        missing_support=missing_support,
        active_option=active_option,
    )
    handoff = _handoff(controller, state)
    assert int(handoff.proposed_retirement_slots[0]) == 0
    assert bool(handoff.proposed_retirement_mask[0])
    phase_one_key = jr.key(101, impl="threefry2x32")
    phase_two_key = jr.key(202, impl="threefry2x32")
    receipt = _receipt(state, handoff, phase_one_key, phase_two_key)
    return _Context(
        controller,
        state,
        handoff,
        receipt,
        phase_one_key,
        phase_two_key,
    )


@pytest.mark.unit
def test_config_resources_and_l0_authority_contract_are_strict() -> None:
    config = AuthorizedOptionRetirementConfig()
    assert AuthorizedOptionRetirementConfig.from_config(config.to_config()) == config
    payload = config.to_config()
    payload["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="scientific_promotion_allowed"):
        AuthorizedOptionRetirementConfig.from_config(payload)
    with pytest.raises(ValueError, match="maximum_completion_reliability"):
        dataclasses.replace(config, maximum_completion_reliability=1.1)

    context = _context()
    budget = context.controller.resource_budget(context.state)
    assert budget.assessment == AUTHORIZED_OPTION_RETIREMENT_ASSESSMENT
    assert budget.pending_proposal_slots == 0
    assert budget.public_rebind_calls_per_applied_retirement == 2
    assert budget.reset_keys_per_applied_retirement == 2
    assert not budget.evidence_authority
    assert not budget.safety_authority
    assert not budget.autonomous_curation_authority
    assert not budget.scientific_promotion_allowed


@pytest.mark.unit
def test_positive_randomized_primitive_margin_is_a_noncompensating_veto() -> None:
    context = _context(positive_margin=True)
    result = context.controller.retire(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert not bool(result.no_positive_randomized_margin[0])
    assert not bool(result.policy_eligible[0])
    assert not bool(result.transaction_applied)
    chex.assert_trees_all_equal(result.state, context.state)


@pytest.mark.unit
def test_missing_context_support_blocks_retirement() -> None:
    context = _context(missing_support=True)
    result = context.controller.retire(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert not bool(result.minimum_context_support[0])
    assert not bool(result.policy_eligible[0])
    assert not bool(result.transaction_applied)
    chex.assert_trees_all_equal(result.state, context.state)


@pytest.mark.unit
def test_active_option_defers_without_consuming_authority_or_keys() -> None:
    context = _context(active_option=True)
    result = context.controller.retire(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert not bool(result.quiescent)
    assert not bool(result.transaction_applied)
    chex.assert_trees_all_equal(result.state, context.state)


@pytest.mark.unit
def test_stale_replayed_authority_and_semantic_mismatch_are_exact_noops() -> None:
    context = _context()
    first = context.controller.retire(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert bool(first.transaction_applied)
    replay = context.controller.retire(
        first.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert not bool(replay.authority_valid)
    assert not bool(replay.transaction_applied)
    chex.assert_trees_all_equal(replay.state, first.state)

    bad_receipt = dataclasses.replace(
        context.receipt,
        descriptor_digest=context.receipt.descriptor_digest.at[0].add(jnp.uint32(1)),
    )
    mismatch = context.controller.retire(
        context.state,
        context.handoff,
        bad_receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert not bool(mismatch.authority_valid)
    assert not bool(mismatch.transaction_applied)
    chex.assert_trees_all_equal(mismatch.state, context.state)

    nonfinite_report = dataclasses.replace(
        context.handoff.report,
        completion_reliability=(
            context.handoff.report.completion_reliability.at[0].set(jnp.nan)
        ),
    )
    nonfinite_handoff = dataclasses.replace(
        context.handoff,
        report=nonfinite_report,
    )
    nonfinite = context.controller.retire(
        context.state,
        nonfinite_handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert not bool(nonfinite.handoff_valid)
    assert not bool(nonfinite.transaction_applied)
    chex.assert_trees_all_equal(nonfinite.state, context.state)


@pytest.mark.unit
def test_two_public_rebinds_scrub_whole_slot_without_identity_or_knowledge_leak() -> None:
    context = _context()
    before = context.state.installation_state
    original_semantics = before.installed_semantic_digests
    original_descriptors = before.installed_bundle.selected_descriptors
    result = context.controller.retire(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert bool(result.transaction_applied)
    assert bool(result.phase_one_applied)
    assert bool(result.phase_two_applied)
    np.testing.assert_array_equal(result.reset_slots, (True, False, False, False))
    after = result.state.installation_state
    assert bool(context.controller.state_valid(result.state))
    assert bool(context.controller.installation.state_valid(after))
    np.testing.assert_array_equal(after.installed_semantic_digests, original_semantics)
    np.testing.assert_array_equal(
        after.lifecycle_state.audit_state.semantic_digests,
        original_semantics,
    )
    np.testing.assert_array_equal(after.installed_bundle.selected_descriptors, original_descriptors)
    assert int(after.installed_bundle.semantic_generation) == int(
        before.installed_bundle.semantic_generation
    )
    assert not bool(result.state.installed_slot_mask[0])
    assert bool(result.cold_mask_active)
    assert not bool(result.extended_action_mask[2])
    assert not bool(result.replacement_installed)

    fresh = jax.jit(context.controller.installation.stomp_agent.init)(
        context.phase_two_key
    )
    post_stomp = after.lifecycle_state.stomp_state
    np.testing.assert_array_equal(
        post_stomp.option_policies.q_weights[0],
        fresh.option_policies.q_weights[0],
    )
    np.testing.assert_array_equal(
        post_stomp.option_policies.traces[0],
        fresh.option_policies.traces[0],
    )
    np.testing.assert_array_equal(
        post_stomp.option_models.next_state_weights[0],
        fresh.option_models.next_state_weights[0],
    )
    head = context.controller.installation.stomp_agent.config.n_primitive_actions
    np.testing.assert_array_equal(
        post_stomp.base_learner_state.head_params.weights[head],
        fresh.base_learner_state.head_params.weights[head],
    )
    chex.assert_trees_all_equal(
        post_stomp.base_learner_state.head_optimizer_states[head],
        fresh.base_learner_state.head_optimizer_states[head],
    )
    chex.assert_trees_all_equal(
        post_stomp.base_learner_state.head_traces[head],
        fresh.base_learner_state.head_traces[head],
    )
    audit = after.lifecycle_state.audit_state
    assert int(audit.execution_starts[0]) == 0
    assert int(audit.model_error_counts[0]) == 0
    assert int(audit.planning_use_counts[0]) == 0
    np.testing.assert_array_equal(
        post_stomp.option_policies.q_weights[1],
        before.lifecycle_state.stomp_state.option_policies.q_weights[1],
    )
    assert int(audit.execution_starts[1]) == int(
        before.lifecycle_state.audit_state.execution_starts[1]
    )


@pytest.mark.unit
@pytest.mark.parametrize("failed_call", [1, 2])
def test_failure_of_either_public_rebind_rolls_back_whole_composition(
    monkeypatch: pytest.MonkeyPatch,
    failed_call: int,
) -> None:
    context = _context()
    original = STOMPOptionLifecycle.rebind
    calls = 0

    def broken_rebind(self: STOMPOptionLifecycle, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raw = original(self, *args, **kwargs)
        if calls == failed_call:
            return dataclasses.replace(raw, applied=jnp.asarray(False, dtype=jnp.bool_))
        return raw

    monkeypatch.setattr(STOMPOptionLifecycle, "rebind", broken_rebind)
    result = context.controller.retire(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    assert calls == 2
    assert not bool(result.transaction_applied)
    chex.assert_trees_all_equal(result.state, context.state)


@pytest.mark.unit
def test_capacity_exhaustion_is_fail_stop_and_never_auto_installs() -> None:
    base = _context()
    controller = AuthorizedOptionRetirementController(
        base.controller.installation,
        dataclasses.replace(base.controller.config, max_retirements=1),
    )
    state = controller.init(
        base.state.installation_state,
        authority_issuer_digest=base.state.expected_authority_issuer_digest,
        controller_owner_digest=base.state.controller_owner_digest,
    )
    handoff = _handoff(controller, state)
    receipt = _receipt(state, handoff, base.phase_one_key, base.phase_two_key)
    first = controller.retire(
        state,
        handoff,
        receipt,
        base.phase_one_key,
        base.phase_two_key,
    )
    assert bool(first.transaction_applied)
    assert bool(first.state.unavailable)
    assert not bool(first.replacement_installed)

    second_handoff = _handoff(controller, first.state, scheduler_step=11)
    second_receipt = _receipt(
        first.state,
        second_handoff,
        jr.key(303, impl="threefry2x32"),
        jr.key(404, impl="threefry2x32"),
        revision=2,
    )
    second = controller.retire(
        first.state,
        second_handoff,
        second_receipt,
        jr.key(303, impl="threefry2x32"),
        jr.key(404, impl="threefry2x32"),
    )
    assert not bool(second.capacity_available)
    assert bool(second.capacity_exhausted)
    assert not bool(second.transaction_applied)
    chex.assert_trees_all_equal(second.state, first.state)


@pytest.mark.unit
def test_checkpoint_round_trip_and_tampering_are_strict() -> None:
    context = _context()
    result = context.controller.retire(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    payload = context.controller.checkpoint_payload(result.state)
    restore_kwargs = {
        "expected_authority_issuer_digest": result.state.expected_authority_issuer_digest,
        "expected_controller_owner_digest": result.state.controller_owner_digest,
        "expected_descriptor_generation": result.state.descriptor_generation,
        "expected_descriptor_digest": result.state.descriptor_digest,
    }
    restored = context.controller.restore_checkpoint(
        copy.deepcopy(payload),
        **restore_kwargs,
    )
    chex.assert_trees_all_equal(restored, result.state)

    tampered = copy.deepcopy(payload)
    fields = tampered["controller_fields"]
    assert isinstance(fields, dict)
    revision = fields["controller_revision"]
    assert isinstance(revision, dict)
    revision["bytes_hex"] = "00" * 4
    with pytest.raises(ValueError, match="digest|invalid"):
        context.controller.restore_checkpoint(tampered, **restore_kwargs)


@pytest.mark.integration
def test_public_retirement_is_eager_jit_and_scan_replay_safe() -> None:
    context = _context()
    eager = context.controller.retire(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    compiled = jax.jit(context.controller.retire)(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    chex.assert_trees_all_equal(compiled, eager)

    def scan_step(
        state: AuthorizedOptionRetirementState,
        _: jax.Array,
    ) -> tuple[AuthorizedOptionRetirementState, jax.Array]:
        update = context.controller.retire(
            state,
            context.handoff,
            context.receipt,
            context.phase_one_key,
            context.phase_two_key,
        )
        return update.state, update.transaction_applied

    final_state, applied = jax.lax.scan(
        scan_step,
        context.state,
        jnp.arange(2, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(applied, (True, False))
    chex.assert_trees_all_equal(final_state, compiled.state)


@pytest.mark.integration
def test_retired_slot_remains_cold_through_live_materialization_and_control() -> None:
    context = _context()
    retired = context.controller.retire(
        context.state,
        context.handoff,
        context.receipt,
        context.phase_one_key,
        context.phase_two_key,
    )
    state = retired.state
    installation = state.installation_state
    transition = installation.last_materialization_transition_id.at[1].add(jnp.uint32(1))
    inputs = CumulantOptionLiveInputs(
        raw_features=installation.last_raw_features,
        raw_available=jnp.ones_like(installation.last_raw_available),
        controllable_events=jnp.asarray([0.0], dtype=jnp.float32),
        controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
        transition_atoms=jnp.asarray([0.0], dtype=jnp.float32),
        transition_atoms_available=jnp.ones((1,), dtype=jnp.bool_),
        bottleneck_values=jnp.asarray([0.0], dtype=jnp.float32),
        bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
        semantic_generation=state.descriptor_generation,
        source_digest=installation.installed_bundle.source_digest,
        canonical_digest=installation.installed_bundle.canonical_digest,
        transition_id=transition,
        state_observation_count=installation.last_materialization_observation_count
        + jnp.int32(1),
    )
    materialized = context.controller.materialize_live(state, inputs)
    assert bool(materialized.applied)
    started = context.controller.start(materialized.state, materialized.materialization)
    assert started.applied
    assert started.lifecycle_result is not None
    assert int(started.state.installation_state.lifecycle_state.stomp_state.executing_option) != 0
    mask = context.controller.extended_action_mask(started.state)
    assert not bool(mask[2])
