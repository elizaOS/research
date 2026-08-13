# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Single-owner, exact-trace, and atomic option-authority bridge contracts."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_authorized_option_replacement import _context
from test_cumulant_option_scheduler import _receipt as _installation_receipt
from test_cumulant_option_scheduler import _transition as _scheduler_transition

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core import (
    authorized_option_replacement,
    cumulant_option_installation,
    cumulant_option_scheduler,
    prototype_option_authority_bridge,
    stomp_option_lifecycle,
)
from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.option_search_control import (
    OptionSearchControl,
    OptionSearchControlConfig,
)
from alberta_framework.core.options import STOMPAgent, STOMPState
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.prototype_option_authority_bridge import (
    PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ASSESSMENT,
    PrototypeOptionAuthorityBridge,
    PrototypeOptionAuthorityBridgeRetirementPrepared,
    PrototypeOptionAuthorityBridgeState,
    _prototype_oak_state,
)
from alberta_framework.core.stomp_owner_finalization import (
    stomp_owner_finalization_trace_valid,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_module() -> Iterator[None]:
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]


class _BridgeContext(NamedTuple):
    lower: Any
    agent: PrototypeAgent
    bridge: PrototypeOptionAuthorityBridge
    source: PrototypeOptionAuthorityBridgeState
    retirement_prepared: PrototypeOptionAuthorityBridgeRetirementPrepared
    retired: PrototypeOptionAuthorityBridgeState


@pytest.fixture(scope="module")
def bridge_context() -> _BridgeContext:
    lower = _context()
    stomp_config = lower.controller.scheduler.installation.stomp_agent.config
    agent = PrototypeAgent(PrototypeAgentConfig(oak=OaKConfig(stomp=stomp_config)))
    bridge = PrototypeOptionAuthorityBridge(agent, lower.controller)
    pristine = agent.init(jr.key(999))
    receipt = bridge.declare_initial_owner_binding(
        pristine,
        lower.pre_retirement_state,
        binding_authorized=True,
    )
    bound = bridge.bind_initial_prototype_owner(
        pristine,
        lower.pre_retirement_state,
        receipt,
    )
    assert bool(bound.transaction_applied)
    source = bridge.init(bound.prototype_state, lower.pre_retirement_state)
    prepared = bridge.prepare_retirement(
        source,
        lower.retirement_handoff,
        lower.retirement_authority,
        lower.phase_one_key,
        lower.phase_two_key,
    )
    committed = bridge.commit_retirement(source, prepared)
    assert bool(committed.transaction_applied)
    return _BridgeContext(lower, agent, bridge, source, prepared, committed.state)


def _count_exact_stomp_owners(value: object) -> int:
    if type(value) is STOMPState:
        return 1
    if dataclasses.is_dataclass(value):
        return sum(
            _count_exact_stomp_owners(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, Mapping):
        return sum(_count_exact_stomp_owners(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_count_exact_stomp_owners(item) for item in value)
    return 0


def _attached_authority(
    context: _BridgeContext,
    state: PrototypeOptionAuthorityBridgeState,
) -> Any:
    oak = _prototype_oak_state(state.prototype_state.oak_state)
    attached = context.lower.controller.attach_borrowed_stomp(
        state.authority_metadata,
        oak.stomp_state,
    )
    assert bool(attached.transaction_applied)
    return attached.state


def _replacement_preparation(context: _BridgeContext) -> tuple[Any, Any]:
    attached = _attached_authority(context, context.retired)
    arm_inputs, observation, live_inputs = _scheduler_transition(
        attached.scheduler_state,
        context.lower.next_step,
    )
    arm = context.lower.controller.arm(attached, arm_inputs)
    prepared = context.bridge.prepare_replacement(
        context.retired,
        arm,
        observation,
        live_inputs,
    )
    authority = _installation_receipt(attached.scheduler_state, authorized=True)
    receipt = context.lower.controller.authority_receipt(
        prepared.authority_prepared,
        authority,
        replacement_authorized=True,
    )
    return prepared, receipt


def _search_dyna_bridge_context() -> _BridgeContext:
    lower = _context(option_planning_backups_per_step=0)
    stomp_config = lower.controller.scheduler.installation.stomp_agent.config
    agent = PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(stomp=stomp_config),
            option_search_control=OptionSearchControlConfig(backup_budget=1),
            world_model=ActionConditionedWorldModelConfig(
                observation_dim=stomp_config.observation_dim,
                n_actions=stomp_config.n_primitive_actions,
                hidden_sizes=(),
                step_size=0.1,
            ),
            dreaming=DreamingConfig(
                warmup_steps=0,
                max_model_error_ema=1.0e6,
            ),
            buffer_capacity=2,
            n_dreams_per_step=1,
        )
    )
    bridge = PrototypeOptionAuthorityBridge(agent, lower.controller)
    pristine = agent.init(jr.key(0xD1A))
    receipt = bridge.declare_initial_owner_binding(
        pristine,
        lower.pre_retirement_state,
        binding_authorized=True,
    )
    bound = bridge.bind_initial_prototype_owner(
        pristine,
        lower.pre_retirement_state,
        receipt,
    )
    assert bool(bound.transaction_applied)
    source = bridge.init(bound.prototype_state, lower.pre_retirement_state)
    prepared = bridge.prepare_retirement(
        source,
        lower.retirement_handoff,
        lower.retirement_authority,
        lower.phase_one_key,
        lower.phase_two_key,
    )
    committed = bridge.commit_retirement(source, prepared)
    assert bool(committed.transaction_applied)
    return _BridgeContext(lower, agent, bridge, source, prepared, committed.state)


def _start_observation(context: _BridgeContext) -> jax.Array:
    return jnp.linspace(
        -0.25,
        0.25,
        context.agent.config.oak.observation_dim,
        dtype=jnp.float32,
    )


def _next_transition(
    state: PrototypeOptionAuthorityBridgeState,
    next_observation: jax.Array,
) -> PrototypeTransition:
    prototype = state.prototype_state
    return PrototypeTransition(
        observation=prototype.current_raw_observation,
        action=prototype.current_action,
        decision_id=prototype.current_decision_id,
        reward=jnp.asarray(0.25, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
    )


def test_persistent_composition_has_one_stomp_owner_and_strict_checkpoint(
    bridge_context: _BridgeContext,
) -> None:
    context = bridge_context
    assert _count_exact_stomp_owners(context.source) == 1
    assert _count_exact_stomp_owners(context.source.authority_metadata) == 0
    assert (
        _count_exact_stomp_owners(
            context.source.authority_metadata.scheduler_metadata
        )
        == 0
    )
    assert (
        _count_exact_stomp_owners(
            context.source.authority_metadata.scheduler_metadata
            .installation_metadata
        )
        == 0
    )
    assert (
        _count_exact_stomp_owners(
            context.source.authority_metadata.scheduler_metadata
            .installation_metadata.lifecycle_metadata
        )
        == 0
    )
    assert _count_exact_stomp_owners(context.retired) == 1
    assert _count_exact_stomp_owners(context.retired.authority_metadata) == 0

    budget = context.bridge.resource_budget(context.retired)
    assert budget.persistent_stomp_state_owners == 1
    assert budget.detached_metadata_stomp_state_owners == 0
    assert budget.persistent_prepared_transactions == 0
    assert budget.real_control_stomp_updates_per_ordinary_transition == 1
    assert budget.configured_imagined_stomp_updates_per_ordinary_transition == 0
    assert budget.max_total_stomp_updates_per_ordinary_transition == 1
    assert budget.option_search_learner_updates_per_ordinary_transition == 0
    assert budget.stomp_updates_per_audit_adoption == 0
    assert budget.stomp_updates_per_retirement_transaction == 0
    assert budget.stomp_updates_per_replacement_transaction == 0
    assert budget.derivation_recomputed_on_audit_adoption is False
    assert budget.caller_authority_required is True
    assert budget.caller_authenticated is False
    assert budget.checksum_authenticated is False
    assert budget.assessment == PROTOTYPE_OPTION_AUTHORITY_BRIDGE_ASSESSMENT
    assert budget.benefit_claim is False
    assert budget.evidence_authority is False
    assert budget.promotion_authority is False
    assert budget.safety_authority is False
    assert budget.go_no_go_authority is False
    assert budget.retirement_authority is False
    assert budget.discovery_authority is False
    assert budget.dispatch_authority is False
    assert budget.autonomous_curation_authority is False
    assert budget.scientific_promotion_allowed is False

    payload = context.bridge.checkpoint_payload(context.retired)
    restored = context.bridge.restore_checkpoint(payload)
    chex.assert_trees_all_equal(restored, context.retired)
    assert payload["checksum_authenticated"] is False
    assert payload["caller_authenticated"] is False
    altered = dict(payload)
    altered["state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="state hash"):
        context.bridge.restore_checkpoint(altered)


def test_initial_owner_binding_is_explicit_directional_and_source_bound(
    bridge_context: _BridgeContext,
) -> None:
    context = bridge_context
    pristine = context.agent.init(jr.key(0xB17D))
    with pytest.raises(ValueError, match="explicit.*initial binding receipt"):
        context.bridge.init(pristine, context.lower.pre_retirement_state)

    receipt = context.bridge.declare_initial_owner_binding(
        pristine,
        context.lower.pre_retirement_state,
        binding_authorized=True,
    )
    bound = context.bridge.bind_initial_prototype_owner(
        pristine,
        context.lower.pre_retirement_state,
        receipt,
    )
    assert bool(bound.transaction_applied)
    assert bool(bound.canonical_owner_adopted)
    assert not bool(bound.caller_authenticated)

    idempotent = context.bridge.bind_initial_prototype_owner(
        pristine,
        context.lower.pre_retirement_state,
        receipt,
    )
    assert bool(idempotent.transaction_applied)
    chex.assert_trees_all_equal(idempotent.prototype_state, bound.prototype_state)

    initialized = context.bridge.init(
        bound.prototype_state,
        context.lower.pre_retirement_state,
    )
    assert bool(context.bridge.state_valid(initialized))

    tampered = receipt.replace(
        receipt_checksum=receipt.receipt_checksum.at[0].add(jnp.uint32(1))
    )
    rejected = context.bridge.bind_initial_prototype_owner(
        pristine,
        context.lower.pre_retirement_state,
        tampered,
    )
    assert not bool(rejected.receipt_integrity_valid)
    assert not bool(rejected.transaction_applied)
    chex.assert_trees_all_equal(rejected.prototype_state, pristine)

    stale_source = context.agent.start(
        bound.prototype_state,
        _start_observation(context),
    )
    replay = context.bridge.bind_initial_prototype_owner(
        stale_source,
        context.lower.pre_retirement_state,
        receipt,
    )
    assert not bool(replay.source_prototype_matches)
    assert not bool(replay.transaction_applied)
    chex.assert_trees_all_equal(replay.prototype_state, stale_source)


def test_retirement_is_two_phase_source_bound_and_atomic(
    bridge_context: _BridgeContext,
) -> None:
    context = bridge_context
    prepared = context.retirement_prepared
    chex.assert_trees_all_equal(prepared.source_state, context.source)
    chex.assert_trees_all_equal(context.source, prepared.source_state)
    assert bool(prepared.source_binding_valid)
    assert bool(prepared.authority_result.transaction_applied)
    assert bool(prepared.oak_rebind.transaction_applied)
    assert bool(prepared.preparation_valid)
    np.testing.assert_array_equal(prepared.reset_slots, [True, False, False, False])

    committed = context.bridge.commit_retirement(context.source, prepared)
    assert bool(committed.transaction_applied)
    assert bool(committed.prepared_integrity_valid)
    assert bool(committed.preparation_derivation_valid)
    assert bool(committed.exact_owner_rebind)
    assert bool(committed.cold_mask_applied)
    assert not bool(committed.caller_authenticated)
    np.testing.assert_array_equal(
        committed.state.extended_action_mask,
        [True, True, False, True, True, True],
    )
    assert bool(context.bridge.state_valid(committed.state))

    tampered = prepared.replace(reset_slots=jnp.roll(prepared.reset_slots, 1))
    tampered = context.bridge._with_retirement_prepared_checksum(tampered)
    rejected = context.bridge.commit_retirement(context.source, tampered)
    assert bool(rejected.prepared_integrity_valid)
    assert not bool(rejected.preparation_derivation_valid)
    assert not bool(rejected.transaction_applied)
    chex.assert_trees_all_equal(rejected.state, context.source)

    stale = context.bridge.commit_retirement(context.retired, prepared)
    assert not bool(stale.destination_matches_source)
    assert not bool(stale.transaction_applied)
    chex.assert_trees_all_equal(stale.state, context.retired)


def test_replacement_replays_lower_preparation_and_rebinds_same_owner(
    bridge_context: _BridgeContext,
) -> None:
    context = bridge_context
    prepared, receipt = _replacement_preparation(context)
    assert bool(prepared.source_binding_valid)
    assert bool(prepared.preparation_valid)
    assert bool(
        prepared.authority_prepared.diagnostics.candidate_ready_for_authority
    )
    committed = context.bridge.commit_replacement(
        context.retired,
        prepared,
        receipt,
    )
    assert bool(committed.transaction_applied)
    assert bool(committed.lower_preparation_derivation_valid)
    assert bool(committed.ordinary_advance_applied)
    assert bool(committed.replacement_applied)
    assert bool(committed.exact_owner_rebind)
    assert bool(committed.cold_mask_applied)
    assert not bool(committed.caller_authenticated)
    assert bool(context.bridge.state_valid(committed.state))
    assert bool(jnp.all(committed.state.extended_action_mask))
    assert _count_exact_stomp_owners(committed.state) == 1
    assert _count_exact_stomp_owners(committed.state.authority_metadata) == 0

    inner = prepared.authority_prepared.replace(
        target_slot=prepared.authority_prepared.target_slot + jnp.int32(1)
    )
    inner = context.lower.controller._with_prepared_checksum(inner)
    tampered = prepared.replace(authority_prepared=inner)
    # A coherent outer reseal is deliberately possible because the checksum
    # is unkeyed. The lower complete provenance replay remains authoritative.
    tampered = tampered.replace(
        prepared_checksum=prototype_option_authority_bridge._checksum_arrays(
            context.bridge._replacement_prepared_payload_arrays(tampered)
        )
    )
    rejected = context.bridge.commit_replacement(context.retired, tampered, receipt)
    assert bool(rejected.prepared_integrity_valid)
    assert not bool(rejected.lower_preparation_derivation_valid)
    assert not bool(rejected.transaction_applied)
    chex.assert_trees_all_equal(rejected.state, context.retired)


def test_one_stomp_call_receives_cold_mask_for_behavior_bootstrap_and_planning(
    bridge_context: _BridgeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = bridge_context
    start_masks: list[jax.Array] = []
    update_masks: list[jax.Array] = []
    planning_flags: list[bool] = []
    original_start = STOMPAgent.start_with_extended_action_mask
    original_update = STOMPAgent.update

    def counted_start(
        stomp: STOMPAgent,
        state: STOMPState,
        observation: jax.Array,
        extended_action_mask: jax.Array,
    ) -> Any:
        start_masks.append(extended_action_mask)
        return original_start(stomp, state, observation, extended_action_mask)

    def counted_update(stomp: STOMPAgent, *args: Any, **kwargs: Any) -> Any:
        update_masks.append(kwargs["extended_action_mask"])
        planning_flags.append(bool(kwargs["enable_planning"]))
        return original_update(stomp, *args, **kwargs)

    monkeypatch.setattr(STOMPAgent, "start_with_extended_action_mask", counted_start)
    observation = _start_observation(context)
    started = context.bridge.start(context.retired, observation)
    assert bool(started.transaction_applied)
    assert len(start_masks) == 1
    np.testing.assert_array_equal(start_masks[0], context.retired.extended_action_mask)
    oak = _prototype_oak_state(started.state.prototype_state.oak_state)
    assert int(oak.stomp_state.executing_option) != 0

    monkeypatch.setattr(STOMPAgent, "update", counted_update)
    next_observation = observation + jnp.float32(0.1)
    updated = context.bridge.update_transition(
        started.state,
        _next_transition(started.state, next_observation),
    )
    assert len(update_masks) == 1
    assert planning_flags == [True]
    np.testing.assert_array_equal(update_masks[0], started.state.extended_action_mask)
    assert int(updated.stomp_update_evaluations) == 1
    assert int(updated.prototype.oak_stomp_update_evaluations) == 1
    assert bool(updated.prototype.oak_stomp_update_available)
    assert not bool(updated.lifecycle.derivation_recomputed)
    assert bool(updated.lifecycle.caller_authority_required)
    assert not bool(updated.lifecycle.caller_authenticated)
    assert not bool(updated.control_transition_rolled_back_by_bridge)
    final_oak = _prototype_oak_state(updated.state.prototype_state.oak_state)
    chex.assert_trees_all_equal(
        updated.prototype.oak_stomp_update_result.state,
        final_oak.stomp_state,
    )
    np.testing.assert_array_equal(
        updated.prototype.oak_bootstrap_observation,
        next_observation,
    )
    np.testing.assert_array_equal(
        updated.prototype.oak_decision_observation,
        next_observation,
    )
    assert bool(updated.authority_metadata_advanced)
    assert bool(context.bridge.state_valid(updated.state))


def test_invalid_resealed_mask_is_never_committed_or_used_by_transient_control(
    bridge_context: _BridgeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = bridge_context
    observation = _start_observation(context)
    started = context.bridge.start(context.retired, observation).state
    forged_mask = started.extended_action_mask.at[2].set(True)
    forged = context.bridge._with_checksum(
        started.replace(extended_action_mask=forged_mask)
    )
    assert not bool(context.bridge.state_valid(forged))

    observed_masks: list[jax.Array] = []
    original_update = STOMPAgent.update

    def observed_update(stomp: STOMPAgent, *args: Any, **kwargs: Any) -> Any:
        observed_masks.append(kwargs["extended_action_mask"])
        return original_update(stomp, *args, **kwargs)

    monkeypatch.setattr(STOMPAgent, "update", observed_update)
    result = context.bridge.update_transition(
        forged,
        _next_transition(forged, observation + jnp.float32(0.2)),
    )

    assert len(observed_masks) == 1
    np.testing.assert_array_equal(
        observed_masks[0],
        [True, True, False, False, False, False],
    )
    assert not bool(result.prototype_control_applied)
    assert not bool(result.transaction_applied)
    assert not bool(result.authority_metadata_advanced)
    assert not bool(result.authority_desynchronized)
    chex.assert_trees_all_equal(result.state, forged)


def test_bridge_forwards_all_prototype_sidecars_without_reinterpretation(
    bridge_context: _BridgeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = bridge_context
    observation = _start_observation(context)
    started = context.bridge.start(context.retired, observation).state
    gradient_audit = object()
    memory = object()
    partner_input = object()
    partner_feedback = object()
    captured: dict[str, object] = {}
    original_update = PrototypeAgent.update_transition

    def observed_update(
        prototype: PrototypeAgent,
        prototype_state: Any,
        transition: Any,
        candidate_update_audit_evidence: Any = None,
        *,
        gradient_joy_evidence: Any = None,
        experiential_memory_input: Any = None,
        partner_policy_fusion_input: Any = None,
        partner_policy_fusion_feedback: Any = None,
        extended_action_mask: Any = None,
    ) -> Any:
        captured.update(
            gradient_audit=(
                candidate_update_audit_evidence
                if candidate_update_audit_evidence is not None
                else gradient_joy_evidence
            ),
            memory=experiential_memory_input,
            partner_input=partner_policy_fusion_input,
            partner_feedback=partner_policy_fusion_feedback,
            extended_action_mask=extended_action_mask,
        )
        # The fixture's minimal Prototype deliberately has these optional
        # mechanisms disabled.  After observing the bridge boundary, invoke
        # the same real controller with the corresponding absent sidecars.
        return original_update(
            prototype,
            prototype_state,
            transition,
            candidate_update_audit_evidence=None,
            gradient_joy_evidence=None,
            experiential_memory_input=None,
            partner_policy_fusion_input=None,
            partner_policy_fusion_feedback=None,
            extended_action_mask=extended_action_mask,
        )

    monkeypatch.setattr(PrototypeAgent, "update_transition", observed_update)
    result = context.bridge.update_transition(
        started,
        _next_transition(started, observation - jnp.float32(0.2)),
        gradient_audit,
        experiential_memory_input=memory,
        partner_policy_fusion_input=partner_input,
        partner_policy_fusion_feedback=partner_feedback,
    )

    assert captured["gradient_audit"] is gradient_audit
    assert captured["memory"] is memory
    assert captured["partner_input"] is partner_input
    assert captured["partner_feedback"] is partner_feedback
    np.testing.assert_array_equal(
        captured["extended_action_mask"],
        started.extended_action_mask,
    )
    assert bool(result.transaction_applied)


def test_bridge_carries_one_cold_mask_through_search_dyna_and_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _search_dyna_bridge_context()
    observation = _start_observation(context)
    started = context.bridge.start(context.retired, observation).state
    search_masks: list[jax.Array] = []
    dream_masks: list[jax.Array] = []
    original_search = OptionSearchControl.apply
    original_dreams = PrototypeAgent._run_dreams_with_count

    def observed_search(
        search: OptionSearchControl,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        search_masks.append(kwargs["extended_action_mask"])
        return original_search(search, *args, **kwargs)

    def observed_dreams(
        prototype: PrototypeAgent,
        oak_state: Any,
        world_model_state: Any,
        buffer_state: Any,
        rng_key: Any,
        extended_action_mask: Any = None,
    ) -> Any:
        dream_masks.append(extended_action_mask)
        return original_dreams(
            prototype,
            oak_state,
            world_model_state,
            buffer_state,
            rng_key,
            extended_action_mask,
        )

    monkeypatch.setattr(OptionSearchControl, "apply", observed_search)
    monkeypatch.setattr(PrototypeAgent, "_run_dreams_with_count", observed_dreams)
    result = context.bridge.update_transition(
        started,
        _next_transition(started, observation + jnp.float32(0.05)),
    )

    assert len(search_masks) == 1
    assert len(dream_masks) == 1
    np.testing.assert_array_equal(search_masks[0], started.extended_action_mask)
    np.testing.assert_array_equal(dream_masks[0], started.extended_action_mask)
    assert int(result.real_control_stomp_update_evaluations) == 1
    assert int(result.imagined_stomp_update_evaluations) == 1
    assert int(result.total_stomp_update_evaluations) == 2
    # The search stage is evaluated once but this transition has no supported
    # residual to commit. Evaluation and applied-work accounting stay distinct.
    assert int(result.option_search_learner_updates) == 0
    search_diagnostics = result.prototype.option_search_control_diagnostics
    assert search_diagnostics is not None
    assert int(search_diagnostics.applied_count) == 0
    trace = result.prototype.oak_owner_finalization_trace
    assert bool(stomp_owner_finalization_trace_valid(trace))
    assert bool(trace.stages[0].configured)
    assert bool(trace.stages[0].evaluated)
    assert bool(trace.stages[0].classified_delta_valid)
    assert int(trace.stages[0].learner_updates_applied) == 0
    assert bool(trace.stages[2].configured)
    assert bool(trace.stages[2].evaluated)
    assert bool(trace.stages[2].classified_delta_valid)
    assert int(trace.stages[2].stomp_update_evaluations) == 1
    assert bool(result.lifecycle_owner_finalization.metadata_finalized)
    assert bool(result.authority_metadata_advanced)
    assert bool(result.transaction_applied)


def test_desynchronized_audit_never_rolls_back_valid_prototype_control(
    bridge_context: _BridgeContext,
) -> None:
    context = bridge_context
    observation = _start_observation(context)
    started = context.bridge.start(context.retired, observation).state
    desynchronized = context.bridge._with_checksum(
        started.replace(
            authority_synchronized=jnp.asarray(False, dtype=jnp.bool_),
            authority_error=jnp.asarray(1, dtype=jnp.int32),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    assert bool(context.bridge.state_valid(desynchronized))
    next_observation = observation - jnp.float32(0.15)
    result = context.bridge.update_transition(
        desynchronized,
        _next_transition(desynchronized, next_observation),
    )
    assert bool(result.prototype_control_applied)
    assert bool(result.transaction_applied)
    assert bool(result.authority_desynchronized)
    assert not bool(result.authority_metadata_advanced)
    assert not bool(result.control_transition_rolled_back_by_bridge)
    chex.assert_trees_all_equal(
        result.state.authority_metadata,
        desynchronized.authority_metadata,
    )
    assert int(result.state.prototype_state.step_words[1]) == (
        int(desynchronized.prototype_state.step_words[1]) + 1
    )
    assert bool(context.bridge.state_valid(result.state))

    blocked = context.bridge.prepare_retirement(
        result.state,
        context.lower.retirement_handoff,
        context.lower.retirement_authority,
        context.lower.phase_one_key,
        context.lower.phase_two_key,
    )
    assert not bool(blocked.preparation_valid)


@pytest.mark.parametrize(
    "audit_kwargs",
    (
        {"context": 2**40},
        {"treatment_propensity": jnp.asarray(jnp.nan, dtype=jnp.float32)},
    ),
)
def test_dynamic_audit_malformation_preserves_exact_prototype_destination(
    bridge_context: _BridgeContext,
    audit_kwargs: dict[str, Any],
) -> None:
    context = bridge_context
    observation = _start_observation(context)
    started = context.bridge.start(context.retired, observation).state
    transition = _next_transition(
        started,
        observation + jnp.asarray(0.125, dtype=jnp.float32),
    )
    expected = context.agent.update_transition(
        started.prototype_state,
        transition,
        extended_action_mask=started.extended_action_mask,
    )
    result = context.bridge.update_transition(
        started,
        transition,
        **audit_kwargs,
    )
    assert not bool(result.audit_inputs_valid)
    assert bool(result.prototype_control_applied)
    assert bool(result.transaction_applied)
    assert bool(result.authority_desynchronized)
    assert not bool(result.authority_metadata_advanced)
    assert not bool(result.lifecycle.metadata_advanced)
    assert not bool(result.lifecycle_owner_finalization.metadata_finalized)
    assert not bool(result.control_transition_rolled_back_by_bridge)
    chex.assert_trees_all_equal(
        result.state.prototype_state,
        expected.state,
    )
    chex.assert_trees_all_equal(
        result.state.authority_metadata,
        started.authority_metadata,
    )


def test_borrowed_and_bridge_module_exports_have_root_identity() -> None:
    modules = (
        authorized_option_replacement,
        cumulant_option_installation,
        cumulant_option_scheduler,
        stomp_option_lifecycle,
        prototype_option_authority_bridge,
    )
    for implementation in modules:
        assert implementation.__all__
        assert len(implementation.__all__) == len(set(implementation.__all__))
        for name in implementation.__all__:
            expected = getattr(implementation, name)
            assert getattr(core, name) is expected
            assert getattr(alberta, name) is expected
            assert name in core.__all__
            assert name in alberta.__all__
    assert len(core.__all__) == len(set(core.__all__))
    assert len(alberta.__all__) == len(set(alberta.__all__))
