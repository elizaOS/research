# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Single-owner contracts for the versioned repeated Prototype bridge."""

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
from test_authorized_option_replacement import _context as _one_shot_context
from test_authorized_option_retirement import _receipt as _retirement_receipt
from test_cumulant_option_scheduler import _receipt as _installation_receipt
from test_cumulant_option_scheduler import _transition as _scheduler_transition

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPAgent, STOMPState
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.prototype_option_authority_bridge import (
    PrototypeOptionAuthorityBridge,
    _prototype_oak_state,
)
from alberta_framework.core.prototype_repeated_option_authority_bridge import (
    PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ASSESSMENT,
    PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA,
    PrototypeRepeatedOptionAuthorityBridge,
    PrototypeRepeatedOptionAuthorityBridgeState,
)
from alberta_framework.core.repeated_option_lifecycle import (
    RepeatedOptionLifecycle,
    RepeatedOptionLifecycleConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_module() -> Iterator[None]:
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]


class _Context(NamedTuple):
    lower: Any
    agent: PrototypeAgent
    v1: PrototypeOptionAuthorityBridge
    lifecycle: RepeatedOptionLifecycle
    adapter: PrototypeRepeatedOptionAuthorityBridge
    source: PrototypeRepeatedOptionAuthorityBridgeState


@pytest.fixture(scope="module")
def context() -> _Context:
    lower = _one_shot_context(max_installations=8)
    stomp_config = lower.controller.scheduler.installation.stomp_agent.config
    agent = PrototypeAgent(PrototypeAgentConfig(oak=OaKConfig(stomp=stomp_config)))
    v1 = PrototypeOptionAuthorityBridge(agent, lower.controller)
    pristine = agent.init(jr.key(0xA22))
    binding_receipt = v1.declare_initial_owner_binding(
        pristine,
        lower.pre_retirement_state,
        binding_authorized=True,
    )
    bound = v1.bind_initial_prototype_owner(
        pristine,
        lower.pre_retirement_state,
        binding_receipt,
    )
    assert bool(bound.transaction_applied)
    bridge_state = v1.init(bound.prototype_state, lower.pre_retirement_state)
    lifecycle = RepeatedOptionLifecycle(
        lower.controller,
        RepeatedOptionLifecycleConfig(max_cycles=2),
    )
    repeated = lifecycle.init(lower.pre_retirement_state)
    adapter = PrototypeRepeatedOptionAuthorityBridge(v1, lifecycle)
    source = adapter.init(bridge_state, repeated)
    return _Context(lower, agent, v1, lifecycle, adapter, source)


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


def _attached_repeated(
    context: _Context,
    state: PrototypeRepeatedOptionAuthorityBridgeState,
) -> Any:
    repeated, attached = context.adapter._attach_source(state)
    assert bool(attached)
    return repeated


def _retire(
    context: _Context,
    state: PrototypeRepeatedOptionAuthorityBridgeState,
    handoff: Any,
    *,
    cycle_seed: int,
    authority_revision: int,
    phase_one_seed: int,
    phase_two_seed: int,
) -> tuple[Any, Any, jax.Array, jax.Array, jax.Array]:
    repeated = _attached_repeated(context, state)
    cycle_key = jr.key(cycle_seed, impl="threefry2x32")
    phase_one = jr.key(phase_one_seed, impl="threefry2x32")
    phase_two = jr.key(phase_two_seed, impl="threefry2x32")
    projected = context.lifecycle.replacement._as_retirement_state(repeated.cycle_state)
    child_receipt = _retirement_receipt(
        projected,
        handoff,
        phase_one,
        phase_two,
        revision=authority_revision,
    )
    receipt = context.adapter.retirement_authority_receipt(
        state,
        child_receipt,
        cycle_key,
    )
    prepared = context.adapter.prepare_retirement(
        state,
        handoff,
        receipt,
        cycle_key,
        phase_one,
        phase_two,
    )
    committed = context.adapter.commit_retirement(state, prepared)
    assert bool(committed.transaction_applied)
    return committed, prepared, cycle_key, phase_one, phase_two


def _prepare_replacement(
    context: _Context,
    state: PrototypeRepeatedOptionAuthorityBridgeState,
    cycle_key: jax.Array,
    step: int,
    *,
    authorized: bool,
) -> tuple[Any, Any]:
    repeated = _attached_repeated(context, state)
    scheduler_state = repeated.cycle_state.scheduler_state
    arm_inputs, observation, live_inputs = _scheduler_transition(scheduler_state, step)
    arm = context.adapter.arm(state, arm_inputs)
    prepared = context.adapter.prepare_replacement(
        state,
        arm,
        observation,
        live_inputs,
    )
    installation_authority = _installation_receipt(
        scheduler_state,
        authorized=authorized,
    )
    receipt = context.adapter.replacement_authority_receipt(
        state,
        prepared,
        installation_authority,
        cycle_key,
        replacement_authorized=authorized,
    )
    return prepared, receipt


def _start_observation(context: _Context) -> jax.Array:
    return jnp.linspace(
        -0.25,
        0.25,
        context.agent.config.oak.observation_dim,
        dtype=jnp.float32,
    )


def _next_transition(
    state: PrototypeRepeatedOptionAuthorityBridgeState,
    next_observation: jax.Array,
) -> PrototypeTransition:
    prototype = state.bridge_state.prototype_state
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


def test_persistent_state_has_one_owner_strict_checkpoint_and_exact_resources(
    context: _Context,
) -> None:
    state = context.source
    assert bool(context.adapter.state_valid(state))
    assert _count_exact_stomp_owners(state) == 1
    assert _count_exact_stomp_owners(state.bridge_state) == 1
    assert _count_exact_stomp_owners(state.bridge_state.authority_metadata) == 0
    assert _count_exact_stomp_owners(state.lifecycle_metadata) == 0

    budget = context.adapter.resource_budget(state)
    assert budget.persistent_stomp_state_owners == 1
    assert budget.detached_authority_metadata_stomp_state_owners == 0
    assert budget.repeated_overlay_stomp_state_owners == 0
    assert budget.borrowed_stomp_bindings == 1
    assert budget.persistent_prepared_transactions == 0
    assert budget.completed_cycles == 0
    assert budget.remaining_cycles == budget.max_cycles == 2
    assert budget.stomp_updates_per_audit_adoption == 0
    assert budget.stomp_updates_per_retirement_transaction == 0
    assert budget.stomp_updates_per_replacement_transaction == 0
    assert budget.retirement_prepare_host_only is True
    assert budget.retirement_commit_host_only is True
    assert budget.replacement_prepare_host_only is True
    assert budget.replacement_commit_host_only is True
    assert budget.checkpoint_host_only is True
    assert budget.assessment == PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_ASSESSMENT
    assert not budget.benefit_claim
    assert not budget.evidence_authority
    assert not budget.promotion_authority
    assert not budget.safety_authority
    assert not budget.go_no_go_authority
    assert not budget.retirement_authority
    assert not budget.replacement_authority
    assert not budget.discovery_authority
    assert not budget.dispatch_authority
    assert not budget.autonomous_curation_authority
    assert not budget.scientific_promotion_allowed

    payload = context.adapter.checkpoint_payload(state)
    assert payload["schema_version"] == PROTOTYPE_REPEATED_OPTION_AUTHORITY_BRIDGE_CHECKPOINT_SCHEMA
    restored = context.adapter.restore_checkpoint(
        payload,
        expected_completed_cycles=0,
        expected_revision=0,
    )
    chex.assert_trees_all_equal(restored, state)
    with pytest.raises(ValueError, match="invalid, stale, or rebound"):
        context.adapter.restore_checkpoint(
            payload,
            expected_completed_cycles=1,
            expected_revision=0,
        )
    with pytest.raises(ValueError, match="invalid, stale, or rebound"):
        context.adapter.restore_checkpoint(
            payload,
            expected_completed_cycles=0,
            expected_revision=1,
        )
    corrupted = dict(payload)
    corrupted["state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="state hash"):
        context.adapter.restore_checkpoint(
            corrupted,
            expected_completed_cycles=0,
            expected_revision=0,
        )


def test_retirement_and_replacement_atomically_rebind_the_only_oak_owner(
    context: _Context,
) -> None:
    retired, retirement_prepared, cycle_key, _, _ = _retire(
        context,
        context.source,
        context.lower.retirement_handoff,
        cycle_seed=1201,
        authority_revision=1,
        phase_one_seed=801,
        phase_two_seed=802,
    )
    assert bool(retirement_prepared.preparation_valid)
    assert bool(retired.exact_owner_rebind)
    assert bool(retired.cold_mask_applied)
    np.testing.assert_array_equal(
        retired.state.bridge_state.extended_action_mask,
        [True, True, False, True, True, True],
    )
    repeated = _attached_repeated(context, retired.state)
    assert bool(repeated.cycle_key_active)
    assert int(repeated.total_retirements) == 1
    assert int(repeated.completed_cycles) == 0
    np.testing.assert_array_equal(repeated.active_cycle_key_data, jr.key_data(cycle_key))
    oak = _prototype_oak_state(retired.state.bridge_state.prototype_state.oak_state)
    chex.assert_trees_all_equal(
        oak.stomp_state,
        repeated.cycle_state.scheduler_state.installation_state.lifecycle_state.stomp_state,
    )
    assert _count_exact_stomp_owners(retired.state) == 1

    stale_retirement = context.adapter.commit_retirement(
        retired.state,
        retirement_prepared,
    )
    assert not bool(stale_retirement.destination_matches_source)
    assert not bool(stale_retirement.transaction_applied)
    chex.assert_trees_all_equal(stale_retirement.state, retired.state)

    prepared, receipt = _prepare_replacement(
        context,
        retired.state,
        cycle_key,
        context.lower.next_step,
        authorized=True,
    )
    assert bool(prepared.preparation_valid)
    accepted = context.adapter.commit_replacement(
        retired.state,
        prepared,
        receipt,
        cycle_key,
    )
    assert accepted.lifecycle_result is not None
    assert accepted.oak_rebind is not None
    assert accepted.transaction_applied
    assert accepted.ordinary_advance_applied
    assert accepted.replacement_applied
    assert accepted.cycle_completed
    assert accepted.exact_owner_rebind
    assert accepted.cold_mask_applied
    assert not accepted.caller_authenticated
    assert bool(context.adapter.state_valid(accepted.state))
    assert _count_exact_stomp_owners(accepted.state) == 1
    np.testing.assert_array_equal(
        accepted.state.bridge_state.extended_action_mask,
        jnp.ones((6,), dtype=jnp.bool_),
    )
    repeated = _attached_repeated(context, accepted.state)
    assert int(repeated.completed_cycles) == 1
    assert int(repeated.total_retirements) == 1
    assert int(repeated.total_replacements) == 1
    assert not bool(repeated.cycle_key_active)
    np.testing.assert_array_equal(repeated.cycle_key_history[0], jr.key_data(cycle_key))
    assert bool(jnp.any(repeated.last_retirement_authority_revision_words != 0))
    assert bool(jnp.any(repeated.last_replacement_authority_revision_words != 0))

    stale_replacement = context.adapter.commit_replacement(
        accepted.state,
        prepared,
        receipt,
        cycle_key,
    )
    assert not stale_replacement.destination_matches_source
    assert not stale_replacement.transaction_applied
    chex.assert_trees_all_equal(stale_replacement.state, accepted.state)


def test_decline_adopts_ordinary_advance_but_retry_requires_fresh_provenance(
    context: _Context,
) -> None:
    retired, _, cycle_key, _, _ = _retire(
        context,
        context.source,
        context.lower.retirement_handoff,
        cycle_seed=1221,
        authority_revision=1,
        phase_one_seed=821,
        phase_two_seed=822,
    )
    prepared, receipt = _prepare_replacement(
        context,
        retired.state,
        cycle_key,
        context.lower.next_step,
        authorized=False,
    )
    declined = context.adapter.commit_replacement(
        retired.state,
        prepared,
        receipt,
        cycle_key,
    )
    assert declined.lifecycle_result is not None
    assert declined.transaction_applied
    assert declined.ordinary_advance_applied
    assert not declined.replacement_applied
    assert not declined.cycle_completed
    assert bool(context.adapter.state_valid(declined.state))
    repeated = _attached_repeated(context, declined.state)
    assert bool(repeated.cycle_key_active)
    assert int(repeated.completed_cycles) == 0
    assert bool(repeated.cycle_state.scheduler_state.retry_due)
    np.testing.assert_array_equal(repeated.active_cycle_key_data, jr.key_data(cycle_key))
    np.testing.assert_array_equal(
        declined.state.bridge_state.extended_action_mask,
        retired.state.bridge_state.extended_action_mask,
    )

    replay = context.adapter.commit_replacement(
        declined.state,
        prepared,
        receipt,
        cycle_key,
    )
    assert replay.lifecycle_result is None
    assert not replay.destination_matches_source
    assert not replay.transaction_applied
    chex.assert_trees_all_equal(replay.state, declined.state)

    current = declined.state
    accepted = None
    fresh_prepared = None
    fresh_receipt = None
    for _ in range(5):
        current_repeated = _attached_repeated(context, current)
        step = int(current_repeated.cycle_state.scheduler_state.step_words[1])
        scheduler_state = current_repeated.cycle_state.scheduler_state
        arm_inputs, observation, live_inputs = _scheduler_transition(scheduler_state, step)
        arm = context.adapter.arm(current, arm_inputs)
        candidate = context.adapter.prepare_replacement(
            current,
            arm,
            observation,
            live_inputs,
        )
        ready = bool(
            candidate.lifecycle_prepared.replacement_prepared.diagnostics.candidate_ready_for_authority
        )
        authority = _installation_receipt(scheduler_state, authorized=ready)
        candidate_receipt = context.adapter.replacement_authority_receipt(
            current,
            candidate,
            authority,
            cycle_key,
            replacement_authorized=ready,
        )
        result = context.adapter.commit_replacement(
            current,
            candidate,
            candidate_receipt,
            cycle_key,
        )
        if result.cycle_completed:
            accepted = result
            fresh_prepared = candidate
            fresh_receipt = candidate_receipt
            break
        assert result.ordinary_advance_applied
        current = result.state
    assert accepted is not None
    assert fresh_prepared is not None
    assert fresh_receipt is not None
    assert not bool(jnp.array_equal(fresh_prepared.prepared_checksum, prepared.prepared_checksum))
    assert not bool(
        jnp.array_equal(
            fresh_receipt.source_state_checksum,
            receipt.source_state_checksum,
        )
    )
    assert int(_attached_repeated(context, accepted.state).completed_cycles) == 1


def test_control_consumes_one_raw_stomp_result_and_forwards_every_sidecar(
    context: _Context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _start_observation(context)
    started = context.adapter.start(context.source, observation)
    assert bool(started.transaction_applied)
    assert bool(started.repeated_metadata_advanced)
    assert not bool(started.repeated_desynchronized)
    assert _count_exact_stomp_owners(started.state) == 1

    gradient_audit = object()
    memory = object()
    partner_input = object()
    partner_feedback = object()
    captured: dict[str, object] = {}
    stomp_calls: list[object] = []
    original_prototype_update = PrototypeAgent.update_transition
    original_stomp_update = STOMPAgent.update

    def observed_stomp_update(stomp: STOMPAgent, *args: Any, **kwargs: Any) -> Any:
        stomp_calls.append(kwargs.get("extended_action_mask"))
        return original_stomp_update(stomp, *args, **kwargs)

    def observed_prototype_update(
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
        return original_prototype_update(
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

    monkeypatch.setattr(STOMPAgent, "update", observed_stomp_update)
    monkeypatch.setattr(PrototypeAgent, "update_transition", observed_prototype_update)
    next_observation = observation + jnp.float32(0.1)
    result = context.adapter.update_transition(
        started.state,
        _next_transition(started.state, next_observation),
        gradient_audit,
        experiential_memory_input=memory,
        partner_policy_fusion_input=partner_input,
        partner_policy_fusion_feedback=partner_feedback,
        context=0,
        idle_candidate_option=0,
        idle_initiation_eligible=True,
        comparator_randomized=True,
        treatment_propensity=0.5,
    )

    assert captured["gradient_audit"] is gradient_audit
    assert captured["memory"] is memory
    assert captured["partner_input"] is partner_input
    assert captured["partner_feedback"] is partner_feedback
    np.testing.assert_array_equal(
        captured["extended_action_mask"],
        started.state.bridge_state.extended_action_mask,
    )
    assert len(stomp_calls) == 1
    np.testing.assert_array_equal(
        stomp_calls[0],
        started.state.bridge_state.extended_action_mask,
    )
    assert int(result.bridge.stomp_update_evaluations) == 1
    assert int(result.bridge.prototype.oak_stomp_update_evaluations) == 1
    assert not bool(result.bridge.lifecycle.derivation_recomputed)
    assert bool(result.bridge.authority_metadata_advanced)
    assert bool(result.repeated_metadata_advanced)
    assert not bool(result.repeated_desynchronized)
    assert not bool(result.control_transition_rolled_back_by_adapter)
    assert bool(result.transaction_applied)
    final_oak = _prototype_oak_state(result.state.bridge_state.prototype_state.oak_state)
    chex.assert_trees_all_equal(
        result.bridge.prototype.oak_stomp_update_result.state,
        final_oak.stomp_state,
    )
    repeated = _attached_repeated(context, result.state)
    chex.assert_trees_all_equal(
        repeated.cycle_state.scheduler_state.installation_state.lifecycle_state.stomp_state,
        final_oak.stomp_state,
    )
    assert _count_exact_stomp_owners(result.state) == 1


def test_desynchronized_overlay_never_rolls_back_valid_control_or_accepts_authority(
    context: _Context,
) -> None:
    observation = _start_observation(context)
    started = context.adapter.start(context.source, observation).state
    desynchronized = context.adapter._next_state(
        started,
        started.bridge_state,
        started.lifecycle_metadata,
        synchronized=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert bool(context.adapter.state_valid(desynchronized))
    assert not bool(desynchronized.repeated_synchronized)

    result = context.adapter.update_transition(
        desynchronized,
        _next_transition(desynchronized, observation - jnp.float32(0.1)),
    )
    assert bool(result.bridge.transaction_applied)
    assert bool(result.transaction_applied)
    assert bool(result.repeated_desynchronized)
    assert not bool(result.repeated_metadata_advanced)
    assert not bool(result.control_transition_rolled_back_by_adapter)
    assert not bool(result.state.repeated_synchronized)
    assert int(result.state.revision) == int(desynchronized.revision) + 1
    chex.assert_trees_all_equal(
        result.state.lifecycle_metadata,
        desynchronized.lifecycle_metadata,
    )
    assert not bool(
        jnp.array_equal(
            result.state.bridge_state.binding_checksum,
            desynchronized.bridge_state.binding_checksum,
        )
    )
    with pytest.raises(ValueError, match="not synchronized"):
        context.adapter.retirement_authority_receipt(
            result.state,
            context.lower.retirement_authority,
            jr.key(1250, impl="threefry2x32"),
        )


def test_array_only_validation_is_eager_jit_and_scan_safe(
    context: _Context,
) -> None:
    eager_valid = context.adapter.state_valid(context.source)
    compiled_valid = jax.jit(context.adapter.state_valid)(context.source)
    _, scanned_valid = jax.lax.scan(
        lambda state, _: (state, context.adapter.state_valid(state)),
        context.source,
        xs=None,
        length=2,
    )
    np.testing.assert_array_equal(compiled_valid, eager_valid)
    np.testing.assert_array_equal(scanned_valid, (True, True))
