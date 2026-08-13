# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Red-first contracts for the no-planner repeated-option HCCL composite.

The persistent composite owns the reviewed HCCL/two-live-memory state and two
coordinator-free repeated-option metadata bundles.  Preparation evaluates the
HCCL and live donors once, then consumes each nested raw STOMP result once.
Adoption is integrity-only and all-or-none.  Delight is unavailable: this
surface does not ask whether a gradient sparks joy and runs no actor backward.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_authorized_option_replacement import (
    _context as _one_shot_context,
)
from test_authorized_option_replacement import (
    _transition as _swap_transition,
)
from test_authorized_option_retirement import _receipt as _retirement_receipt
from test_cumulant_option_scheduler import _receipt as _installation_receipt
from test_hccl_two_live_memory_bridge import (
    _replace_stored_record,
    _tree_exact,
)
from test_hccl_two_live_memory_prepare_adopt_bridge import (
    _config as _inner_config,
)
from test_hccl_two_live_memory_prepare_adopt_bridge import (
    _event_input,
)

import alberta_framework
import alberta_framework.core as core_api
from alberta_framework.core.external_coordinator_repeated_option_sidecar import (
    ExternalCoordinatorRepeatedOptionBorrowedMetadata,
    ExternalCoordinatorRepeatedOptionLiveActionProjectionResult,
    ExternalCoordinatorRepeatedOptionSidecar,
)
from alberta_framework.core.hccl_two_live_memory_bridge import (
    HCCLTwoLiveMemoryBridgeConfig,
    HCCLTwoLiveMemoryBridgeState,
    _tree_exact_equal,
)
from alberta_framework.core.hccl_two_live_memory_prepare_adopt_bridge import (
    HCCLTwoLiveMemoryPrepareAdoptBridge,
)
from alberta_framework.core.hccl_two_live_memory_repeated_option_prepare_adopt_bridge import (
    HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_STATUS,
    HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared,
    HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt,
    HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge,
    HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction,
)
from alberta_framework.core.options import STOMPState
from alberta_framework.core.prototype_agent import (
    PrototypeCachedPrimitiveActionReplacement,
)
from alberta_framework.core.prototype_option_authority_bridge import (
    _prototype_oak_state,
)
from alberta_framework.core.repeated_option_lifecycle import (
    RepeatedOptionLifecycle,
    RepeatedOptionLifecycleConfig,
    RepeatedOptionLifecycleState,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

N_AGENTS = 2
N_ACTIONS = 2
RESERVED_SUFFIX = 12


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> object:
    with jax.disable_jit():
        yield


class _Rig(NamedTuple):
    lower: Any
    inner: HCCLTwoLiveMemoryPrepareAdoptBridge
    sidecars: tuple[
        ExternalCoordinatorRepeatedOptionSidecar,
        ExternalCoordinatorRepeatedOptionSidecar,
    ]
    bridge: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge
    source: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState


def _config(stomp: Any) -> HCCLTwoLiveMemoryBridgeConfig:
    source = _inner_config()

    def with_stomp(live: Any) -> Any:
        coordinator = live.coordinator
        prototype = coordinator.inner.prototype
        feature = prototype.prototype_feature_lifecycle
        assert feature is not None
        next_prototype = dataclasses.replace(
            prototype,
            oak=dataclasses.replace(prototype.oak, stomp=stomp),
            prototype_feature_lifecycle=dataclasses.replace(
                feature,
                n_options=stomp.n_options,
                option_subtask_feature_indices=tuple(
                    spec.feature_index for spec in stomp.subtask_specs
                ),
            ),
        )
        return dataclasses.replace(
            live,
            coordinator=dataclasses.replace(
                coordinator,
                inner=dataclasses.replace(
                    coordinator.inner,
                    prototype=next_prototype,
                ),
            ),
        )

    return dataclasses.replace(
        source,
        agent_0=with_stomp(source.agent_0),
        agent_1=with_stomp(source.agent_1),
    )


def _nested_stomp(state: Any) -> STOMPState:
    prototype = state.coordinator_state.inner_state.prototype_state
    return _prototype_oak_state(prototype.oak_state).stomp_state


def _start_repeated_owner_for_hccl_child(
    sidecar: ExternalCoordinatorRepeatedOptionSidecar,
    lifecycle: RepeatedOptionLifecycle,
    lower: Any,
    coordinator_key: jax.Array,
    destination_live_state: Any,
) -> RepeatedOptionLifecycleState:
    """Bind false/false first, then reproduce the HCCL coordinator start."""

    destination_coordinator = destination_live_state.coordinator_state
    unstarted_coordinator = sidecar.coordinator.init(coordinator_key)
    repeated_source = lifecycle.init(lower.pre_retirement_state)
    source_lifecycle = (
        repeated_source.cycle_state.scheduler_state.installation_state
        .lifecycle_state
    )
    assert not bool(unstarted_coordinator.started)
    assert not bool(source_lifecycle.started)
    assert bool(destination_coordinator.started)

    projected, projected_valid = sidecar._project_repeated_stomp_owner(
        repeated_source,
        sidecar._oak_state(unstarted_coordinator).stomp_state,
    )
    assert bool(projected_valid)
    unstarted_sidecar = sidecar.init(unstarted_coordinator, projected)
    assert bool(jnp.all(unstarted_sidecar.extended_action_mask))

    started = sidecar.start(
        unstarted_sidecar,
        destination_coordinator.current_raw_observation,
    )
    assert bool(started.transaction_applied)
    assert bool(started.coordinator_started)
    assert bool(started.lifecycle_metadata_applied)
    assert bool(jnp.all(started.state.extended_action_mask))
    _tree_exact(started.state.coordinator_state, destination_coordinator)

    repeated, attached = sidecar._attach_source(started.state)
    assert bool(attached)
    destination_lifecycle = (
        repeated.cycle_state.scheduler_state.installation_state.lifecycle_state
    )
    assert bool(destination_lifecycle.started)
    assert bool(destination_coordinator.started == destination_lifecycle.started)
    assert int(destination_lifecycle.stomp_state.executing_option) == -1
    assert int(destination_lifecycle.audit_state.active_option) == -1
    assert not bool(destination_lifecycle.audit_state.trial_active)
    _tree_exact(_nested_stomp(destination_live_state), destination_lifecycle.stomp_state)
    return repeated


@pytest.fixture(scope="module")
def rig() -> _Rig:
    lower = _one_shot_context(
        max_installations=8,
        option_planning_backups_per_step=0,
        reserved_observation_suffix=RESERVED_SUFFIX,
    )
    shared_stomp = lower.controller.scheduler.installation.stomp_agent.config
    assert shared_stomp.option_planning_backups_per_step == 0
    assert shared_stomp.observation_dim == 18
    inner = HCCLTwoLiveMemoryPrepareAdoptBridge(_config(shared_stomp))
    root_key = jr.key(8)
    _, agent_0_key, agent_1_key = jr.split(root_key, 3)
    inner_state = inner.init(root_key)
    assert int(_nested_stomp(inner_state.agent_0_state).executing_option) == -1
    assert int(_nested_stomp(inner_state.agent_1_state).executing_option) == -1
    lifecycle_0 = RepeatedOptionLifecycle(
        lower.controller,
        RepeatedOptionLifecycleConfig(max_cycles=2),
    )
    lifecycle_1 = RepeatedOptionLifecycle(
        lower.controller,
        RepeatedOptionLifecycleConfig(max_cycles=2),
    )
    sidecar_0 = ExternalCoordinatorRepeatedOptionSidecar(
        inner.inner.agent_0.coordinator,
        lifecycle_0,
    )
    sidecar_1 = ExternalCoordinatorRepeatedOptionSidecar(
        inner.inner.agent_1.coordinator,
        lifecycle_1,
    )
    repeated_0 = _start_repeated_owner_for_hccl_child(
        sidecar_0,
        lifecycle_0,
        lower,
        agent_0_key,
        inner_state.agent_0_state,
    )
    repeated_1 = _start_repeated_owner_for_hccl_child(
        sidecar_1,
        lifecycle_1,
        lower,
        agent_1_key,
        inner_state.agent_1_state,
    )
    bridge = HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge(
        inner,
        sidecar_0,
        sidecar_1,
    )
    source = bridge.init(inner_state, repeated_0, repeated_1)
    return _Rig(lower, inner, (sidecar_0, sidecar_1), bridge, source)


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


def _prepare(
    rig: _Rig,
    state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    *,
    downstream: tuple[bool, bool] = (True, True),
) -> HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction:
    event = rig.bridge.prepare_event(state)
    binding = rig.bridge.bind_live_memory_actions(state, event)
    return rig.bridge.prepare_transaction(
        state,
        event,
        binding,
        _event_input(70),
        _event_input(71),
        next_decision_hard_action_masks=jnp.ones(
            (N_AGENTS, N_ACTIONS), dtype=jnp.bool_
        ),
        agent_0_downstream_candidate_valid=downstream[0],
        agent_1_downstream_candidate_valid=downstream[1],
    )


def _adopt(
    rig: _Rig,
    state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    prepared: HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction,
    *,
    outer: bool = True,
) -> Any:
    return rig.bridge.adopt_prepared_transaction(
        state,
        prepared,
        rig.bridge.integrity_receipt(prepared),
        downstream_candidate_valid=outer,
    )


def _seed_actual_memory_action_change(
    rig: _Rig,
) -> tuple[
    HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    HCCLTwoLiveMemoryRepeatedOptionPreparedTransaction,
]:
    first_prepared = _prepare(rig, rig.source)
    first = _adopt(rig, rig.source, first_prepared)
    assert bool(first.update_applied)

    event = rig.bridge.prepare_event(first.state)
    binding = rig.bridge.bind_live_memory_actions(first.state, event)
    preview = rig.bridge.prepare_transaction(
        first.state,
        event,
        binding,
        _event_input(72),
        _event_input(73),
        next_decision_hard_action_masks=jnp.ones(
            (N_AGENTS, N_ACTIONS), dtype=jnp.bool_
        ),
    )
    base_actions = jnp.stack(
        tuple(
            cast(Any, agent.inner_facts.live_prepared.coordinator_result)
            .state.current_action
            for agent in (preview.agent_0, preview.agent_1)
        )
    )
    retrieved_actions = 1 - base_actions
    seeded_inner = first.state.inner_state
    for agent, first_agent, preview_agent in (
        (0, first_prepared.agent_0, preview.agent_0),
        (1, first_prepared.agent_1, preview.agent_1),
    ):
        memory_result = first_agent.inner_facts.live_prepared.learned_memory_result
        assert memory_result is not None
        seeded_inner = _replace_stored_record(
            rig.inner.inner,
            seeded_inner,
            agent=agent,
            slot=int(memory_result.slot),
            query_key=preview_agent.inner_facts.live_prepared.query_key,
            action=int(retrieved_actions[agent]),
        )
    seeded = rig.bridge._with_checksum(
        first.state.replace(inner_state=seeded_inner)
    )
    assert bool(rig.bridge.state_valid(seeded))
    return seeded, _prepare(rig, seeded)


def test_persistent_schema_contains_only_hccl_and_two_detached_metadata_bundles(
    rig: _Rig,
) -> None:
    shared_stomp = rig.lower.controller.scheduler.installation.stomp_agent.config
    assert shared_stomp.option_planning_backups_per_step == 0
    assert rig.inner.config.agent_0.coordinator.inner.prototype.oak.stomp is shared_stomp
    assert rig.inner.config.agent_1.coordinator.inner.prototype.oak.stomp is shared_stomp
    for sidecar in rig.sidecars:
        assert (
            sidecar.lifecycle.replacement.scheduler.installation.stomp_agent.config
            is shared_stomp
        )
        assert sidecar.coordinator.inner.prototype.config.oak.stomp is shared_stomp
    assert tuple(
        field.name
        for field in dataclasses.fields(HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState)
    ) == (
        "inner_state",
        "agent_0_metadata",
        "agent_1_metadata",
        "revision",
        "binding_checksum",
    )
    assert tuple(
        field.name
        for field in dataclasses.fields(ExternalCoordinatorRepeatedOptionBorrowedMetadata)
    ) == (
        "authority_metadata",
        "lifecycle_metadata",
        "extended_action_mask",
        "revision",
        "binding_checksum",
    )
    assert type(rig.source.inner_state) is HCCLTwoLiveMemoryBridgeState
    assert _count_exact_stomp_owners(rig.source) == 2
    assert _count_exact_stomp_owners(rig.source.agent_0_metadata) == 0
    assert _count_exact_stomp_owners(rig.source.agent_1_metadata) == 0
    assert bool(rig.bridge.state_valid(rig.source))

    payload = rig.bridge.to_config()
    assert payload["mechanism_status"] == HCCL_TWO_LIVE_MEMORY_REPEATED_OPTION_PREPARE_ADOPT_STATUS
    assert payload["hccl_state_owners"] == 1
    assert payload["live_memory_adapter_state_owners"] == 2
    assert payload["external_coordinator_state_owners"] == 2
    assert payload["prototype_state_owners"] == 2
    assert payload["oak_state_owners"] == 2
    assert payload["stomp_state_owners"] == 2
    assert payload["detached_metadata_stomp_state_owners"] == 0
    assert payload["persistent_cold_states"] == 0
    assert payload["planner_state_owners"] == 0
    assert payload["planner_action_relation"] == "P=M-no-planner-rung"
    assert payload["prepare_outer_metadata_attachment_evaluations"] == 8
    assert payload["adopt_outer_metadata_attachment_evaluations"] == 10
    assert payload["atomic_selected_sidecar_overlays"] == 1
    assert payload["atomic_candidate_installation_evaluations"] == 1
    assert payload["atomic_oak_option_slot_rebind_calls"] == 1
    assert payload["atomic_world_or_learner_reevaluations"] == 0
    assert payload["delight_available"] is False
    assert payload["additional_actor_backward_calls"] == 0
    assert "sparks_joy" not in payload


def test_prepare_evaluates_once_consumes_each_raw_result_and_binds_final_owner(
    rig: _Rig,
) -> None:
    prepared = _prepare(rig, rig.source)
    assert bool(prepared.preparation_valid)
    _tree_exact(prepared.source_state, rig.source)
    np.testing.assert_array_equal(prepared.work.live_prepare_calls, [1, 1])
    np.testing.assert_array_equal(prepared.work.coordinator_update_calls, [1, 1])
    np.testing.assert_array_equal(prepared.work.prototype_update_calls, [1, 1])
    np.testing.assert_array_equal(
        prepared.work.raw_stomp_update_evaluations,
        [1, 1],
    )
    np.testing.assert_array_equal(
        prepared.work.lifecycle_observation_evaluations,
        [1, 1],
    )
    np.testing.assert_array_equal(
        prepared.work.additional_stomp_update_evaluations,
        [0, 0],
    )
    np.testing.assert_array_equal(prepared.work.kondo_calls, [0, 0])
    np.testing.assert_array_equal(prepared.work.actor_backward_calls, [0, 0])
    assert int(prepared.work.outer_metadata_attachment_evaluations) == 8
    assert bool(prepared.no_planner_rung_valid)
    np.testing.assert_array_equal(prepared.final_live_owner_digests_bound, [True, True])
    assert not bool(prepared.delight_available)
    assert int(prepared.actor_backward_calls) == 0

    for index, agent in enumerate((prepared.agent_0, prepared.agent_1)):
        assert bool(agent.raw_stomp_result_consumed_once)
        assert bool(agent.finalization_digests_bound)
        assert bool(agent.candidate_metadata_owner_bound)
        np.testing.assert_array_equal(
            agent.downstream_receipt.extended_action_mask,
            agent.source_metadata.extended_action_mask,
        )
        trace = agent.inner_facts.owner_finalization_trace
        np.testing.assert_array_equal(agent.downstream_receipt.raw_stomp_digest, trace.raw_digest)
        np.testing.assert_array_equal(
            agent.downstream_receipt.final_stomp_digest,
            trace.final_digest,
        )
        np.testing.assert_array_equal(
            agent.downstream_receipt.owner_finalization_trace_checksum,
            trace.trace_checksum,
        )
        child = (
            prepared.candidate_state.inner_state.agent_0_state
            if index == 0
            else prepared.candidate_state.inner_state.agent_1_state
        )
        attached = rig.sidecars[index].attach_borrowed_metadata(
            child.coordinator_state,
            agent.candidate_metadata,
        )
        assert bool(attached.transaction_applied)


def test_actual_memory_action_replacement_projects_exact_full_owner_and_adopts(
    rig: _Rig,
) -> None:
    source, prepared = _seed_actual_memory_action_change(rig)
    assert bool(prepared.preparation_valid)
    for index, agent in enumerate((prepared.agent_0, prepared.agent_1)):
        live = agent.inner_facts.live_prepared
        replacement = live.cached_action_replacement
        assert type(replacement) is PrototypeCachedPrimitiveActionReplacement
        assert bool(replacement.committed)
        base_coordinator = cast(Any, live.coordinator_result).state
        final_coordinator = live.candidate_state.coordinator_state
        assert not bool(_tree_exact_equal(base_coordinator, final_coordinator))
        assert int(base_coordinator.current_action) != int(final_coordinator.current_action)
        projection = agent.live_action_projection
        assert type(projection) is ExternalCoordinatorRepeatedOptionLiveActionProjectionResult
        assert bool(projection.replacement_supplied)
        assert bool(projection.replacement_committed)
        assert bool(projection.replacement_stomp_delta_exact)
        assert bool(projection.replacement_prototype_exact)
        assert bool(projection.coordinator_wrapper_delta_exact)
        assert bool(projection.stomp_owner_matches_final_live)
        assert bool(projection.stomp_clocks_preserved)
        np.testing.assert_array_equal(
            projection.replacement_stomp_digest,
            projection.final_live_stomp_digest,
        )
        np.testing.assert_array_equal(
            agent.final_live_stomp_digest,
            projection.final_live_stomp_digest,
        )
        assert bool(agent.final_live_owner_digest_bound)
        assert bool(projection.metadata_rebased)
        assert int(projection.additional_coordinator_evaluations) == 0
        assert int(projection.additional_prototype_evaluations) == 0
        assert int(projection.additional_stomp_update_evaluations) == 0
        assert int(projection.additional_lifecycle_observations) == 0
        final_prototype = final_coordinator.inner_state.prototype_state
        _tree_exact(final_prototype, replacement.state)
        attached = rig.sidecars[index].attach_borrowed_metadata(
            final_coordinator,
            projection.metadata,
        )
        assert bool(attached.transaction_applied)

    result = _adopt(rig, source, prepared)
    assert bool(result.update_applied)
    _tree_exact(result.state, prepared.candidate_state)
    np.testing.assert_array_equal(result.downstream_receipts_valid, [True, True])
    np.testing.assert_array_equal(result.final_live_owner_bindings_valid, [True, True])
    np.testing.assert_array_equal(result.lifecycle_metadata_updates_applied, [True, True])
    np.testing.assert_array_equal(result.adoption_work.coordinator_update_calls, [0, 0])
    np.testing.assert_array_equal(result.adoption_work.prototype_update_calls, [0, 0])
    np.testing.assert_array_equal(result.adoption_work.stomp_update_evaluations, [0, 0])
    np.testing.assert_array_equal(
        result.adoption_work.lifecycle_observation_evaluations,
        [0, 0],
    )
    np.testing.assert_array_equal(result.adoption_work.memory_donor_reevaluations, [0, 0])
    np.testing.assert_array_equal(result.adoption_work.actor_backward_calls, [0, 0])
    assert int(result.adoption_work.outer_metadata_attachment_evaluations) == 10


def test_tampered_or_foreign_live_replacement_and_outer_refusals_roll_back(
    rig: _Rig,
) -> None:
    source, prepared = _seed_actual_memory_action_change(rig)
    live_0 = prepared.agent_0.inner_facts.live_prepared
    replacement_0 = live_0.cached_action_replacement
    replacement_1 = prepared.agent_1.inner_facts.live_prepared.cached_action_replacement
    assert type(replacement_0) is PrototypeCachedPrimitiveActionReplacement
    assert type(replacement_1) is PrototypeCachedPrimitiveActionReplacement

    bad_replacement = replacement_0.replace(
        action=1 - replacement_0.action,
    )
    bad_projection = rig.sidecars[0].project_live_cached_action_replacement(
        prepared.agent_0.sidecar_attempt.state,
        bad_replacement,
        live_0.candidate_state.coordinator_state,
    )
    assert not bool(bad_projection.transaction_applied)
    _tree_exact(bad_projection.state, prepared.agent_0.sidecar_attempt.state)
    foreign_projection = rig.sidecars[0].project_live_cached_action_replacement(
        prepared.agent_0.sidecar_attempt.state,
        replacement_1,
        live_0.candidate_state.coordinator_state,
    )
    assert not bool(foreign_projection.transaction_applied)
    _tree_exact(foreign_projection.state, prepared.agent_0.sidecar_attempt.state)

    tampered_agent = prepared.agent_0.replace(
        live_action_projection=bad_projection,
        candidate_metadata=bad_projection.metadata,
        candidate_coordinator_matches_live=jnp.asarray(False, dtype=jnp.bool_),
        candidate_metadata_owner_bound=jnp.asarray(False, dtype=jnp.bool_),
        preparation_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    candidate = rig.bridge._with_checksum(
        prepared.candidate_state.replace(
            agent_0_metadata=bad_projection.metadata,
        )
    )
    tampered = prepared.replace(
        agent_0=tampered_agent,
        candidate_state=candidate,
        candidate_metadata_bindings_valid=prepared.candidate_metadata_bindings_valid.at[
            0
        ].set(False),
        preparation_valid=jnp.asarray(False, dtype=jnp.bool_),
        content_tag_words=jnp.zeros((8,), dtype=jnp.uint32),
    )
    tampered = tampered.replace(
        content_tag_words=rig.bridge._prepared_tag(tampered)
    )
    rejected = _adopt(rig, source, tampered)
    assert not bool(rejected.update_applied)
    _tree_exact(rejected.state, source)

    # Agent 1 metadata is internally checksum-valid, but it is foreign to the
    # final agent 0 coordinator.  Re-sealing both outer checksums must not turn
    # that borrowed binding into a valid owner relationship.
    foreign_metadata = prepared.agent_1.candidate_metadata
    foreign_agent = prepared.agent_0.replace(
        candidate_metadata=foreign_metadata,
        candidate_metadata_owner_bound=jnp.asarray(True, dtype=jnp.bool_),
        preparation_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    foreign_candidate = rig.bridge._with_checksum(
        prepared.candidate_state.replace(agent_0_metadata=foreign_metadata)
    )
    assert not bool(rig.bridge.state_valid(foreign_candidate))
    foreign_prepared = prepared.replace(
        agent_0=foreign_agent,
        candidate_state=foreign_candidate,
        candidate_metadata_bindings_valid=jnp.ones((2,), dtype=jnp.bool_),
        preparation_valid=jnp.asarray(True, dtype=jnp.bool_),
        content_tag_words=jnp.zeros((8,), dtype=jnp.uint32),
    )
    foreign_prepared = foreign_prepared.replace(
        content_tag_words=rig.bridge._prepared_tag(foreign_prepared)
    )
    foreign_result = _adopt(rig, source, foreign_prepared)
    assert not bool(foreign_result.update_applied)
    _tree_exact(foreign_result.state, source)

    declined = _prepare(rig, source, downstream=(True, False))
    assert not bool(declined.preparation_valid)
    declined_result = _adopt(rig, source, declined)
    assert not bool(declined_result.update_applied)
    _tree_exact(declined_result.state, source)
    vetoed = _adopt(rig, source, prepared, outer=False)
    assert not bool(vetoed.update_applied)
    assert bool(vetoed.outer_veto)
    _tree_exact(vetoed.state, source)
    replay = _adopt(rig, prepared.candidate_state, prepared)
    assert not bool(replay.update_applied)
    _tree_exact(replay.state, prepared.candidate_state)


def _atomic_prepared(
    rig: _Rig,
    state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
    agent: int,
) -> tuple[HCCLTwoLiveMemoryRepeatedOptionAtomicSwapPrepared, Any, jax.Array]:
    attachment = rig.bridge._attachments(state)[agent]
    repeated, attached = rig.sidecars[agent]._attach_source(attachment.state)
    assert bool(attached)
    authority_state = repeated.cycle_state
    scheduler = rig.lower.controller.scheduler
    handoff = scheduler._retirement_handoff(
        authority_state.scheduler_state.discovery_state,
        authority_state.scheduler_state.installation_state,
        authority_state.scheduler_state.step_words,
        available=jnp.asarray(True, dtype=jnp.bool_),
    )
    phase_one = jr.key(0xA100 + agent)
    phase_two = jr.key(0xA200 + agent)
    retirement = _retirement_receipt(
        rig.lower.controller._as_retirement_state(authority_state),
        handoff,
        phase_one,
        phase_two,
    )
    arm_inputs, observation, live = _swap_transition(
        authority_state.scheduler_state,
        rig.lower.next_step,
    )
    cycle_key = jr.key(0xA300 + agent)
    prepared = rig.bridge.prepare_agent_atomic_swap(
        state,
        agent_index=agent,
        cycle_key=cycle_key,
        retirement_handoff=handoff,
        retirement_authority=retirement,
        phase_one_key=phase_one,
        phase_two_key=phase_two,
        arm_inputs=arm_inputs,
        observation=observation,
        live_inputs=live,
    )
    retired = prepared.sidecar_prepared.atomic_swap_prepared.retirement_result.state
    installation = _installation_receipt(
        retired.scheduler_state,
        authorized=True,
    )
    return prepared, installation, cycle_key


def _quiescent_atomic_agent(
    rig: _Rig,
    state: HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptState,
) -> int:
    """Select a real-started child that satisfies retirement's exact gate."""

    for agent in range(N_AGENTS):
        attachment = rig.bridge._attachment(state, agent)
        repeated, attached = rig.sidecars[agent]._attach_source(attachment.state)
        assert bool(attached)
        lifecycle = (
            repeated.cycle_state.scheduler_state.installation_state.lifecycle_state
        )
        if bool(
            (lifecycle.stomp_state.executing_option < 0)
            & (lifecycle.audit_state.active_option < 0)
            & (~lifecycle.audit_state.trial_active)
        ):
            return agent
    raise AssertionError("real-started fixture has no quiescent atomic-swap child")


def test_per_agent_atomic_swap_preserves_every_unselected_owner_and_refuses_replay(
    rig: _Rig,
) -> None:
    agent = _quiescent_atomic_agent(rig, rig.source)
    other = 1 - agent
    prepared, installation, cycle_key = _atomic_prepared(rig, rig.source, agent)
    diagnostics = prepared.sidecar_prepared.atomic_swap_prepared.diagnostics
    assert bool(diagnostics.source_state_valid)
    assert bool(diagnostics.source_all_slots_installed)
    assert bool(diagnostics.transient_retirement_applied)
    assert bool(diagnostics.transient_retirement_state_valid)
    assert bool(diagnostics.exact_one_transient_cold_slot)
    assert bool(diagnostics.replacement_preparation_valid)
    assert bool(diagnostics.fresh_candidate_available)
    assert bool(diagnostics.exact_target_semantic_change)
    assert bool(diagnostics.live_slots_semantically_preserved)
    assert bool(diagnostics.atomic_swap_ready)
    assert bool(prepared.preparation_valid)
    assert int(prepared.work.selected_sidecar_overlays) == 1
    assert int(prepared.work.outer_metadata_attachment_evaluations) == 3
    assert int(prepared.work.sidecar_atomic_prepare_calls) == 1
    assert int(prepared.work.authorized_atomic_prepare_derivations) == 1
    assert int(prepared.work.retirement_filter_derivations) == 1
    assert int(prepared.work.scheduler_observations) == 1
    assert int(prepared.work.replacement_candidate_preparations) == 1
    assert int(prepared.work.candidate_installation_evaluations) == 0
    assert int(prepared.work.oak_option_slot_rebind_calls) == 0
    authority = rig.bridge.authorize_agent_atomic_swap(
        rig.source,
        prepared,
        installation,
        cycle_key,
        swap_authorized=True,
    )
    result = rig.bridge.adopt_agent_atomic_swap(
        rig.source,
        prepared,
        authority,
        cycle_key,
    )
    assert bool(result.transaction_applied)
    assert bool(result.selected_agent_preserved_memory)
    assert bool(result.selected_agent_preserved_pending_binding)
    assert bool(result.other_agent_preserved)
    assert bool(result.hccl_world_attribution_preserved)
    assert bool(result.primitive_action_masks_preserved)
    assert bool(result.exact_final_owner_binding)
    assert not bool(result.cold_state_persisted)
    _tree_exact(result.state.inner_state.hccl_state, rig.source.inner_state.hccl_state)
    result_children = (
        result.state.inner_state.agent_0_state,
        result.state.inner_state.agent_1_state,
    )
    source_children = (
        rig.source.inner_state.agent_0_state,
        rig.source.inner_state.agent_1_state,
    )
    _tree_exact(result_children[other], source_children[other])
    _tree_exact(
        result_children[agent].learned_memory_state,
        source_children[agent].learned_memory_state,
    )
    _tree_exact(
        result_children[agent].pending_binding,
        source_children[agent].pending_binding,
    )
    assert _count_exact_stomp_owners(result.state) == 2
    assert int(result.work.stomp_update_evaluations) == 0
    assert int(result.work.memory_donor_reevaluations) == 0
    assert int(result.work.actor_backward_calls) == 0
    assert int(result.work.selected_sidecar_overlays) == 1
    assert int(result.work.outer_metadata_attachment_evaluations) == 6
    assert int(result.work.sidecar_atomic_prepare_calls) == 1
    assert int(result.work.authorized_atomic_prepare_rederivations) == 2
    assert int(result.work.retirement_filter_rederivations) == 2
    assert int(result.work.scheduler_observations) == 3
    assert int(result.work.replacement_candidate_preparations) == 3
    assert int(result.work.candidate_installation_evaluations) == 1
    assert int(result.work.oak_option_slot_rebind_calls) == 1

    replay = rig.bridge.adopt_agent_atomic_swap(
        result.state,
        prepared,
        authority,
        cycle_key,
    )
    assert not bool(replay.transaction_applied)
    _tree_exact(replay.state, result.state)
    veto = rig.bridge.adopt_agent_atomic_swap(
        rig.source,
        prepared,
        authority,
        cycle_key,
        downstream_candidate_valid=False,
    )
    assert not bool(veto.transaction_applied)
    _tree_exact(veto.state, rig.source)
    declined_authority = rig.bridge.authorize_agent_atomic_swap(
        rig.source,
        prepared,
        installation,
        cycle_key,
        swap_authorized=False,
    )
    declined = rig.bridge.adopt_agent_atomic_swap(
        rig.source,
        prepared,
        declined_authority,
        cycle_key,
    )
    assert not bool(declined.transaction_applied)
    _tree_exact(declined.state, rig.source)
    tampered = prepared.replace(
        source_state_receipt_words=prepared.source_state_receipt_words.at[0].add(
            jnp.uint32(1)
        )
    )
    tampered_result = rig.bridge.adopt_agent_atomic_swap(
        rig.source,
        tampered,
        authority,
        cycle_key,
    )
    assert not bool(tampered_result.transaction_applied)
    _tree_exact(tampered_result.state, rig.source)


def test_resource_contract_and_public_exports_are_exact(rig: _Rig) -> None:
    budget = rig.bridge.resource_budget(rig.source)
    assert budget.hccl_state_owners == 1
    assert budget.live_memory_adapter_state_owners == 2
    assert budget.external_coordinator_state_owners == 2
    assert budget.prototype_state_owners == 2
    assert budget.oak_state_owners == 2
    assert budget.stomp_state_owners == 2
    assert budget.detached_metadata_stomp_state_owners == 0
    assert budget.persistent_cold_states == 0
    assert budget.persisted_preparations == 0
    assert budget.planner_state_owners == 0
    assert budget.prepare_hccl_stage_calls == 1
    assert budget.prepare_live_adapter_calls == 2
    assert budget.prepare_lifecycle_observation_calls == 2
    assert budget.prepare_additional_stomp_evaluations == 0
    assert budget.prepare_outer_metadata_attachment_evaluations == 8
    assert budget.adopt_world_or_learner_reevaluations == 0
    assert budget.adopt_outer_metadata_attachment_evaluations == 10
    assert budget.atomic_swap_selected_sidecar_overlays == 1
    assert budget.atomic_prepare_outer_metadata_attachment_evaluations == 3
    assert budget.atomic_authorize_outer_metadata_attachment_evaluations == 3
    assert budget.atomic_adopt_outer_metadata_attachment_evaluations == 6
    assert budget.atomic_total_outer_metadata_attachment_evaluations == 12
    assert budget.atomic_prepare_retirement_filter_derivations == 1
    assert budget.atomic_adopt_retirement_filter_rederivations == 2
    assert budget.atomic_total_retirement_filter_derivations == 3
    assert budget.atomic_prepare_scheduler_observations == 1
    assert budget.atomic_adopt_scheduler_observations == 3
    assert budget.atomic_total_scheduler_observations == 4
    assert budget.atomic_prepare_replacement_candidate_preparations == 1
    assert budget.atomic_adopt_replacement_candidate_preparations == 3
    assert budget.atomic_total_replacement_candidate_preparations == 4
    assert budget.atomic_total_candidate_installation_evaluations == 1
    assert budget.atomic_total_oak_option_slot_rebind_calls == 1
    assert budget.atomic_swap_world_or_learner_reevaluations == 0
    assert budget.delight_available is False
    assert budget.additional_delight_evaluations == 0
    assert budget.additional_actor_backward_calls == 0
    assert budget.output_write_calls == 0
    assert budget.artifact_bytes_written == 0
    assert budget.persistent_state_nbytes == (
        budget.inner_hccl_two_live_memory_state_nbytes
        + budget.detached_sidecar_metadata_nbytes
    )

    assert (
        core_api.HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge
        is HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge
    )
    assert (
        alberta_framework.HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge
        is HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge
    )
    assert (
        core_api.HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt
        is HCCLTwoLiveMemoryRepeatedOptionPreparationReceipt
    )
