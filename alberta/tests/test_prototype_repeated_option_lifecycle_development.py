# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Bounded mechanism contracts for the repeated Prototype development harness."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NamedTuple

import jax
import jax.random as jr
import pytest
from test_authorized_option_replacement import _context as _one_shot_context

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.prototype_agent import PrototypeAgent, PrototypeAgentConfig
from alberta_framework.core.prototype_option_authority_bridge import (
    PrototypeOptionAuthorityBridge,
)
from alberta_framework.core.prototype_repeated_option_authority_bridge import (
    PrototypeRepeatedOptionAuthorityBridge,
    PrototypeRepeatedOptionAuthorityBridgeState,
)
from alberta_framework.core.repeated_option_lifecycle import (
    RepeatedOptionLifecycle,
    RepeatedOptionLifecycleConfig,
)
from alberta_framework.evaluation.prototype_repeated_option_lifecycle_development import (
    PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_ASSESSMENT,
    PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_MECHANISM_STATUS,
    PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_OUTCOME_STATUS,
    PrototypeRepeatedOptionLifecycleDevelopmentConfig,
    PrototypeRepeatedOptionLifecycleDevelopmentHarness,
    ReplacementAttemptExhaustedError,
)

pytestmark = [pytest.mark.integration, pytest.mark.development, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_module() -> Iterator[None]:
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]


class _Rig(NamedTuple):
    lower: Any
    bridge: PrototypeRepeatedOptionAuthorityBridge
    source: PrototypeRepeatedOptionAuthorityBridgeState


@pytest.fixture(scope="module")
def rig() -> _Rig:
    lower = _one_shot_context(max_installations=8)
    stomp_config = lower.controller.scheduler.installation.stomp_agent.config
    agent = PrototypeAgent(PrototypeAgentConfig(oak=OaKConfig(stomp=stomp_config)))
    v1 = PrototypeOptionAuthorityBridge(agent, lower.controller)
    pristine = agent.init(jr.key(0xA22))
    receipt = v1.declare_initial_owner_binding(
        pristine,
        lower.pre_retirement_state,
        binding_authorized=True,
    )
    bound = v1.bind_initial_prototype_owner(
        pristine,
        lower.pre_retirement_state,
        receipt,
    )
    assert bool(bound.transaction_applied)
    v1_state = v1.init(bound.prototype_state, lower.pre_retirement_state)
    lifecycle = RepeatedOptionLifecycle(
        lower.controller,
        RepeatedOptionLifecycleConfig(max_cycles=2),
    )
    bridge = PrototypeRepeatedOptionAuthorityBridge(v1, lifecycle)
    source = bridge.init(v1_state, lifecycle.init(lower.pre_retirement_state))
    return _Rig(lower, bridge, source)


def test_schedule_is_versioned_calibration_consumed_and_mechanically_capped() -> None:
    first = PrototypeRepeatedOptionLifecycleDevelopmentConfig()
    second = PrototypeRepeatedOptionLifecycleDevelopmentConfig()
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.control_update_caps == (1, 5)
    assert first.stop_on_first_post_use_primitive_fallback
    assert first.checkpoint_after_completed_cycles == 1
    assert first.control_discount > 0.0
    assert first.censored_execution_boundary is True
    assert first.to_config()["calibration_consumed"] is True
    assert first.to_config()["preregistered"] is False
    assert first.to_config()["cycle_one_cap_derivation"] == "option_budget_plus_one"
    assert first.to_config()["replacement_attempt_cap_derivation"] == (
        "scheduler_config.max_install_attempts"
    )
    with pytest.raises(ValueError, match="fixed"):
        PrototypeRepeatedOptionLifecycleDevelopmentConfig(control_reward=-0.5)
    with pytest.raises(TypeError):
        PrototypeRepeatedOptionLifecycleDevelopmentHarness(object())


def test_replacement_attempt_cap_is_declared_before_candidate_outcomes(rig: _Rig) -> None:
    harness = PrototypeRepeatedOptionLifecycleDevelopmentHarness(rig.bridge)
    scheduler = rig.bridge.lifecycle.replacement.scheduler
    assert harness.config.max_replacement_attempts == scheduler.config.max_install_attempts
    assert harness.config.max_replacement_attempts == 8


def test_consumed_two_cycle_schedule_fails_closed_at_exact_scheduler_bound(
    rig: _Rig,
) -> None:
    harness = PrototypeRepeatedOptionLifecycleDevelopmentHarness(rig.bridge)
    assert harness.config.to_config()["outcome_status"] == (
        PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_OUTCOME_STATUS
    )
    with pytest.raises(ReplacementAttemptExhaustedError) as captured:
        harness.run(rig.source)

    failure = captured.value
    assert failure.cycle_index == 1
    assert failure.attempts == 8
    assert failure.scheduler_attempt_cap == 8
    assert failure.scheduler_attempt_cap == (
        rig.bridge.lifecycle.replacement.scheduler.config.max_install_attempts
    )
    assert failure.completed_cycles_before == 1
    assert failure.cycle_zero_completed
    assert len(failure.completed_cycle_traces) == 1
    first = failure.completed_cycle_traces[0]
    assert first.completed_cycles_before == 0
    assert first.completed_cycles_after == 1
    assert first.retirement_revision_words == (0, 1)
    assert first.replacement_revision_words == (0, 2)
    assert first.replacement_attempts == 2
    assert first.option_use_count > 0
    assert first.control_return_count > 0
    assert first.primitive_fallback_count > 0
    assert first.stale_retirement_replay_rejected
    assert first.declined_replacement_observed
    assert first.stale_replacement_replay_rejected
    assert first.fresh_retry_required
    assert first.persistent_stomp_state_owners == 1

    assert len(failure.failed_cycle_control_events) >= 1
    assert any(event.option_used for event in failure.failed_cycle_control_events)
    assert any(event.control_returned for event in failure.failed_cycle_control_events)
    assert any(event.primitive_fallback for event in failure.failed_cycle_control_events)
    assert all(
        event.censored_execution_boundary
        for event in failure.failed_cycle_control_events
    )
    assert all(
        event.stomp_update_evaluations == 1
        for event in failure.failed_cycle_control_events
    )

    attempts = failure.attempt_diagnostics
    assert len(attempts) == 8
    assert tuple(item.attempt_index for item in attempts) == tuple(range(1, 9))
    assert all(item.ordinary_advance_applied for item in attempts)
    assert all(item.proposal_due for item in attempts)
    assert all(item.proposal_ready for item in attempts)
    assert all(not item.candidate_ready_for_authority for item in attempts)
    assert all(
        item.rejection_reason == "semantic_change_mask_mismatch"
        for item in attempts
    )
    assert tuple(item.scheduler_step_before for item in attempts) == tuple(
        (0, value) for value in range(6, 14)
    )
    assert tuple(item.scheduler_step_after for item in attempts) == tuple(
        (0, value) for value in range(7, 15)
    )
    assert tuple(item.scheduler_observation_count_before for item in attempts) == tuple(
        range(6, 14)
    )
    assert tuple(item.scheduler_observation_count_after for item in attempts) == tuple(
        range(7, 15)
    )
    assert all(len(item.selected_candidate_indices) == 4 for item in attempts)
    assert all(len(item.selected_descriptors) == 4 for item in attempts)
    assert all(
        all(len(descriptor) == 4 for descriptor in item.selected_descriptors)
        for item in attempts
    )
    assert all(len(item.changed_slots) == 4 for item in attempts)
    assert all(item.semantic_generation == 3 for item in attempts)
    assert all(item.source_digest_words == (659918, 20958) for item in attempts)
    assert all(item.selected_candidate_indices == (1, 3, 4, 5) for item in attempts)
    assert all(
        item.selected_descriptors
        == (
            (0, 0, -1, 11),
            (1, 0, 1, 20),
            (2, 0, 1, 30),
            (3, 0, 1, 40),
        )
        for item in attempts
    )
    assert all(item.changed_slots == (False, False, False, False) for item in attempts)
    assert all(
        item.scheduler_observation_count_after
        == item.scheduler_observation_count_before + 1
        for item in attempts
    )
    assert all(
        item.installed_slot_mask_before == item.installed_slot_mask_after
        for item in attempts
    )
    assert all(item.cold_slot_mask_before == item.cold_slot_mask_after for item in attempts)
    assert all(sum(item.cold_slot_mask_before) == 1 for item in attempts)
    assert all(
        item.installed_slot_mask_before == (True, False, True, True)
        for item in attempts
    )
    assert all(
        item.scheduler_install_attempts_before == (0, 2)
        and item.scheduler_install_attempts_after == (0, 2)
        for item in attempts
    )
    assert all(item.installation_count_before == 3 for item in attempts)
    assert all(
        item.installation_count_before == item.installation_count_after
        for item in attempts
    )

    assert failure.source_state_valid_before
    assert failure.source_state_valid_after
    assert failure.source_state_unchanged
    assert failure.source_sha256_before == failure.source_sha256_after
    assert len(failure.source_sha256_before) == 64
    assert bool(jax.device_get(rig.bridge.state_valid(rig.source)))
    repeated, attached = rig.bridge._attach_source(rig.source)
    assert bool(jax.device_get(attached))
    assert int(jax.device_get(repeated.completed_cycles)) == 0
    assert not bool(jax.device_get(rig.source.bridge_state.prototype_state.started))

    assert failure.checkpoint_created
    assert failure.checkpoint_suffix_assessment == "not_assessed"
    assert failure.checkpoint_suffix_parity is None
    assert not failure.report_produced
    assert not failure.winner_selected
    assert not failure.benefit_claim
    assert not failure.efficacy_claim
    assert not failure.promotion_authority
    assert PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_MECHANISM_STATUS.endswith(
        "mechanism-only"
    )
    assert PROTOTYPE_REPEATED_OPTION_LIFECYCLE_DEVELOPMENT_ASSESSMENT == "not_assessed"
