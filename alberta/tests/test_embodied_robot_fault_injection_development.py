"""Unit contracts for the strict nonpromoting WP9 robot fault audit."""

from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.evaluation import embodied_robot_fault_injection_development as dev

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module")
def mechanical_run() -> tuple[Any, Any, tuple[dev.FaultTraceRecord, ...]]:
    return dev._run_schedule()


def _record(
    trace: tuple[dev.FaultTraceRecord, ...],
    fault: str,
) -> dev.FaultTraceRecord:
    return next(record for record in trace if record.event.fault == fault)


def test_protocol_is_frozen_single_lane_development_only_and_nonpromoting() -> None:
    assert dev.ASSESSMENT == "not_assessed"
    assert dev.DEVELOPMENT_ONLY
    assert not dev.SCIENTIFIC_PROMOTION_ALLOWED
    assert not dev.OUTPUT_WRITES_ALLOWED
    assert not dev.ARTIFACT_WRITER_AVAILABLE
    assert not dev.PHYSICAL_SAFETY_CLAIM
    assert not dev.DEPLOYMENT_AUTHORITY
    assert not dev.EFFICACY_CLAIM
    assert not dev.CALLER_AUTHENTICATION_PERFORMED
    assert dev.EXTERNAL_CALLER_AUTHENTICATION_REQUIRED
    assert dev.SYNTHETIC_TELEMETRY_AUDIT_SCHEDULE
    assert not dev.DYNAMICS_SIMULATION_PERFORMED
    assert not dev.GEOMETRY_PROOF
    assert not dev.LEARNER_ADAPTATION_LATENCY_AVAILABLE
    assert dev.RECOVERY_DELAYS_ARE_ENVELOPE_ACTION_AVAILABILITY_ONLY
    assert dev.SIMULATED_COMMAND_EXECUTION_IS_ACCOUNTING_ONLY
    assert dev.SHADOW_SUCCESS_INPUT_IS_ACTION_AVAILABILITY_PROXY
    assert dev.PHYSICAL_DISPATCHES == 0
    assert dev.RNG_DRAWS == 0
    assert dev.EVIDENCE_SEEDS == ()
    assert dev.ACCEPTANCE_THRESHOLDS == ()
    assert dev.COMPARISON_MODE == "single_strict_audit_lane"
    assert not dev.NO_CANDIDATE_ARM_EXECUTED
    assert "not a matched" in dev.NO_CANDIDATE_ARM_REASON
    assert all("write" not in name for name in dev.__all__)


@pytest.mark.parametrize(
    "change",
    [
        {"num_events": 29},
        {"checkpoint_split": 21},
        {"controller_revision": 72},
        {"shadow_calibration_error": 0.1},
    ],
)
def test_protocol_configuration_rejects_retuning(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="frozen"):
        dev.RobotFaultInjectionConfig(**change)


def test_schedule_covers_every_declared_fault_without_executing_held_out_family() -> None:
    schedule = dev.build_fault_schedule()
    assert schedule == dev.SCHEDULE
    assert len(schedule) == dev.CONFIG.num_events == 30
    phases = {event.phase for event in schedule}
    assert {
        "observation_drift",
        "dynamics_wear_drift",
        "timing_faults",
        "reward_delay",
        "sensor_faults",
        "bridge_faults",
        "unsafe_candidates",
        "emergency_stop",
        "checkpoint_recovery",
        "authority_control",
        "final_recovery",
    } <= phases
    faults = {event.fault for event in schedule}
    assert {
        "stale_telemetry",
        "deadline_miss",
        "delayed_untrusted_reward",
        "sensor_nan",
        "sensor_out_of_bounds",
        "sensor_failure",
        "bridge_disconnect",
        "bridge_reconnect",
        "unsafe_position",
        "unsafe_torque",
        "unsafe_clearance",
        "assert_stop",
        "stationary_reset",
        "authority_rollback",
        "stationary_reset_after_rollback",
    } <= faults
    held_out = dev.HELD_OUT_CHANGE_FAMILY
    assert held_out.declared
    assert not held_out.executed
    assert held_out.event_count == 0
    assert held_out.evidence_seeds == ()
    assert held_out.assessment == "not_assessed"


def test_only_envelope_available_commands_are_executed_and_fallback_is_exact(
    mechanical_run: tuple[Any, Any, tuple[dev.FaultTraceRecord, ...]],
) -> None:
    envelope, _, trace = mechanical_run
    fallback_bits = dev._command_bits(envelope.fallback_command)
    for record in trace:
        assert record.command_executed_in_simulation == record.action_available
        assert record.simulated_command_execution_is_accounting_only
        assert record.physical_dispatches == 0
        if record.action_available:
            assert record.executed_command is not None
        else:
            assert record.executed_command is None
        if record.fallback_used:
            assert record.fallback_certified
            assert record.executed_command == fallback_bits
    for fault in ("unsafe_position", "unsafe_torque", "unsafe_clearance"):
        record = _record(trace, fault)
        assert record.fallback_used
        assert record.action_available
        assert not record.proposed_envelope_safe


def test_timing_sensor_bridge_stop_and_suspension_faults_fail_closed(
    mechanical_run: tuple[Any, Any, tuple[dev.FaultTraceRecord, ...]],
) -> None:
    _, _, trace = mechanical_run
    stale = _record(trace, "stale_telemetry")
    deadline = _record(trace, "deadline_miss")
    assert not stale.telemetry_fresh and not stale.action_available
    assert not deadline.control_deadline_valid and not deadline.action_available
    for fault in (
        "sensor_nan",
        "sensor_out_of_bounds",
        "sensor_failure",
        "bridge_disconnect",
        "assert_stop",
        "latched_stop",
        "suspended_after_rollback",
    ):
        record = _record(trace, fault)
        assert record.hard_violation
        assert not record.action_available
    reconnect = _record(trace, "bridge_reconnect")
    assert reconnect.bridge_connected
    assert reconnect.action_available


def test_delayed_reward_is_exact_untrusted_metadata_and_never_an_override(
    mechanical_run: tuple[Any, Any, tuple[dev.FaultTraceRecord, ...]],
) -> None:
    _, _, trace = mechanical_run
    delayed = _record(trace, "delayed_untrusted_reward")
    source = trace[delayed.reward_source_index]
    assert delayed.reward_delay_events == 2
    assert delayed.reward_source_index == 9
    assert delayed.reward_float32_bits == source.reward_float32_bits
    assert delayed.metadata_finite
    assert delayed.action_available


def test_stop_reset_rollback_and_checkpoint_are_exact_causal_events(
    mechanical_run: tuple[Any, Any, tuple[dev.FaultTraceRecord, ...]],
) -> None:
    envelope, state, trace = mechanical_run
    asserted = _record(trace, "assert_stop")
    latched = _record(trace, "latched_stop")
    reset = _record(trace, "stationary_reset")
    restored_action = _record(trace, "post_restore_recovery")
    rollback = _record(trace, "authority_rollback")
    suspended = _record(trace, "suspended_after_rollback")
    second_reset = _record(trace, "stationary_reset_after_rollback")
    final = _record(trace, "first_action_after_reset")

    assert asserted.emergency_stop_latch_applied
    assert asserted.emergency_stop_latched_after
    assert latched.emergency_stop_latched_after
    assert reset.checkpoint_resumed_before
    assert reset.checkpoint_restore_exact
    assert reset.checkpoint_revision == reset.state_revision_before
    assert len(reset.checkpoint_state_digest) == 64
    assert reset.reset_applied and reset.reset_stationary_safe
    assert reset.telemetry is not None and asserted.telemetry is not None
    assert reset.telemetry.telemetry_id > asserted.telemetry.telemetry_id
    assert reset.telemetry.sample_tick > asserted.telemetry.sample_tick
    assert restored_action.action_available
    assert rollback.rollback_applied
    assert suspended.emergency_stop_latched_after
    assert not suspended.action_available
    assert second_reset.reset_applied and second_reset.reset_stationary_safe
    assert final.action_available
    assert int(state.reset_count) == 2
    assert int(state.rollback_count) == 1
    assert bool(envelope.state_valid(state))


def test_controller_and_learner_witness_is_uninterrupted(
    mechanical_run: tuple[Any, Any, tuple[dev.FaultTraceRecord, ...]],
) -> None:
    _, _, trace = mechanical_run
    for record in trace:
        assert record.controller_revision_before == dev.CONFIG.controller_revision
        assert record.controller_revision_after == dev.CONFIG.controller_revision
        assert record.controller_identity_before == dev.CONFIG.controller_state_identity
        assert record.controller_identity_after == dev.CONFIG.controller_state_identity
        assert record.learner_reset_count == 0
        assert record.learner_state_mutations == 0
        assert not record.caller_authentication_performed
        assert record.external_caller_authentication_required
        assert record.synthetic_telemetry_audit_schedule
        assert not record.dynamics_simulation_performed
        assert not record.geometry_proof
        assert record.rng_draws == 0


def test_shadow_ring_is_diagnostic_only_and_raw_bits_preserve_nan(
    mechanical_run: tuple[Any, Any, tuple[dev.FaultTraceRecord, ...]],
) -> None:
    _, _, trace = mechanical_run
    evaluate = [record for record in trace if record.event.operation == "evaluate"]
    assert all(record.shadow_recorded for record in evaluate)
    assert all(record.shadow_success_is_action_availability_proxy for record in trace)
    assert all(record.shadow_success_input == record.action_available for record in evaluate)
    assert all(
        not record.shadow_success_input for record in trace if record.event.operation != "evaluate"
    )
    assert all(not record.deployment_readout_authority for record in trace)
    nan_record = _record(trace, "sensor_nan")
    assert nan_record.telemetry is not None
    first_bits = nan_record.telemetry.joint_position[0]
    assert first_bits & 0x7F800000 == 0x7F800000
    assert first_bits & 0x007FFFFF != 0


def test_summary_reconstructs_from_raw_trace_and_fixed_state_budget(
    mechanical_run: tuple[Any, Any, tuple[dev.FaultTraceRecord, ...]],
) -> None:
    envelope, state, trace = mechanical_run
    summary = dev._summarize(envelope, state, trace)
    evaluate = [record for record in trace if record.event.operation == "evaluate"]
    assert summary.event_count == len(trace) == 30
    assert summary.evaluate_calls == len(evaluate) == 27
    assert summary.shadow_evaluation_calls == len(evaluate)
    assert summary.shadow_record_calls == len(evaluate)
    assert summary.shadow_record_applied_count == len(evaluate)
    assert summary.simulated_executed_command_count == sum(
        record.action_available for record in evaluate
    )
    assert summary.fallback_count == sum(record.fallback_used for record in evaluate)
    assert summary.unavailable_count == sum(not record.action_available for record in evaluate)
    assert summary.intervention_count == summary.fallback_count + summary.unavailable_count
    assert summary.physical_dispatch_count == 0
    assert summary.primary_reset_recovery_delay_events == 1
    assert summary.rollback_reset_recovery_delay_events == 1
    assert summary.bridge_reconnect_recovery_delay_events == 0
    assert not summary.adaptation_latency_assessed
    assert summary.adaptation_latency_events is None
    assert "not exercised" in summary.learner_adaptation_latency_unavailable_reason
    assert summary.recovery_delays_are_envelope_action_availability_only
    assert summary.simulated_command_execution_is_accounting_only
    assert summary.shadow_success_input_is_action_availability_proxy
    assert summary.persistent_state_bytes == envelope.resource_budget(state).persistent_state_nbytes
    assert summary.rng_draws == 0


def test_mechanical_trace_payload_is_finite_canonical_json(
    mechanical_run: tuple[Any, Any, tuple[dev.FaultTraceRecord, ...]],
) -> None:
    _, _, trace = mechanical_run
    encoded = dev._canonical_json_bytes([record.to_dict() for record in trace])
    assert b"NaN" not in encoded
    assert b"not_assessed" not in encoded
    assert len(encoded) > 1_000
