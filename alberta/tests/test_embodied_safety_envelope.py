# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Unit contracts for the non-learning WP9 embodied hard envelope."""

from __future__ import annotations

import copy
import dataclasses

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.embodied_safety_envelope import (
    EMBODIED_SAFETY_ENVELOPE_ACTION_DISPATCH_AUTHORITY,
    EMBODIED_SAFETY_ENVELOPE_CALLER_AUTHENTICATION,
    EMBODIED_SAFETY_ENVELOPE_DEPLOYMENT_AUTHORITY,
    EMBODIED_SAFETY_ENVELOPE_LEARNED_COST_OVERRIDE_AUTHORITY,
    EMBODIED_SAFETY_ENVELOPE_LEARNING_MUTATION_AUTHORITY,
    EMBODIED_SAFETY_ENVELOPE_PHYSICAL_SAFETY_CLAIM,
    EMBODIED_SAFETY_ENVELOPE_PROMOTION_AUTHORITY,
    EMBODIED_SAFETY_ENVELOPE_SCIENTIFIC_PROMOTION_ALLOWED,
    ENVELOPE_REASON_CAPACITY,
    ENVELOPE_REASON_EMERGENCY_STOP,
    ENVELOPE_REASON_PERSISTENT_STATE,
    HANDSHAKE_REASON_AUTHORITY,
    AuthorityBoundEnvelopeHandshake,
    EmbodiedCommand,
    EmbodiedEnvelopeDecision,
    EmbodiedSafetyEnvelope,
    EmbodiedSafetyEnvelopeConfig,
    EmbodiedSafetyEnvelopeState,
    EmbodiedShadowEvaluation,
    EmbodiedTelemetry,
)

pytestmark = pytest.mark.unit

SOURCE = jnp.arange(11, 19, dtype=jnp.uint32)
MODEL = jnp.full((8,), 0x11, dtype=jnp.uint32)
OPTIMIZER = jnp.full((8,), 0x22, dtype=jnp.uint32)
LIFECYCLE = jnp.full((8,), 0x33, dtype=jnp.uint32)
PARTNER = jnp.full((8,), 0x44, dtype=jnp.uint32)


def _words(value: int) -> jnp.ndarray:
    return jnp.asarray([0, value], dtype=jnp.uint32)


def _config(**overrides: object) -> EmbodiedSafetyEnvelopeConfig:
    values: dict[str, object] = {
        "n_joints": 2,
        "joint_position_lower": (-1.0, -2.0),
        "joint_position_upper": (1.0, 2.0),
        "max_abs_joint_velocity": (1.0, 2.0),
        "max_abs_joint_torque": (3.0, 4.0),
        "workspace_lower": (-1.0, -1.0, 0.0),
        "workspace_upper": (1.0, 1.0, 2.0),
        "min_collision_clearance": 0.1,
        "fallback_joint_position": (0.0, 0.0),
        "fallback_joint_velocity": (0.0, 0.0),
        "fallback_joint_torque": (0.0, 0.0),
        "fallback_workspace_position": (0.0, 0.0, 1.0),
        "fallback_collision_clearance": 1.0,
        "reset_stationary_velocity_tolerance": 0.01,
        "max_telemetry_age_ticks": 5,
        "max_control_deadline_ticks": 3,
        "shadow_window": 4,
        "min_shadow_samples": 3,
        "min_shadow_success_lcb": 0.5,
        "wilson_z": 1.0,
        "max_shadow_calibration_error": 0.2,
        "max_shadow_latency_ticks": 4,
        "max_decisions": 32,
        "max_committed_actions": 32,
        "max_shadow_records": 32,
        "max_handshakes_per_kind": 8,
        "reset_authority_digest": (1, 2, 3, 4, 5, 6, 7, 8),
        "rollback_authority_digest": (8, 7, 6, 5, 4, 3, 2, 1),
    }
    values.update(overrides)
    return EmbodiedSafetyEnvelopeConfig(**values)  # type: ignore[arg-type]


def _envelope(**overrides: object) -> EmbodiedSafetyEnvelope:
    return EmbodiedSafetyEnvelope(_config(**overrides))


def _state(envelope: EmbodiedSafetyEnvelope) -> EmbodiedSafetyEnvelopeState:
    return envelope.init(source_digest=SOURCE)


def _telemetry(
    identity: int = 1,
    sample_tick: int = 10,
    *,
    position: tuple[float, float] = (0.0, 0.0),
    velocity: tuple[float, float] = (0.0, 0.0),
    torque: tuple[float, float] = (0.0, 0.0),
    workspace: tuple[float, float, float] = (0.0, 0.0, 1.0),
    clearance: float = 0.5,
    connected: bool = True,
    emergency_stop: bool = False,
) -> EmbodiedTelemetry:
    return EmbodiedTelemetry(
        joint_position=jnp.asarray(position, dtype=jnp.float32),
        joint_velocity=jnp.asarray(velocity, dtype=jnp.float32),
        joint_torque=jnp.asarray(torque, dtype=jnp.float32),
        workspace_position=jnp.asarray(workspace, dtype=jnp.float32),
        collision_clearance=jnp.asarray(clearance, dtype=jnp.float32),
        bridge_connected=jnp.asarray(connected, dtype=jnp.bool_),
        emergency_stop=jnp.asarray(emergency_stop, dtype=jnp.bool_),
        telemetry_id=_words(identity),
        sample_tick=_words(sample_tick),
    )


def _command(
    *,
    position: tuple[float, float] = (0.2, 0.3),
    velocity: tuple[float, float] = (0.1, 0.2),
    torque: tuple[float, float] = (0.3, 0.4),
    workspace: tuple[float, float, float] = (0.2, 0.2, 1.0),
    clearance: float = 0.4,
) -> EmbodiedCommand:
    return EmbodiedCommand(
        joint_position=jnp.asarray(position, dtype=jnp.float32),
        joint_velocity=jnp.asarray(velocity, dtype=jnp.float32),
        joint_torque=jnp.asarray(torque, dtype=jnp.float32),
        workspace_position=jnp.asarray(workspace, dtype=jnp.float32),
        collision_clearance=jnp.asarray(clearance, dtype=jnp.float32),
    )


def _evaluate(
    envelope: EmbodiedSafetyEnvelope,
    state: EmbodiedSafetyEnvelopeState,
    telemetry: EmbodiedTelemetry,
    command: EmbodiedCommand,
    *,
    decision: int = 1,
    action: int = 1,
    now: int = 12,
    deadline: int = 15,
    reward: float | jnp.ndarray = 7.0,
    learned_cost: float | jnp.ndarray = -1_000.0,
) -> EmbodiedEnvelopeDecision:
    return envelope.evaluate(
        state,
        telemetry,
        command,
        decision_id=_words(decision),
        action_id=_words(action),
        control_tick=_words(now),
        control_deadline_tick=_words(deadline),
        model_version=MODEL,
        optimizer_version=OPTIMIZER,
        lifecycle_version=LIFECYCLE,
        untrusted_reward=reward,
        partner_metadata_digest=PARTNER,
        learned_cost_estimate=learned_cost,
    )


def _handshake(
    envelope: EmbodiedSafetyEnvelope,
    state: EmbodiedSafetyEnvelopeState,
    *,
    nonce: int,
    reset: bool,
) -> AuthorityBoundEnvelopeHandshake:
    authority = (
        envelope.config.reset_authority_digest
        if reset
        else envelope.config.rollback_authority_digest
    )
    return AuthorityBoundEnvelopeHandshake(
        nonce=_words(nonce),
        authority_digest=jnp.asarray(authority, dtype=jnp.uint32),
        source_digest=state.source_digest,
        config_digest=envelope.config_digest,
        observed_state_revision=state.revision,
        observed_state_checksum=state.state_checksum,
    )


def _shadow(
    envelope: EmbodiedSafetyEnvelope,
    state: EmbodiedSafetyEnvelopeState,
    *,
    decision: int,
    success: bool = True,
    calibration: float = 0.2,
    latency: int = 4,
    command: EmbodiedCommand | None = None,
) -> EmbodiedShadowEvaluation:
    return envelope.evaluate_shadow(
        state,
        _telemetry(identity=decision, sample_tick=10),
        _command() if command is None else command,
        decision_id=_words(decision),
        control_tick=_words(12),
        control_deadline_tick=_words(15),
        model_version=MODEL,
        optimizer_version=OPTIMIZER,
        lifecycle_version=LIFECYCLE,
        observed_success=success,
        calibration_error=calibration,
        latency_ticks=latency,
        untrusted_reward=99.0,
        partner_metadata_digest=PARTNER,
        learned_cost_estimate=-999.0,
    )


def test_config_fallback_resource_and_zero_authority_are_exact() -> None:
    envelope = _envelope()
    state = _state(envelope)
    assert bool(envelope.state_valid(state))
    assert EmbodiedSafetyEnvelopeConfig.from_config(envelope.to_config()) == envelope.config
    budget = envelope.resource_budget(state)
    assert budget.persistent_state_nbytes > 0
    assert budget.n_joints == 2
    assert budget.workspace_dimensions == 3
    assert budget.shadow_window == 4
    assert budget.random_generator_calls_per_operation == 0
    assert budget.action_dispatches_per_operation == 0
    assert budget.learning_state_mutations_per_operation == 0
    assert budget.physical_safety_claim is False
    assert budget.caller_authentication is False
    assert EMBODIED_SAFETY_ENVELOPE_ACTION_DISPATCH_AUTHORITY is False
    assert EMBODIED_SAFETY_ENVELOPE_DEPLOYMENT_AUTHORITY is False
    assert EMBODIED_SAFETY_ENVELOPE_LEARNING_MUTATION_AUTHORITY is False
    assert EMBODIED_SAFETY_ENVELOPE_LEARNED_COST_OVERRIDE_AUTHORITY is False
    assert EMBODIED_SAFETY_ENVELOPE_PHYSICAL_SAFETY_CLAIM is False
    assert EMBODIED_SAFETY_ENVELOPE_PROMOTION_AUTHORITY is False
    assert EMBODIED_SAFETY_ENVELOPE_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert EMBODIED_SAFETY_ENVELOPE_CALLER_AUTHENTICATION is False
    with pytest.raises(ValueError, match="fallback joint position"):
        _config(fallback_joint_position=(2.0, 0.0))
    with pytest.raises(ValueError, match="wilson_z squared"):
        _config(wilson_z=3.0e19)
    with pytest.raises(ValueError, match="must be distinct"):
        _config(rollback_authority_digest=(1, 2, 3, 4, 5, 6, 7, 8))


def test_boundary_inclusive_safe_command_logs_exact_versions_and_metadata() -> None:
    envelope = _envelope()
    command = _command(
        position=(-1.0, 2.0),
        velocity=(-1.0, 2.0),
        torque=(3.0, -4.0),
        workspace=(-1.0, 1.0, 2.0),
        clearance=0.1,
    )
    result = _evaluate(envelope, _state(envelope), _telemetry(), command)
    assert bool(result.transaction_applied)
    assert bool(result.action_available)
    assert bool(result.proposed_accepted)
    assert not bool(result.fallback_used)
    assert not bool(result.learned_cost_override_used)
    chex.assert_trees_all_equal(result.command, command)
    assert bool(envelope.state_valid(result.state))
    np.testing.assert_array_equal(result.state.last_committed_decision_id, _words(1))
    np.testing.assert_array_equal(result.state.last_action_id, _words(1))
    np.testing.assert_array_equal(result.state.last_model_version, MODEL)
    np.testing.assert_array_equal(result.state.last_optimizer_version, OPTIMIZER)
    np.testing.assert_array_equal(result.state.last_lifecycle_version, LIFECYCLE)
    np.testing.assert_array_equal(
        result.state.last_logged_config_digest,
        envelope.config_digest,
    )
    np.testing.assert_array_equal(result.state.last_partner_metadata_digest, PARTNER)
    assert float(result.state.last_untrusted_reward) == 7.0
    assert float(result.state.last_learned_cost_estimate) == -1_000.0


@pytest.mark.parametrize(
    "command,gate",
    [
        (_command(position=(float("nan"), 0.0)), "proposed_position_finite"),
        (_command(position=(1.01, 0.0)), "proposed_position_in_bounds"),
        (_command(velocity=(1.01, 0.0)), "proposed_velocity_in_bounds"),
        (_command(torque=(3.01, 0.0)), "proposed_torque_in_bounds"),
        (_command(workspace=(1.01, 0.0, 1.0)), "proposed_workspace_in_bounds"),
        (_command(clearance=0.099), "proposed_clearance_in_bounds"),
    ],
)
def test_every_unsafe_proposed_dimension_uses_only_certified_fallback(
    command: EmbodiedCommand,
    gate: str,
) -> None:
    envelope = _envelope()
    result = _evaluate(envelope, _state(envelope), _telemetry(), command)
    assert bool(result.transaction_applied)
    assert bool(result.action_available)
    assert bool(result.fallback_used)
    assert not bool(result.proposed_accepted)
    assert not bool(getattr(result, gate))
    assert bool(result.hard_violation)
    chex.assert_trees_all_equal(result.command, envelope.fallback_command)
    assert bool(envelope.state_valid(result.state))


@pytest.mark.parametrize(
    "telemetry,now,deadline,gate",
    [
        (_telemetry(position=(1.01, 0.0)), 12, 15, "current_position_in_bounds"),
        (_telemetry(velocity=(1.01, 0.0)), 12, 15, "current_velocity_in_bounds"),
        (_telemetry(torque=(3.01, 0.0)), 12, 15, "current_torque_in_bounds"),
        (_telemetry(workspace=(1.01, 0.0, 1.0)), 12, 15, "current_workspace_in_bounds"),
        (_telemetry(clearance=0.099), 12, 15, "current_clearance_in_bounds"),
        (_telemetry(connected=False), 12, 15, "bridge_connected"),
        (_telemetry(sample_tick=1), 12, 15, "telemetry_fresh"),
        (_telemetry(), 12, 16, "control_deadline_valid"),
    ],
)
def test_current_disconnect_stale_deadline_and_collision_cannot_certify_fallback(
    telemetry: EmbodiedTelemetry,
    now: int,
    deadline: int,
    gate: str,
) -> None:
    envelope = _envelope()
    result = _evaluate(
        envelope,
        _state(envelope),
        telemetry,
        _command(),
        now=now,
        deadline=deadline,
    )
    assert bool(result.transaction_applied)
    assert not bool(result.action_available)
    assert not bool(result.fallback_certified)
    assert not bool(getattr(result, gate))
    assert bool(result.hard_violation)
    assert int(result.state.rejected_action_count) == 1
    assert bool(envelope.state_valid(result.state))


def test_corrupt_persistent_state_makes_every_action_unavailable_and_is_atomic() -> None:
    envelope = _envelope()
    state = _state(envelope)
    corrupt = dataclasses.replace(
        state,
        state_checksum=jnp.zeros((2,), dtype=jnp.uint32),
    )
    result = _evaluate(envelope, corrupt, _telemetry(), _command())
    assert not bool(result.persistent_state_valid)
    assert not bool(result.transaction_applied)
    assert not bool(result.action_available)
    assert int(result.unavailable_reason) == ENVELOPE_REASON_PERSISTENT_STATE
    chex.assert_trees_all_equal(result.state, corrupt)


def test_emergency_stop_latch_cannot_be_suppressed_by_replay_or_invalid_metadata() -> None:
    envelope = _envelope()
    first = _evaluate(envelope, _state(envelope), _telemetry(), _command())
    stopped = _evaluate(
        envelope,
        first.state,
        _telemetry(identity=2, sample_tick=11, emergency_stop=True),
        _command(),
        decision=1,
        action=1,
        reward=jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    assert not bool(stopped.transaction_applied)
    assert bool(stopped.emergency_stop_latch_applied)
    assert not bool(stopped.action_available)
    assert bool(stopped.state.emergency_stop_latched)
    assert int(stopped.state.emergency_stop_latch_count) == 1
    assert int(stopped.state.decision_count) == int(first.state.decision_count)
    assert bool(envelope.state_valid(stopped.state))

    same_sample_reset = envelope.authority_bound_reset(
        stopped.state,
        _handshake(envelope, stopped.state, nonce=1, reset=True),
        _telemetry(identity=2, sample_tick=11),
        control_tick=_words(12),
        control_deadline_tick=_words(15),
    )
    assert not bool(same_sample_reset.applied)
    assert not bool(same_sample_reset.stationary_safe)
    chex.assert_trees_all_equal(same_sample_reset.state, stopped.state)

    later = _evaluate(
        envelope,
        stopped.state,
        _telemetry(identity=3, sample_tick=12),
        _command(),
        decision=2,
        action=2,
    )
    assert bool(later.transaction_applied)
    assert not bool(later.action_available)
    assert bool(later.state.emergency_stop_latched)


def test_emergency_stop_latch_has_revision_headroom_after_decision_capacity() -> None:
    envelope = _envelope(max_decisions=1, max_committed_actions=1)
    first = _evaluate(envelope, _state(envelope), _telemetry(), _command())
    stopped = _evaluate(
        envelope,
        first.state,
        _telemetry(identity=2, sample_tick=11, emergency_stop=True),
        _command(),
        decision=2,
        action=2,
    )
    assert not bool(stopped.transaction_applied)
    assert bool(stopped.emergency_stop_latch_applied)
    assert not bool(stopped.decision_capacity_available)
    assert bool(stopped.state.emergency_stop_latched)
    assert int(stopped.state.emergency_stop_latch_count) == 1
    assert bool(envelope.state_valid(stopped.state))


def test_decision_capacity_and_identity_replay_fail_closed() -> None:
    envelope = _envelope(max_decisions=1, max_committed_actions=1)
    first = _evaluate(envelope, _state(envelope), _telemetry(), _command())
    assert bool(first.action_available)
    replay = _evaluate(
        envelope,
        first.state,
        _telemetry(identity=2, sample_tick=11),
        _command(),
        decision=1,
        action=2,
        now=12,
        deadline=15,
    )
    assert not bool(replay.action_available)
    assert int(replay.unavailable_reason) == ENVELOPE_REASON_CAPACITY
    chex.assert_trees_all_equal(replay.state, first.state)


def test_uint64_identity_ceiling_rejects_replay_without_wraparound() -> None:
    envelope = _envelope()
    maximum = jnp.asarray([0xFFFFFFFF, 0xFFFFFFFF], dtype=jnp.uint32)
    telemetry = dataclasses.replace(_telemetry(), telemetry_id=maximum)
    first = envelope.evaluate(
        _state(envelope),
        telemetry,
        _command(),
        decision_id=maximum,
        action_id=maximum,
        control_tick=_words(12),
        control_deadline_tick=_words(15),
        model_version=MODEL,
        optimizer_version=OPTIMIZER,
        lifecycle_version=LIFECYCLE,
        untrusted_reward=1.0,
        partner_metadata_digest=PARTNER,
        learned_cost_estimate=-1.0,
    )
    assert bool(first.action_available)
    replay = envelope.evaluate(
        first.state,
        telemetry,
        _command(),
        decision_id=maximum,
        action_id=maximum,
        control_tick=_words(12),
        control_deadline_tick=_words(15),
        model_version=MODEL,
        optimizer_version=OPTIMIZER,
        lifecycle_version=LIFECYCLE,
        untrusted_reward=1.0,
        partner_metadata_digest=PARTNER,
        learned_cost_estimate=-1.0,
    )
    assert not bool(replay.transaction_applied)
    assert not bool(replay.action_available)
    chex.assert_trees_all_equal(replay.state, first.state)


def test_emergency_stop_latches_until_authority_bound_stationary_reset() -> None:
    envelope = _envelope()
    stopped = _evaluate(
        envelope,
        _state(envelope),
        _telemetry(emergency_stop=True),
        _command(),
    )
    assert bool(stopped.transaction_applied)
    assert not bool(stopped.action_available)
    assert bool(stopped.state.emergency_stop_latched)
    assert int(stopped.unavailable_reason) == ENVELOPE_REASON_EMERGENCY_STOP
    still_stopped = _evaluate(
        envelope,
        stopped.state,
        _telemetry(identity=2, sample_tick=11),
        _command(),
        decision=2,
        action=1,
    )
    assert not bool(still_stopped.action_available)
    assert bool(still_stopped.state.emergency_stop_latched)

    handshake = _handshake(envelope, still_stopped.state, nonce=1, reset=True)
    bad = dataclasses.replace(
        handshake,
        authority_digest=jnp.full((8,), 99, dtype=jnp.uint32),
    )
    rejected = envelope.authority_bound_reset(
        still_stopped.state,
        bad,
        _telemetry(identity=3, sample_tick=12),
        control_tick=_words(12),
        control_deadline_tick=_words(15),
    )
    assert not bool(rejected.applied)
    assert int(rejected.unavailable_reason) == HANDSHAKE_REASON_AUTHORITY

    moving = envelope.authority_bound_reset(
        still_stopped.state,
        handshake,
        _telemetry(identity=3, sample_tick=12, velocity=(0.02, 0.0)),
        control_tick=_words(12),
        control_deadline_tick=_words(15),
    )
    assert not bool(moving.applied)
    assert not bool(moving.stationary_safe)
    reset = envelope.authority_bound_reset(
        still_stopped.state,
        handshake,
        _telemetry(identity=3, sample_tick=12),
        control_tick=_words(12),
        control_deadline_tick=_words(15),
    )
    assert bool(reset.applied)
    assert not bool(reset.state.emergency_stop_latched)
    assert bool(envelope.state_valid(reset.state))
    replay = envelope.authority_bound_reset(
        reset.state,
        handshake,
        _telemetry(identity=4, sample_tick=13),
        control_tick=_words(13),
        control_deadline_tick=_words(16),
    )
    assert not bool(replay.applied)
    assert bool(replay.replay_rejected)
    chex.assert_trees_all_equal(replay.state, reset.state)


def test_authority_bound_rollback_suspends_without_erasing_diagnostics() -> None:
    envelope = _envelope()
    committed = _evaluate(envelope, _state(envelope), _telemetry(), _command())
    before = committed.state
    handshake = _handshake(envelope, before, nonce=1, reset=False)
    rolled = envelope.authority_bound_rollback(before, handshake)
    assert bool(rolled.applied)
    assert bool(rolled.state.deployment_suspended)
    assert bool(rolled.state.emergency_stop_latched)
    assert int(rolled.state.decision_count) == int(before.decision_count)
    assert int(rolled.state.committed_action_count) == int(before.committed_action_count)
    np.testing.assert_array_equal(
        rolled.state.last_committed_decision_id,
        before.last_committed_decision_id,
    )
    assert bool(envelope.state_valid(rolled.state))
    replay = envelope.authority_bound_rollback(rolled.state, handshake)
    assert not bool(replay.applied)
    assert bool(replay.replay_rejected)
    chex.assert_trees_all_equal(replay.state, rolled.state)


def test_checkpoint_is_exact_source_config_bound_and_tamper_evident() -> None:
    envelope = _envelope()
    state = _evaluate(envelope, _state(envelope), _telemetry(), _command()).state
    payload = envelope.checkpoint_payload(state)
    restored = envelope.restore_checkpoint(
        copy.deepcopy(payload),
        expected_source_digest=SOURCE,
        trusted_state_revision=state.revision,
        trusted_state_digest=jnp.asarray(
            copy.deepcopy(payload["state_digest"]),
            dtype=jnp.uint8,
        ),
    )
    chex.assert_trees_all_equal(restored, state)
    tampered = copy.deepcopy(payload)
    tampered["state_digest"] = jnp.zeros((32,), dtype=jnp.uint8)
    with pytest.raises(ValueError, match="tampered"):
        envelope.restore_checkpoint(
            tampered,
            expected_source_digest=SOURCE,
            trusted_state_revision=state.revision,
            trusted_state_digest=jnp.asarray(
                copy.deepcopy(payload["state_digest"]),
                dtype=jnp.uint8,
            ),
        )
    with pytest.raises(ValueError, match="tampered"):
        envelope.restore_checkpoint(
            payload,
            expected_source_digest=SOURCE + 1,
            trusted_state_revision=state.revision,
            trusted_state_digest=jnp.asarray(
                copy.deepcopy(payload["state_digest"]),
                dtype=jnp.uint8,
            ),
        )


def test_external_restore_anchor_rejects_old_snapshot_that_erases_stop_and_nonce() -> None:
    envelope = _envelope()
    ordinary = _evaluate(envelope, _state(envelope), _telemetry(), _command()).state
    old_payload = envelope.checkpoint_payload(ordinary)
    rolled = envelope.authority_bound_rollback(
        ordinary,
        _handshake(envelope, ordinary, nonce=1, reset=False),
    )
    assert bool(rolled.applied)
    assert bool(rolled.state.emergency_stop_latched)
    assert bool(rolled.state.has_rollback_nonce)
    latest_payload = envelope.checkpoint_payload(rolled.state)
    trusted_digest = jnp.asarray(
        copy.deepcopy(latest_payload["state_digest"]),
        dtype=jnp.uint8,
    )

    with pytest.raises(ValueError, match="stale"):
        envelope.restore_checkpoint(
            old_payload,
            expected_source_digest=SOURCE,
            trusted_state_revision=rolled.state.revision,
            trusted_state_digest=trusted_digest,
        )
    restored = envelope.restore_checkpoint(
        latest_payload,
        expected_source_digest=SOURCE,
        trusted_state_revision=rolled.state.revision,
        trusted_state_digest=trusted_digest,
    )
    assert bool(restored.emergency_stop_latched)
    assert bool(restored.has_rollback_nonce)
    np.testing.assert_array_equal(restored.last_rollback_nonce, _words(1))


def test_shadow_is_pure_and_recent_ring_gate_is_conservative() -> None:
    envelope = _envelope()
    state = _state(envelope)
    learning_state = {"weights": jnp.asarray([1.0, 2.0], dtype=jnp.float32)}
    learning_before = copy.deepcopy(learning_state)
    for decision in range(1, 4):
        before = state
        outcome = _shadow(envelope, state, decision=decision)
        chex.assert_trees_all_equal(state, before)
        assert int(outcome.dispatches) == 0
        assert int(outcome.learning_state_mutations) == 0
        assert not bool(outcome.deployment_authority)
        recorded = envelope.record_shadow(state, outcome)
        assert bool(recorded.applied)
        state = recorded.state
    chex.assert_trees_all_equal(learning_state, learning_before)
    gate = envelope.deployment_gate(state)
    assert int(gate.sample_count) == 3
    assert int(gate.success_count) == 3
    assert int(gate.hard_violation_count) == 0
    assert float(gate.performance_success_lcb) >= 0.5
    assert float(gate.max_calibration_error) == pytest.approx(0.2)
    assert int(gate.max_latency_ticks) == 4
    assert bool(gate.deployment_ready)
    assert not bool(gate.deployment_authority)
    assert not bool(gate.learned_cost_override_used)

    unsafe = _shadow(
        envelope,
        state,
        decision=4,
        command=_command(clearance=0.01),
    )
    assert bool(unsafe.hard_violation)
    state = envelope.record_shadow(state, unsafe).state
    blocked = envelope.deployment_gate(state)
    assert int(blocked.hard_violation_count) == 1
    assert not bool(blocked.hard_zero)
    assert not bool(blocked.deployment_ready)


def test_shadow_stale_binding_replay_and_finite_record_cap_are_atomic() -> None:
    envelope = _envelope(
        shadow_window=2,
        min_shadow_samples=2,
        max_shadow_records=2,
    )
    state = _state(envelope)
    stale = _shadow(envelope, state, decision=1)
    tampered = dataclasses.replace(
        stale,
        hard_violation=~stale.hard_violation,
    )
    rejected_tamper = envelope.record_shadow(state, tampered)
    assert not bool(rejected_tamper.transaction_valid)
    assert not bool(rejected_tamper.applied)
    chex.assert_trees_all_equal(rejected_tamper.state, state)
    first = envelope.record_shadow(state, stale)
    assert bool(first.applied)
    stale_again = envelope.record_shadow(first.state, stale)
    assert not bool(stale_again.applied)
    chex.assert_trees_all_equal(stale_again.state, first.state)
    second_outcome = _shadow(envelope, first.state, decision=2)
    second = envelope.record_shadow(first.state, second_outcome)
    assert bool(second.applied)
    capped_outcome = _shadow(envelope, second.state, decision=3)
    capped = envelope.record_shadow(second.state, capped_outcome)
    assert not bool(capped.capacity_available)
    assert not bool(capped.applied)
    chex.assert_trees_all_equal(capped.state, second.state)
