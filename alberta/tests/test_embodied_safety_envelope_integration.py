# mypy: disable-error-code="attr-defined,call-arg,no-untyped-call,type-var"
"""Eager/JIT/scan parity for the non-learning WP9 hard envelope."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.embodied_safety_envelope import (
    EmbodiedCommand,
    EmbodiedSafetyEnvelope,
    EmbodiedSafetyEnvelopeConfig,
    EmbodiedSafetyEnvelopeState,
    EmbodiedTelemetry,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SOURCE = jnp.arange(1, 9, dtype=jnp.uint32)
MODEL = jnp.full((8,), 0x11, dtype=jnp.uint32)
OPTIMIZER = jnp.full((8,), 0x22, dtype=jnp.uint32)
LIFECYCLE = jnp.full((8,), 0x33, dtype=jnp.uint32)
PARTNER = jnp.full((8,), 0x44, dtype=jnp.uint32)


@pytest.fixture(autouse=True)
def _clear_jax_caches_after_test() -> Iterator[None]:
    yield
    jax.clear_caches()


def _words(values: list[int]) -> jax.Array:
    return jnp.stack(
        tuple(jnp.asarray([0, value], dtype=jnp.uint32) for value in values)
    )


def _envelope() -> EmbodiedSafetyEnvelope:
    return EmbodiedSafetyEnvelope(
        EmbodiedSafetyEnvelopeConfig(
            n_joints=2,
            joint_position_lower=(-1.0, -2.0),
            joint_position_upper=(1.0, 2.0),
            max_abs_joint_velocity=(1.0, 2.0),
            max_abs_joint_torque=(3.0, 4.0),
            workspace_lower=(-1.0, -1.0, 0.0),
            workspace_upper=(1.0, 1.0, 2.0),
            min_collision_clearance=0.1,
            fallback_joint_position=(0.0, 0.0),
            fallback_joint_velocity=(0.0, 0.0),
            fallback_joint_torque=(0.0, 0.0),
            fallback_workspace_position=(0.0, 0.0, 1.0),
            fallback_collision_clearance=1.0,
            reset_stationary_velocity_tolerance=0.01,
            max_telemetry_age_ticks=5,
            max_control_deadline_ticks=3,
            shadow_window=4,
            min_shadow_samples=3,
            min_shadow_success_lcb=0.5,
            wilson_z=1.0,
            max_shadow_calibration_error=0.2,
            max_shadow_latency_ticks=4,
            max_decisions=16,
            max_committed_actions=16,
            max_shadow_records=16,
            max_handshakes_per_kind=4,
            reset_authority_digest=(1, 2, 3, 4, 5, 6, 7, 8),
            rollback_authority_digest=(8, 7, 6, 5, 4, 3, 2, 1),
        )
    )


def _scan_step(
    envelope: EmbodiedSafetyEnvelope,
) -> Callable[
    [
        EmbodiedSafetyEnvelopeState,
        tuple[
            EmbodiedTelemetry,
            EmbodiedCommand,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ],
    tuple[EmbodiedSafetyEnvelopeState, tuple[jax.Array, ...]],
]:
    def step(
        state: EmbodiedSafetyEnvelopeState,
        inputs: tuple[
            EmbodiedTelemetry,
            EmbodiedCommand,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ) -> tuple[EmbodiedSafetyEnvelopeState, tuple[jax.Array, ...]]:
        telemetry, command, decision, action, now, deadline, reward, cost = inputs
        result = envelope.evaluate(
            state,
            telemetry,
            command,
            decision_id=decision,
            action_id=action,
            control_tick=now,
            control_deadline_tick=deadline,
            model_version=MODEL,
            optimizer_version=OPTIMIZER,
            lifecycle_version=LIFECYCLE,
            untrusted_reward=reward,
            partner_metadata_digest=PARTNER,
            learned_cost_estimate=cost,
        )
        return result.state, (
            result.transaction_applied,
            result.action_available,
            result.proposed_accepted,
            result.fallback_used,
            result.hard_violation,
            result.emergency_stop_latched_after,
            result.command.joint_position,
        )

    return step


def test_eager_jit_scan_parity_preserves_hard_boundary_and_stop_latch() -> None:
    envelope = _envelope()
    initial = envelope.init(source_digest=SOURCE)
    telemetry = EmbodiedTelemetry(
        joint_position=jnp.zeros((4, 2), dtype=jnp.float32),
        joint_velocity=jnp.zeros((4, 2), dtype=jnp.float32),
        joint_torque=jnp.zeros((4, 2), dtype=jnp.float32),
        workspace_position=jnp.tile(
            jnp.asarray([[0.0, 0.0, 1.0]], dtype=jnp.float32),
            (4, 1),
        ),
        collision_clearance=jnp.full((4,), 0.5, dtype=jnp.float32),
        bridge_connected=jnp.asarray([True, True, False, True]),
        emergency_stop=jnp.asarray([False, False, False, True]),
        telemetry_id=_words([1, 2, 3, 4]),
        sample_tick=_words([10, 11, 12, 13]),
    )
    commands = EmbodiedCommand(
        joint_position=jnp.asarray(
            [[1.0, 2.0], [0.2, 0.3], [0.2, 0.3], [0.2, 0.3]],
            dtype=jnp.float32,
        ),
        joint_velocity=jnp.asarray(
            [[1.0, -2.0], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
            dtype=jnp.float32,
        ),
        joint_torque=jnp.asarray(
            [[3.0, -4.0], [3.1, 0.0], [0.3, 0.4], [0.3, 0.4]],
            dtype=jnp.float32,
        ),
        workspace_position=jnp.asarray(
            [[-1.0, 1.0, 2.0], [0.2, 0.2, 1.0], [0.2, 0.2, 1.0], [0.2, 0.2, 1.0]],
            dtype=jnp.float32,
        ),
        collision_clearance=jnp.asarray([0.1, 0.4, 0.4, 0.4], dtype=jnp.float32),
    )
    inputs = (
        telemetry,
        commands,
        _words([1, 2, 3, 3]),
        _words([1, 2, 3, 3]),
        _words([15, 16, 17, 18]),
        _words([18, 19, 20, 21]),
        jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        jnp.asarray([-100.0, -100.0, -100.0, -100.0], dtype=jnp.float32),
    )
    step = _scan_step(envelope)
    eager_state = initial
    eager_facts: list[tuple[jax.Array, ...]] = []
    for index in range(4):
        row = jax.tree.map(lambda value: value[index], inputs)
        eager_state, facts = step(eager_state, row)
        eager_facts.append(facts)

    scanned_state, scanned_facts = jax.jit(
        lambda state, values: jax.lax.scan(step, state, values)
    )(initial, inputs)
    chex.assert_trees_all_equal(scanned_state, eager_state)
    expected = jax.tree.map(lambda *values: jnp.stack(values), *eager_facts)
    chex.assert_trees_all_equal(scanned_facts, expected)
    np.testing.assert_array_equal(scanned_facts[0], [True, True, True, False])
    np.testing.assert_array_equal(scanned_facts[1], [True, True, False, False])
    np.testing.assert_array_equal(scanned_facts[2], [True, False, False, False])
    np.testing.assert_array_equal(scanned_facts[3], [False, True, False, False])
    np.testing.assert_array_equal(scanned_facts[4], [False, True, True, True])
    assert bool(scanned_state.emergency_stop_latched)
    assert int(scanned_state.decision_count) == 3
    assert int(scanned_state.committed_action_count) == 2
    assert int(scanned_state.fallback_action_count) == 1
    assert int(scanned_state.hard_violation_count) == 2
    assert int(scanned_state.emergency_stop_latch_count) == 1
    assert bool(envelope.state_valid(scanned_state))
