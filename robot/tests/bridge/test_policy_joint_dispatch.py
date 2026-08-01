"""Focused dispatch tests for profile-bounded direct joint policy ticks."""

from __future__ import annotations

import asyncio
import json

import pytest

from eliza_robot.bridge.protocol import (
    CommandEnvelope,
    ResponseEnvelope,
    utc_now_iso,
)
from eliza_robot.bridge.safety import MotionSafetySupervisor
from eliza_robot.bridge.server import (
    PolicyLoopState,
    RuntimeConfig,
    _handle_policy_command,
)
from eliza_robot.profiles.schema import load_profile


class _RecordingSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, payload: str) -> None:
        self.messages.append(json.loads(payload))


class _RecordingBackend:
    backend_name = "recording"

    def __init__(self) -> None:
        self.commands: list[CommandEnvelope] = []

    def capabilities(self) -> dict:
        return {
            "servo_set": True,
            "walk_set": True,
            "walk_command": True,
            "head_set": True,
            "motion_safety": {
                "environment": "nonphysical",
                "torque_limit_status": "not_applicable",
                "all_motion_stop": True,
                "stop_out_of_band": True,
                "walk_stop": True,
                "known_joint_pose_at_connect": True,
                "pose_remains_trusted_after_stop": True,
            },
        }

    def physical_motion_resources(self) -> tuple[str, ...]:
        return ()

    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        self.commands.append(command)
        return ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=True,
            backend=self.backend_name,
            message="ok",
            data={},
        )

    async def handle_emergency_stop(
        self,
        command: CommandEnvelope,
    ) -> ResponseEnvelope:
        return await self.handle_command(command)


class _FailingStopBackend(_RecordingBackend):
    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        self.commands.append(command)
        is_stop = (
            command.command == "walk.command"
            and command.payload.get("action") == "stop"
        )
        return ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=not is_stop,
            backend=self.backend_name,
            message="stop failed" if is_stop else "ok",
            data={},
        )


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        queue_size=8,
        max_commands_per_sec=100,
        deadman_timeout_sec=5.0,
        trace_log_path="",
        profile_id="hiwonder-ainex",
    )


def _active_policy_state() -> PolicyLoopState:
    profile = load_profile("hiwonder-ainex")
    return PolicyLoopState(
        active=True,
        last_joint_positions={
            joint.name: float(joint.home_rad)
            for joint in profile.kinematics.joints
        },
    )


def _tick(action: dict) -> CommandEnvelope:
    return CommandEnvelope(
        request_id="joint-tick",
        timestamp=utc_now_iso(),
        command="policy.tick",
        payload={"action": action},
    )


@pytest.mark.asyncio
async def test_direct_joint_dispatch_uses_only_the_validated_named_payload() -> None:
    profile = load_profile("hiwonder-ainex")
    state = _active_policy_state()
    backend = _RecordingBackend()
    socket = _RecordingSocket()
    raw_positions = {"r_hip_pitch": 0}

    response = await _handle_policy_command(
        socket,
        backend,
        _tick(
            {
                "joint_positions": raw_positions,
                "duration": 0.1,
            }
        ),
        state,
        None,
        _runtime_config(),
        profile,
    )

    assert response.ok
    servo_commands = [
        command for command in backend.commands if command.command == "servo.set"
    ]
    assert len(servo_commands) == 1
    dispatched = servo_commands[0].payload
    assert dispatched["positions"] == [{"id": 8, "position": 500}]
    assert "joint_positions" not in dispatched
    assert dispatched["duration"] == 0.1
    assert response.data["clamped"]["joint_positions"] == {
        "r_hip_pitch": 0.0
    }
    assert state.last_joint_positions["r_hip_pitch"] == 0.0


@pytest.mark.asyncio
async def test_direct_joint_dispatch_rejects_mixed_target_formats() -> None:
    profile = load_profile("hiwonder-ainex")
    state = _active_policy_state()
    backend = _RecordingBackend()
    socket = _RecordingSocket()

    response = await _handle_policy_command(
        socket,
        backend,
        _tick(
            {
                "joint_positions": {"r_hip_pitch": 0.0},
                "positions": [{"id": 8, "position": 1000}],
                "duration": 0.1,
            }
        ),
        state,
        None,
        _runtime_config(),
        profile,
    )

    assert not response.ok
    assert "alternate or mixed positions" in response.message
    assert not state.active
    assert [command.command for command in backend.commands] == ["walk.command"]
    assert backend.commands[0].payload == {"action": "stop"}


@pytest.mark.parametrize(
    ("joint_positions", "reason_fragment"),
    [
        ([{"id": 8, "position": 1000}], "non-empty object keyed by joint name"),
        ({"not_a_profile_joint": 0.0}, "unknown joint"),
        ({"r_hip_pitch": float("nan")}, "joint_positions"),
        ({"r_hip_pitch": 2.5}, "outside profile range"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_direct_joint_bypasses_stop_policy_without_servo_dispatch(
    joint_positions,
    reason_fragment: str,
) -> None:
    profile = load_profile("hiwonder-ainex")
    state = _active_policy_state()
    backend = _RecordingBackend()
    socket = _RecordingSocket()

    response = await _handle_policy_command(
        socket,
        backend,
        _tick({"joint_positions": joint_positions, "duration": 0.1}),
        state,
        None,
        _runtime_config(),
        profile,
    )

    assert not response.ok
    assert reason_fragment in response.message
    assert not state.active
    assert [command.command for command in backend.commands] == ["walk.command"]
    assert backend.commands[0].payload == {"action": "stop"}
    assert any(
        message.get("event") == "safety.policy_guard"
        for message in socket.messages
    )


@pytest.mark.parametrize(
    "duration",
    [True, "0.1", float("nan"), float("inf"), 0.0, -0.1, 5.1],
)
@pytest.mark.asyncio
async def test_invalid_direct_joint_duration_stops_without_dispatch(
    duration,
) -> None:
    profile = load_profile("hiwonder-ainex")
    state = _active_policy_state()
    backend = _RecordingBackend()
    socket = _RecordingSocket()

    response = await _handle_policy_command(
        socket,
        backend,
        _tick(
            {
                "joint_positions": {"r_hip_pitch": 0.1},
                "duration": duration,
            }
        ),
        state,
        None,
        _runtime_config(),
        profile,
    )

    assert not response.ok
    assert "duration" in response.message
    assert not state.active
    assert [command.command for command in backend.commands] == ["walk.command"]


@pytest.mark.asyncio
async def test_last_allowed_tick_immediately_stops_at_max_steps() -> None:
    profile = load_profile("hiwonder-ainex")
    state = _active_policy_state()
    state.max_steps = 1
    backend = _RecordingBackend()
    socket = _RecordingSocket()

    response = await _handle_policy_command(
        socket,
        backend,
        _tick({"walk_x": 0.01, "walk_y": 0.0, "walk_yaw": 0.0}),
        state,
        None,
        _runtime_config(),
        profile,
    )

    assert response.ok
    assert state.step == 1
    assert not state.active
    assert [command.command for command in backend.commands] == [
        "walk.set",
        "walk.command",
    ]
    assert backend.commands[-1].payload == {"action": "stop"}
    assert any(
        message.get("event") == "policy.status"
        and message["data"].get("reason") == "max_steps_reached"
        for message in socket.messages
    )


@pytest.mark.asyncio
async def test_policy_stop_cancels_loop_and_reports_pending_when_backend_stop_fails() -> None:
    profile = load_profile("hiwonder-ainex")
    state = _active_policy_state()
    backend = _FailingStopBackend()
    socket = _RecordingSocket()
    supervisor = MotionSafetySupervisor(
        backend,  # type: ignore[arg-type]
        profile,
        owner_id="failing-stop",
    )
    assert supervisor.acquire_motion()
    cancelled = False

    async def _autonomous_loop() -> None:
        nonlocal cancelled
        try:
            await asyncio.Future()
        finally:
            cancelled = True

    state._loop_task = asyncio.create_task(_autonomous_loop())
    await asyncio.sleep(0)
    response = await _handle_policy_command(
        socket,
        backend,
        CommandEnvelope(
            request_id="stop",
            timestamp=utc_now_iso(),
            command="policy.stop",
            payload={"reason": "manual_preempt"},
        ),
        state,
        None,
        _runtime_config(),
        profile,
        supervisor,
    )

    assert cancelled
    assert state._loop_task is None
    assert not response.ok
    assert response.data["stop_ok"] is False
    assert supervisor.emergency_stop_pending
    assert supervisor.owns_motion
    status = next(message for message in socket.messages if message.get("event") == "policy.status")
    assert status["data"]["state"] == "emergency_stop_pending"
    queried = await _handle_policy_command(
        socket,
        backend,
        CommandEnvelope(
            request_id="status",
            timestamp=utc_now_iso(),
            command="policy.status",
            payload={},
        ),
        state,
        None,
        _runtime_config(),
        profile,
        supervisor,
    )
    assert queried.data["state"] == "emergency_stop_pending"
    assert queried.data["emergency_stop_pending"] is True
    assert queried.data["owns_motion"] is True
