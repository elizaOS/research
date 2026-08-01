"""End-to-end coverage for the two control modes Eliza exposes to users:

  - "Joystick mode" — Eliza drives the robot via `walk.set` + `walk.command`
    envelopes. These map to the Hiwonder Bezier gait controller (real robot)
    or the joint-target sim loop (MuJoCo). The bridge protocol guarantees a
    consistent surface regardless of the backend.

  - "Trained mode" — Eliza launches a learned text-conditioned policy via
    `policy.start` and ticks it via `policy.tick`. The bridge applies safety
    clamps, dispatches the resulting joint targets to the backend, and emits
    `telemetry.policy` / `policy.status` events.

Both modes are tested here against the in-memory MockBackend so the suite
runs without GPU or MuJoCo dependencies.
"""

from __future__ import annotations

import json

import pytest
from websockets.asyncio.client import connect

from eliza_robot.bridge.protocol import CommandEnvelope, utc_now_iso


def _cmd(command: str, payload: dict, preempt: bool = False) -> str:
    envelope = CommandEnvelope(
        request_id=f"test-{command}",
        timestamp=utc_now_iso(),
        command=command,
        payload=payload,
        preempt=preempt,
    )
    return json.dumps(envelope.to_json())


async def _drain_until_response(ws, request_id: str, max_frames: int = 30):
    """Pull frames from the socket until we see the matching response."""
    for _ in range(max_frames):
        frame = json.loads(await ws.recv())
        if frame.get("type") == "response" and frame.get("request_id") == request_id:
            return frame
    raise AssertionError(f"no response for request_id={request_id}")


async def _drain_until_event(ws, event_name: str, max_frames: int = 30):
    for _ in range(max_frames):
        frame = json.loads(await ws.recv())
        if frame.get("type") == "event" and frame.get("event") == event_name:
            return frame
    raise AssertionError(f"no event named {event_name}")


@pytest.mark.asyncio
async def test_joystick_mode_walk_forward(mock_server: str) -> None:
    """Joystick mode: walk.set + walk.command:start drives the bridge into
    a walking state. Telemetry reflects the commanded velocity.
    """
    async with connect(mock_server) as ws:
        # Consume session.hello first.
        hello = json.loads(await ws.recv())
        assert hello["event"] == "session.hello"

        # walk.set
        await ws.send(
            _cmd(
                "walk.set",
                {"speed": 2, "height": 0.036, "x": 0.04, "y": 0.0, "yaw": 0.0},
            )
        )
        set_response = None
        for _ in range(30):
            frame = json.loads(await ws.recv())
            if frame.get("type") == "response" and frame.get("request_id") == "test-walk.set":
                set_response = frame
                break
        assert set_response is not None and set_response["ok"]

        # walk.command:start
        await ws.send(_cmd("walk.command", {"action": "start"}))
        start_response = None
        for _ in range(30):
            frame = json.loads(await ws.recv())
            if frame.get("type") == "response" and frame.get("request_id") == "test-walk.command":
                start_response = frame
                break
        assert start_response is not None and start_response["ok"]
        assert start_response["data"]["is_walking"] is True

        # Eventually a telemetry.basic event should report walking + walk_x
        telemetry = None
        for _ in range(40):
            frame = json.loads(await ws.recv())
            if (
                frame.get("type") == "event"
                and frame.get("event") == "telemetry.basic"
                and frame["data"]["is_walking"] is True
            ):
                telemetry = frame
                break
        assert telemetry is not None
        assert telemetry["data"]["walk_x"] == pytest.approx(0.04)
        assert telemetry["data"]["walk_speed"] == 2


@pytest.mark.asyncio
async def test_trained_mode_policy_tick_dispatch(mock_server: str) -> None:
    """Trained mode: policy.start opens a session; policy.tick with a
    learned action payload propagates to the backend (servo.set on the
    mock) and emits telemetry.policy.
    """
    async with connect(mock_server) as ws:
        # session.hello
        json.loads(await ws.recv())

        # policy.start with text-conditioned task
        await ws.send(
            _cmd(
                "policy.start",
                {
                    "task": "walk_forward",
                    "canonical_action": "walk_forward",
                    "trace_id": "trace-1",
                    "planner_step_id": "step-1",
                    "hz": 10,
                    "max_steps": 100,
                },
            )
        )

        start_response = None
        for _ in range(30):
            frame = json.loads(await ws.recv())
            if (
                frame.get("type") == "response"
                and frame.get("request_id") == "test-policy.start"
            ):
                start_response = frame
                break
        assert start_response is not None and start_response["ok"]
        assert start_response["data"]["task"] == "walk_forward"

        # Send a tick with a joint_positions action (direct joint control mode)
        await ws.send(
            _cmd(
                "policy.tick",
                {
                    "action": {
                        "joint_positions": {
                            "r_hip_pitch": -0.3,
                            "l_hip_pitch": -0.3,
                        },
                        "duration": 0.1,
                    }
                },
            )
        )

        tick_response = None
        for _ in range(30):
            frame = json.loads(await ws.recv())
            if (
                frame.get("type") == "response"
                and frame.get("request_id") == "test-policy.tick"
            ):
                tick_response = frame
                break
        assert tick_response is not None
        assert tick_response["ok"], f"tick failed: {tick_response.get('message')}"
        assert tick_response["data"]["step"] == 1
        assert tick_response["data"]["clamped"]["joint_positions"] == {
            "r_hip_pitch": -0.3,
            "l_hip_pitch": -0.3,
        }

        # policy.stop cleans up
        await ws.send(_cmd("policy.stop", {"reason": "test_done"}))
        stop_response = None
        for _ in range(30):
            frame = json.loads(await ws.recv())
            if (
                frame.get("type") == "response"
                and frame.get("request_id") == "test-policy.stop"
            ):
                stop_response = frame
                break
        assert stop_response is not None and stop_response["ok"]


@pytest.mark.asyncio
async def test_trained_mode_enforces_delta_from_last_valid_joint_dispatch(
    mock_server: str,
) -> None:
    """Per-step deltas advance only from successfully dispatched joint targets."""
    async with connect(mock_server) as ws:
        json.loads(await ws.recv())

        await ws.send(
            _cmd(
                "policy.start",
                {
                    "task": "bounded_joint_motion",
                    "hz": 10,
                    "max_steps": 100,
                },
            )
        )
        start_response = await _drain_until_response(ws, "test-policy.start")
        assert start_response["ok"]

        for target in (0.2, 0.49):
            await ws.send(
                _cmd(
                    "policy.tick",
                    {
                        "action": {
                            "joint_positions": {"r_hip_pitch": target},
                            "duration": 0.1,
                        }
                    },
                )
            )
            tick_response = await _drain_until_response(ws, "test-policy.tick")
            assert tick_response["ok"], tick_response["message"]

        await ws.send(
            _cmd(
                "policy.tick",
                {
                    "action": {
                        # 0.31 rad from the last successful 0.49 target.
                        "joint_positions": {"r_hip_pitch": 0.8},
                        "duration": 0.1,
                    }
                },
            )
        )
        rejected = await _drain_until_response(ws, "test-policy.tick")
        assert not rejected["ok"]
        assert "delta" in rejected["message"]
        assert "profile max 0.3" in rejected["message"]

        await ws.send(_cmd("policy.status", {}))
        status = await _drain_until_response(ws, "test-policy.status")
        assert not status["data"]["active"]


@pytest.mark.asyncio
async def test_manual_command_preempts_active_policy(mock_server: str) -> None:
    """Manual `walk.command:stop` while a policy is active must auto-stop
    the policy AND stop the walk — Eliza should never have to fight a
    learned policy to reclaim control.
    """
    async with connect(mock_server) as ws:
        json.loads(await ws.recv())  # session.hello

        await ws.send(
            _cmd(
                "policy.start",
                {"task": "wave_to_human", "hz": 10, "max_steps": 50},
            )
        )
        # Drain start response.
        for _ in range(30):
            frame = json.loads(await ws.recv())
            if (
                frame.get("type") == "response"
                and frame.get("request_id") == "test-policy.start"
            ):
                break

        # Manual preempt while policy is "running"
        await ws.send(_cmd("walk.command", {"action": "stop"}, preempt=True))

        # Eventually we should see policy.status: state=idle, reason=manual_preempt
        saw_preempt = False
        for _ in range(50):
            frame = json.loads(await ws.recv())
            if (
                frame.get("type") == "event"
                and frame.get("event") == "policy.status"
                and frame["data"].get("state") == "idle"
                and frame["data"].get("reason") == "manual_preempt"
            ):
                saw_preempt = True
                break
        assert saw_preempt, "manual command did not preempt active policy"
