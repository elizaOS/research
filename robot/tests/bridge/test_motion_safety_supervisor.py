"""Adversarial tests for the bridge's single guarded motion boundary."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import sys
import threading
import time
import types

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from eliza_robot.bridge.backends.base import (
    _physical_motion_authority_error,
    _supervised_motion_dispatch_authority,
)
from eliza_robot.bridge.backends.calibrated import CalibratedBackend, JointCalibration
from eliza_robot.bridge.backends.mock_backend import MockBackend
from eliza_robot.bridge.backends.noise_injector import NoiseInjectorBackend, NoiseProfile
from eliza_robot.bridge.protocol import (
    CommandEnvelope,
    EventEnvelope,
    ResponseEnvelope,
    utc_now_iso,
)
from eliza_robot.bridge.safety import MotionOwnershipRegistry, MotionSafetySupervisor
from eliza_robot.bridge.server import (
    PolicyLoopState,
    RuntimeConfig,
    SharedBackendRuntime,
    _cancel_policy_and_stop,
    _command_worker,
    _deadman_pump,
    _event_pump,
    _handle_policy_command,
    _handler,
    _request_is_authorized,
    _validate_bind_security,
    _validate_physical_auth_token,
    _validated_physical_resource,
)
from eliza_robot.bridge.validation import validate_command_payload
from eliza_robot.profiles.schema import RobotProfile, load_profile
from eliza_robot.rl.text_conditioned import inference_loop
from eliza_robot.rl.text_conditioned.inference_loop import InferenceLoopConfig


def _command(command: str, payload: dict, request_id: str = "test") -> CommandEnvelope:
    return CommandEnvelope(
        request_id=request_id,
        timestamp=utc_now_iso(),
        command=command,
        payload=payload,
    )


def _safe_capabilities() -> dict:
    return {
        "servo_set": True,
        "walk_set": True,
        "walk_command": True,
        "head_set": True,
        "action_play": True,
        "motion_safety": {
            "imu_roll": True,
            "imu_pitch": True,
            "battery_mv": True,
            "environment": "nonphysical",
            "torque_limit_status": "not_applicable",
            "all_motion_stop": True,
            "stop_out_of_band": True,
            "walk_stop": True,
            "known_joint_pose_at_connect": True,
            "pose_remains_trusted_after_stop": True,
            "hard_envelope_complete": False,
        },
    }


class _Backend:
    backend_name = "adversarial"

    def __init__(
        self,
        *,
        capabilities: dict | None = None,
        reject_motion: bool = False,
        failed_stops: int = 0,
    ) -> None:
        self._capabilities = capabilities if capabilities is not None else _safe_capabilities()
        self.reject_motion = reject_motion
        self.failed_stops = failed_stops
        self.stop_attempts = 0
        self.commands: list[CommandEnvelope] = []
        self.stop_succeeded = asyncio.Event()
        self.connect_calls = 0
        self.shutdown_calls = 0
        self.poll_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def capabilities(self) -> dict:
        return dict(self._capabilities)

    def physical_motion_resources(self) -> tuple[str, ...]:
        return ()

    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        self.commands.append(command)
        is_stop = command.command == "walk.command" and command.payload.get("action") == "stop"
        if is_stop:
            self.stop_attempts += 1
            ok = self.stop_attempts > self.failed_stops
            if ok:
                self.stop_succeeded.set()
        else:
            ok = not self.reject_motion
        return ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=ok,
            backend=self.backend_name,
            message="ok" if ok else "injected rejection",
            data={},
        )

    async def handle_emergency_stop(
        self,
        command: CommandEnvelope,
    ) -> ResponseEnvelope:
        return await self.handle_command(command)

    async def poll_events(self) -> list[EventEnvelope]:
        self.poll_calls += 1
        return []


class _Socket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, raw: str) -> None:
        self.messages.append(json.loads(raw))


async def _recv_response(ws: object, request_id: str) -> dict:
    async def _receive() -> dict:
        while True:
            raw = await ws.recv()  # type: ignore[attr-defined]
            message = json.loads(raw)
            if (
                isinstance(message, dict)
                and message.get("type") == "response"
                and message.get("request_id") == request_id
            ):
                return message

    return await asyncio.wait_for(_receive(), timeout=2.0)


class _DelayedBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.motion_started = asyncio.Event()
        self.cancel_motion = asyncio.Event()
        self.stop_called_at = 0.0

    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        is_stop = command.command == "walk.command" and command.payload.get("action") == "stop"
        if command.command == "head.set":
            self.commands.append(command)
            self.motion_started.set()
            await asyncio.wait_for(self.cancel_motion.wait(), timeout=1.0)
            return ResponseEnvelope(
                command.request_id,
                utc_now_iso(),
                True,
                self.backend_name,
                "cancelled",
                {},
            )
        if is_stop:
            self.stop_called_at = time.monotonic()
            self.cancel_motion.set()
        return await super().handle_command(command)


class _SlowUnwindBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.motion_started = asyncio.Event()
        self.release_motion = asyncio.Event()

    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        if command.command == "head.set":
            self.commands.append(command)
            self.motion_started.set()
            await self.release_motion.wait()
            return ResponseEnvelope(
                command.request_id,
                utc_now_iso(),
                True,
                self.backend_name,
                "late success",
                {},
            )
        return await super().handle_command(command)

    async def handle_emergency_stop(
        self,
        command: CommandEnvelope,
    ) -> ResponseEnvelope:
        # Acknowledge independently while the ordinary call is still unwinding.
        return await _Backend.handle_command(self, command)


class _PostDispatchStopFailureBackend(_SlowUnwindBackend):
    """First stop succeeds, forced late-dispatch stop fails, retry succeeds."""

    def __init__(self) -> None:
        super().__init__()
        self.retry_started = asyncio.Event()
        self.release_retry = asyncio.Event()

    async def handle_emergency_stop(
        self,
        command: CommandEnvelope,
    ) -> ResponseEnvelope:
        self.commands.append(command)
        self.stop_attempts += 1
        if self.stop_attempts == 3:
            self.retry_started.set()
            await self.release_retry.wait()
        ok = self.stop_attempts in {1, 3}
        return ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=ok,
            backend=self.backend_name,
            message="ok" if ok else "injected forced-stop failure",
            data={},
        )


class _PhysicalBackend(_Backend):
    def physical_motion_resources(self) -> tuple[str, ...]:
        return ("physical:test-robot",)


class _MultiplePhysicalBackend(_Backend):
    def physical_motion_resources(self) -> tuple[str, ...]:
        return ("physical:test-robot", "physical:second-robot")


class _BlockingStopBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.stop_started = asyncio.Event()
        self.release_stop = asyncio.Event()

    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        is_stop = command.command == "walk.command" and command.payload.get("action") == "stop"
        if is_stop:
            self.stop_started.set()
            await self.release_stop.wait()
        return await super().handle_command(command)


class _DelayedServoBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.servo_started = asyncio.Event()
        self.release_servo = asyncio.Event()

    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        if command.command == "servo.set":
            self.commands.append(command)
            self.servo_started.set()
            await self.release_servo.wait()
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=True,
                backend=self.backend_name,
                message="ok",
                data={},
            )
        return await super().handle_command(command)


class _MalformedAckBackend(_Backend):
    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        is_stop = command.command == "walk.command" and command.payload.get("action") == "stop"
        if is_stop:
            return await super().handle_command(command)
        self.commands.append(command)
        return ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=float("nan"),  # type: ignore[arg-type]
            backend=self.backend_name,
            message="malformed acknowledgement",
            data={},
        )


class _PollingFailureBackend(_Backend):
    async def poll_events(self) -> list[EventEnvelope]:
        raise RuntimeError("injected telemetry transport failure")


class _MalformedTelemetryBackend(_Backend):
    async def poll_events(self) -> list[EventEnvelope]:
        return [object()]  # type: ignore[list-item]


class _AuthSocket:
    def __init__(self, authorization: str | None) -> None:
        headers = {} if authorization is None else {"Authorization": authorization}
        self.request = types.SimpleNamespace(headers=headers)


def _supervisor(
    backend: _Backend,
    registry: MotionOwnershipRegistry | None = None,
    *,
    owner: str = "owner-a",
    resource: str = "robot-under-test",
) -> MotionSafetySupervisor:
    return MotionSafetySupervisor(
        backend,  # type: ignore[arg-type]
        load_profile("hiwonder-ainex"),
        owner_id=owner,
        resource_id=resource,
        registry=registry or MotionOwnershipRegistry(),
    )


@pytest.mark.parametrize(
    "value",
    [True, False, float("inf"), float("nan"), 10**1000],
)
def test_all_motion_numbers_reject_bool_and_nonfinite(value: object) -> None:
    with pytest.raises(ValueError):
        validate_command_payload(
            _command(
                "head.set",
                {"pan": value, "tilt": 0.0, "duration": 0.1},
            )
        )


def test_nested_motion_numbers_reject_huge_values_without_overflow() -> None:
    from eliza_robot.asimov_1.constants import ASIMOV1_FIRMWARE_JOINT_ORDER

    huge = 10**1000
    with pytest.raises(ValueError, match="finite"):
        validate_command_payload(
            _command(
                "servo.set",
                {"duration": 0.1, "joint_positions": {"r_hip_pitch": huge}},
            )
        )
    with pytest.raises(ValueError, match="finite"):
        validate_command_payload(
            _command(
                "asimov.trajectory",
                {"joint_positions": {ASIMOV1_FIRMWARE_JOINT_ORDER[0]: huge}},
            )
        )
    with pytest.raises(ValueError, match="finite"):
        validate_command_payload(
            _command(
                "asimov.trajectory",
                {
                    "positions": [0.0] * len(ASIMOV1_FIRMWARE_JOINT_ORDER),
                    "kp": [huge] * len(ASIMOV1_FIRMWARE_JOINT_ORDER),
                },
            )
        )


@pytest.mark.parametrize("speed", [True, False, 1.9, 4.1, 10**100])
def test_walk_speed_requires_exact_bounded_integer(speed: object) -> None:
    with pytest.raises(ValueError):
        validate_command_payload(
            _command(
                "walk.set",
                {"speed": speed, "height": 0.036, "x": 0.0, "y": 0.0, "yaw": 0.0},
            )
        )


@pytest.mark.parametrize("max_steps", [True, False, 1.5, 10**100])
def test_policy_max_steps_requires_exact_bounded_integer(max_steps: object) -> None:
    with pytest.raises(ValueError):
        validate_command_payload(_command("policy.start", {"task": "walk", "max_steps": max_steps}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deadman_timeout_sec", float("inf")),
        ("deadman_timeout_sec", 0.0),
        ("mujoco_target_xyz", (0.0, float("nan"), 0.0)),
        ("queue_size", True),
        ("camera_width", 100_000),
        ("rosbridge_port", 70_000),
    ],
)
def test_runtime_config_rejects_nonfinite_and_unbounded_values(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "queue_size": 8,
        "max_commands_per_sec": 30,
        "deadman_timeout_sec": 1.0,
        "trace_log_path": "",
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        RuntimeConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [
        ("kinematics", "joints", 0, "lower_rad"),
        ("kinematics", "joints", 0, "actuator_torque_nm"),
        ("gait", "cycle_hz"),
        ("sensors", "imu_noise_std"),
        ("sensors", "cameras", 0, "extrinsics_rpy_xyz"),
        ("control", "rate_hz"),
        ("actions", "groups", "wave", "duration_s"),
        ("actions", "groups", "wave", "frames", 0, "t"),
        ("safety", "fall_pitch_rad"),
        ("safety", "deadman_timeout_s"),
    ],
)
def test_every_profile_float_group_rejects_nonfinite_values(
    path: tuple[str | int, ...],
) -> None:
    raw = copy.deepcopy(load_profile("hiwonder-ainex").model_dump())
    cursor: object = raw
    for component in path[:-1]:
        cursor = cursor[component]  # type: ignore[index]
    final = path[-1]
    if final == "extrinsics_rpy_xyz":
        cursor[final] = (0.0, 0.0, float("inf"), 0.0, 0.0, 0.0)  # type: ignore[index]
    else:
        cursor[final] = float("inf")  # type: ignore[index]
    with pytest.raises(ValueError):
        RobotProfile.model_validate(raw)


def test_physical_bind_requires_loopback_and_authentication() -> None:
    with pytest.raises(ValueError, match="must bind to loopback"):
        _validate_bind_security("0.0.0.0", "ros_real", "")
    with pytest.raises(ValueError, match="must bind to loopback"):
        _validate_bind_security("0.0.0.0", "ros_real", "too-short")
    with pytest.raises(ValueError, match="32..4096 visible ASCII"):
        _validate_bind_security("127.0.0.1", "ros_real", "")
    with pytest.raises(ValueError, match="ELIZA_ROBOT_PHYSICAL_RESOURCE_ID"):
        _validate_bind_security("127.0.0.1", "ros_real", "s" * 32)
    _validate_bind_security(
        "127.0.0.1",
        "ros_real",
        "s" * 32,
        "lab-ainex-01",
    )
    _validate_bind_security("0.0.0.0", "mock", "")


@pytest.mark.parametrize(
    "token",
    ["s" * 31, "s" * 4097, "s" * 31 + " ", "s" * 31 + "\n", "s" * 31 + "é"],
)
def test_physical_bearer_secret_must_be_header_safe(token: str) -> None:
    with pytest.raises(ValueError, match="32..4096 visible ASCII"):
        _validate_physical_auth_token(token)


def test_server_physical_identity_must_exactly_match_backend() -> None:
    backend = _PhysicalBackend()
    valid = RuntimeConfig(
        queue_size=4,
        max_commands_per_sec=10,
        deadman_timeout_sec=1.0,
        trace_log_path="",
        physical_resource_id="test-robot",
    )
    assert _validated_physical_resource(backend, valid) == "physical:test-robot"  # type: ignore[arg-type]

    mismatched = RuntimeConfig(
        queue_size=4,
        max_commands_per_sec=10,
        deadman_timeout_sec=1.0,
        trace_log_path="",
        physical_resource_id="different-robot",
    )
    with pytest.raises(ValueError, match="does not match"):
        _validated_physical_resource(backend, mismatched)  # type: ignore[arg-type]

    nonphysical = RuntimeConfig(4, 10, 1.0, "")
    assert _validated_physical_resource(_Backend(), nonphysical) is None  # type: ignore[arg-type]


def test_bearer_authentication_is_exact_and_fail_closed() -> None:
    assert _request_is_authorized(_AuthSocket("Bearer secret"), "secret")  # type: ignore[arg-type]
    assert not _request_is_authorized(_AuthSocket("Bearer secret "), "secret")  # type: ignore[arg-type]
    assert not _request_is_authorized(_AuthSocket(None), "secret")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_embedded_physical_handler_rejects_empty_auth_before_connect() -> None:
    backend = _PhysicalBackend()
    config = RuntimeConfig(
        queue_size=4,
        max_commands_per_sec=10,
        deadman_timeout_sec=1.0,
        trace_log_path="",
        physical_resource_id="test-robot",
    )

    with pytest.raises(ValueError, match="ELIZA_ROBOT_BRIDGE_AUTH_TOKEN"):
        await _handler(  # type: ignore[arg-type]
            _AuthSocket(None),
            lambda: backend,  # type: ignore[arg-type,return-value]
            config,
        )

    assert backend.connect_calls == 0


@pytest.mark.asyncio
async def test_websocket_authentication_rejects_missing_token_before_hello() -> None:
    backend = _Backend()
    runtime = SharedBackendRuntime(lambda: backend, poll_hz=100.0)
    await runtime.start()
    token = "a" * 32
    config = RuntimeConfig(
        queue_size=4,
        max_commands_per_sec=10,
        deadman_timeout_sec=5.0,
        trace_log_path="",
        auth_token=token,
    )
    server = await serve(
        lambda ws: _handler(ws, runtime, config),
        "127.0.0.1",
        0,
    )
    sockets = server.sockets
    assert sockets
    port = int(sockets[0].getsockname()[1])
    try:
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            rejected = json.loads(await ws.recv())
            assert rejected["ok"] is False
            assert "authentication required" in rejected["message"]
        async with connect(
            f"ws://127.0.0.1:{port}",
            additional_headers={"Authorization": f"Bearer {token}"},
        ) as ws:
            hello = json.loads(await ws.recv())
            assert hello["event"] == "session.hello"
    finally:
        server.close()
        await server.wait_closed()
        await runtime.close()

    assert backend.commands == []


@pytest.mark.asyncio
async def test_supervisor_revalidates_motion_payload_before_ownership() -> None:
    backend = _Backend()
    supervisor = _supervisor(backend, resource="central-validation")

    response = await supervisor.guarded_dispatch(
        _command("head.set", {"pan": True, "tilt": 0.0, "duration": 0.1})
    )

    assert not response.ok
    assert "payload validation failed" in response.message
    assert backend.commands == []
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_invalid_motion_from_current_owner_immediately_stops() -> None:
    backend = _Backend()
    supervisor = _supervisor(backend, resource="owned-invalid-motion")
    assert (
        await supervisor.guarded_dispatch(
            _command(
                "walk.set",
                {
                    "speed": 2,
                    "height": 0.036,
                    "x": 0.01,
                    "y": 0.0,
                    "yaw": 0.0,
                },
            )
        )
    ).ok

    rejected = await supervisor.guarded_dispatch(
        _command("head.set", {"pan": float("inf"), "tilt": 0.0, "duration": 0.1})
    )

    assert not rejected.ok
    assert "emergency_stop_ok=True" in rejected.message
    assert [command.command for command in backend.commands] == [
        "walk.set",
        "walk.command",
    ]
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_asimov_stand_is_guarded_motion_and_damp_is_a_stop() -> None:
    backend = _Backend(
        capabilities={
            "commands": ["asimov.mode"],
        }
    )
    supervisor = MotionSafetySupervisor(
        backend,  # type: ignore[arg-type]
        load_profile("asimov-1"),
        owner_id="asimov-owner",
        resource_id="asimov-mode",
        registry=MotionOwnershipRegistry(),
    )

    stand = await supervisor.guarded_dispatch(_command("asimov.mode", {"mode": "STAND"}))
    assert not stand.ok
    assert "out-of-band cancellation" in stand.message
    assert backend.commands == []

    damp = await supervisor.guarded_dispatch(_command("asimov.mode", {"mode": "DAMP"}))
    assert damp.ok
    assert len(backend.commands) == 1
    assert backend.commands[0].command == "walk.command"
    assert backend.commands[0].payload == {"action": "stop"}


def test_servo_mixed_payload_is_rejected_before_dispatch() -> None:
    command = _command(
        "servo.set",
        {
            "duration": 0.1,
            "joint_positions": {"r_hip_pitch": 0.1},
            "positions": [{"id": 8, "position": 1000}],
        },
    )
    with pytest.raises(ValueError, match="exactly one"):
        validate_command_payload(command)

    with pytest.raises(ValueError, match="exactly one"):
        validate_command_payload(
            _command(
                "asimov.trajectory",
                {"positions": [], "joint_positions": {}},
            )
        )
    with pytest.raises(ValueError, match="must not be empty"):
        validate_command_payload(_command("asimov.trajectory", {"joint_positions": {}}))


@pytest.mark.asyncio
async def test_manual_servo_enforces_profile_limit_and_delta_without_side_effect() -> None:
    backend = _Backend()
    supervisor = _supervisor(backend)

    response = await supervisor.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.31}},
        ),
        require_servo_capability=True,
    )

    assert not response.ok
    assert "delta" in response.message
    assert backend.commands == []
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_effective_pulse_validation_cannot_fall_back_to_named_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eliza_robot.bridge.isaaclab import joint_map

    backend = _Backend()
    supervisor = _supervisor(backend, resource="effective-pulse")
    monkeypatch.setattr(joint_map, "pulse_to_radians", lambda _pulse, _servo_id: 1.0)

    response = await supervisor.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.1}},
        )
    )

    assert not response.ok
    assert "effective pulse target rejected" in response.message
    assert backend.commands == []


@pytest.mark.asyncio
async def test_servo_revalidates_fresh_pose_after_waiting_for_command_lock() -> None:
    capabilities = _safe_capabilities()
    capabilities["motion_safety"]["joint_positions"] = True
    backend = _Backend(capabilities=capabilities)
    registry = MotionOwnershipRegistry()
    supervisor = _supervisor(backend, registry, resource="pose-race")
    command_lock = registry.command_lock("pose-race")
    await command_lock.acquire()
    motion_task = asyncio.create_task(
        supervisor.guarded_dispatch(
            _command(
                "servo.set",
                {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.2}},
            )
        )
    )
    try:
        await asyncio.sleep(0)
        observed = {
            joint.name: float(joint.home_rad)
            for joint in load_profile("hiwonder-ainex").kinematics.joints
        }
        observed["r_hip_pitch"] = -0.2
        assert (
            supervisor.telemetry_violation(
                EventEnvelope(
                    event="telemetry.basic",
                    timestamp=utc_now_iso(),
                    backend="test",
                    data={
                        "imu_roll": 0.0,
                        "imu_pitch": 0.0,
                        "battery_mv": 12000,
                        "joint_positions": observed,
                    },
                )
            )
            is None
        )
    finally:
        command_lock.release()

    response = await motion_task
    assert not response.ok
    assert "changed before dispatch" in response.message
    assert [command.command for command in backend.commands] == ["walk.command"]


@pytest.mark.asyncio
async def test_in_flight_pose_telemetry_does_not_certify_post_command_pose() -> None:
    capabilities = _safe_capabilities()
    capabilities["motion_safety"]["joint_positions"] = True
    backend = _DelayedServoBackend()
    backend._capabilities = capabilities
    supervisor = _supervisor(backend, resource="pose-response-race")
    motion_task = asyncio.create_task(
        supervisor.guarded_dispatch(
            _command(
                "servo.set",
                {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.2}},
            )
        )
    )
    try:
        await asyncio.wait_for(backend.servo_started.wait(), timeout=0.2)
        observed = {
            joint.name: float(joint.home_rad)
            for joint in load_profile("hiwonder-ainex").kinematics.joints
        }
        observed["r_hip_pitch"] = -0.1
        assert (
            supervisor.telemetry_violation(
                EventEnvelope(
                    event="telemetry.basic",
                    timestamp=utc_now_iso(),
                    backend="test",
                    data={
                        "imu_roll": 0.0,
                        "imu_pitch": 0.0,
                        "battery_mv": 12000,
                        "joint_positions": observed,
                    },
                )
            )
            is None
        )
    finally:
        backend.release_servo.set()

    response = await motion_task
    assert response.ok
    assert supervisor.last_joint_positions == {}


@pytest.mark.asyncio
async def test_profile_and_backend_capability_mismatch_have_zero_side_effects() -> None:
    profile_backend = _Backend()
    profile_supervisor = MotionSafetySupervisor(
        profile_backend,  # type: ignore[arg-type]
        load_profile("unitree-h1"),
        owner_id="profile-mismatch",
    )
    profile_response = await profile_supervisor.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"left_hip_yaw_joint": 0.0}},
        ),
        require_servo_capability=True,
    )
    assert not profile_response.ok
    assert "does not permit" in profile_response.message
    assert profile_backend.commands == []
    assert not profile_supervisor.owns_motion

    capabilities = _safe_capabilities()
    capabilities.pop("head_set")
    backend = _Backend(capabilities=capabilities)
    backend_supervisor = _supervisor(backend, resource="backend-mismatch")
    backend_response = await backend_supervisor.guarded_dispatch(
        _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 0.1})
    )
    assert not backend_response.ok
    assert "does not advertise" in backend_response.message
    assert backend.commands == []
    assert not backend_supervisor.owns_motion


@pytest.mark.asyncio
async def test_backend_not_ok_causes_stop_and_never_advances_joint_state() -> None:
    backend = _Backend(reject_motion=True)
    supervisor = _supervisor(backend)

    response = await supervisor.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.2}},
        ),
        require_servo_capability=True,
    )

    assert not response.ok
    assert [command.command for command in backend.commands] == [
        "servo.set",
        "walk.command",
    ]
    assert supervisor.last_joint_positions["r_hip_pitch"] == 0.0
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_malformed_backend_acknowledgement_fails_closed() -> None:
    backend = _MalformedAckBackend()
    supervisor = _supervisor(backend, resource="malformed-ack")

    response = await supervisor.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.2}},
        )
    )

    assert not response.ok
    assert "exact boolean" in response.message
    assert [command.command for command in backend.commands] == [
        "servo.set",
        "walk.command",
    ]
    assert supervisor.last_joint_positions["r_hip_pitch"] == 0.0
    assert not supervisor.owns_motion


@pytest.mark.parametrize("service_result", [False, 1, None])
def test_remote_rosbridge_walk_service_requires_exact_true_ack(
    monkeypatch: pytest.MonkeyPatch,
    service_result: object,
) -> None:
    from eliza_robot.bridge.backends.ainex_remote import AinexRemoteBackend

    roslibpy = types.ModuleType("roslibpy")
    roslibpy.ServiceRequest = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roslibpy", roslibpy)

    class _Service:
        def call(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"result": service_result}

    backend = AinexRemoteBackend(physical_resource_id="ack-test")
    backend._walking_cmd_srv = _Service()  # noqa: SLF001
    with pytest.raises(RuntimeError, match="exact result=true"):
        backend._dispatch(_command("walk.command", {"action": "stop"}))  # noqa: SLF001


@pytest.mark.parametrize("service_result", [False, 1, None])
def test_native_ros_walk_service_requires_exact_true_ack(
    monkeypatch: pytest.MonkeyPatch,
    service_result: object,
) -> None:
    from eliza_robot.bridge.backends.ros_backend import RosBridgeBackend

    ainex = types.ModuleType("ainex_interfaces")
    ainex_msg = types.ModuleType("ainex_interfaces.msg")
    ainex_msg.AppWalkingParam = object  # type: ignore[attr-defined]
    ainex_msg.HeadState = object  # type: ignore[attr-defined]
    std = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.String = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ainex_interfaces", ainex)
    monkeypatch.setitem(sys.modules, "ainex_interfaces.msg", ainex_msg)
    monkeypatch.setitem(sys.modules, "std_msgs", std)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msg)

    backend = RosBridgeBackend("ros_real", physical_resource_id="ack-test")
    backend._walking_command_srv = (  # noqa: SLF001
        lambda _action: types.SimpleNamespace(result=service_result)
    )
    with pytest.raises(RuntimeError, match="exact result=true"):
        backend._dispatch_blocking(  # noqa: SLF001
            _command("walk.command", {"action": "stop"})
        )


@pytest.mark.asyncio
async def test_worker_drops_motion_queued_before_backend_failure() -> None:
    backend = _Backend(reject_motion=True)
    supervisor = _supervisor(backend, resource="failed-queue")
    socket = _Socket()
    queue: asyncio.Queue[CommandEnvelope] = asyncio.Queue()
    queue.put_nowait(
        _command(
            "walk.set",
            {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0},
            request_id="first",
        )
    )
    queue.put_nowait(
        _command(
            "walk.set",
            {"speed": 2, "height": 0.036, "x": 0.02, "y": 0.0, "yaw": 0.0},
            request_id="stale-second",
        )
    )
    worker = asyncio.create_task(
        _command_worker(
            socket,  # type: ignore[arg-type]
            supervisor,
            queue,
            trace_logger=None,
        )
    )
    try:
        await asyncio.wait_for(queue.join(), timeout=0.5)
    finally:
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

    assert [command.request_id for command in backend.commands] == [
        "first",
        "first-backend-rejected-stop",
    ]
    assert [message["request_id"] for message in socket.messages] == ["first"]


@pytest.mark.asyncio
async def test_motion_ownership_is_exclusive_and_stop_hands_it_off() -> None:
    registry = MotionOwnershipRegistry()
    first_backend = _Backend()
    second_backend = _Backend()
    first = _supervisor(first_backend, registry, owner="first")
    second = _supervisor(second_backend, registry, owner="second")

    first_response = await first.guarded_dispatch(
        _command("walk.set", {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0})
    )
    blocked = await second.guarded_dispatch(
        _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 0.1})
    )

    assert first_response.ok
    assert not blocked.ok
    assert second_backend.commands == []
    assert (await first.emergency_stop_once("handoff")).ok
    assert (
        await second.guarded_dispatch(
            _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 0.1})
        )
    ).ok


@pytest.mark.asyncio
async def test_shared_runtime_routes_cross_session_stop_to_owner_backend() -> None:
    backend = _Backend()
    factory_calls = 0

    def _factory() -> _Backend:
        nonlocal factory_calls
        factory_calls += 1
        return backend

    runtime = SharedBackendRuntime(_factory, poll_hz=100.0)
    await runtime.start()
    registry = MotionOwnershipRegistry()
    owner = _supervisor(
        runtime.backend,  # type: ignore[arg-type]
        registry,
        owner="actual-owner",
        resource="shared-runtime-owner",
    )
    stopper = _supervisor(
        runtime.backend,  # type: ignore[arg-type]
        registry,
        owner="other-session",
        resource="shared-runtime-owner",
    )
    try:
        assert (
            await owner.guarded_dispatch(
                _command(
                    "walk.set",
                    {
                        "speed": 2,
                        "height": 0.036,
                        "x": 0.01,
                        "y": 0.0,
                        "yaw": 0.0,
                    },
                )
            )
        ).ok
        assert (await stopper.emergency_stop_once("cross-session")).ok
    finally:
        await runtime.close()

    assert factory_calls == 1
    assert backend.connect_calls == 1
    assert backend.shutdown_calls == 1
    assert [command.command for command in backend.commands] == [
        "walk.set",
        "walk.command",
    ]
    assert owner.motion_revoked
    assert not owner.owns_motion


@pytest.mark.asyncio
async def test_stop_generation_rejects_unqueued_pre_stop_intent_atomically() -> None:
    registry = MotionOwnershipRegistry()
    backend = _Backend()
    owner = _supervisor(
        backend,
        registry,
        owner="generation-owner",
        resource="generation-race",
    )
    stopper = _supervisor(
        backend,
        registry,
        owner="generation-stopper",
        resource="generation-race",
    )
    assert owner.acquire_motion()
    generation_when_message_arrived = owner.motion_generation

    assert (await stopper.emergency_stop_once("generation-race")).ok
    assert not owner.accept_fresh_motion(generation_when_message_arrived)
    assert owner.motion_revoked

    # A genuinely new post-stop message can rearm, but an external stop has
    # invalidated the old session's connect-time pose belief.
    assert owner.accept_fresh_motion(owner.motion_generation)
    assert owner.last_joint_positions == {}


@pytest.mark.asyncio
async def test_shared_runtime_has_one_telemetry_consumer_and_purges_all_queues() -> None:
    backend = _Backend()
    runtime = SharedBackendRuntime(lambda: backend, poll_hz=100.0)
    first_events = runtime.subscribe()
    second_events = runtime.subscribe()
    first_commands: asyncio.Queue[CommandEnvelope] = asyncio.Queue()
    second_commands: asyncio.Queue[CommandEnvelope] = asyncio.Queue()
    runtime.register_command_queue(first_commands)
    runtime.register_command_queue(second_commands)
    first_commands.put_nowait(_command("head.set", {}))
    second_commands.put_nowait(_command("head.set", {}))
    await runtime.start()
    try:
        first_delivery, second_delivery = await asyncio.gather(
            asyncio.wait_for(first_events.get(), timeout=0.5),
            asyncio.wait_for(second_events.get(), timeout=0.5),
        )
        polls_before_read = backend.poll_calls
        _ = await runtime.latest_events()
        assert backend.poll_calls == polls_before_read
        assert first_delivery.error == second_delivery.error == ""
        assert runtime.purge_command_queues() == 2
        assert first_commands.empty()
        assert second_commands.empty()
    finally:
        first_events.task_done()
        second_events.task_done()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start_command", "start_payload", "stop_command", "stop_payload"),
    [
        ("walk.command", {"action": "start"}, "walk.command", {"action": "stop"}),
        ("policy.start", {"task": "test"}, "policy.stop", {"reason": "test"}),
    ],
)
async def test_stop_commands_bypass_saturated_rate_limiter_and_queue(
    start_command: str,
    start_payload: dict,
    stop_command: str,
    stop_payload: dict,
) -> None:
    backend = _Backend()
    runtime = SharedBackendRuntime(lambda: backend, poll_hz=100.0)
    await runtime.start()
    config = RuntimeConfig(
        queue_size=1,
        max_commands_per_sec=1,
        deadman_timeout_sec=5.0,
        trace_log_path="",
    )
    server = await serve(
        lambda ws: _handler(ws, runtime, config),
        "127.0.0.1",
        0,
    )
    sockets = server.sockets
    assert sockets
    port = int(sockets[0].getsockname()[1])
    try:
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            _ = await ws.recv()
            await ws.send(json.dumps(_command(start_command, start_payload, "start").to_json()))
            assert (await _recv_response(ws, "start"))["ok"] is True

            await ws.send(json.dumps(_command(stop_command, stop_payload, "stop").to_json()))
            assert (await _recv_response(ws, "stop"))["ok"] is True
    finally:
        server.close()
        await server.wait_closed()
        await runtime.close()

    assert backend.commands[-1].payload == {"action": "stop"}


@pytest.mark.asyncio
async def test_rate_limited_motion_from_owner_emergency_stops() -> None:
    backend = _Backend()
    runtime = SharedBackendRuntime(lambda: backend, poll_hz=100.0)
    await runtime.start()
    config = RuntimeConfig(
        queue_size=1,
        max_commands_per_sec=1,
        deadman_timeout_sec=5.0,
        trace_log_path="",
    )
    server = await serve(
        lambda ws: _handler(ws, runtime, config),
        "127.0.0.1",
        0,
    )
    sockets = server.sockets
    assert sockets
    port = int(sockets[0].getsockname()[1])
    try:
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            _ = await ws.recv()
            await ws.send(
                json.dumps(
                    _command(
                        "walk.command",
                        {"action": "start"},
                        "start",
                    ).to_json()
                )
            )
            assert (await _recv_response(ws, "start"))["ok"] is True
            await ws.send(
                json.dumps(
                    _command(
                        "head.set",
                        {"pan": 0.1, "tilt": 0.0, "duration": 0.1},
                        "rejected-motion",
                    ).to_json()
                )
            )
            rejected = await _recv_response(ws, "rejected-motion")
            assert rejected["ok"] is False
            assert "emergency_stop_ok=True" in rejected["message"]
    finally:
        server.close()
        await server.wait_closed()
        await runtime.close()

    assert [command.command for command in backend.commands] == [
        "walk.command",
        "walk.command",
    ]
    assert backend.commands[-1].payload == {"action": "stop"}


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_shared_telemetry_failure_latches_and_stops_owned_motion(
    malformed: bool,
) -> None:
    backend = _MalformedTelemetryBackend() if malformed else _PollingFailureBackend()
    runtime = SharedBackendRuntime(lambda: backend, poll_hz=100.0)
    await runtime.start()
    config = RuntimeConfig(
        queue_size=4,
        max_commands_per_sec=10,
        deadman_timeout_sec=5.0,
        trace_log_path="",
    )
    server = await serve(
        lambda ws: _handler(ws, runtime, config),
        "127.0.0.1",
        0,
    )
    sockets = server.sockets
    assert sockets
    port = int(sockets[0].getsockname()[1])
    try:
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            _ = await ws.recv()
            await ws.send(
                json.dumps(
                    _command(
                        "walk.command",
                        {"action": "start"},
                        "telemetry-start",
                    ).to_json()
                )
            )
            assert (await _recv_response(ws, "telemetry-start"))["ok"] is True
            await asyncio.wait_for(backend.stop_succeeded.wait(), timeout=1.0)
    finally:
        server.close()
        await server.wait_closed()
        await runtime.close()

    assert backend.commands[-1].payload == {"action": "stop"}


@pytest.mark.asyncio
async def test_failed_and_external_stops_latch_process_wide() -> None:
    registry = MotionOwnershipRegistry()
    owner_backend = _Backend(failed_stops=1)
    owner = _supervisor(owner_backend, registry, owner="running-owner", resource="shared-stop")
    stopper = _supervisor(
        owner_backend,
        registry,
        owner="external-stopper",
        resource="shared-stop",
    )

    assert (
        await owner.guarded_dispatch(
            _command(
                "walk.set",
                {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0},
            )
        )
    ).ok
    owner_command_count = len(owner_backend.commands)

    failed_stop = await stopper.emergency_stop_once("external-failure")
    assert not failed_stop.ok
    assert owner.emergency_stop_pending
    assert len(owner_backend.commands) == owner_command_count + 1
    after_failed_stop_count = len(owner_backend.commands)
    blocked_pending = await owner.guarded_dispatch(
        _command(
            "walk.set",
            {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0},
        )
    )
    assert not blocked_pending.ok
    assert len(owner_backend.commands) == after_failed_stop_count

    assert (await stopper.emergency_stop_once("external-retry")).ok
    assert owner.motion_revoked
    blocked_revoked = await owner.guarded_dispatch(
        _command(
            "walk.set",
            {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0},
        )
    )
    assert not blocked_revoked.ok
    assert "another session stopped" in blocked_revoked.message
    assert len(owner_backend.commands) == after_failed_stop_count + 1

    owner.rearm_motion()
    assert not owner.motion_revoked
    assert owner.last_joint_positions == {}
    assert "trusted_current_joint_pose" in str(owner.policy_start_capability_error())
    assert (
        await owner.guarded_dispatch(
            _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 0.1})
        )
    ).ok


@pytest.mark.asyncio
async def test_deadman_retries_failed_stop_until_backend_acknowledges() -> None:
    backend = _Backend(failed_stops=2)
    supervisor = _supervisor(backend)
    assert (
        await supervisor.guarded_dispatch(
            _command("walk.set", {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0})
        )
    ).ok
    socket = _Socket()
    task = asyncio.create_task(
        _deadman_pump(
            socket,  # type: ignore[arg-type]
            supervisor,
            PolicyLoopState(),
            deadman_timeout_sec=0.01,
        )
    )
    try:
        await asyncio.wait_for(backend.stop_succeeded.wait(), timeout=1.0)
        for _ in range(20):
            if len(socket.messages) >= 3:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert backend.stop_attempts == 3
    notified_results = [message["data"]["response_ok"] for message in socket.messages]
    # The process-wide retry loop may consume an intermediate attempt without
    # coupling actuator safety to this session's notification channel.
    assert notified_results[0] is False
    assert notified_results[-1] is True
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_out_of_band_deadman_interrupts_delayed_motion_before_duration() -> None:
    backend = _DelayedBackend()
    supervisor = _supervisor(backend, resource="delayed-motion")
    socket = _Socket()
    started_at = time.monotonic()
    motion_task = asyncio.create_task(
        supervisor.guarded_dispatch(
            _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 1.0})
        )
    )
    await asyncio.wait_for(backend.motion_started.wait(), timeout=0.2)
    deadman_task = asyncio.create_task(
        _deadman_pump(
            socket,  # type: ignore[arg-type]
            supervisor,
            PolicyLoopState(),
            deadman_timeout_sec=0.01,
        )
    )
    try:
        response = await asyncio.wait_for(motion_task, timeout=0.5)
    finally:
        deadman_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await deadman_task

    assert backend.stop_called_at - started_at < 0.5
    assert not response.ok
    assert "cancelled while backend dispatch was in flight" in response.message


@pytest.mark.asyncio
async def test_confirmed_stop_cannot_rearm_while_old_backend_call_unwinds() -> None:
    backend = _SlowUnwindBackend()
    supervisor = _supervisor(backend, resource="slow-unwind")
    motion_task = asyncio.create_task(
        supervisor.guarded_dispatch(
            _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 1.0})
        )
    )
    await asyncio.wait_for(backend.motion_started.wait(), timeout=0.2)

    stop_response = await supervisor.emergency_stop_once("slow-unwind")
    assert stop_response.ok
    assert not supervisor.accept_fresh_motion(supervisor.motion_generation)

    backend.release_motion.set()
    motion_response = await asyncio.wait_for(motion_task, timeout=0.2)
    assert not motion_response.ok
    assert "cancelled while backend dispatch was in flight" in motion_response.message
    assert supervisor.accept_fresh_motion(supervisor.motion_generation)


@pytest.mark.asyncio
async def test_cancelled_dispatch_stops_immediately_and_again_after_io_settles() -> None:
    backend = _SlowUnwindBackend()
    supervisor = _supervisor(backend, resource="cancelled-thread-like-dispatch")
    motion_task = asyncio.create_task(
        supervisor.guarded_dispatch(
            _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 1.0})
        )
    )
    await asyncio.wait_for(backend.motion_started.wait(), timeout=0.2)

    motion_task.cancel()
    await asyncio.wait_for(backend.stop_succeeded.wait(), timeout=0.2)
    assert backend.stop_attempts == 1
    assert not motion_task.done()

    backend.release_motion.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(motion_task, timeout=0.5)
    assert backend.stop_attempts == 2
    assert not supervisor.emergency_stop_pending
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_failed_forced_post_dispatch_stop_retries_without_an_owner() -> None:
    backend = _PostDispatchStopFailureBackend()
    supervisor = _supervisor(backend, resource="ownerless-forced-stop-retry")
    motion_task = asyncio.create_task(
        supervisor.guarded_dispatch(
            _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 1.0})
        )
    )
    await asyncio.wait_for(backend.motion_started.wait(), timeout=0.2)

    first_stop = await supervisor.emergency_stop_once("first-stop")
    assert first_stop.ok
    assert not supervisor.owns_motion

    backend.release_motion.set()
    motion_response = await asyncio.wait_for(motion_task, timeout=0.5)
    assert not motion_response.ok
    assert backend.stop_attempts == 2
    assert supervisor.emergency_stop_pending
    assert not supervisor.owns_motion

    await asyncio.wait_for(backend.retry_started.wait(), timeout=0.5)
    backend.release_retry.set()
    for _ in range(50):
        if not supervisor.emergency_stop_pending:
            break
        await asyncio.sleep(0.01)
    assert backend.stop_attempts == 3
    assert not supervisor.emergency_stop_pending
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_explicit_physical_stop_always_reaches_the_actuator() -> None:
    backend = _PhysicalBackend()
    supervisor = _supervisor(backend, resource="physical:test-robot")
    stop = _command("walk.command", {"action": "stop"})

    assert (await supervisor.guarded_dispatch(stop)).ok
    assert (await supervisor.guarded_dispatch(stop)).ok
    assert backend.stop_attempts == 2


def test_physical_supervisor_rejects_forged_or_multiple_resource_identity() -> None:
    profile = load_profile("hiwonder-ainex")
    with pytest.raises(ValueError, match="exactly match"):
        MotionSafetySupervisor(
            _PhysicalBackend(),  # type: ignore[arg-type]
            profile,
            owner_id="forged-owner",
            resource_id="physical:forged-robot",
            registry=MotionOwnershipRegistry(),
        )
    with pytest.raises(ValueError, match="exactly one"):
        MotionSafetySupervisor(
            _MultiplePhysicalBackend(),  # type: ignore[arg-type]
            profile,
            owner_id="multi-owner",
            registry=MotionOwnershipRegistry(),
        )


def test_physical_authority_is_resource_bound_single_use_and_fail_closed() -> None:
    motion = _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 1.0})
    with _supervised_motion_dispatch_authority(motion, ("physical:expected",)):
        assert (
            _physical_motion_authority_error(
                "physical-test",
                "physical:wrong",
                motion,
            )
            is not None
        )
        assert (
            _physical_motion_authority_error(
                "physical-test",
                "physical:expected",
                motion,
            )
            is None
        )
        assert (
            _physical_motion_authority_error(
                "physical-test",
                "physical:expected",
                motion,
            )
            is not None
        )

    unknown = _command("future.motion.command", {})
    assert (
        _physical_motion_authority_error(
            "physical-test",
            "physical:expected",
            unknown,
        )
        is not None
    )
    assert (
        _physical_motion_authority_error(
            "physical-test",
            "physical:expected",
            _command("walk.command", {"action": "stop"}),
        )
        is None
    )


@pytest.mark.asyncio
async def test_physical_motion_stays_blocked_while_hard_envelope_is_incomplete() -> None:
    backend = _PhysicalBackend()
    supervisor = _supervisor(backend, resource="physical:test-robot")

    response = await supervisor.guarded_dispatch(
        _command(
            "walk.set",
            {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0},
        )
    )

    assert not response.ok
    assert "hard safety envelope is incomplete" in response.message
    assert backend.commands == []


def test_physical_policy_start_stays_blocked_while_hard_envelope_is_incomplete() -> None:
    backend = _PhysicalBackend()
    supervisor = _supervisor(backend, resource="physical:test-robot")

    error = supervisor.policy_start_capability_error()

    assert error is not None
    assert "physical hard safety envelope" in error
    assert "workspace" in error
    assert "self_collision" in error


@pytest.mark.asyncio
async def test_non_oob_motion_is_denied_and_fallback_stop_latches_before_lock() -> None:
    capabilities = {
        "walk_set": True,
        "walk_command": True,
        "motion_safety": {"walk_stop": True},
    }
    denied_backend = _Backend(capabilities=capabilities)
    denied_supervisor = _supervisor(
        denied_backend,
        resource="non-oob-motion-denied",
    )
    denied = await denied_supervisor.guarded_dispatch(
        _command(
            "walk.set",
            {
                "speed": 2,
                "height": 0.036,
                "x": 0.01,
                "y": 0.0,
                "yaw": 0.0,
            },
        )
    )
    assert not denied.ok
    assert "out-of-band cancellation" in denied.message
    assert denied_backend.commands == []

    registry = MotionOwnershipRegistry()
    backend = _Backend(capabilities=capabilities)
    owner = _supervisor(
        backend,
        registry,
        owner="slow-owner",
        resource="slow-non-oob",
    )
    contender = _supervisor(
        backend,
        registry,
        owner="contender",
        resource="slow-non-oob",
    )
    assert owner.acquire_motion()
    command_lock = registry.command_lock("slow-non-oob")
    await command_lock.acquire()
    stop_task = asyncio.create_task(owner.emergency_stop_once("slow-inflight"))
    await asyncio.sleep(0)
    assert registry.stop_in_progress("slow-non-oob")
    assert not contender.acquire_motion()

    command_lock.release()
    stop_response = await stop_task
    assert stop_response.ok
    assert backend.commands[-1].payload == {"action": "stop"}


@pytest.mark.asyncio
async def test_out_of_band_stop_does_not_wait_for_stubborn_policy_cancellation() -> None:
    backend = _Backend()
    supervisor = _supervisor(backend, resource="stubborn-policy")
    assert (
        await supervisor.guarded_dispatch(
            _command(
                "walk.set",
                {
                    "speed": 2,
                    "height": 0.036,
                    "x": 0.01,
                    "y": 0.0,
                    "yaw": 0.0,
                },
            )
        )
    ).ok
    release = asyncio.Event()

    async def _stubborn_policy() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()

    policy_task = asyncio.create_task(_stubborn_policy())
    await asyncio.sleep(0)
    state = PolicyLoopState(active=True, _loop_task=policy_task)
    stop_task = asyncio.create_task(_cancel_policy_and_stop(state, supervisor, "stubborn-policy"))
    await asyncio.wait_for(backend.stop_succeeded.wait(), timeout=0.1)
    assert not policy_task.done()
    response = await asyncio.wait_for(stop_task, timeout=0.5)
    assert response.ok
    assert not supervisor.owns_motion

    state.active = False
    blocked_restart = await _handle_policy_command(
        _Socket(),  # type: ignore[arg-type]
        backend,  # type: ignore[arg-type]
        _command("policy.start", {"task": "stale-restart"}),
        state,
        None,
        RuntimeConfig(8, 30, 1.0, ""),
        load_profile("hiwonder-ainex"),
        supervisor,
    )
    assert not blocked_restart.ok
    assert "cancellation has not completed" in blocked_restart.message
    assert supervisor.motion_revoked

    release.set()
    await asyncio.wait_for(policy_task, timeout=0.2)


@pytest.mark.asyncio
async def test_stop_in_progress_blocks_acquisition_and_cancellation_latches_failure() -> None:
    registry = MotionOwnershipRegistry()
    backend = _BlockingStopBackend()
    owner = _supervisor(backend, registry, owner="owner", resource="blocking-stop")
    contender = _supervisor(
        _Backend(),
        registry,
        owner="contender",
        resource="blocking-stop",
    )
    assert (
        await owner.guarded_dispatch(
            _command(
                "walk.set",
                {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0},
            )
        )
    ).ok

    stop_task = asyncio.create_task(owner.emergency_stop_once("blocking"))
    await asyncio.wait_for(backend.stop_started.wait(), timeout=0.2)
    blocked = await contender.guarded_dispatch(
        _command(
            "walk.set",
            {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0},
        )
    )
    assert not blocked.ok
    assert "stop is in progress" in blocked.message

    stop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop_task
    assert owner.emergency_stop_pending
    assert owner.owns_motion


@pytest.mark.asyncio
@pytest.mark.parametrize("poll_failure", [False, True])
async def test_event_pump_stops_on_missing_or_failed_safety_telemetry(
    poll_failure: bool,
) -> None:
    backend = _PollingFailureBackend() if poll_failure else _Backend()
    supervisor = _supervisor(
        backend,
        resource=f"telemetry-heartbeat-{poll_failure}",
    )
    assert (
        await supervisor.guarded_dispatch(
            _command(
                "walk.set",
                {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0},
            )
        )
    ).ok
    if not poll_failure:
        supervisor._ownership_started_monotonic -= 2.0
    socket = _Socket()
    task = asyncio.create_task(
        _event_pump(
            socket,  # type: ignore[arg-type]
            backend,  # type: ignore[arg-type]
            supervisor,
            PolicyLoopState(active=True),
            hz=100.0,
        )
    )
    try:
        await asyncio.wait_for(backend.stop_succeeded.wait(), timeout=0.5)
        for _ in range(20):
            if socket.messages:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    reasons = [message["data"]["reason"] for message in socket.messages]
    assert any("telemetry" in reason for reason in reasons)
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_event_pump_stops_on_malformed_telemetry_batch() -> None:
    backend = _MalformedTelemetryBackend()
    supervisor = _supervisor(backend, resource="malformed-telemetry")
    assert (
        await supervisor.guarded_dispatch(
            _command(
                "walk.set",
                {
                    "speed": 2,
                    "height": 0.036,
                    "x": 0.01,
                    "y": 0.0,
                    "yaw": 0.0,
                },
            )
        )
    ).ok
    socket = _Socket()
    task = asyncio.create_task(
        _event_pump(
            socket,  # type: ignore[arg-type]
            backend,  # type: ignore[arg-type]
            supervisor,
            PolicyLoopState(active=True),
            hz=100.0,
        )
    )
    try:
        await asyncio.wait_for(backend.stop_succeeded.wait(), timeout=0.5)
        for _ in range(20):
            if socket.messages:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert any("valid EventEnvelope" in message["data"]["reason"] for message in socket.messages)
    assert not supervisor.owns_motion


def test_capability_contract_fails_closed_and_mock_is_explicitly_nonphysical() -> None:
    missing = _supervisor(_Backend(capabilities={}), resource="missing")
    assert "does not declare" in str(missing.policy_start_capability_error())

    false_torque = _supervisor(
        _Backend(
            capabilities={
                "motion_safety": {
                    "imu_roll": True,
                    "imu_pitch": True,
                    "battery_mv": True,
                    "torque_limit_enforced": False,
                }
            }
        ),
        resource="false-torque",
    )
    assert "torque_limit_enforced" in str(false_torque.policy_start_capability_error())

    physical_caps = _safe_capabilities()
    physical_safety = physical_caps["motion_safety"]
    physical_safety["environment"] = "physical"
    physical_safety.pop("torque_limit_status")
    physical_safety["torque_limit_enforced"] = True
    missing_limit = _supervisor(
        _Backend(capabilities=physical_caps),
        resource="missing-torque-limit",
    )
    assert "torque_limit_nm" in str(missing_limit.policy_start_capability_error())
    physical_safety["torque_limit_nm"] = 1000.0
    excessive_limit = _supervisor(
        _Backend(capabilities=physical_caps),
        resource="excessive-torque-limit",
    )
    assert "torque_limit_nm<=" in str(excessive_limit.policy_start_capability_error())

    mock = MotionSafetySupervisor(
        MockBackend(),
        load_profile("hiwonder-ainex"),
        owner_id="mock",
    )
    assert mock.policy_start_capability_error() is None
    safety = MockBackend().capabilities()["motion_safety"]
    assert safety["environment"] == "nonphysical"
    assert safety["torque_limit_status"] == "not_applicable"
    assert "torque_limit_enforced" not in safety
    report = mock.capability_report()
    assert report["hard_envelope_complete"] is False
    assert report["promotion_ready"] is False


def test_out_of_band_capability_flag_without_endpoint_is_not_trusted() -> None:
    backend = _Backend()
    backend.handle_emergency_stop = None  # type: ignore[method-assign,assignment]
    supervisor = _supervisor(backend, resource="flag-without-endpoint")

    assert "stop_out_of_band" in str(supervisor.policy_start_capability_error())
    assert "out-of-band cancellation" in str(
        supervisor._interruptible_stop_error("head.set")  # noqa: SLF001
    )
    assert "out-of-band cancellation" in str(
        supervisor._interruptible_stop_error("walk.set")  # noqa: SLF001
    )


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ({"imu_pitch": 0.0, "battery_mv": 12000}, "imu_roll"),
        ({"imu_roll": float("nan"), "imu_pitch": 0.0, "battery_mv": 12000}, "non-finite"),
        ({"imu_roll": 1.0, "imu_pitch": 0.0, "battery_mv": 12000}, "fall roll"),
        ({"imu_roll": 0.0, "imu_pitch": 0.0, "battery_mv": 6500}, "battery"),
    ],
)
def test_trusted_telemetry_missing_nonfinite_and_limits_fail_closed(
    data: dict,
    reason: str,
) -> None:
    supervisor = _supervisor(_Backend(), resource=f"telemetry-{reason}")
    assert supervisor.acquire_motion()
    violation = supervisor.telemetry_violation(
        EventEnvelope(
            event="telemetry.basic",
            timestamp=utc_now_iso(),
            backend="test",
            data=data,
        )
    )
    assert reason in str(violation)


def test_invalid_attested_pose_clears_connect_pose_before_ownership() -> None:
    capabilities = _safe_capabilities()
    capabilities["motion_safety"]["joint_positions"] = True
    supervisor = _supervisor(
        _Backend(capabilities=capabilities),
        resource="invalid-preownership-pose",
    )
    assert supervisor.last_joint_positions

    violation = supervisor.telemetry_violation(
        EventEnvelope(
            event="telemetry.basic",
            timestamp=utc_now_iso(),
            backend="test",
            data={
                "imu_roll": 0.0,
                "imu_pitch": 0.0,
                "battery_mv": 12000,
                "joint_positions": {},
            },
        )
    )

    assert violation is None
    assert supervisor.last_joint_positions == {}
    assert "trusted_current_joint_pose" in str(supervisor.policy_start_capability_error())


@pytest.mark.asyncio
async def test_direct_joint_motion_requires_fresh_trusted_pose_not_registry_history() -> None:
    registry = MotionOwnershipRegistry()
    first = _supervisor(_Backend(), registry, owner="old-owner", resource="pose-resource")
    assert (
        await first.guarded_dispatch(
            _command(
                "servo.set",
                {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.2}},
            ),
            require_servo_capability=True,
        )
    ).ok
    assert (await first.emergency_stop_once("owner-left")).ok

    capabilities = _safe_capabilities()
    safety = capabilities["motion_safety"]
    del safety["known_joint_pose_at_connect"]
    safety["joint_positions"] = True
    backend = _Backend(capabilities=capabilities)
    second = _supervisor(
        backend,
        registry,
        owner="new-owner",
        resource="pose-resource",
    )
    blocked = await second.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.21}},
        ),
        require_servo_capability=True,
    )
    assert not blocked.ok
    assert "trusted current joint pose" in blocked.message
    assert backend.commands == []

    profile = load_profile("hiwonder-ainex")
    observed = {joint.name: float(joint.home_rad) for joint in profile.kinematics.joints}
    assert (
        second.telemetry_violation(
            EventEnvelope(
                event="telemetry.basic",
                timestamp=utc_now_iso(),
                backend=backend.backend_name,
                data={
                    "imu_roll": 0.0,
                    "imu_pitch": 0.0,
                    "battery_mv": 12000,
                    "joint_positions": observed,
                },
            )
        )
        is None
    )
    accepted = await second.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.21}},
        ),
        require_servo_capability=True,
    )
    assert accepted.ok


@pytest.mark.asyncio
async def test_other_session_motion_invalidates_an_older_connect_pose_claim() -> None:
    registry = MotionOwnershipRegistry()
    capabilities = _safe_capabilities()
    capabilities["motion_safety"]["joint_positions"] = True
    first_backend = _Backend(capabilities=capabilities)
    second_backend = _Backend(capabilities=capabilities)
    first = _supervisor(
        first_backend,
        registry,
        owner="first-generation",
        resource="pose-generation",
    )
    second = _supervisor(
        second_backend,
        registry,
        owner="second-generation",
        resource="pose-generation",
    )

    assert (
        await first.guarded_dispatch(
            _command(
                "servo.set",
                {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.2}},
            )
        )
    ).ok
    assert (await first.emergency_stop_once("generation-handoff")).ok

    stale = await second.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": -0.2}},
        )
    )
    assert not stale.ok
    assert "trusted current joint pose" in stale.message
    assert second_backend.commands == []

    observed = {
        joint.name: float(joint.home_rad)
        for joint in load_profile("hiwonder-ainex").kinematics.joints
    }
    observed["r_hip_pitch"] = 0.2
    assert (
        second.telemetry_violation(
            EventEnvelope(
                event="telemetry.basic",
                timestamp=utc_now_iso(),
                backend="test",
                data={
                    "imu_roll": 0.0,
                    "imu_pitch": 0.0,
                    "battery_mv": 12000,
                    "joint_positions": observed,
                },
            )
        )
        is None
    )
    refreshed = await second.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.0}},
        )
    )
    assert refreshed.ok


@pytest.mark.asyncio
async def test_connect_time_home_attestation_cannot_be_reused_after_motion() -> None:
    registry = MotionOwnershipRegistry()
    backend = _Backend()
    first = _supervisor(
        backend,
        registry,
        owner="first-home-claim",
        resource="single-use-home",
    )
    assert (
        await first.guarded_dispatch(
            _command(
                "servo.set",
                {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.2}},
            ),
            require_servo_capability=True,
        )
    ).ok
    assert (await first.emergency_stop_once("handoff")).ok

    second = _supervisor(
        backend,
        registry,
        owner="second-home-claim",
        resource="single-use-home",
    )
    blocked = await second.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.21}},
        ),
        require_servo_capability=True,
    )
    assert not blocked.ok
    assert "trusted current joint pose" in blocked.message


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_name", ["calibrated", "noise"])
async def test_motion_mutating_wrappers_invalidate_inherited_safety(
    wrapper_name: str,
) -> None:
    inner = _Backend()
    if wrapper_name == "calibrated":
        wrapper = CalibratedBackend(
            inner,  # type: ignore[arg-type]
            {
                "r_hip_pitch": JointCalibration(
                    name="r_hip_pitch",
                    strength=2.0,
                    offset=0.0,
                    rmse=0.0,
                )
            },
        )
    else:
        wrapper = NoiseInjectorBackend(
            inner,  # type: ignore[arg-type]
            profile=NoiseProfile(deterministic_only=True),
        )
        wrapper._motor_strengths[0] = 1.5

    # The wrapper would turn an admissible .25-rad request into .5/.375 rad,
    # beyond the profile's .3-rad per-step cap if called behind the guard.
    await wrapper.handle_command(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.25}},
        )
    )
    assert float(inner.commands[-1].payload["joint_positions"]["r_hip_pitch"]) > 0.3
    inner.commands.clear()

    supervisor = MotionSafetySupervisor(
        wrapper,
        load_profile("hiwonder-ainex"),
        owner_id=f"{wrapper_name}-owner",
    )
    response = await supervisor.guarded_dispatch(
        _command(
            "servo.set",
            {"duration": 0.1, "joint_positions": {"r_hip_pitch": 0.25}},
        ),
        require_servo_capability=True,
    )
    assert not response.ok
    assert "out-of-band cancellation" in response.message
    assert inner.commands == []
    assert "motion_safety" not in wrapper.capabilities()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        InferenceLoopConfig(hz=0.0),
        InferenceLoopConfig(hz=float("nan")),
        InferenceLoopConfig(hz=1_000_000.0),
        InferenceLoopConfig(max_steps=True),
        InferenceLoopConfig(max_steps=1.5),  # type: ignore[arg-type]
        InferenceLoopConfig(action_scale=float("inf")),
        InferenceLoopConfig(action_scale=100.0),
        InferenceLoopConfig(safety_clip_rad=0.0),
        InferenceLoopConfig(safety_clip_rad=100.0),
    ],
)
async def test_invalid_direct_inference_config_never_acquires_motion(
    config: InferenceLoopConfig,
) -> None:
    backend = _Backend()
    supervisor = _supervisor(backend, resource=f"bad-config-{id(config)}")
    with pytest.raises(ValueError):
        await inference_loop.run_inference(
            backend,  # type: ignore[arg-type]
            "unused-checkpoint",
            "walk",
            config=config,
            supervisor=supervisor,
        )
    assert not supervisor.owns_motion
    assert backend.commands == []


class _StubManifest:
    profile_id = "hiwonder-ainex"
    output_dim = 24
    action_scale = 0.1
    proprio_dim = 45


class _StubPolicy:
    def __init__(self, *_args, **_kwargs) -> None:
        self.manifest = _StubManifest()

    def resolve_task(self, _text: str) -> tuple[str, None, float]:
        return "walk_forward", None, 1.0

    def act(self, *_args, **_kwargs) -> tuple[object, None]:
        import numpy as np

        return np.zeros(24, dtype=np.float32), None


class _StopFailingMock(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[CommandEnvelope] = []

    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        self.commands.append(command)
        if command.command == "walk.command" and command.payload.get("action") == "stop":
            return ResponseEnvelope(
                command.request_id,
                utc_now_iso(),
                False,
                self.backend_name,
                "stop actuator unavailable",
                {},
            )
        return await super().handle_command(command)


class _RecordingMock(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[CommandEnvelope] = []

    async def handle_command(self, command: CommandEnvelope) -> ResponseEnvelope:
        self.commands.append(command)
        return await super().handle_command(command)


@pytest.mark.asyncio
async def test_inference_stop_failure_cannot_return_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference_loop, "TextConditionedPolicy", _StubPolicy)
    backend = _StopFailingMock()
    supervisor = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="stop-failure",
    )

    with pytest.raises(RuntimeError, match="emergency stop failed"):
        await inference_loop.run_inference(
            backend,
            "unused-checkpoint",
            "walk_forward",
            config=InferenceLoopConfig(hz=30.0, max_steps=1),
            supervisor=supervisor,
        )

    assert supervisor.owns_motion
    assert supervisor.emergency_stop_pending
    assert any(command.command == "servo.set" for command in backend.commands)
    assert backend.commands[-1].payload == {"action": "stop"}
    command_count = len(backend.commands)
    blocked = await supervisor.guarded_dispatch(
        _command("head.set", {"pan": 0.1, "tilt": 0.0, "duration": 0.1})
    )
    assert not blocked.ok
    assert "emergency stop acknowledgement is pending" in blocked.message
    assert len(backend.commands) == command_count


@pytest.mark.asyncio
async def test_nonfinite_inference_warmup_never_acquires_or_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NonfinitePolicy(_StubPolicy):
        def act(self, *_args, **_kwargs) -> tuple[object, None]:
            import numpy as np

            return np.full(24, np.nan, dtype=np.float32), None

    monkeypatch.setattr(inference_loop, "TextConditionedPolicy", _NonfinitePolicy)
    backend = _RecordingMock()
    supervisor = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="nonfinite-action",
    )

    with pytest.raises(RuntimeError, match="non-finite policy action"):
        await inference_loop.run_inference(
            backend,
            "unused-checkpoint",
            "walk_forward",
            config=InferenceLoopConfig(hz=30.0, max_steps=1),
            supervisor=supervisor,
        )

    assert backend.commands == []
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_nonfinite_live_inference_action_stops_without_servo_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NonfiniteAfterWarmupPolicy(_StubPolicy):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            self.calls = 0

        def act(self, *_args, **_kwargs) -> tuple[object, None]:
            import numpy as np

            self.calls += 1
            if self.calls == 1:
                return np.zeros(24, dtype=np.float32), None
            return np.full(24, np.nan, dtype=np.float32), None

    monkeypatch.setattr(
        inference_loop,
        "TextConditionedPolicy",
        _NonfiniteAfterWarmupPolicy,
    )
    backend = _RecordingMock()
    supervisor = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="nonfinite-live-action",
    )

    with pytest.raises(RuntimeError, match="non-finite policy action"):
        await inference_loop.run_inference(
            backend,
            "unused-checkpoint",
            "walk_forward",
            config=InferenceLoopConfig(hz=30.0, max_steps=1),
            supervisor=supervisor,
        )

    assert [command.command for command in backend.commands] == ["walk.command"]
    assert not supervisor.owns_motion


@pytest.mark.asyncio
async def test_inference_prepare_failure_never_acquires_or_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingPolicy:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("checkpoint initialization failed")

    monkeypatch.setattr(inference_loop, "TextConditionedPolicy", _FailingPolicy)
    backend = _RecordingMock()
    supervisor = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="prepare-failure",
    )

    with pytest.raises(RuntimeError, match="checkpoint initialization failed"):
        await inference_loop.run_inference(
            backend,
            "unused-checkpoint",
            "walk_forward",
            config=InferenceLoopConfig(hz=30.0, max_steps=1),
            supervisor=supervisor,
        )

    assert not supervisor.owns_motion
    assert backend.commands == []


@pytest.mark.asyncio
async def test_inference_cancellation_during_slow_prepare_never_acquires_or_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_started = threading.Event()
    release_prepare = threading.Event()
    prepare_finished = threading.Event()

    class _SlowPolicy(_StubPolicy):
        def __init__(self, *_args, **_kwargs) -> None:
            prepare_started.set()
            try:
                if not release_prepare.wait(timeout=2.0):
                    raise RuntimeError("test did not release policy preparation")
                super().__init__()
            finally:
                prepare_finished.set()

    monkeypatch.setattr(inference_loop, "TextConditionedPolicy", _SlowPolicy)
    backend = _RecordingMock()
    supervisor = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="cancelled-prepare",
    )
    task = asyncio.create_task(
        inference_loop.run_inference(
            backend,
            "unused-checkpoint",
            "walk_forward",
            config=InferenceLoopConfig(hz=30.0, max_steps=1),
            supervisor=supervisor,
        )
    )
    deadman_task = asyncio.create_task(
        _deadman_pump(
            _Socket(),  # type: ignore[arg-type]
            supervisor,
            PolicyLoopState(active=True, _loop_task=task),
            deadman_timeout_sec=0.05,
        )
    )
    try:
        assert await asyncio.to_thread(prepare_started.wait, 1.0)
        # Preparation intentionally outlives the configured deadman. The pump
        # must remain inert because the uninitialized policy owns no motion.
        await asyncio.sleep(0.15)
        assert not supervisor.owns_motion
        assert backend.commands == []
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not supervisor.owns_motion
        assert backend.commands == []
    finally:
        deadman_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await deadman_task
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        release_prepare.set()
        assert await asyncio.to_thread(prepare_finished.wait, 1.0)

    # Completion of the detached worker after cancellation must not revive the
    # cancelled coroutine or acquire motion later.
    await asyncio.sleep(0)
    assert task.cancelled()
    assert not supervisor.owns_motion
    assert backend.commands == []


@pytest.mark.asyncio
async def test_competing_owner_during_prepare_wins_without_loser_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_started = threading.Event()
    release_prepare = threading.Event()

    class _SlowPolicy(_StubPolicy):
        def __init__(self, *_args, **_kwargs) -> None:
            prepare_started.set()
            if not release_prepare.wait(timeout=2.0):
                raise RuntimeError("test did not release policy preparation")
            super().__init__()

    monkeypatch.setattr(inference_loop, "TextConditionedPolicy", _SlowPolicy)
    backend = _RecordingMock()
    registry = MotionOwnershipRegistry()
    candidate = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="preparing-candidate",
        resource_id="prepare-race",
        registry=registry,
    )
    contender = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="winning-contender",
        resource_id="prepare-race",
        registry=registry,
    )
    task = asyncio.create_task(
        inference_loop.run_inference(
            backend,
            "unused-checkpoint",
            "walk_forward",
            config=InferenceLoopConfig(hz=30.0, max_steps=1),
            supervisor=candidate,
        )
    )
    try:
        assert await asyncio.to_thread(prepare_started.wait, 1.0)
        response = await contender.guarded_dispatch(
            _command("walk.command", {"action": "start"})
        )
        assert response.ok
        assert contender.owns_motion
    finally:
        release_prepare.set()

    with pytest.raises(RuntimeError, match="intent is stale"):
        await task

    assert not candidate.owns_motion
    assert contender.owns_motion
    assert [command.command for command in backend.commands] == ["walk.command"]
    assert backend.commands[0].payload == {"action": "start"}

    stop_response = await contender.emergency_stop_once("prepare-race-cleanup")
    assert stop_response.ok


@pytest.mark.asyncio
async def test_server_side_policy_start_does_not_acquire_before_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _RecordingMock()
    supervisor = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="server-prepare",
    )
    state = PolicyLoopState()
    observed: dict[str, object] = {}

    async def _fake_run_inference(
        _backend: object,
        _checkpoint: object,
        _text: object,
        **kwargs: object,
    ) -> dict[str, object]:
        observed["owns_motion"] = supervisor.owns_motion
        observed["expected_generation"] = kwargs.get("expected_motion_generation")
        return {"steps_completed": 0}

    monkeypatch.setattr(inference_loop, "run_inference", _fake_run_inference)
    expected_generation = supervisor.motion_generation
    response = await _handle_policy_command(
        _Socket(),  # type: ignore[arg-type]
        backend,
        _command("policy.start", {"task": "walk_forward", "max_steps": 1}),
        state,
        None,
        RuntimeConfig(
            queue_size=8,
            max_commands_per_sec=30,
            deadman_timeout_sec=1.0,
            trace_log_path="",
            policy_checkpoint="unused-checkpoint",
        ),
        load_profile("hiwonder-ainex"),
        supervisor,
        motion_generation_at_receive=expected_generation,
    )
    assert response.ok
    assert state._loop_task is not None
    await state._loop_task

    assert observed == {
        "owns_motion": False,
        "expected_generation": expected_generation,
    }
    assert not supervisor.owns_motion
    assert backend.commands == []


@pytest.mark.asyncio
async def test_physical_policy_start_hard_envelope_rejects_before_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _PhysicalBackend()
    supervisor = MotionSafetySupervisor(
        backend,  # type: ignore[arg-type]
        load_profile("hiwonder-ainex"),
        owner_id="physical-policy-start",
        registry=MotionOwnershipRegistry(),
    )
    state = PolicyLoopState()

    async def _unexpected_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("physical policy runner must not be submitted")

    monkeypatch.setattr(inference_loop, "run_inference", _unexpected_run)
    response = await _handle_policy_command(
        _Socket(),  # type: ignore[arg-type]
        backend,  # type: ignore[arg-type]
        _command("policy.start", {"task": "walk_forward", "max_steps": 1}),
        state,
        None,
        RuntimeConfig(
            queue_size=8,
            max_commands_per_sec=30,
            deadman_timeout_sec=1.0,
            trace_log_path="",
            policy_checkpoint="unused-checkpoint",
        ),
        load_profile("hiwonder-ainex"),
        supervisor,
    )

    assert not response.ok
    assert "physical hard safety envelope" in response.message
    assert state._loop_task is None
    assert not supervisor.owns_motion
    assert backend.commands == []
