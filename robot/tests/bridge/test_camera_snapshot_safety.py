"""Safety and resource-bound tests for the bridge camera snapshot path."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from eliza_robot.bridge.backends.mock_backend import MockBackend
from eliza_robot.bridge.camera_snapshot import (
    CameraSnapshotBusyError,
    CameraSnapshotCoordinator,
    CameraSnapshotError,
    CameraSnapshotLimits,
    CameraSnapshotTimeoutError,
)
from eliza_robot.bridge.protocol import CommandEnvelope, ResponseEnvelope, utc_now_iso
from eliza_robot.bridge.safety import MotionOwnershipRegistry, MotionSafetySupervisor
from eliza_robot.bridge.server import (
    PolicyLoopState,
    RuntimeConfig,
    SharedBackendRuntime,
    _deadman_pump,
    _handler,
    _physical_snapshot_blocker,
)
from eliza_robot.profiles.schema import load_profile


def _command(command: str, payload: dict[str, object]) -> CommandEnvelope:
    return CommandEnvelope(
        request_id=f"camera-safety-{command}",
        timestamp=utc_now_iso(),
        command=command,
        payload=payload,
    )


class _RecordingWebSocket:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)
        self.frames.append(parsed)


class _BlockingSnapshotBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.capture_started = threading.Event()
        self.release_capture = threading.Event()
        self.stop_seen = asyncio.Event()

    def snapshot_camera(self, _camera: str = "head") -> np.ndarray | None:
        self.capture_started.set()
        if not self.release_capture.wait(timeout=2.0):
            raise RuntimeError("test did not release blocking camera")
        return np.zeros((2, 2, 3), dtype=np.uint8)

    async def handle_emergency_stop(self, cmd: CommandEnvelope) -> ResponseEnvelope:
        self.stop_seen.set()
        return await super().handle_emergency_stop(cmd)


class _PhysicalSnapshotBackend(MockBackend):
    def physical_motion_resources(self) -> tuple[str, ...]:
        return ("physical:test-camera",)


async def _wait_for_thread_event(event: threading.Event) -> None:
    for _ in range(200):
        if event.is_set():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("worker thread did not start")


async def _receive_response(websocket: Any, request_id: str, timeout: float) -> dict[str, object]:
    async with asyncio.timeout(timeout):
        while True:
            frame = json.loads(await websocket.recv())
            if frame.get("type") == "response" and frame.get("request_id") == request_id:
                return frame


@pytest.mark.asyncio
async def test_snapshot_timeout_keeps_orphan_registered_and_rejects_concurrency() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0
    coordinator = CameraSnapshotCoordinator(
        CameraSnapshotLimits(
            timeout_sec=0.02,
            max_width=4,
            max_height=4,
            max_pixels=16,
            max_png_bytes=1_024,
        )
    )

    def blocking_capture() -> np.ndarray:
        nonlocal calls
        calls += 1
        started.set()
        if not release.wait(timeout=2.0):
            raise RuntimeError("test did not release blocking camera")
        return np.zeros((2, 2, 3), dtype=np.uint8)

    try:
        first = asyncio.create_task(coordinator.capture(blocking_capture))
        await _wait_for_thread_event(started)
        with pytest.raises(CameraSnapshotTimeoutError, match="exceeded"):
            await first
        assert coordinator.busy

        with pytest.raises(CameraSnapshotBusyError, match="still busy"):
            await coordinator.capture(lambda: np.zeros((1, 1, 3), dtype=np.uint8))
        assert calls == 1
    finally:
        release.set()

    for _ in range(200):
        if not coordinator.busy:
            break
        await asyncio.sleep(0.001)
    assert not coordinator.busy
    recovered = await coordinator.capture(lambda: np.zeros((1, 1, 3), dtype=np.uint8))
    assert (recovered.width, recovered.height) == (1, 1)


@pytest.mark.asyncio
async def test_blocking_snapshot_does_not_delay_deadman_emergency_stop() -> None:
    backend = _BlockingSnapshotBackend()
    runtime = SharedBackendRuntime(
        lambda: backend,
        camera_snapshot_limits=CameraSnapshotLimits(
            timeout_sec=1.0,
            max_width=4,
            max_height=4,
            max_pixels=16,
            max_png_bytes=1_024,
        ),
    )
    supervisor = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="camera-deadman-owner",
        resource_id="camera-deadman-resource",
        registry=MotionOwnershipRegistry(),
    )
    motion = await supervisor.guarded_dispatch(
        _command(
            "walk.set",
            {"speed": 2, "height": 0.036, "x": 0.01, "y": 0.0, "yaw": 0.0},
        )
    )
    assert motion.ok

    websocket = _RecordingWebSocket()
    deadman = asyncio.create_task(
        _deadman_pump(
            websocket,  # type: ignore[arg-type]
            supervisor,
            PolicyLoopState(),
            deadman_timeout_sec=0.05,
        )
    )
    snapshot = asyncio.create_task(
        runtime.capture_camera_snapshot(lambda: backend.snapshot_camera("head"))
    )
    try:
        await _wait_for_thread_event(backend.capture_started)
        await asyncio.wait_for(backend.stop_seen.wait(), timeout=0.5)
        assert not backend.release_capture.is_set()
        assert not snapshot.done()
        for _ in range(100):
            if any(
                frame.get("event") == "safety.deadman_triggered"
                for frame in websocket.frames
            ):
                break
            await asyncio.sleep(0.001)
        assert any(frame.get("event") == "safety.deadman_triggered" for frame in websocket.frames)
    finally:
        backend.release_capture.set()
        deadman.cancel()
        await asyncio.gather(deadman, return_exceptions=True)
    encoded = await snapshot
    assert (encoded.width, encoded.height) == (2, 2)


@pytest.mark.asyncio
async def test_blocking_snapshot_does_not_delay_same_session_priority_stop() -> None:
    backend = _BlockingSnapshotBackend()
    runtime = SharedBackendRuntime(
        lambda: backend,
        poll_hz=100.0,
        camera_snapshot_limits=CameraSnapshotLimits(
            timeout_sec=1.0,
            max_width=4,
            max_height=4,
            max_pixels=16,
            max_png_bytes=1_024,
        ),
    )
    config = RuntimeConfig(
        queue_size=8,
        max_commands_per_sec=30,
        deadman_timeout_sec=10.0,
        trace_log_path="",
    )
    await runtime.start()
    server = await serve(lambda websocket: _handler(websocket, runtime, config), "127.0.0.1", 0)
    assert server.sockets
    port = int(server.sockets[0].getsockname()[1])

    try:
        async with connect(f"ws://127.0.0.1:{port}") as websocket:
            await websocket.recv()
            camera = _command("camera.snapshot", {})
            await websocket.send(json.dumps(camera.to_json()))
            await _wait_for_thread_event(backend.capture_started)

            stop = _command("walk.command", {"action": "stop"})
            await websocket.send(json.dumps(stop.to_json()))
            stop_response = await _receive_response(websocket, stop.request_id, timeout=0.5)
            assert stop_response["ok"] is True
            assert backend.stop_seen.is_set()
            assert not backend.release_capture.is_set()

            backend.release_capture.set()
            camera_response = await _receive_response(websocket, camera.request_id, timeout=0.5)
            assert camera_response["ok"] is True
    finally:
        backend.release_capture.set()
        server.close()
        await server.wait_closed()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (np.zeros((2, 3, 3), dtype=np.uint8), "dimensions exceed limit"),
        (np.zeros((2, 2, 3), dtype=np.uint8), "has 4 pixels"),
        (np.zeros((1, 1, 3), dtype=np.float32), "dtype must be uint8"),
    ],
)
async def test_snapshot_frame_bounds_fail_closed(frame: np.ndarray, message: str) -> None:
    limits = CameraSnapshotLimits(
        max_width=2,
        max_height=2,
        max_pixels=3,
        max_png_bytes=1_024,
    )
    with pytest.raises(CameraSnapshotError, match=message):
        await CameraSnapshotCoordinator(limits).capture(lambda: frame)


@pytest.mark.asyncio
async def test_snapshot_encoded_byte_limit_fails_closed() -> None:
    limits = CameraSnapshotLimits(
        max_width=2,
        max_height=2,
        max_pixels=4,
        max_png_bytes=16,
    )
    with pytest.raises(CameraSnapshotError, match="encoded snapshot"):
        await CameraSnapshotCoordinator(limits).capture(
            lambda: np.zeros((2, 2, 3), dtype=np.uint8)
        )


def test_runtime_camera_configuration_rejects_excessive_pixel_area() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        RuntimeConfig(
            queue_size=8,
            max_commands_per_sec=30,
            deadman_timeout_sec=1.0,
            trace_log_path="",
            camera_width=4_096,
            camera_height=4_096,
        )


@pytest.mark.parametrize("state", ["owned", "pending"])
def test_physical_snapshot_is_blocked_while_actuator_resource_is_not_idle(
    state: str,
) -> None:
    backend = _PhysicalSnapshotBackend()
    registry = MotionOwnershipRegistry()
    supervisor = MotionSafetySupervisor(
        backend,
        load_profile("hiwonder-ainex"),
        owner_id="camera-owner",
        resource_id="physical:test-camera",
        registry=registry,
    )
    assert _physical_snapshot_blocker(supervisor) is None
    if state == "owned":
        assert registry.acquire(supervisor.resource_id, supervisor.owner_id)
    else:
        assert registry.begin_stop(supervisor.resource_id, force=True)
        registry.record_stop_result(supervisor.resource_id, supervisor.owner_id, ok=False)

    blocker = _physical_snapshot_blocker(supervisor)
    assert blocker is not None
    assert "blocked" in blocker
