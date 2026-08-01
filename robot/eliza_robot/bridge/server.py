"""Unified websocket server for AiNex real/sim backends."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import ipaddress
import json
import logging
import math
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

# isaac_backend and ros_backend may pull lazy ROS/IsaacLab modules at call-time;
# their top-level imports are safe. mujoco_backend is resolved lazily so the
# bridge can boot without mujoco installed.
from eliza_robot.bridge.backends.base import (
    BridgeBackend,
    canonical_physical_resource_id,
)
from eliza_robot.bridge.backends.isaac_backend import IsaacBackend
from eliza_robot.bridge.backends.mock_backend import MockBackend
from eliza_robot.bridge.backends.ros_backend import RosBridgeBackend
from eliza_robot.bridge.camera_snapshot import (
    CAMERA_SNAPSHOT_MAX_HEIGHT,
    CAMERA_SNAPSHOT_MAX_PIXELS,
    CAMERA_SNAPSHOT_MAX_WIDTH,
    CameraCapture,
    CameraFrame,
    CameraSnapshotCoordinator,
    CameraSnapshotLimits,
    CameraSnapshotUnavailableError,
    EncodedCameraSnapshot,
)
from eliza_robot.bridge.protocol import (
    CommandEnvelope,
    EventEnvelope,
    ResponseEnvelope,
    parse_command,
    utc_now_iso,
)
from eliza_robot.bridge.safety import (
    GLOBAL_MOTION_OWNERSHIP,
    CommandRateLimiter,
    MotionSafetySupervisor,
    PolicyHeartbeatMonitor,
)
from eliza_robot.bridge.trace_log import TraceLogger, safe_to_record
from eliza_robot.bridge.types import JsonDict, JsonValue
from eliza_robot.bridge.validation import validate_command_payload
from eliza_robot.profiles.schema import RobotProfile, load_profile

BackendFactory = Callable[[], BridgeBackend]
TelemetryReader = Callable[[], Awaitable[list[EventEnvelope]]]

logger = logging.getLogger(__name__)

_PHYSICAL_BACKENDS = frozenset({"ros", "ros_real", "ainex_remote", "ros_remote", "asimov_remote"})
_SAFETY_NOTIFICATION_TIMEOUT_SEC = 0.25


def _profile_to_jsondict(profile: RobotProfile) -> JsonDict:
    """Serialize a RobotProfile to a JSON-safe dict.

    Pydantic's `model_dump()` keeps `Path` objects as-is, which `json.dumps`
    rejects. Stringify them here so the wire payload is plain JSON.
    """
    raw = profile.model_dump()
    assets = raw.get("assets")
    if isinstance(assets, dict):
        for key, value in list(assets.items()):
            assets[key] = str(value)
    return raw


def _effective_deadman_timeout(config: RuntimeConfig, profile: RobotProfile) -> float:
    """A deployment request may tighten, but never relax, profile safety."""
    return min(
        float(config.deadman_timeout_sec),
        float(profile.safety.deadman_timeout_s),
    )


@dataclass
class PolicyLoopState:
    """Tracks the state of an active policy loop within a session."""

    active: bool = False
    task: str = ""
    trace_id: str = ""
    planner_step_id: str = ""
    canonical_action: str = ""
    target_entity_id: str = ""
    target_label: str = ""
    hz: float = 10.0
    max_steps: int = 10000
    step: int = 0
    heartbeat: PolicyHeartbeatMonitor | None = None
    last_joint_positions: dict[str, float] = field(default_factory=dict)
    _loop_task: asyncio.Task[None] | None = None
    _supervisor: MotionSafetySupervisor | None = None


@dataclass
class RuntimeConfig:
    queue_size: int
    max_commands_per_sec: int
    deadman_timeout_sec: float
    trace_log_path: str
    profile_id: str = "hiwonder-ainex"
    # MuJoCo backend knobs (only consulted when backend == "mujoco").
    mujoco_target_xyz: tuple[float, float, float] = (2.0, 0.0, 0.05)
    # When set, `camera.snapshot` reads from a v4l2 device (e.g. Obsbot)
    # instead of (or in addition to) the backend's snapshot_camera(). -1 = off.
    camera_device: int = -1
    camera_width: int = 640
    camera_height: int = 480
    # Remote AiNex rosbridge connection (--backend ainex_remote).
    rosbridge_host: str = "192.168.1.218"
    rosbridge_port: int = 9090
    asimov_livekit_url: str = ""
    asimov_livekit_token: str = ""
    # Optional server-side text-conditioned policy checkpoint. When set,
    # policy.start runs the checkpoint in-process and dispatches servo targets;
    # when unset, the bridge preserves the external policy.tick protocol.
    policy_checkpoint: str = ""
    # Deployment authentication is intentionally sourced from configuration or
    # an environment secret, never emitted in hello/log payloads.
    auth_token: str = field(default="", repr=False)
    # Stable, non-secret actuator identity used to share ownership across
    # multiple server runtimes or backend aliases targeting the same robot.
    physical_resource_id: str = ""

    def __post_init__(self) -> None:
        def _bounded_int(name: str, value: object, lo: int, hi: int) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < lo or value > hi:
                raise ValueError(f"{name} must be in {lo}..{hi}")
            return value

        def _bounded_float(
            name: str,
            value: object,
            lo: float,
            hi: float,
        ) -> float:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be numeric")
            try:
                parsed = float(value)
            except OverflowError as exc:
                raise ValueError(f"{name} must be finite") from exc
            if not math.isfinite(parsed) or parsed < lo or parsed > hi:
                raise ValueError(f"{name} must be finite and in {lo}..{hi}")
            return parsed

        _bounded_int("queue_size", self.queue_size, 1, 10_000)
        _bounded_int(
            "max_commands_per_sec",
            self.max_commands_per_sec,
            1,
            10_000,
        )
        _bounded_float(
            "deadman_timeout_sec",
            self.deadman_timeout_sec,
            0.05,
            3_600.0,
        )
        if not isinstance(self.mujoco_target_xyz, tuple) or len(self.mujoco_target_xyz) != 3:
            raise ValueError("mujoco_target_xyz must be a 3-tuple")
        for axis, coordinate in zip("xyz", self.mujoco_target_xyz, strict=True):
            _bounded_float(f"mujoco_target_{axis}", coordinate, -1_000.0, 1_000.0)
        _bounded_int("camera_device", self.camera_device, -1, 1_000_000)
        _bounded_int("camera_width", self.camera_width, 1, CAMERA_SNAPSHOT_MAX_WIDTH)
        _bounded_int("camera_height", self.camera_height, 1, CAMERA_SNAPSHOT_MAX_HEIGHT)
        if self.camera_width * self.camera_height > CAMERA_SNAPSHOT_MAX_PIXELS:
            raise ValueError(
                "camera_width * camera_height must not exceed "
                f"{CAMERA_SNAPSHOT_MAX_PIXELS} pixels"
            )
        _bounded_int("rosbridge_port", self.rosbridge_port, 1, 65_535)
        for name in (
            "trace_log_path",
            "profile_id",
            "rosbridge_host",
            "asimov_livekit_url",
            "asimov_livekit_token",
            "policy_checkpoint",
            "auth_token",
            "physical_resource_id",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if not self.profile_id:
            raise ValueError("profile_id must not be empty")
        if not self.rosbridge_host:
            raise ValueError("rosbridge_host must not be empty")
        if self.physical_resource_id:
            canonical_physical_resource_id(self.physical_resource_id)


@dataclass(frozen=True)
class _TelemetryDelivery:
    events: tuple[EventEnvelope, ...] = ()
    error: str = ""


class SharedBackendRuntime:
    """One backend lifecycle and one telemetry consumer per server process."""

    def __init__(
        self,
        backend_factory: BackendFactory,
        *,
        poll_hz: float = 2.0,
        poll_timeout_sec: float = 1.0,
        camera_snapshot_limits: CameraSnapshotLimits | None = None,
    ) -> None:
        def _bounded_float(name: str, value: object, upper: float) -> float:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be numeric")
            try:
                parsed = float(value)
            except OverflowError as exc:
                raise ValueError(f"{name} must be finite") from exc
            if not math.isfinite(parsed) or parsed <= 0.0 or parsed > upper:
                raise ValueError(f"{name} must be finite and in (0, {upper:g}]")
            return parsed

        poll_hz_value = _bounded_float("poll_hz", poll_hz, 1_000.0)
        poll_timeout_value = _bounded_float(
            "poll_timeout_sec",
            poll_timeout_sec,
            10.0,
        )
        self.backend = backend_factory()
        self._period = 1.0 / poll_hz_value
        self._poll_timeout_sec = poll_timeout_value
        self.motion_resource_token = uuid.uuid4().hex
        self._camera_snapshots = CameraSnapshotCoordinator(camera_snapshot_limits)
        self._telemetry_queues: set[asyncio.Queue[_TelemetryDelivery]] = set()
        self._command_queues: set[asyncio.Queue[CommandEnvelope]] = set()
        self._latest_events: tuple[EventEnvelope, ...] = ()
        self._poll_task: asyncio.Task[None] | None = None
        self._connected = False

    async def start(self) -> None:
        if self._connected:
            return
        await self.backend.connect()
        self._connected = True
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def close(self) -> None:
        poll_task = self._poll_task
        self._poll_task = None
        if poll_task is not None:
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task
        if self._connected:
            self._connected = False
            await self.backend.shutdown()

    def subscribe(self) -> asyncio.Queue[_TelemetryDelivery]:
        queue: asyncio.Queue[_TelemetryDelivery] = asyncio.Queue(maxsize=1)
        self._telemetry_queues.add(queue)
        if self._latest_events:
            queue.put_nowait(_TelemetryDelivery(events=self._latest_events))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[_TelemetryDelivery]) -> None:
        self._telemetry_queues.discard(queue)

    def register_command_queue(
        self,
        queue: asyncio.Queue[CommandEnvelope],
    ) -> None:
        self._command_queues.add(queue)

    def unregister_command_queue(
        self,
        queue: asyncio.Queue[CommandEnvelope],
    ) -> None:
        self._command_queues.discard(queue)

    def purge_command_queues(self) -> int:
        """Discard every command accepted before a process-wide stop."""
        removed = 0
        for queue in tuple(self._command_queues):
            while not queue.empty():
                try:
                    _ = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                queue.task_done()
                removed += 1
        return removed

    async def latest_events(self) -> list[EventEnvelope]:
        """Return the fanout pump's latest snapshot without consuming it."""
        return list(self._latest_events)

    async def capture_camera_snapshot(
        self,
        capture: CameraCapture,
    ) -> EncodedCameraSnapshot:
        """Capture and encode through the process-wide single-worker gate."""
        return await self._camera_snapshots.capture(capture)

    def _broadcast(self, delivery: _TelemetryDelivery) -> None:
        for queue in tuple(self._telemetry_queues):
            retained: _TelemetryDelivery | None = None
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    previous = queue.get_nowait()
                    queue.task_done()
                    # A transport/malformed-data fault is a safety edge, not a
                    # latest-value sample. Never let a later healthy poll
                    # overwrite it before the owning session observes it.
                    if previous.error and not delivery.error:
                        retained = previous
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(retained or delivery)

    async def _poll_loop(self) -> None:
        while True:
            try:
                raw_events = await asyncio.wait_for(
                    self.backend.poll_events(),
                    timeout=self._poll_timeout_sec,
                )
                if not isinstance(raw_events, list) or not all(
                    isinstance(event, EventEnvelope) and isinstance(event.data, dict)
                    for event in raw_events
                ):
                    raise RuntimeError(
                        "backend telemetry must be a list of valid EventEnvelope objects"
                    )
                events = tuple(raw_events)
                self._latest_events = events
                self._broadcast(_TelemetryDelivery(events=events))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._broadcast(_TelemetryDelivery(error=f"telemetry polling failed: {exc}"))
            await asyncio.sleep(self._period)


def _load_config_file(path: str) -> JsonDict:
    if path == "":
        return {}
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a JSON object")
    return raw


def _coerce_runtime_config(args: argparse.Namespace, config_obj: JsonDict) -> RuntimeConfig:
    queue_size = args.queue_size
    max_commands_per_sec = args.max_commands_per_sec
    deadman_timeout_sec = args.deadman_timeout_sec
    trace_log_path = args.trace_log_path

    safety_value = config_obj.get("safety")
    if isinstance(safety_value, dict):
        queue_size_value = safety_value.get("queue_size")
        if isinstance(queue_size_value, int):
            queue_size = queue_size_value
        rate_value = safety_value.get("command_rate_limit_hz")
        if isinstance(rate_value, int):
            max_commands_per_sec = rate_value
        deadman_value = safety_value.get("deadman_timeout_sec")
        if isinstance(deadman_value, int | float):
            deadman_timeout_sec = float(deadman_value)

    logging_value = config_obj.get("logging")
    if isinstance(logging_value, dict):
        trace_log_value = logging_value.get("trace_log_path")
        if isinstance(trace_log_value, str):
            trace_log_path = trace_log_value

    asimov_livekit_url = getattr(args, "asimov_livekit_url", "") or os.environ.get(
        "ASIMOV_LIVEKIT_URL", ""
    )
    asimov_livekit_token = getattr(args, "asimov_livekit_token", "") or os.environ.get(
        "ASIMOV_LIVEKIT_TOKEN", ""
    )
    policy_checkpoint = getattr(args, "policy_checkpoint", "") or os.environ.get(
        "ELIZA_ROBOT_POLICY_CHECKPOINT", ""
    )
    auth_token = os.environ.get("ELIZA_ROBOT_BRIDGE_AUTH_TOKEN", "")
    physical_resource_id = os.environ.get("ELIZA_ROBOT_PHYSICAL_RESOURCE_ID", "")
    policy_value = config_obj.get("policy")
    if isinstance(policy_value, dict):
        ckpt_value = policy_value.get("checkpoint")
        if isinstance(ckpt_value, str) and ckpt_value:
            policy_checkpoint = ckpt_value

    return RuntimeConfig(
        queue_size=queue_size,
        max_commands_per_sec=max_commands_per_sec,
        deadman_timeout_sec=deadman_timeout_sec,
        trace_log_path=trace_log_path,
        profile_id=getattr(args, "profile", "hiwonder-ainex"),
        mujoco_target_xyz=(
            getattr(args, "mujoco_target_x", 2.0),
            getattr(args, "mujoco_target_y", 0.0),
            getattr(args, "mujoco_target_z", 0.05),
        ),
        camera_device=getattr(args, "camera_device", -1),
        camera_width=getattr(args, "camera_width", 640),
        camera_height=getattr(args, "camera_height", 480),
        rosbridge_host=getattr(args, "rosbridge_host", "192.168.1.218"),
        rosbridge_port=getattr(args, "rosbridge_port", 9090),
        asimov_livekit_url=asimov_livekit_url,
        asimov_livekit_token=asimov_livekit_token,
        policy_checkpoint=policy_checkpoint,
        auth_token=auth_token,
        physical_resource_id=physical_resource_id,
    )


def _build_backend_factory(name: str, config: RuntimeConfig) -> BackendFactory:
    if name == "mock":
        return MockBackend
    if name == "ros":
        return lambda: RosBridgeBackend(
            "ros_real",
            physical_resource_id=config.physical_resource_id,
        )
    if name == "ros_real":
        return lambda: RosBridgeBackend(
            "ros_real",
            physical_resource_id=config.physical_resource_id,
        )
    if name == "ros_sim":
        return lambda: RosBridgeBackend("ros_sim")
    if name == "isaac":
        return IsaacBackend
    if name == "mujoco":
        # Lazy import so the bridge can boot without mujoco installed.
        # DemoEnv (CPU MuJoCo) is the default sim — it loads the profile's
        # primitives model, spawns a target ball, and exposes the same
        # joint-target + telemetry surface the real robot does.
        def _build_mujoco_backend() -> BridgeBackend:
            from eliza_robot.bridge.backends.mujoco_backend import MuJocoBackend
            from eliza_robot.sim.mujoco.demo_env import DemoEnv

            env = DemoEnv(target_position=config.mujoco_target_xyz)
            return MuJocoBackend(env, profile_id=config.profile_id)

        return _build_mujoco_backend
    if name in {"ainex_remote", "ros_remote"}:
        # Drives a physical AiNex over its rosbridge_suite without needing
        # rospy locally. Host/port come from RuntimeConfig.
        def _build_remote_backend() -> BridgeBackend:
            from eliza_robot.bridge.backends.ainex_remote import AinexRemoteBackend

            return AinexRemoteBackend(
                host=config.rosbridge_host,
                port=config.rosbridge_port,
                physical_resource_id=config.physical_resource_id,
            )

        return _build_remote_backend
    if name in {"asimov_mock", "asimov_remote"}:

        def _build_asimov_backend() -> BridgeBackend:
            from eliza_robot.bridge.backends.asimov_remote import AsimovRemoteBackend

            return AsimovRemoteBackend(
                profile_id=config.profile_id,
                mock=name == "asimov_mock",
                livekit_url=config.asimov_livekit_url,
                livekit_token=config.asimov_livekit_token,
                physical_resource_id=(
                    config.physical_resource_id if name == "asimov_remote" else None
                ),
            )

        return _build_asimov_backend
    if name == "asimov_mujoco":

        def _build_asimov_mujoco_backend() -> BridgeBackend:
            from eliza_robot.bridge.backends.asimov_mujoco import AsimovMujocoBackend

            return AsimovMujocoBackend(profile_id=config.profile_id)

        return _build_asimov_mujoco_backend
    raise ValueError(f"unsupported backend: {name}")


def _is_loopback_bind(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_physical_auth_token(auth_token: str) -> None:
    """Require one header-safe bearer secret for a physical endpoint."""
    if not 32 <= len(auth_token) <= 4_096:
        raise ValueError(
            "physical backends require ELIZA_ROBOT_BRIDGE_AUTH_TOKEN "
            "with 32..4096 visible ASCII characters"
        )
    if any(not 0x21 <= ord(character) <= 0x7E for character in auth_token):
        raise ValueError(
            "physical backends require ELIZA_ROBOT_BRIDGE_AUTH_TOKEN "
            "with 32..4096 visible ASCII characters"
        )


def _validate_bind_security(
    host: str,
    backend_name: str,
    auth_token: str,
    physical_resource_id: str = "",
) -> None:
    """Keep physical control behind local bearer auth and a secure tunnel."""
    if backend_name not in _PHYSICAL_BACKENDS:
        return
    if not _is_loopback_bind(host):
        raise ValueError(
            "physical backends must bind to loopback; expose them only through "
            "an authenticated secure tunnel because the bridge serves plaintext ws://"
        )
    _validate_physical_auth_token(auth_token)
    try:
        canonical_physical_resource_id(physical_resource_id)
    except ValueError as exc:
        raise ValueError(
            "physical backends require an explicit canonical "
            "ELIZA_ROBOT_PHYSICAL_RESOURCE_ID"
        ) from exc


def _validated_physical_resource(
    backend: BridgeBackend,
    config: RuntimeConfig,
) -> str | None:
    """Bind server ownership to the backend's one configured actuator ID."""
    resources = backend.physical_motion_resources()
    if not resources:
        return None
    if not isinstance(resources, tuple) or len(resources) != 1:
        raise ValueError("a bridge backend must expose exactly one physical actuator resource")
    try:
        expected = canonical_physical_resource_id(config.physical_resource_id)
    except ValueError as exc:
        raise ValueError(
            "physical backend requires ELIZA_ROBOT_PHYSICAL_RESOURCE_ID"
        ) from exc
    if resources[0] != expected:
        raise ValueError(
            "configured physical resource identity does not match the backend actuator identity"
        )
    return expected


def _request_is_authorized(ws: ServerConnection, auth_token: str) -> bool:
    if not auth_token:
        return True
    request = getattr(ws, "request", None)
    headers = getattr(request, "headers", None)
    if headers is None:
        return False
    try:
        supplied = headers.get("Authorization")
    except Exception:
        return False
    if not isinstance(supplied, str):
        return False
    return hmac.compare_digest(supplied, f"Bearer {auth_token}")


def _capture_camera_frame(
    backend: BridgeBackend,
    config: RuntimeConfig,
    camera_name: str,
) -> CameraFrame | None:
    """Read one frame synchronously; callers must run this outside the event loop."""
    frame: CameraFrame | None = None
    use_external = config.camera_device >= 0 and camera_name in {
        "head",
        "external",
        "obsbot",
        "v4l2",
    }
    if camera_name in {"external", "obsbot", "v4l2"} and config.camera_device >= 0:
        from eliza_robot.perception.frame_source import OpenCVSource

        with OpenCVSource(
            device=config.camera_device,
            width=config.camera_width,
            height=config.camera_height,
        ) as source:
            ok, bgr = source.read()
            if ok and bgr is not None and bgr.size > 0:
                frame = bgr[:, :, ::-1].copy()
        return frame

    frame = backend.snapshot_camera(camera_name)
    if frame is not None or not use_external:
        return frame

    from eliza_robot.perception.frame_source import OpenCVSource

    with OpenCVSource(
        device=config.camera_device,
        width=config.camera_width,
        height=config.camera_height,
    ) as source:
        ok, bgr = source.read()
        if ok and bgr is not None and bgr.size > 0:
            return bgr[:, :, ::-1].copy()
    return None


def _physical_snapshot_blocker(supervisor: MotionSafetySupervisor) -> str | None:
    """Reject physical camera work while its actuator resource is not idle."""
    if not supervisor.backend.physical_motion_resources():
        return None
    registry = supervisor.registry
    resource_id = supervisor.resource_id
    if registry.emergency_stop_pending(resource_id):
        return "camera.snapshot blocked while emergency stop acknowledgement is pending"
    if registry.stop_in_progress(resource_id):
        return "camera.snapshot blocked while an actuator stop is in progress"
    if registry.owner(resource_id) is not None:
        return "camera.snapshot blocked while the physical motion resource is owned"
    return None


async def _send_camera_snapshot_response(
    ws: ServerConnection,
    runtime: SharedBackendRuntime,
    backend: BridgeBackend,
    supervisor: MotionSafetySupervisor,
    config: RuntimeConfig,
    command: CommandEnvelope,
    camera_name: str,
) -> None:
    """Finish a snapshot request without stalling the websocket receive loop."""
    physical_blocker = _physical_snapshot_blocker(supervisor)
    if physical_blocker is not None:
        await _safe_send(
            ws,
            _json_error(physical_blocker, request_id=command.request_id),
        )
        return
    try:
        snapshot = await runtime.capture_camera_snapshot(
            partial(_capture_camera_frame, backend, config, camera_name)
        )
    except CameraSnapshotUnavailableError:
        await _safe_send(
            ws,
            ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=(
                    f"camera '{camera_name}' not available on backend {backend.backend_name}"
                ),
                data={},
            ).to_json(),
        )
        return
    except Exception as exc:
        await _safe_send(
            ws,
            _json_error(
                f"camera.snapshot failed: {exc}",
                request_id=command.request_id,
            ),
        )
        return

    await _safe_send(
        ws,
        ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=True,
            backend=backend.backend_name,
            message="ok",
            data={
                "camera": camera_name,
                "width": snapshot.width,
                "height": snapshot.height,
                "format": "png",
                "frame_base64": snapshot.frame_base64,
                "png_bytes": snapshot.png_bytes,
            },
        ).to_json(),
    )


def _json_error(message: str, request_id: str = "unknown") -> JsonDict:
    envelope = ResponseEnvelope(
        request_id=request_id,
        timestamp=utc_now_iso(),
        ok=False,
        backend="bridge",
        message=message,
        data={},
    )
    return envelope.to_json()


async def _safe_send(ws: ServerConnection, payload: JsonValue) -> None:
    if not isinstance(payload, dict):
        raise ValueError("websocket send payload must be dict")
    await ws.send(json.dumps(payload))


async def _bounded_safety_send(ws: ServerConnection, payload: JsonValue) -> bool:
    """Best-effort safety notification that cannot block actuator handling."""
    try:
        await asyncio.wait_for(
            _safe_send(ws, payload),
            timeout=_SAFETY_NOTIFICATION_TIMEOUT_SEC,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("safety notification failed or timed out: %s", exc)
        return False
    return True


async def _cancel_policy_loop(policy_state: PolicyLoopState) -> None:
    """Cancel an autonomous loop without ever cancelling the current task."""
    task = policy_state._loop_task
    if task is None or task is asyncio.current_task():
        return
    if task.done():
        policy_state._loop_task = None
        return
    task.cancel()
    # Never let a task that mishandles cancellation delay the actuator stop.
    # The caller has already latched or concurrently dispatched that stop.
    done, _ = await asyncio.wait({task}, timeout=0.25)
    if task in done:
        # Inference cleanup can surface a failed-stop RuntimeError from its
        # finally block. The process-wide supervisor already latches that
        # failure; callers must remain able to issue their own immediate retry.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()
        policy_state._loop_task = None


def _policy_loop_unterminated(policy_state: PolicyLoopState) -> bool:
    """Return whether a cancelled loop could still emit stale motion."""
    task = policy_state._loop_task
    if task is None:
        return False
    if not task.done():
        return True
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()
    policy_state._loop_task = None
    return False


async def _cancel_policy_and_stop(
    policy_state: PolicyLoopState,
    supervisor: MotionSafetySupervisor,
    reason: str,
    *,
    force_physical: bool = False,
) -> ResponseEnvelope:
    """Dispatch a latched stop concurrently with bounded policy cancellation."""
    supervisor.latch_stop_request()
    stop_task = asyncio.create_task(
        supervisor.emergency_stop_once(
            reason,
            force_physical=force_physical,
        )
    )
    await _cancel_policy_loop(policy_state)
    return await stop_task


async def _event_pump(
    ws: ServerConnection,
    backend: BridgeBackend,
    supervisor: MotionSafetySupervisor,
    policy_state: PolicyLoopState,
    hz: float,
) -> None:
    if (
        isinstance(hz, bool)
        or not isinstance(hz, int | float)
        or not math.isfinite(float(hz))
        or hz <= 0.0
        or hz > 1_000.0
    ):
        raise ValueError("event pump hz must be finite and in (0, 1000]")
    period = 1.0 / float(hz)

    async def _fail_closed(reason: str) -> None:
        policy_state.active = False
        stop_response = await _cancel_policy_and_stop(
            policy_state,
            supervisor,
            "telemetry",
        )
        await _bounded_safety_send(
            ws,
            EventEnvelope(
                event="safety.policy_guard",
                timestamp=utc_now_iso(),
                backend=backend.backend_name,
                data={
                    "reason": reason,
                    "response_ok": stop_response.ok,
                },
            ).to_json(),
        )

    while True:
        try:
            if supervisor.emergency_stop_pending:
                await _fail_closed("emergency stop acknowledgement pending")
            events = await backend.poll_events()
            if not isinstance(events, list) or not all(
                isinstance(event, EventEnvelope) and isinstance(event.data, dict)
                for event in events
            ):
                raise RuntimeError(
                    "backend telemetry must be a list of valid EventEnvelope objects"
                )
            for event in events:
                violation = supervisor.telemetry_violation(event)
                if violation is not None:
                    await _fail_closed(violation)
                if not await _bounded_safety_send(ws, event.to_json()):
                    raise RuntimeError("telemetry notification timed out")
            if supervisor.required_telemetry_stale():
                await _fail_closed("required safety telemetry heartbeat timed out")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if supervisor.owns_motion:
                await _fail_closed(f"telemetry polling failed: {exc}")
            await asyncio.sleep(period)
            continue
        await asyncio.sleep(period)


async def _fanout_event_pump(
    ws: ServerConnection,
    runtime: SharedBackendRuntime,
    queue: asyncio.Queue[_TelemetryDelivery],
    supervisor: MotionSafetySupervisor,
    policy_state: PolicyLoopState,
) -> None:
    """Consume one session's fanout queue; never poll the backend directly."""

    async def _fail_closed(reason: str) -> None:
        runtime.purge_command_queues()
        policy_state.active = False
        stop_response = await _cancel_policy_and_stop(
            policy_state,
            supervisor,
            "telemetry",
        )
        await _bounded_safety_send(
            ws,
            EventEnvelope(
                event="safety.policy_guard",
                timestamp=utc_now_iso(),
                backend=runtime.backend.backend_name,
                data={
                    "reason": reason,
                    "response_ok": stop_response.ok,
                },
            ).to_json(),
        )

    while True:
        try:
            delivery = await asyncio.wait_for(queue.get(), timeout=0.1)
        except TimeoutError:
            if supervisor.emergency_stop_pending:
                await _fail_closed("emergency stop acknowledgement pending")
            elif supervisor.required_telemetry_stale():
                await _fail_closed("required safety telemetry heartbeat timed out")
            continue
        try:
            if supervisor.emergency_stop_pending:
                await _fail_closed("emergency stop acknowledgement pending")
            if delivery.error:
                if supervisor.owns_motion:
                    await _fail_closed(delivery.error)
                continue
            for event in delivery.events:
                violation = supervisor.telemetry_violation(event)
                if violation is not None:
                    await _fail_closed(violation)
                if not await _bounded_safety_send(ws, event.to_json()):
                    raise RuntimeError("telemetry notification timed out")
            if supervisor.required_telemetry_stale():
                await _fail_closed("required safety telemetry heartbeat timed out")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if supervisor.owns_motion:
                await _fail_closed(f"telemetry fanout failed: {exc}")
        finally:
            queue.task_done()


async def _command_worker(
    ws: ServerConnection,
    supervisor: MotionSafetySupervisor,
    command_queue: asyncio.Queue[CommandEnvelope],
    trace_logger: TraceLogger | None,
) -> None:
    backend = supervisor.backend
    while True:
        command = await command_queue.get()
        response = await supervisor.guarded_dispatch(
            command,
            require_servo_capability=command.command == "servo.set",
        )
        await _safe_send(ws, response.to_json())
        if not response.ok:
            # A rejected motion has already triggered (or is blocked behind)
            # the fail-closed stop path. Commands queued before that failure
            # are stale intent and must never auto-reacquire motion afterward.
            while not command_queue.empty():
                _ = command_queue.get_nowait()
                command_queue.task_done()
        if trace_logger is not None:
            trace_logger.write(
                {
                    "kind": "command_response",
                    "timestamp": utc_now_iso(),
                    "backend": backend.backend_name,
                    "request_id": command.request_id,
                    "command": command.command,
                    "response": safe_to_record(response.to_json()),
                }
            )
        command_queue.task_done()


async def _deadman_pump(
    ws: ServerConnection,
    supervisor: MotionSafetySupervisor,
    policy_state: PolicyLoopState,
    deadman_timeout_sec: float,
) -> None:
    if (
        isinstance(deadman_timeout_sec, bool)
        or not isinstance(deadman_timeout_sec, int | float)
        or not math.isfinite(float(deadman_timeout_sec))
        or deadman_timeout_sec <= 0.0
        or deadman_timeout_sec > 3_600.0
    ):
        raise ValueError("deadman_timeout_sec must be finite and in (0, 3600]")
    fired = False
    while True:
        await asyncio.sleep(0.1)
        if supervisor.emergency_stop_pending:
            policy_state.active = False
            response = await _cancel_policy_and_stop(
                policy_state,
                supervisor,
                "pending-stop-retry",
            )
            await _bounded_safety_send(
                ws,
                EventEnvelope(
                    event="safety.stop_retry",
                    timestamp=utc_now_iso(),
                    backend=supervisor.backend.backend_name,
                    data={"response_ok": response.ok},
                ).to_json(),
            )
            continue
        if not supervisor.owns_motion:
            fired = False
            continue
        age = supervisor.activity_age_sec()
        if age < deadman_timeout_sec:
            fired = False
            continue
        if fired:
            continue
        policy_state.active = False
        response = await _cancel_policy_and_stop(
            policy_state,
            supervisor,
            "deadman",
        )
        # A failed stop retains ownership and is retried on the next pump
        # iteration. Never mark the deadman as fired until stop is confirmed.
        fired = response.ok
        await _bounded_safety_send(
            ws,
            EventEnvelope(
                event="safety.deadman_triggered",
                timestamp=utc_now_iso(),
                backend=supervisor.backend.backend_name,
                data={"response_ok": response.ok, "age_sec": age},
            ).to_json(),
        )


async def _handle_policy_command(
    ws: ServerConnection,
    backend: BridgeBackend,
    command: CommandEnvelope,
    policy_state: PolicyLoopState,
    trace_logger: TraceLogger | None,
    config: RuntimeConfig,
    profile: RobotProfile,
    supervisor: MotionSafetySupervisor | None = None,
    telemetry_reader: TelemetryReader | None = None,
    purge_commands: Callable[[], int] | None = None,
    motion_generation_at_receive: int | None = None,
) -> ResponseEnvelope:
    """Handle policy lifecycle commands (policy.start/stop/tick/status)."""
    if supervisor is None:
        supervisor = policy_state._supervisor
        if supervisor is None:
            supervisor = MotionSafetySupervisor(
                backend,
                profile,
                owner_id=f"direct-policy-{id(policy_state)}",
            )
            if policy_state.active:
                supervisor.acquire_motion()
    policy_state._supervisor = supervisor

    def _prepare_stop() -> None:
        supervisor.latch_stop_request()
        if purge_commands is not None:
            purge_commands()

    async def _owned_rejection_suffix(reason: str) -> str:
        if not supervisor.owns_motion:
            return ""
        _prepare_stop()
        policy_state.active = False
        response = await _cancel_policy_and_stop(policy_state, supervisor, reason)
        return f"; emergency_stop_ok={response.ok}"

    if command.command == "policy.start":
        if policy_state.active:
            stop_suffix = await _owned_rejection_suffix("duplicate-policy-start")
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=f"policy already active{stop_suffix}",
                data={"task": policy_state.task},
            )
        if _policy_loop_unterminated(policy_state):
            stop_suffix = await _owned_rejection_suffix("unterminated-policy-loop")
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=(
                    "policy start blocked: prior policy cancellation has not completed"
                    f"{stop_suffix}"
                ),
                data={},
            )
        raw_hz = command.payload.get("hz", 10.0)
        raw_max_steps = command.payload.get("max_steps", 10000)
        hz_value = 0.0
        if not isinstance(raw_hz, bool) and isinstance(raw_hz, int | float):
            with contextlib.suppress(OverflowError):
                hz_value = float(raw_hz)
        if not np.isfinite(hz_value) or hz_value < 1.0 or hz_value > 30.0:
            stop_suffix = await _owned_rejection_suffix("invalid-policy-start-hz")
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=(f"policy start blocked: hz must be finite and in 1..30{stop_suffix}"),
                data={},
            )
        if (
            isinstance(raw_max_steps, bool)
            or not isinstance(raw_max_steps, int)
            or raw_max_steps < 1
            or raw_max_steps > 100_000
        ):
            stop_suffix = await _owned_rejection_suffix("invalid-policy-start-max-steps")
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=(
                    f"policy start blocked: max_steps must be an integer in 1..100000{stop_suffix}"
                ),
                data={},
            )
        if supervisor.emergency_stop_pending:
            stop_suffix = await _owned_rejection_suffix("policy-start-stop-retry")
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=(
                    f"policy start blocked: emergency stop acknowledgement pending{stop_suffix}"
                ),
                data={},
            )
        capability_error = supervisor.policy_start_capability_error()
        if capability_error is not None:
            stop_suffix = await _owned_rejection_suffix("policy-start-capability-rejected")
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=f"policy start blocked: {capability_error}{stop_suffix}",
                data={},
            )
        expected_generation = (
            supervisor.motion_generation
            if motion_generation_at_receive is None
            else motion_generation_at_receive
        )
        server_side_policy = bool(config.policy_checkpoint)
        if not server_side_policy and not supervisor.accept_fresh_motion(
            expected_generation
        ):
            stop_suffix = await _owned_rejection_suffix("stale-policy-start")
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=(
                    "policy start blocked: motion intent is stale, a stop is pending, "
                    f"or the resource is owned by another session{stop_suffix}"
                ),
                data={},
            )
        policy_state.active = True
        policy_state.task = str(command.payload.get("task", ""))
        policy_state.trace_id = str(command.payload.get("trace_id", ""))
        policy_state.planner_step_id = str(command.payload.get("planner_step_id", ""))
        policy_state.canonical_action = str(command.payload.get("canonical_action", ""))
        policy_state.target_entity_id = str(command.payload.get("target_entity_id", ""))
        policy_state.target_label = str(command.payload.get("target_label", ""))
        policy_state.hz = hz_value
        policy_state.max_steps = raw_max_steps
        policy_state.step = 0
        policy_state.heartbeat = None
        policy_state.last_joint_positions = {
            joint.name: float(joint.home_rad) for joint in profile.kinematics.joints
        }

        if config.policy_checkpoint:

            async def _run_server_side_policy() -> None:
                try:
                    from eliza_robot.rl.text_conditioned.inference_loop import (
                        InferenceLoopConfig,
                        run_inference,
                    )

                    result = await run_inference(
                        backend,
                        config.policy_checkpoint,
                        policy_state.task,
                        config=InferenceLoopConfig(
                            hz=policy_state.hz,
                            max_steps=policy_state.max_steps,
                            profile_id=config.profile_id,
                        ),
                        supervisor=supervisor,
                        telemetry_reader=telemetry_reader,
                        expected_motion_generation=expected_generation,
                    )
                    policy_state.step = int(result.get("steps_completed", policy_state.step))
                    policy_state.active = False
                    await _safe_send(
                        ws,
                        EventEnvelope(
                            event="policy.status",
                            timestamp=utc_now_iso(),
                            backend=backend.backend_name,
                            data={
                                "state": "idle",
                                "reason": "completed",
                                "steps_completed": policy_state.step,
                                "trace_id": policy_state.trace_id,
                                "planner_step_id": policy_state.planner_step_id,
                                "canonical_action": policy_state.canonical_action,
                                "target_entity_id": policy_state.target_entity_id,
                                "target_label": policy_state.target_label,
                                "checkpoint": config.policy_checkpoint,
                                "result": result,
                            },
                        ).to_json(),
                    )
                    if trace_logger is not None:
                        trace_logger.write(
                            {
                                "kind": "policy_autonomous_complete",
                                "timestamp": utc_now_iso(),
                                "trace_id": policy_state.trace_id,
                                "steps_completed": policy_state.step,
                                "checkpoint": config.policy_checkpoint,
                            }
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    policy_state.active = False
                    # Preparation and warmup deliberately happen before motion
                    # acquisition. Do not stop another session when stale
                    # generation, corrupt-checkpoint, or cancellation failure
                    # occurs without this supervisor ever becoming the owner.
                    stop_ok = True
                    if supervisor.owns_motion or supervisor.emergency_stop_pending:
                        _prepare_stop()
                        stop_response = await supervisor.emergency_stop_once(
                            "autonomous-error"
                        )
                        stop_ok = stop_response.ok
                    await _safe_send(
                        ws,
                        EventEnvelope(
                            event="policy.status",
                            timestamp=utc_now_iso(),
                            backend=backend.backend_name,
                            data={
                                "state": (
                                    "emergency_stop_pending" if not stop_ok else "idle"
                                ),
                                "reason": "error",
                                "error": str(exc),
                                "stop_ok": stop_ok,
                                "steps_completed": policy_state.step,
                                "trace_id": policy_state.trace_id,
                                "planner_step_id": policy_state.planner_step_id,
                                "canonical_action": policy_state.canonical_action,
                                "target_entity_id": policy_state.target_entity_id,
                                "target_label": policy_state.target_label,
                                "checkpoint": config.policy_checkpoint,
                            },
                        ).to_json(),
                    )
                    if trace_logger is not None:
                        trace_logger.write(
                            {
                                "kind": "policy_autonomous_error",
                                "timestamp": utc_now_iso(),
                                "trace_id": policy_state.trace_id,
                                "error": str(exc),
                                "checkpoint": config.policy_checkpoint,
                            }
                        )

            policy_state._loop_task = asyncio.create_task(_run_server_side_policy())
        else:
            policy_state.heartbeat = PolicyHeartbeatMonitor(
                timeout_sec=min(
                    2.0,
                    _effective_deadman_timeout(config, profile),
                )
            )
            policy_state.heartbeat.record_tick()
            # Ensure walking is started for externally ticked policy mode.
            start_cmd = CommandEnvelope(
                request_id=f"{command.request_id}-walk-start",
                timestamp=utc_now_iso(),
                command="walk.command",
                payload={"action": "start"},
            )
            start_response = await supervisor.guarded_dispatch(start_cmd)
            if not start_response.ok:
                policy_state.active = False
                policy_state.heartbeat = None
                return ResponseEnvelope(
                    request_id=command.request_id,
                    timestamp=utc_now_iso(),
                    ok=False,
                    backend=backend.backend_name,
                    message=f"policy start failed: {start_response.message}",
                    data={},
                )

        await _safe_send(
            ws,
            EventEnvelope(
                event="policy.status",
                timestamp=utc_now_iso(),
                backend=backend.backend_name,
                data={
                    "state": "running",
                    "task": policy_state.task,
                    "step": 0,
                    "trace_id": policy_state.trace_id,
                    "planner_step_id": policy_state.planner_step_id,
                    "canonical_action": policy_state.canonical_action,
                    "target_entity_id": policy_state.target_entity_id,
                    "target_label": policy_state.target_label,
                },
            ).to_json(),
        )

        if trace_logger is not None:
            trace_logger.write(
                {
                    "kind": "policy_start",
                    "timestamp": utc_now_iso(),
                    "task": policy_state.task,
                    "trace_id": policy_state.trace_id,
                    "planner_step_id": policy_state.planner_step_id,
                    "canonical_action": policy_state.canonical_action,
                    "target_entity_id": policy_state.target_entity_id,
                    "target_label": policy_state.target_label,
                    "hz": policy_state.hz,
                    "max_steps": policy_state.max_steps,
                    "checkpoint": config.policy_checkpoint,
                    "server_side_policy": bool(config.policy_checkpoint),
                }
            )

        return ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=True,
            backend=backend.backend_name,
            message="policy started",
            data={
                "task": policy_state.task,
                "trace_id": policy_state.trace_id,
                "planner_step_id": policy_state.planner_step_id,
                "canonical_action": policy_state.canonical_action,
                "target_entity_id": policy_state.target_entity_id,
                "target_label": policy_state.target_label,
                "hz": policy_state.hz,
                "checkpoint": config.policy_checkpoint,
                "server_side_policy": bool(config.policy_checkpoint),
            },
        )

    if command.command == "policy.stop":
        reason = str(command.payload.get("reason", "explicit_stop"))
        was_active = policy_state.active
        _prepare_stop()
        policy_state.active = False
        stop_response = await _cancel_policy_and_stop(
            policy_state,
            supervisor,
            reason,
            force_physical=True,
        )

        await _safe_send(
            ws,
            EventEnvelope(
                event="policy.status",
                timestamp=utc_now_iso(),
                backend=backend.backend_name,
                data={
                    "state": "idle" if stop_response.ok else "emergency_stop_pending",
                    "reason": reason,
                    "stop_ok": stop_response.ok,
                    "steps_completed": policy_state.step,
                    "trace_id": policy_state.trace_id,
                    "planner_step_id": policy_state.planner_step_id,
                    "canonical_action": policy_state.canonical_action,
                    "target_entity_id": policy_state.target_entity_id,
                    "target_label": policy_state.target_label,
                    "checkpoint": config.policy_checkpoint,
                },
            ).to_json(),
        )

        if trace_logger is not None:
            trace_logger.write(
                {
                    "kind": "policy_stop",
                    "timestamp": utc_now_iso(),
                    "trace_id": policy_state.trace_id,
                    "planner_step_id": policy_state.planner_step_id,
                    "canonical_action": policy_state.canonical_action,
                    "target_entity_id": policy_state.target_entity_id,
                    "target_label": policy_state.target_label,
                    "reason": reason,
                    "steps_completed": policy_state.step,
                    "stop_ok": stop_response.ok,
                }
            )

        return ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=stop_response.ok,
            backend=backend.backend_name,
            message=(
                "policy stopped"
                if stop_response.ok and was_active
                else "policy was not active"
                if stop_response.ok
                else f"policy stop failed: {stop_response.message}"
            ),
            data={
                "reason": reason,
                "stop_ok": stop_response.ok,
                "steps_completed": policy_state.step,
                "trace_id": policy_state.trace_id,
                "planner_step_id": policy_state.planner_step_id,
                "canonical_action": policy_state.canonical_action,
                "target_entity_id": policy_state.target_entity_id,
                "target_label": policy_state.target_label,
                "checkpoint": config.policy_checkpoint,
            },
        )

    if command.command == "policy.tick":
        if not policy_state.active:
            stop_suffix = await _owned_rejection_suffix("inactive-policy-tick")
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=f"policy not active{stop_suffix}",
                data={},
            )

        # Check the limit before accepting another action. Reaching the limit
        # is a terminal transition and therefore always dispatches stop.
        if policy_state.step >= policy_state.max_steps:
            policy_state.active = False
            _prepare_stop()
            stop_response = await supervisor.emergency_stop_once("max-steps")
            await _bounded_safety_send(
                ws,
                EventEnvelope(
                    event="policy.status",
                    timestamp=utc_now_iso(),
                    backend=backend.backend_name,
                    data={
                        "state": ("idle" if stop_response.ok else "emergency_stop_pending"),
                        "reason": "max_steps_reached",
                        "stop_ok": stop_response.ok,
                        "steps_completed": policy_state.step,
                        "trace_id": policy_state.trace_id,
                        "planner_step_id": policy_state.planner_step_id,
                        "canonical_action": policy_state.canonical_action,
                        "target_entity_id": policy_state.target_entity_id,
                        "target_label": policy_state.target_label,
                    },
                ).to_json(),
            )
            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=False,
                backend=backend.backend_name,
                message=(
                    "max steps reached, policy stopped"
                    if stop_response.ok
                    else f"max steps reached; stop failed: {stop_response.message}"
                ),
                data={"step": policy_state.step, "stop_ok": stop_response.ok},
            )

        # Safety-gate the action payload
        action_payload = command.payload.get("action", {})
        if isinstance(action_payload, dict):
            guard = supervisor.check_policy_action(action_payload)
            if not guard.allowed:
                policy_state.active = False
                _prepare_stop()
                stop_response = await supervisor.emergency_stop_once("policy-guard")
                await _safe_send(
                    ws,
                    EventEnvelope(
                        event="safety.policy_guard",
                        timestamp=utc_now_iso(),
                        backend=backend.backend_name,
                        data={
                            "reason": guard.reason,
                            "step": policy_state.step,
                            "response_ok": stop_response.ok,
                        },
                    ).to_json(),
                )
                return ResponseEnvelope(
                    request_id=command.request_id,
                    timestamp=utc_now_iso(),
                    ok=False,
                    backend=backend.backend_name,
                    message=f"safety guard blocked: {guard.reason}",
                    data={"step": policy_state.step},
                )

            # Apply clamped action
            clamped = guard.clamped

            # Direct joint control mode: dispatch servo.set with joint positions
            if "joint_positions" in action_payload:
                validated_joint_positions = clamped.get("joint_positions")
                if not isinstance(validated_joint_positions, dict):
                    raise RuntimeError(
                        "policy joint safety guard returned no validated joint positions"
                    )
                duration = clamped.get("duration")
                if not isinstance(duration, float):
                    raise RuntimeError("policy joint safety guard returned no validated duration")

                servo_cmd = CommandEnvelope(
                    request_id=f"{command.request_id}-servo",
                    timestamp=utc_now_iso(),
                    command="servo.set",
                    payload={
                        "joint_positions": dict(validated_joint_positions),
                        "duration": duration,
                    },
                )
                response = await supervisor.guarded_dispatch(servo_cmd)
                if response.ok:
                    policy_state.last_joint_positions = supervisor.last_joint_positions
            else:
                # Legacy walk.set mode
                walk_cmd = CommandEnvelope(
                    request_id=f"{command.request_id}-walk",
                    timestamp=utc_now_iso(),
                    command="walk.set",
                    payload={
                        "speed": clamped.get("walk_speed", 2),
                        "height": clamped.get("walk_height", 0.036),
                        "x": clamped.get("walk_x", 0.0),
                        "y": clamped.get("walk_y", 0.0),
                        "yaw": clamped.get("walk_yaw", 0.0),
                    },
                )
                response = await supervisor.guarded_dispatch(walk_cmd)

            # Apply head if present
            if response.ok and ("head_pan" in clamped or "head_tilt" in clamped):
                head_cmd = CommandEnvelope(
                    request_id=f"{command.request_id}-head",
                    timestamp=utc_now_iso(),
                    command="head.set",
                    payload={
                        "pan": clamped.get("head_pan", 0.0),
                        "tilt": clamped.get("head_tilt", 0.0),
                        "duration": 0.1,
                    },
                )
                head_response = await supervisor.guarded_dispatch(head_cmd)
                if not head_response.ok:
                    response = head_response

            if response.ok:
                policy_state.step += 1
                if policy_state.heartbeat is not None:
                    policy_state.heartbeat.record_tick()
            else:
                # guarded_dispatch already attempted an emergency stop. Keep
                # the policy terminal so a rejected backend cannot be ignored.
                policy_state.active = False

            if response.ok and policy_state.step >= policy_state.max_steps:
                policy_state.active = False
                _prepare_stop()
                max_stop_response = await supervisor.emergency_stop_once("max-steps")
                await _safe_send(
                    ws,
                    EventEnvelope(
                        event="policy.status",
                        timestamp=utc_now_iso(),
                        backend=backend.backend_name,
                        data={
                            "state": ("idle" if max_stop_response.ok else "emergency_stop_pending"),
                            "reason": "max_steps_reached",
                            "stop_ok": max_stop_response.ok,
                            "steps_completed": policy_state.step,
                            "trace_id": policy_state.trace_id,
                            "planner_step_id": policy_state.planner_step_id,
                            "canonical_action": policy_state.canonical_action,
                            "target_entity_id": policy_state.target_entity_id,
                            "target_label": policy_state.target_label,
                        },
                    ).to_json(),
                )
                if not max_stop_response.ok:
                    response = ResponseEnvelope(
                        request_id=command.request_id,
                        timestamp=utc_now_iso(),
                        ok=False,
                        backend=backend.backend_name,
                        message=(
                            "max steps reached but emergency stop failed: "
                            f"{max_stop_response.message}"
                        ),
                        data={},
                    )

            if guard.reason and trace_logger is not None:
                trace_logger.write(
                    {
                        "kind": "policy_tick_clamped",
                        "timestamp": utc_now_iso(),
                        "trace_id": policy_state.trace_id,
                        "step": policy_state.step,
                        "reason": guard.reason,
                    }
                )

            # Emit telemetry
            await _safe_send(
                ws,
                EventEnvelope(
                    event="telemetry.policy",
                    timestamp=utc_now_iso(),
                    backend=backend.backend_name,
                    data={
                        "step": policy_state.step,
                        "trace_id": policy_state.trace_id,
                        "planner_step_id": policy_state.planner_step_id,
                        "canonical_action": policy_state.canonical_action,
                        "target_entity_id": policy_state.target_entity_id,
                        "target_label": policy_state.target_label,
                        "clamped": clamped,
                        "guard_reason": guard.reason,
                    },
                ).to_json(),
            )

            if trace_logger is not None:
                trace_logger.write(
                    {
                        "kind": "policy_tick",
                        "timestamp": utc_now_iso(),
                        "trace_id": policy_state.trace_id,
                        "planner_step_id": policy_state.planner_step_id,
                        "canonical_action": policy_state.canonical_action,
                        "target_entity_id": policy_state.target_entity_id,
                        "target_label": policy_state.target_label,
                        "step": policy_state.step,
                        "action": safe_to_record(action_payload),
                        "clamped": safe_to_record(clamped),
                        "response_ok": response.ok,
                    }
                )

            return ResponseEnvelope(
                request_id=command.request_id,
                timestamp=utc_now_iso(),
                ok=response.ok,
                backend=backend.backend_name,
                message="policy tick applied" if response.ok else response.message,
                data={
                    "step": policy_state.step,
                    "trace_id": policy_state.trace_id,
                    "planner_step_id": policy_state.planner_step_id,
                    "canonical_action": policy_state.canonical_action,
                    "target_entity_id": policy_state.target_entity_id,
                    "target_label": policy_state.target_label,
                    "clamped": clamped,
                    **response.data,
                },
            )

        policy_state.active = False
        _prepare_stop()
        stop_response = await supervisor.emergency_stop_once("invalid-policy-payload")
        await _safe_send(
            ws,
            EventEnvelope(
                event="safety.policy_guard",
                timestamp=utc_now_iso(),
                backend=backend.backend_name,
                data={
                    "reason": "policy.tick requires action dict in payload",
                    "step": policy_state.step,
                    "response_ok": stop_response.ok,
                },
            ).to_json(),
        )
        return ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=False,
            backend=backend.backend_name,
            message="policy.tick requires action dict in payload",
            data={},
        )

    if command.command == "policy.status":
        return ResponseEnvelope(
            request_id=command.request_id,
            timestamp=utc_now_iso(),
            ok=True,
            backend=backend.backend_name,
            message="ok",
            data={
                "state": (
                    "emergency_stop_pending"
                    if supervisor.emergency_stop_pending
                    else "running"
                    if policy_state.active
                    else "idle"
                ),
                "active": policy_state.active,
                "emergency_stop_pending": supervisor.emergency_stop_pending,
                "owns_motion": supervisor.owns_motion,
                "task": policy_state.task,
                "trace_id": policy_state.trace_id,
                "planner_step_id": policy_state.planner_step_id,
                "canonical_action": policy_state.canonical_action,
                "target_entity_id": policy_state.target_entity_id,
                "target_label": policy_state.target_label,
                "step": policy_state.step,
                "hz": policy_state.hz,
                "checkpoint": config.policy_checkpoint,
                "server_side_policy": bool(config.policy_checkpoint),
            },
        )

    return ResponseEnvelope(
        request_id=command.request_id,
        timestamp=utc_now_iso(),
        ok=False,
        backend=backend.backend_name,
        message=f"unknown policy command: {command.command}",
        data={},
    )


async def _policy_heartbeat_pump(
    ws: ServerConnection,
    supervisor: MotionSafetySupervisor,
    policy_state: PolicyLoopState,
) -> None:
    """Monitor policy heartbeat and trigger fallback if stale."""
    while True:
        await asyncio.sleep(0.5)
        if policy_state.heartbeat is None:
            continue
        if policy_state.heartbeat.is_stale() and supervisor.owns_motion:
            policy_state.active = False
            response = await _cancel_policy_and_stop(
                policy_state,
                supervisor,
                "policy-heartbeat",
            )
            await _bounded_safety_send(
                ws,
                EventEnvelope(
                    event="safety.policy_guard",
                    timestamp=utc_now_iso(),
                    backend=supervisor.backend.backend_name,
                    data={
                        "reason": "policy_heartbeat_timeout",
                        "age_sec": policy_state.heartbeat.age_sec(),
                        "step": policy_state.step,
                        "response_ok": response.ok,
                    },
                ).to_json(),
            )
            await _bounded_safety_send(
                ws,
                EventEnvelope(
                    event="policy.status",
                    timestamp=utc_now_iso(),
                    backend=supervisor.backend.backend_name,
                    data={
                        "state": "idle" if response.ok else "emergency_stop_pending",
                        "reason": "heartbeat_timeout",
                        "stop_ok": response.ok,
                        "steps_completed": policy_state.step,
                        "trace_id": policy_state.trace_id,
                        "planner_step_id": policy_state.planner_step_id,
                        "canonical_action": policy_state.canonical_action,
                        "target_entity_id": policy_state.target_entity_id,
                        "target_label": policy_state.target_label,
                    },
                ).to_json(),
            )
            # Failed stops retain ownership. Keep monitoring the same stale
            # heartbeat so the next iteration retries until the backend acks.
            if response.ok:
                policy_state.heartbeat = None


async def _handler(
    ws: ServerConnection,
    backend_source: BackendFactory | SharedBackendRuntime,
    config: RuntimeConfig,
) -> None:
    if not _request_is_authorized(ws, config.auth_token):
        with contextlib.suppress(Exception):
            await _safe_send(ws, _json_error("authentication required"))
            await ws.close(code=1008, reason="authentication required")
        return

    profile = load_profile(config.profile_id)
    effective_deadman_timeout_sec = _effective_deadman_timeout(config, profile)
    if isinstance(backend_source, SharedBackendRuntime):
        runtime = backend_source
        owns_runtime = False
    else:
        runtime = SharedBackendRuntime(
            backend_source,
            poll_timeout_sec=max(
                0.1,
                min(effective_deadman_timeout_sec / 2.0, 2.0),
            ),
        )
        owns_runtime = True
    backend = runtime.backend
    physical_resource = _validated_physical_resource(backend, config)
    if physical_resource is not None:
        # `_handler` is also used by embedded/factory-owned runtimes.  Enforce
        # the same secret contract here, before such a runtime can connect,
        # rather than relying solely on the CLI server's bind validation.
        _validate_physical_auth_token(config.auth_token)
    if owns_runtime:
        # Identity validation is deliberately before the first transport
        # connection. A missing/mismatched hardware-unit ID must never dial.
        await runtime.start()
    supervisor = MotionSafetySupervisor(
        backend,
        profile,
        owner_id=uuid.uuid4().hex,
        resource_id=(
            physical_resource
            if physical_resource is not None
            else (
                f"bridge:{config.profile_id}:{backend.backend_name}:"
                f"{runtime.motion_resource_token}"
            )
        ),
        registry=GLOBAL_MOTION_OWNERSHIP,
    )
    limiter = CommandRateLimiter(max_commands_per_sec=config.max_commands_per_sec)
    command_queue: asyncio.Queue[CommandEnvelope] = asyncio.Queue(maxsize=config.queue_size)
    telemetry_queue = runtime.subscribe()
    runtime.register_command_queue(command_queue)
    policy_state = PolicyLoopState()
    trace_logger: TraceLogger | None = None
    if config.trace_log_path != "":
        trace_logger = TraceLogger(path=Path(config.trace_log_path))

    async def _stop_for_rejected_intent(reason: str) -> ResponseEnvelope:
        """Purge intent, terminate policy state, and dispatch a bounded stop."""
        runtime.purge_command_queues()
        policy_state.active = False
        return await _cancel_policy_and_stop(policy_state, supervisor, reason)

    await _safe_send(
        ws,
        EventEnvelope(
            event="session.hello",
            timestamp=utc_now_iso(),
            backend=backend.backend_name,
            data={
                "capabilities": backend.capabilities(),
                "safety_supervisor": supervisor.capability_report(),
                "queue_size": config.queue_size,
                "max_commands_per_sec": config.max_commands_per_sec,
                "deadman_timeout_sec": effective_deadman_timeout_sec,
                "configured_deadman_timeout_sec": config.deadman_timeout_sec,
                "trace_log_path": config.trace_log_path,
            },
        ).to_json(),
    )

    event_task = asyncio.create_task(
        _fanout_event_pump(
            ws,
            runtime,
            telemetry_queue,
            supervisor,
            policy_state,
        )
    )
    worker_task = asyncio.create_task(
        _command_worker(ws, supervisor, command_queue, trace_logger=trace_logger)
    )
    deadman_task = asyncio.create_task(
        _deadman_pump(
            ws,
            supervisor,
            policy_state,
            deadman_timeout_sec=effective_deadman_timeout_sec,
        )
    )
    policy_heartbeat_task = asyncio.create_task(
        _policy_heartbeat_pump(ws, supervisor, policy_state)
    )
    camera_response_task: asyncio.Task[None] | None = None
    try:
        async for raw_message in ws:
            request_id = "unknown"
            parsed: dict[str, object] | None = None
            received_motion_generation = supervisor.motion_generation
            try:
                parsed = json.loads(raw_message)
                if not isinstance(parsed, dict):
                    raise ValueError("payload must be a JSON object")
                request_id_value = parsed.get("request_id")
                if isinstance(request_id_value, str):
                    request_id = request_id_value
                command = parse_command(parsed)
                is_priority_stop = command.command == "policy.stop" or supervisor.is_stop_command(
                    command
                )
                try:
                    validate_command_payload(command)
                except ValueError as exc:
                    stop_response: ResponseEnvelope | None = None
                    if supervisor.owns_motion and (
                        supervisor.is_motion_command(command)
                        or supervisor.is_stop_command(command)
                        or command.command.startswith("policy.")
                    ):
                        stop_response = await _stop_for_rejected_intent(
                            "server-validation-rejection"
                        )
                    suffix = (
                        "" if stop_response is None else f"; emergency_stop_ok={stop_response.ok}"
                    )
                    await _safe_send(
                        ws,
                        _json_error(
                            f"{exc}{suffix}",
                            request_id=request_id,
                        ),
                    )
                    continue

                # Stops are safety traffic: they bypass both the limiter and
                # every ordinary command queue, synchronously fence new motion,
                # and discard intent accepted before the stop.
                if is_priority_stop:
                    supervisor.latch_stop_request()
                    runtime.purge_command_queues()
                    if command.command == "policy.stop":
                        response = await _handle_policy_command(
                            ws,
                            backend,
                            command,
                            policy_state,
                            trace_logger,
                            config,
                            profile,
                            supervisor,
                            telemetry_reader=runtime.latest_events,
                            purge_commands=runtime.purge_command_queues,
                            motion_generation_at_receive=received_motion_generation,
                        )
                    else:
                        was_active = policy_state.active
                        policy_state.active = False
                        stop_dispatch_task = asyncio.create_task(
                            supervisor.guarded_dispatch(command)
                        )
                        await _cancel_policy_loop(policy_state)
                        response = await stop_dispatch_task
                        if was_active:
                            await _safe_send(
                                ws,
                                EventEnvelope(
                                    event="policy.status",
                                    timestamp=utc_now_iso(),
                                    backend=backend.backend_name,
                                    data={
                                        "state": (
                                            "idle" if response.ok else "emergency_stop_pending"
                                        ),
                                        "reason": "manual_preempt",
                                        "steps_completed": policy_state.step,
                                        "trace_id": policy_state.trace_id,
                                        "planner_step_id": policy_state.planner_step_id,
                                        "canonical_action": policy_state.canonical_action,
                                        "target_entity_id": policy_state.target_entity_id,
                                        "target_label": policy_state.target_label,
                                        "stop_ok": response.ok,
                                    },
                                ).to_json(),
                            )
                    await _safe_send(ws, response.to_json())
                    continue

                limit_result = limiter.check()
                if not limit_result.allowed:
                    stop_response = None
                    if supervisor.owns_motion and (
                        supervisor.is_motion_command(command)
                        or command.command in {"policy.start", "policy.tick"}
                    ):
                        stop_response = await _stop_for_rejected_intent("rate-limited-motion")
                    suffix = (
                        "" if stop_response is None else f", emergency_stop_ok={stop_response.ok}"
                    )
                    await _safe_send(
                        ws,
                        _json_error(
                            "rate limit exceeded, "
                            f"retry_after_sec={limit_result.retry_after_sec:.3f}"
                            f"{suffix}",
                            request_id=request_id,
                        ),
                    )
                    continue

                # Server-level commands are answered inline without touching
                # the backend command queue. `profile.describe` returns the
                # active RobotProfile so plugins can self-configure.
                if command.command == "profile.describe":
                    requested_id = command.payload.get("id")
                    target_profile_id = (
                        requested_id
                        if isinstance(requested_id, str) and requested_id
                        else config.profile_id
                    )
                    try:
                        described_profile = load_profile(target_profile_id)
                        await _safe_send(
                            ws,
                            ResponseEnvelope(
                                request_id=command.request_id,
                                timestamp=utc_now_iso(),
                                ok=True,
                                backend=backend.backend_name,
                                message="ok",
                                data={"profile": _profile_to_jsondict(described_profile)},
                            ).to_json(),
                        )
                    except Exception as exc:
                        await _safe_send(
                            ws,
                            _json_error(
                                f"profile.describe failed: {exc}",
                                request_id=request_id,
                            ),
                        )
                    continue

                if command.command == "camera.snapshot":
                    requested_cam = command.payload.get("camera")
                    cam_name = (
                        requested_cam
                        if isinstance(requested_cam, str) and requested_cam
                        else "head"
                    )
                    physical_blocker = _physical_snapshot_blocker(supervisor)
                    if physical_blocker is not None:
                        await _safe_send(
                            ws,
                            _json_error(physical_blocker, request_id=request_id),
                        )
                        continue
                    if camera_response_task is not None:
                        if not camera_response_task.done():
                            await _safe_send(
                                ws,
                                _json_error(
                                    "camera.snapshot failed: a snapshot request is already active",
                                    request_id=request_id,
                                ),
                            )
                            continue
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            camera_response_task.result()
                    camera_response_task = asyncio.create_task(
                        _send_camera_snapshot_response(
                            ws,
                            runtime,
                            backend,
                            supervisor,
                            config,
                            command,
                            cam_name,
                        )
                    )
                    continue

                # Policy commands are handled directly (not queued)
                if command.command.startswith("policy."):
                    # A policy start supersedes manual intent. Purge all queues
                    # and, when this session already owns motion, fence any
                    # in-flight manual command with a confirmed stop first.
                    if command.command == "policy.start":
                        runtime.purge_command_queues()
                        if supervisor.owns_motion and not policy_state.active:
                            stop_response = await _cancel_policy_and_stop(
                                policy_state,
                                supervisor,
                                "policy-preempt",
                            )
                            if not stop_response.ok:
                                await _safe_send(
                                    ws,
                                    _json_error(
                                        "policy start blocked because backend stop failed",
                                        request_id=request_id,
                                    ),
                                )
                                continue
                            # This command deliberately caused the stop above,
                            # so its new intent belongs to the post-stop generation.
                            received_motion_generation = supervisor.motion_generation

                    response = await _handle_policy_command(
                        ws,
                        backend,
                        command,
                        policy_state,
                        trace_logger,
                        config,
                        profile,
                        supervisor,
                        telemetry_reader=runtime.latest_events,
                        purge_commands=runtime.purge_command_queues,
                        motion_generation_at_receive=received_motion_generation,
                    )
                    await _safe_send(ws, response.to_json())
                    continue

                # Manual commands preempt policy mode
                if policy_state.active and command.command in {
                    "walk.set",
                    "walk.command",
                    "head.set",
                    "action.play",
                    "servo.set",
                    "asimov.mode",
                    "asimov.velocity",
                    "asimov.trajectory",
                }:
                    runtime.purge_command_queues()
                    policy_state.active = False
                    stop_response = await _cancel_policy_and_stop(
                        policy_state,
                        supervisor,
                        "manual-preempt",
                    )
                    await _safe_send(
                        ws,
                        EventEnvelope(
                            event="policy.status",
                            timestamp=utc_now_iso(),
                            backend=backend.backend_name,
                            data={
                                "state": ("idle" if stop_response.ok else "emergency_stop_pending"),
                                "reason": "manual_preempt",
                                "steps_completed": policy_state.step,
                                "trace_id": policy_state.trace_id,
                                "planner_step_id": policy_state.planner_step_id,
                                "canonical_action": policy_state.canonical_action,
                                "target_entity_id": policy_state.target_entity_id,
                                "target_label": policy_state.target_label,
                                "stop_ok": stop_response.ok,
                            },
                        ).to_json(),
                    )
                    if not stop_response.ok:
                        await _safe_send(
                            ws,
                            _json_error(
                                "manual preempt blocked because backend stop failed",
                                request_id=request_id,
                            ),
                        )
                        continue
                    # Manual preemption is part of processing this command, so
                    # accept it against the generation created by that stop.
                    received_motion_generation = supervisor.motion_generation

                if command.preempt:
                    while not command_queue.empty():
                        _ = command_queue.get_nowait()
                        command_queue.task_done()
                is_stop_command = (
                    command.command == "walk.command"
                    and command.payload.get("action") in {"stop", "disable", "disable_control"}
                ) or (
                    command.command == "asimov.mode"
                    and str(command.payload.get("mode", "")).upper() == "DAMP"
                )
                is_fresh_manual_motion = (
                    command.command
                    in {
                        "walk.set",
                        "walk.command",
                        "head.set",
                        "action.play",
                        "servo.set",
                        "asimov.mode",
                        "asimov.velocity",
                        "asimov.trajectory",
                    }
                    and not is_stop_command
                )
                if is_fresh_manual_motion and _policy_loop_unterminated(policy_state):
                    await _safe_send(
                        ws,
                        _json_error(
                            "motion blocked: prior policy cancellation has not completed",
                            request_id=request_id,
                        ),
                    )
                    continue
                if is_fresh_manual_motion and not supervisor.accept_fresh_motion(
                    received_motion_generation
                ):
                    # This message is a fresh deliberate manual intent. Rearm
                    # only here, after any policy-preempt stop; queued commands
                    # that predate an external stop remain revoked and fail.
                    await _safe_send(
                        ws,
                        _json_error(
                            "motion blocked: intent became stale, a stop is pending, "
                            "or another session owns the resource",
                            request_id=request_id,
                        ),
                    )
                    continue
                try:
                    command_queue.put_nowait(command)
                except asyncio.QueueFull:
                    stop_response = None
                    if supervisor.owns_motion and supervisor.is_motion_command(command):
                        stop_response = await _stop_for_rejected_intent("motion-queue-full")
                    suffix = (
                        "" if stop_response is None else f"; emergency_stop_ok={stop_response.ok}"
                    )
                    await _safe_send(
                        ws,
                        _json_error(
                            f"command queue is full{suffix}",
                            request_id=request_id,
                        ),
                    )
                    continue
                if trace_logger is not None:
                    trace_logger.write(
                        {
                            "kind": "command_enqueued",
                            "timestamp": utc_now_iso(),
                            "backend": backend.backend_name,
                            "request_id": command.request_id,
                            "command": command.command,
                            "preempt": command.preempt,
                            "payload": safe_to_record(command.payload),
                            "queue_size": command_queue.qsize(),
                        }
                    )
            except Exception as exc:
                raw_command = parsed.get("command") if parsed is not None else None
                stop_response = None
                if (
                    supervisor.owns_motion
                    and isinstance(raw_command, str)
                    and MotionSafetySupervisor.is_motion_name(raw_command)
                ):
                    stop_response = await _stop_for_rejected_intent("malformed-owned-motion")
                suffix = "" if stop_response is None else f"; emergency_stop_ok={stop_response.ok}"
                await _safe_send(
                    ws,
                    _json_error(f"{exc}{suffix}", request_id=request_id),
                )
    except ConnectionClosed:
        pass
    finally:
        # Fence first.  Cancelling a slow policy or pump is an awaitable cleanup
        # operation and must never delay the process-wide stop latch.
        disconnect_stop_task: asyncio.Task[ResponseEnvelope] | None = None
        if supervisor.owns_motion or supervisor.emergency_stop_pending:
            supervisor.latch_stop_request()
            runtime.purge_command_queues()
            disconnect_stop_task = asyncio.create_task(
                supervisor.emergency_stop_once("disconnect-1")
            )
        event_task.cancel()
        worker_task.cancel()
        deadman_task.cancel()
        policy_heartbeat_task.cancel()
        if camera_response_task is not None:
            camera_response_task.cancel()
        policy_state.active = False
        await _cancel_policy_loop(policy_state)
        # Ownership, not the policy flag, decides whether disconnect owes a
        # stop. Max-step and prior safety transitions may already be inactive.
        if disconnect_stop_task is not None:
            stop_response = await disconnect_stop_task
            if not stop_response.ok:
                for attempt in range(2, 4):
                    stop_response = await supervisor.emergency_stop_once(f"disconnect-{attempt}")
                    if stop_response.ok:
                        break
                    await asyncio.sleep(0.1)
        background_tasks: list[asyncio.Task[None]] = [
            event_task,
            worker_task,
            deadman_task,
            policy_heartbeat_task,
        ]
        if camera_response_task is not None:
            background_tasks.append(camera_response_task)
        await asyncio.gather(*background_tasks, return_exceptions=True)
        runtime.unregister_command_queue(command_queue)
        runtime.unsubscribe(telemetry_queue)
        if owns_runtime:
            await runtime.close()


async def _run_server(host: str, port: int, backend: str, config: RuntimeConfig) -> None:
    _validate_bind_security(
        host,
        backend,
        config.auth_token,
        config.physical_resource_id,
    )
    profile = load_profile(config.profile_id)
    effective_deadman_timeout_sec = _effective_deadman_timeout(config, profile)
    backend_factory = _build_backend_factory(backend, config)
    runtime = SharedBackendRuntime(
        backend_factory,
        poll_timeout_sec=max(
            0.1,
            min(effective_deadman_timeout_sec / 2.0, 2.0),
        ),
    )
    _validated_physical_resource(runtime.backend, config)
    await runtime.start()
    try:
        async with serve(
            lambda ws: _handler(ws, runtime, config),
            host=host,
            port=port,
        ):
            logger.info(
                "bridge websocket listening on ws://%s:%d backend=%s "
                "queue_size=%d max_commands_per_sec=%d deadman_timeout_sec=%s",
                host,
                port,
                backend,
                config.queue_size,
                config.max_commands_per_sec,
                effective_deadman_timeout_sec,
            )
            await asyncio.Future()
    finally:
        await runtime.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AiNex unified websocket bridge")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="listen host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9100,
        help="listen port",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=[
            "mock",
            "mujoco",
            "ros",
            "ros_real",
            "ros_sim",
            "isaac",
            "ainex_remote",
            "ros_remote",
            "asimov_mock",
            "asimov_remote",
            "asimov_mujoco",
        ],
        default="mock",
        help="target backend adapter",
    )
    parser.add_argument(
        "--rosbridge-host",
        type=str,
        default="192.168.1.218",
        help="rosbridge host for --backend ainex_remote (default: 192.168.1.218)",
    )
    parser.add_argument(
        "--rosbridge-port",
        type=int,
        default=9090,
        help="rosbridge port for --backend ainex_remote (default: 9090)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="hiwonder-ainex",
        help="robot profile id (resolves URDF, calibration, safety from "
        "packages/research/robot/profiles/<id>/)",
    )
    parser.add_argument(
        "--asimov-livekit-url",
        type=str,
        default="",
        help="ASIMOV LiveKit websocket URL for --backend asimov_remote",
    )
    parser.add_argument(
        "--asimov-livekit-token",
        type=str,
        default="",
        help="ASIMOV LiveKit access token for --backend asimov_remote",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=256,
        help="max queued commands per websocket session",
    )
    parser.add_argument(
        "--max-commands-per-sec",
        type=int,
        default=30,
        help="rate limit for inbound commands per session",
    )
    parser.add_argument(
        "--deadman-timeout-sec",
        type=float,
        default=1.0,
        help="auto-stop timeout if no heartbeat command is received",
    )
    parser.add_argument(
        "--trace-log-path",
        type=str,
        default="",
        help="optional JSONL path for command/response trace logging",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default="",
        help=(
            "optional text-conditioned checkpoint directory. When set, "
            "policy.start runs the checkpoint server-side; otherwise clients "
            "must send policy.tick actions."
        ),
    )
    parser.add_argument(
        "--mujoco-target-x",
        type=float,
        default=2.0,
        help="MuJoCo backend: target ball X position (m, default 2.0)",
    )
    parser.add_argument(
        "--mujoco-target-y",
        type=float,
        default=0.0,
        help="MuJoCo backend: target ball Y position (m, default 0.0)",
    )
    parser.add_argument(
        "--mujoco-target-z",
        type=float,
        default=0.05,
        help="MuJoCo backend: target ball Z position (m, default 0.05)",
    )
    parser.add_argument(
        "--camera-device",
        type=int,
        default=-1,
        help="v4l2 device index for an external camera (Obsbot etc.). "
        "When >=0, `camera.snapshot` reads from this device instead "
        "of the backend (useful with --backend ros_real).",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="External camera capture width (default 640)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="External camera capture height (default 480)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="optional JSON config path (bridge/config/default_bridge_config.json style)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_obj = _load_config_file(args.config)
    config = _coerce_runtime_config(args, config_obj)
    asyncio.run(_run_server(host=args.host, port=args.port, backend=args.backend, config=config))


if __name__ == "__main__":
    main()
