"""Session-level safety controls for bridge command handling."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from eliza_robot.bridge.backends.base import (
    BridgeBackend,
    _supervised_motion_dispatch_authority,
    canonical_physical_resource_id,
)
from eliza_robot.bridge.protocol import (
    CommandEnvelope,
    EventEnvelope,
    ResponseEnvelope,
    utc_now_iso,
)
from eliza_robot.bridge.validation import validate_command_payload
from eliza_robot.profiles.schema import RobotProfile

logger = logging.getLogger(__name__)
_PROCESS_STOP_RETRY_TASKS: set[asyncio.Task[None]] = set()


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_sec: float


class CommandRateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_commands_per_sec: int) -> None:
        if (
            isinstance(max_commands_per_sec, bool)
            or not isinstance(max_commands_per_sec, int)
            or max_commands_per_sec <= 0
            or max_commands_per_sec > 10_000
        ):
            raise ValueError("max_commands_per_sec must be an integer in 1..10000")
        self._limit = max_commands_per_sec
        self._window_sec = 1.0
        self._timestamps: deque[float] = deque()

    def check(self) -> RateLimitResult:
        now = time.monotonic()
        while self._timestamps and (now - self._timestamps[0]) > self._window_sec:
            self._timestamps.popleft()

        if len(self._timestamps) >= self._limit:
            retry_after_sec = self._window_sec - (now - self._timestamps[0])
            return RateLimitResult(allowed=False, retry_after_sec=max(0.0, retry_after_sec))

        self._timestamps.append(now)
        return RateLimitResult(allowed=True, retry_after_sec=0.0)


def is_deadman_heartbeat_command(command: CommandEnvelope) -> bool:
    """Commands that count as keepalive movement/control activity."""
    return command.command in {
        "walk.set",
        "walk.command",
        "head.set",
        "action.play",
        "servo.set",
        "asimov.mode",
        "asimov.velocity",
        "asimov.trajectory",
        "policy.tick",
    }


# ---------------------------------------------------------------------------
# Policy motion-bound safety checks
# ---------------------------------------------------------------------------

# Maximum absolute deltas per policy tick (prevents runaway commands)
POLICY_WALK_X_MAX = 0.05
POLICY_WALK_Y_MAX = 0.05
POLICY_WALK_YAW_MAX = 10.0
POLICY_WALK_HEIGHT_MIN = 0.015
POLICY_WALK_HEIGHT_MAX = 0.06
POLICY_WALK_SPEED_MIN = 1
POLICY_WALK_SPEED_MAX = 4
POLICY_HEAD_PAN_MAX = 1.5  # radians
POLICY_HEAD_TILT_MAX = 1.0  # radians
POLICY_JOINT_DURATION_DEFAULT = 0.1  # seconds
POLICY_JOINT_DURATION_MAX = 5.0  # seconds; matches servo.set validation


@dataclass
class PolicyGuardResult:
    """Result of a policy motion-bound check."""

    allowed: bool
    reason: str = ""
    clamped: dict[str, Any] = field(default_factory=dict)


def _validate_policy_joint_positions(
    raw_positions: Any,
    *,
    profile: RobotProfile | None,
    previous_joint_positions: Mapping[str, float] | None,
    duration_sec: float,
) -> tuple[dict[str, float], list[str]]:
    """Validate one direct-joint policy command against the active profile.

    Direct ``policy.tick`` commands use named joints only. Rejecting alternate
    list/servo-id shapes here prevents callers from bypassing profile joint
    names and limits. The first command is compared with each joint's home pose;
    subsequent commands are compared with the last successfully dispatched
    target supplied by the policy loop.
    """
    if profile is None:
        return {}, ["joint_positions requires an active robot profile"]
    if not isinstance(raw_positions, dict) or not raw_positions:
        return {}, ["joint_positions must be a non-empty object keyed by joint name"]

    specs = {joint.name: joint for joint in profile.kinematics.joints}
    max_delta = float(profile.control.max_joint_delta_rad_per_step)
    previous = previous_joint_positions or {}
    validated: dict[str, float] = {}
    invalid: list[str] = []

    for raw_name, raw_value in raw_positions.items():
        if not isinstance(raw_name, str) or raw_name not in specs:
            invalid.append(f"unknown joint {raw_name!r}")
            continue

        joint = specs[raw_name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            invalid.append(f"joint_positions[{raw_name!r}]=non-numeric")
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            invalid.append(f"joint_positions[{raw_name!r}]=non-numeric")
            continue
        if not math.isfinite(value):
            invalid.append(f"joint_positions[{raw_name!r}]={value}")
            continue
        if value < joint.lower_rad or value > joint.upper_rad:
            invalid.append(
                f"joint_positions[{raw_name!r}]={value} outside profile range "
                f"[{joint.lower_rad}, {joint.upper_rad}]"
            )
            continue

        try:
            prior_value = float(previous.get(raw_name, joint.home_rad))
        except (TypeError, ValueError, OverflowError):
            invalid.append(f"previous joint target {raw_name!r} is non-numeric")
            continue
        if not math.isfinite(prior_value):
            invalid.append(f"previous joint target {raw_name!r} is non-finite")
            continue
        delta = abs(value - prior_value)
        if delta > max_delta + 1.0e-9:
            invalid.append(
                f"joint_positions[{raw_name!r}] delta {delta} exceeds profile max "
                f"{max_delta} from previous target {prior_value}"
            )
            continue
        effective_velocity = delta / duration_sec
        control_velocity_limit = float(profile.control.max_joint_delta_rad_per_step) * float(
            profile.control.rate_hz
        )
        velocity_limit = min(
            float(joint.velocity_max_rad_s),
            control_velocity_limit,
        )
        if effective_velocity > velocity_limit + 1.0e-9:
            invalid.append(
                f"joint_positions[{raw_name!r}] velocity {effective_velocity} rad/s "
                f"exceeds effective profile/control limit {velocity_limit} rad/s"
            )
            continue

        validated[raw_name] = value

    if invalid:
        return {}, invalid
    return validated, []


def check_policy_motion_bounds(
    action: dict[str, Any],
    *,
    profile: RobotProfile | None = None,
    previous_joint_positions: Mapping[str, float] | None = None,
) -> PolicyGuardResult:
    """Check and clamp a policy action chunk against hard safety limits.

    Returns a PolicyGuardResult with the clamped values. If any value was
    out of bounds, ``allowed`` is still True but ``reason`` describes what
    was clamped. Direct joint commands are stricter: malformed, unknown,
    non-finite, profile-limit, and per-step-delta violations are rejected.
    If the action is fundamentally invalid, ``allowed`` is False.
    """
    clamped: dict[str, Any] = {}
    reasons: list[str] = []
    invalid: list[str] = []

    def _num(name: str, default: float) -> float:
        """Parse a float field; flag non-finite/garbage and substitute a safe 0."""
        raw = action.get(name, default)
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            invalid.append(f"{name}=non-numeric")
            return 0.0
        try:
            v = float(raw)
        except (TypeError, ValueError, OverflowError):
            invalid.append(f"{name}=non-numeric")
            return 0.0
        if not math.isfinite(v):
            invalid.append(f"{name}={v}")
            return 0.0
        return v

    # Walk parameters. A diverged policy commonly emits NaN/inf — these MUST be
    # rejected (allowed=False), not silently clamped, since abs(nan) > MAX is
    # False and a raw NaN would otherwise pass straight through to the robot.
    walk_x = _num("walk_x", 0.0)
    walk_y = _num("walk_y", 0.0)
    walk_yaw = _num("walk_yaw", 0.0)
    walk_height = _num("walk_height", 0.036)  # 0.0 if invalid -> clamped to MIN below
    raw_walk_speed = action.get("walk_speed", 2)
    if isinstance(raw_walk_speed, bool) or not isinstance(raw_walk_speed, int):
        invalid.append("walk_speed=non-integer")
        walk_speed = POLICY_WALK_SPEED_MIN
    else:
        walk_speed = raw_walk_speed

    if abs(walk_x) > POLICY_WALK_X_MAX:
        reasons.append(
            f"walk_x clamped {walk_x:.4f}->{_clamp(walk_x, -POLICY_WALK_X_MAX, POLICY_WALK_X_MAX):.4f}"
        )
        walk_x = _clamp(walk_x, -POLICY_WALK_X_MAX, POLICY_WALK_X_MAX)
    if abs(walk_y) > POLICY_WALK_Y_MAX:
        reasons.append(
            f"walk_y clamped {walk_y:.4f}->{_clamp(walk_y, -POLICY_WALK_Y_MAX, POLICY_WALK_Y_MAX):.4f}"
        )
        walk_y = _clamp(walk_y, -POLICY_WALK_Y_MAX, POLICY_WALK_Y_MAX)
    if abs(walk_yaw) > POLICY_WALK_YAW_MAX:
        reasons.append(
            f"walk_yaw clamped {walk_yaw:.2f}->{_clamp(walk_yaw, -POLICY_WALK_YAW_MAX, POLICY_WALK_YAW_MAX):.2f}"
        )
        walk_yaw = _clamp(walk_yaw, -POLICY_WALK_YAW_MAX, POLICY_WALK_YAW_MAX)
    if walk_height < POLICY_WALK_HEIGHT_MIN or walk_height > POLICY_WALK_HEIGHT_MAX:
        reasons.append(f"walk_height clamped {walk_height:.4f}")
        walk_height = _clamp(walk_height, POLICY_WALK_HEIGHT_MIN, POLICY_WALK_HEIGHT_MAX)
    if walk_speed < POLICY_WALK_SPEED_MIN or walk_speed > POLICY_WALK_SPEED_MAX:
        reasons.append(f"walk_speed clamped {walk_speed}")
        walk_speed = max(POLICY_WALK_SPEED_MIN, min(POLICY_WALK_SPEED_MAX, walk_speed))

    clamped["walk_x"] = walk_x
    clamped["walk_y"] = walk_y
    clamped["walk_yaw"] = walk_yaw
    clamped["walk_height"] = walk_height
    clamped["walk_speed"] = walk_speed

    # Head parameters (optional)
    if "head_pan" in action:
        head_pan = _num("head_pan", 0.0)
        if abs(head_pan) > POLICY_HEAD_PAN_MAX:
            reasons.append(f"head_pan clamped {head_pan:.3f}")
            head_pan = _clamp(head_pan, -POLICY_HEAD_PAN_MAX, POLICY_HEAD_PAN_MAX)
        clamped["head_pan"] = head_pan
    if "head_tilt" in action:
        head_tilt = _num("head_tilt", 0.0)
        if abs(head_tilt) > POLICY_HEAD_TILT_MAX:
            reasons.append(f"head_tilt clamped {head_tilt:.3f}")
            head_tilt = _clamp(head_tilt, -POLICY_HEAD_TILT_MAX, POLICY_HEAD_TILT_MAX)
        clamped["head_tilt"] = head_tilt

    if "joint_positions" in action:
        raw_duration = action.get(
            "duration",
            POLICY_JOINT_DURATION_DEFAULT,
        )
        duration: float | None = None
        if isinstance(raw_duration, bool) or not isinstance(
            raw_duration,
            int | float,
        ):
            invalid.append("duration=non-numeric")
        else:
            try:
                candidate_duration = float(raw_duration)
            except OverflowError:
                candidate_duration = math.inf
            if (
                not math.isfinite(candidate_duration)
                or candidate_duration <= 0.0
                or candidate_duration > POLICY_JOINT_DURATION_MAX
            ):
                invalid.append(
                    f"duration must be finite and in (0, {POLICY_JOINT_DURATION_MAX}] seconds"
                )
            else:
                duration = candidate_duration
                clamped["duration"] = duration
        joint_positions, joint_invalid = _validate_policy_joint_positions(
            action["joint_positions"],
            profile=profile,
            previous_joint_positions=previous_joint_positions,
            duration_sec=(duration if duration is not None else POLICY_JOINT_DURATION_DEFAULT),
        )
        clamped["joint_positions"] = joint_positions
        invalid.extend(joint_invalid)

    # A fundamentally invalid action (NaN/inf/garbage) is rejected: allowed=False
    # and the clamped payload is forced to the safe neutral pose so a caller that
    # ignores `allowed` still sends nothing dangerous.
    if invalid:
        if "joint_positions" in action:
            clamped["joint_positions"] = {}
            clamped.pop("duration", None)
        return PolicyGuardResult(
            allowed=False,
            reason="invalid action rejected: "
            + ", ".join(invalid)
            + ("; " + "; ".join(reasons) if reasons else ""),
            clamped=clamped,
        )

    return PolicyGuardResult(
        allowed=True,
        reason="; ".join(reasons) if reasons else "",
        clamped=clamped,
    )


def _clamp(value: float | int, lo: float | int, hi: float | int) -> float | int:
    if isinstance(value, int) and isinstance(lo, int) and isinstance(hi, int):
        return max(lo, min(hi, value))
    return max(float(lo), min(float(hi), float(value)))


@dataclass
class PolicyHeartbeatMonitor:
    """Tracks policy tick heartbeats and detects stale policy loops."""

    timeout_sec: float = 2.0
    _last_tick: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_sec, bool)
            or not isinstance(self.timeout_sec, int | float)
            or not math.isfinite(float(self.timeout_sec))
            or self.timeout_sec <= 0.0
            or self.timeout_sec > 3_600.0
        ):
            raise ValueError("timeout_sec must be finite and in (0, 3600]")

    def record_tick(self) -> None:
        self._last_tick = time.monotonic()

    def is_stale(self) -> bool:
        if self._last_tick == 0.0:
            return False  # Never started
        return (time.monotonic() - self._last_tick) > self.timeout_sec

    def age_sec(self) -> float:
        if self._last_tick == 0.0:
            return 0.0
        return time.monotonic() - self._last_tick


# ---------------------------------------------------------------------------
# Process-wide motion ownership and guarded backend dispatch
# ---------------------------------------------------------------------------


@dataclass
class _MotionResource:
    owner_id: str | None = None
    emergency_stop_pending: bool = False
    stop_confirmed: bool = False
    stop_in_progress: bool = False
    motion_generation: int = 0
    revoked_owners: set[str] = field(default_factory=set)
    revoked_by: dict[str, str] = field(default_factory=dict)
    command_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stop_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MotionOwnershipRegistry:
    """Atomic process-wide ownership for one physical/simulated robot.

    A websocket session owns motion as a whole, rather than owning only its
    local backend object. This prevents two clients connected to the same
    bridge process from interleaving commands through separate per-session
    queues. A stop is deliberately allowed from any session and clears the
    current owner only after the backend confirms that it succeeded.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._resources: dict[str, _MotionResource] = {}

    def _resource(self, resource_id: str) -> _MotionResource:
        with self._guard:
            resource = self._resources.get(resource_id)
            if resource is None:
                resource = _MotionResource()
                self._resources[resource_id] = resource
            return resource

    def acquire(
        self,
        resource_id: str,
        owner_id: str,
    ) -> bool:
        resource = self._resource(resource_id)
        with self._guard:
            if (
                resource.emergency_stop_pending
                or resource.stop_in_progress
                or owner_id in resource.revoked_owners
            ):
                return False
            if resource.owner_id not in {None, owner_id}:
                return False
            resource.owner_id = owner_id
            resource.stop_confirmed = False
            return True

    def acquire_fresh(
        self,
        resource_id: str,
        owner_id: str,
        *,
        expected_generation: int,
    ) -> tuple[bool, str | None]:
        """Atomically accept intent received in the current stop generation.

        A queue purge cannot remove a command that a websocket handler has
        parsed but not enqueued yet.  Comparing the generation and acquiring
        ownership in the same critical section closes that race: a successful
        stop makes the pending message stale, while a stop that begins just
        after this method sees and revokes the newly recorded owner.
        """
        resource = self._resource(resource_id)
        with self._guard:
            if resource.motion_generation != expected_generation:
                return False, None
            if resource.emergency_stop_pending or resource.stop_in_progress:
                return False, None
            if resource.owner_id not in {None, owner_id}:
                return False, None
            if owner_id in resource.revoked_owners and resource.command_lock.locked():
                # A stop was acknowledged while an older backend call was
                # still unwinding. Do not restore the same owner identity until
                # that stale call has released the process-wide command fence.
                return False, None
            revoked_by = resource.revoked_by.pop(owner_id, None)
            resource.revoked_owners.discard(owner_id)
            resource.owner_id = owner_id
            resource.stop_confirmed = False
            return True, revoked_by

    def rearm(self, resource_id: str, owner_id: str) -> str | None:
        """Clear a stop revocation and return the stop caller, if any."""
        resource = self._resource(resource_id)
        with self._guard:
            revoked_by = resource.revoked_by.pop(owner_id, None)
            resource.revoked_owners.discard(owner_id)
            return revoked_by

    def is_revoked(self, resource_id: str, owner_id: str) -> bool:
        resource = self._resource(resource_id)
        with self._guard:
            return owner_id in resource.revoked_owners

    def emergency_stop_pending(self, resource_id: str) -> bool:
        resource = self._resource(resource_id)
        with self._guard:
            return resource.emergency_stop_pending

    def stop_in_progress(self, resource_id: str) -> bool:
        resource = self._resource(resource_id)
        with self._guard:
            return resource.stop_in_progress

    def motion_generation(self, resource_id: str) -> int:
        resource = self._resource(resource_id)
        with self._guard:
            return resource.motion_generation

    def record_motion_success(self, resource_id: str) -> int:
        resource = self._resource(resource_id)
        with self._guard:
            resource.motion_generation += 1
            return resource.motion_generation

    def begin_stop(self, resource_id: str, *, force: bool = False) -> bool:
        """Latch a stop attempt and exclude new motion acquisition."""
        resource = self._resource(resource_id)
        with self._guard:
            if not force and resource.stop_confirmed and resource.owner_id is None:
                return False
            resource.stop_in_progress = True
            return True

    def record_stop_result(
        self,
        resource_id: str,
        caller_owner_id: str,
        *,
        ok: bool,
    ) -> int:
        """Atomically latch failures or release/revoke on a confirmed stop."""
        resource = self._resource(resource_id)
        with self._guard:
            if not ok:
                resource.emergency_stop_pending = True
                resource.stop_confirmed = False
                resource.stop_in_progress = False
                return resource.motion_generation
            previous_owner = resource.owner_id
            if previous_owner is not None:
                # Every stop invalidates all intent accepted before it.  This
                # includes a stop issued by the owner itself: commands already
                # sitting in a websocket queue must not silently reacquire the
                # resource after the actuator has acknowledged the stop.  A
                # later, freshly received command explicitly rearms its owner.
                resource.revoked_owners.add(previous_owner)
                resource.revoked_by[previous_owner] = caller_owner_id
            resource.owner_id = None
            resource.emergency_stop_pending = False
            resource.stop_confirmed = True
            resource.stop_in_progress = False
            resource.motion_generation += 1
            return resource.motion_generation

    def is_owner(self, resource_id: str, owner_id: str) -> bool:
        resource = self._resource(resource_id)
        with self._guard:
            return resource.owner_id == owner_id

    def owner(self, resource_id: str) -> str | None:
        resource = self._resource(resource_id)
        with self._guard:
            return resource.owner_id

    def command_lock(self, resource_id: str) -> asyncio.Lock:
        return self._resource(resource_id).command_lock

    def stop_lock(self, resource_id: str) -> asyncio.Lock:
        return self._resource(resource_id).stop_lock

    def stop_required(self, resource_id: str) -> bool:
        """Return whether a latched caller still owes an actuator stop."""
        resource = self._resource(resource_id)
        with self._guard:
            return not (resource.stop_confirmed and resource.owner_id is None)


GLOBAL_MOTION_OWNERSHIP = MotionOwnershipRegistry()


class MotionSafetySupervisor:
    """Single fail-closed boundary for every backend motion dispatch."""

    _MOTION_COMMANDS = {
        "walk.set",
        "head.set",
        "action.play",
        "servo.set",
        "asimov.mode",
        "asimov.velocity",
        "asimov.trajectory",
    }
    _REQUIRED_AUTONOMOUS_TELEMETRY = (
        "imu_roll",
        "imu_pitch",
        "battery_mv",
    )
    # RobotProfile does not yet carry either geometry contract.  Keep these
    # load-bearing omissions in one place so manual physical commands and
    # autonomous policy startup fail closed for the same reason.
    _MISSING_HARD_ENVELOPE_PROFILE_CHECKS = ("workspace", "self_collision")

    def __init__(
        self,
        backend: BridgeBackend,
        profile: RobotProfile,
        *,
        owner_id: str,
        resource_id: str | None = None,
        registry: MotionOwnershipRegistry | None = None,
    ) -> None:
        self.backend = backend
        self.profile = profile
        self.owner_id = owner_id
        physical_resources = backend.physical_motion_resources()
        if not isinstance(physical_resources, tuple) or any(
            not isinstance(item, str) for item in physical_resources
        ):
            raise ValueError("physical_motion_resources must return a tuple of strings")
        if len(physical_resources) > 1:
            raise ValueError(
                "one safety supervisor may own exactly one physical actuator resource"
            )
        if physical_resources:
            physical_resource = physical_resources[0]
            prefix = "physical:"
            if not physical_resource.startswith(prefix):
                raise ValueError("physical actuator resource must use the physical: namespace")
            if canonical_physical_resource_id(physical_resource.removeprefix(prefix)) != physical_resource:
                raise ValueError("physical actuator resource is not canonical")
            if resource_id is not None and resource_id != physical_resource:
                raise ValueError(
                    "physical resource override must exactly match the backend actuator identity"
                )
            self.resource_id = physical_resource
        else:
            # Nonphysical direct inference/tests get object isolation unless
            # callers deliberately share one process-local resource id.
            self.resource_id = resource_id or (
                f"{backend.backend_name}:{profile.id}:{id(backend)}"
            )
        self.registry = registry or GLOBAL_MOTION_OWNERSHIP
        self._last_activity_monotonic = 0.0
        self._ownership_started_monotonic = 0.0
        self._last_safety_telemetry_monotonic = 0.0
        self._joint_pose_generation = 0
        self._resource_generation_seen = self.registry.motion_generation(self.resource_id)
        self._stop_retry_task: asyncio.Task[None] | None = None
        self._home = {joint.name: float(joint.home_rad) for joint in profile.kinematics.joints}
        safety = self._safety_capabilities()
        # A connect-time pose belongs to the process-shared motion resource,
        # not to the first websocket that happens to connect. Multiple passive
        # sessions may observe it, but the first successful pose-changing
        # command advances the resource generation and invalidates it for all
        # current and future supervisors.
        known_pose_claim = (
            safety.get("known_joint_pose_at_connect") is True
            and self.registry.motion_generation(self.resource_id) == 0
        )
        self._trusted_joint_positions: dict[str, float] | None = (
            dict(self._home) if known_pose_claim else None
        )

    @property
    def owns_motion(self) -> bool:
        return self.registry.is_owner(self.resource_id, self.owner_id)

    @property
    def emergency_stop_pending(self) -> bool:
        return self.registry.emergency_stop_pending(self.resource_id)

    @property
    def motion_revoked(self) -> bool:
        return self.registry.is_revoked(self.resource_id, self.owner_id)

    @property
    def last_joint_positions(self) -> dict[str, float]:
        self._invalidate_stale_pose()
        return dict(self._trusted_joint_positions or {})

    def _invalidate_stale_pose(self) -> None:
        current_generation = self.registry.motion_generation(self.resource_id)
        if self._resource_generation_seen != current_generation:
            self._trusted_joint_positions = None
            self._resource_generation_seen = current_generation

    def acquire_motion(self) -> bool:
        if self.emergency_stop_pending:
            return False
        already_owned = self.owns_motion
        acquired = self.registry.acquire(self.resource_id, self.owner_id)
        if acquired and not already_owned:
            now = time.monotonic()
            self._ownership_started_monotonic = now
            self._last_activity_monotonic = now
        return acquired

    @property
    def motion_generation(self) -> int:
        """Return the process-wide stop/motion generation for this resource."""
        return self.registry.motion_generation(self.resource_id)

    def accept_fresh_motion(self, expected_generation: int) -> bool:
        """Atomically rearm and acquire a newly received motion command."""
        already_owned = self.owns_motion
        acquired, revoked_by = self.registry.acquire_fresh(
            self.resource_id,
            self.owner_id,
            expected_generation=expected_generation,
        )
        if (
            acquired
            and revoked_by is not None
            and (
                revoked_by != self.owner_id
                or self._safety_capabilities().get("pose_remains_trusted_after_stop") is not True
            )
        ):
            self._trusted_joint_positions = None
            self._resource_generation_seen = self.registry.motion_generation(self.resource_id)
        if acquired and not already_owned:
            now = time.monotonic()
            self._ownership_started_monotonic = now
            self._last_activity_monotonic = now
        return acquired

    def rearm_motion(self) -> None:
        """Rearm this owner for a new deliberate command after external stop."""
        revoked_by = self.registry.rearm(self.resource_id, self.owner_id)
        if revoked_by is not None and (
            revoked_by != self.owner_id
            or self._safety_capabilities().get("pose_remains_trusted_after_stop") is not True
        ):
            # Another session performed the stop, so this supervisor cannot
            # retain a pose belief from before that out-of-band intervention.
            self._trusted_joint_positions = None
            self._resource_generation_seen = self.registry.motion_generation(self.resource_id)

    def latch_stop_request(self) -> bool:
        """Synchronously fence motion before a caller awaits cancellation.

        Server policy/deadman paths use this before waiting on an autonomous
        task.  ``emergency_stop_once`` remains responsible for the bounded
        actuator call and final acknowledgement state.
        """
        return self.registry.begin_stop(self.resource_id)

    def record_activity(self) -> None:
        self._last_activity_monotonic = time.monotonic()

    def activity_age_sec(self) -> float:
        if self._last_activity_monotonic == 0.0:
            return 0.0
        return time.monotonic() - self._last_activity_monotonic

    def required_telemetry_stale(self) -> bool:
        """Return true when owned motion outlives its telemetry grace period."""
        if not self.owns_motion:
            return False
        reference = max(
            self._ownership_started_monotonic,
            self._last_safety_telemetry_monotonic,
        )
        if reference == 0.0:
            return True
        timeout = max(0.1, float(self.profile.safety.deadman_timeout_s))
        return (time.monotonic() - reference) > timeout

    def _safety_capabilities(self) -> dict[str, Any]:
        try:
            capabilities = self.backend.capabilities()
        except Exception:
            return {}
        safety = capabilities.get("motion_safety")
        return dict(safety) if isinstance(safety, dict) else {}

    def profile_command_error(self, command_name: str) -> str | None:
        if command_name not in self.profile.bridge_capabilities:
            return f"profile {self.profile.id!r} does not permit command {command_name!r}"
        return None

    def backend_command_error(self, command_name: str) -> str | None:
        try:
            capabilities = self.backend.capabilities()
        except Exception as exc:
            return f"backend capabilities unavailable: {exc}"
        commands = capabilities.get("commands")
        if isinstance(commands, list):
            if command_name not in commands:
                return f"backend does not advertise command {command_name!r}"
            return None
        capability_key = command_name.replace(".", "_")
        if capabilities.get(capability_key) is not True:
            return f"backend does not advertise capability {capability_key!r}"
        return None

    def _interruptible_stop_error(self, command_name: str) -> str | None:
        if command_name not in self._MOTION_COMMANDS and command_name != "walk.command":
            return None
        safety = self._safety_capabilities()
        class_key = f"{command_name.replace('.', '_')}_cancel"
        has_cancel = (
            safety.get("all_motion_stop") is True
            or safety.get(class_key) is True
            or (command_name in {"walk.set", "walk.command"} and safety.get("walk_stop") is True)
        )
        out_of_band_handler = getattr(self.backend, "handle_emergency_stop", None)
        if (
            not has_cancel
            or safety.get("stop_out_of_band") is not True
            or not callable(out_of_band_handler)
        ):
            return f"backend lacks verified out-of-band cancellation for {command_name}"
        return None

    def _torque_capability_error(self) -> str | None:
        safety = self._safety_capabilities()
        if (
            safety.get("environment") == "nonphysical"
            and safety.get("torque_limit_status") == "not_applicable"
        ):
            return None
        if safety.get("torque_limit_enforced") is not True:
            return "torque_limit_enforced"
        raw_limit = safety.get("torque_limit_nm")
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int | float):
            return "torque_limit_nm"
        try:
            limit = float(raw_limit)
        except OverflowError:
            return "torque_limit_nm"
        profile_limit = float(self.profile.control.safe_torque_clip_nm)
        if not math.isfinite(limit) or limit <= 0.0 or limit > profile_limit:
            return f"torque_limit_nm<={profile_limit}"
        return None

    def policy_start_capability_error(self) -> str | None:
        """Return a precise fail-closed reason for missing safety capability."""
        self._invalidate_stale_pose()
        profile_error = self.profile_command_error("policy.start")
        if profile_error is not None:
            return profile_error
        safety = self._safety_capabilities()
        if not safety:
            return "backend does not declare trustworthy motion_safety capabilities"
        missing = [
            name for name in self._REQUIRED_AUTONOMOUS_TELEMETRY if safety.get(name) is not True
        ]
        torque_error = self._torque_capability_error()
        if torque_error is not None:
            missing.append(torque_error)
        if safety.get("all_motion_stop") is not True:
            missing.append("all_motion_stop")
        if safety.get("stop_out_of_band") is not True or not callable(
            getattr(self.backend, "handle_emergency_stop", None)
        ):
            missing.append("stop_out_of_band")
        if self._trusted_joint_positions is None:
            missing.append("trusted_current_joint_pose")
        if self.backend.physical_motion_resources():
            missing.extend(
                "physical hard safety envelope lacks " + name
                for name in self._MISSING_HARD_ENVELOPE_PROFILE_CHECKS
            )
        if missing:
            return "backend cannot enforce autonomous safety: missing " + ", ".join(missing)
        return None

    def servo_capability_error(self) -> str | None:
        self._invalidate_stale_pose()
        torque_error = self._torque_capability_error()
        if torque_error is not None:
            return (
                "backend cannot enforce profile torque limit for direct servo motion "
                f"({self.profile.control.safe_torque_clip_nm} Nm): {torque_error}"
            )
        interrupt_error = self._interruptible_stop_error("servo.set")
        if interrupt_error is not None:
            return interrupt_error
        if self._trusted_joint_positions is None:
            return (
                "backend has no trusted current joint pose; direct servo delta/velocity "
                "checks cannot use profile home or stale commanded targets as a proxy"
            )
        return None

    def physical_motion_capability_error(
        self,
        command: CommandEnvelope,
    ) -> str | None:
        """Keep every physical motion behind the complete hard envelope."""
        if not self.backend.physical_motion_resources():
            return None
        missing = self._MISSING_HARD_ENVELOPE_PROFILE_CHECKS
        if missing:
            return (
                "physical motion disabled because the hard safety envelope is incomplete: "
                + ", ".join(str(item) for item in missing)
            )
        report = self.capability_report()
        if report["autonomous_motion_ready"] is not True:
            return "physical motion disabled: " + str(report["autonomous_blocker"])
        if command.command == "action.play":
            return (
                "physical scripted actions are unpromoted seed gestures and remain disabled"
            )
        return None

    def capability_report(self) -> dict[str, Any]:
        autonomous_error = self.policy_start_capability_error()
        return {
            "autonomous_motion_ready": autonomous_error is None,
            "autonomous_blocker": autonomous_error or "",
            # RobotProfile currently has no workspace or self-collision
            # geometry/threshold contract. Keep promotion fail-closed even for
            # a backend that satisfies the narrower runtime checks above.
            "hard_envelope_complete": not self._MISSING_HARD_ENVELOPE_PROFILE_CHECKS,
            "promotion_ready": False,
            "missing_profile_checks": list(self._MISSING_HARD_ENVELOPE_PROFILE_CHECKS),
        }

    def check_policy_action(self, action: dict[str, Any]) -> PolicyGuardResult:
        if "positions" in action:
            return PolicyGuardResult(
                allowed=False,
                reason=(
                    "invalid action rejected: policy.tick accepts only named "
                    "joint_positions; alternate or mixed positions payloads are forbidden"
                ),
                clamped={},
            )
        if "joint_positions" in action:
            capability_error = self.servo_capability_error()
            if capability_error is not None:
                return PolicyGuardResult(
                    allowed=False,
                    reason=f"invalid action rejected: {capability_error}",
                    clamped={},
                )
        return check_policy_motion_bounds(
            action,
            profile=self.profile,
            previous_joint_positions=self.last_joint_positions,
        )

    def telemetry_violation(self, event: EventEnvelope) -> str | None:
        """Evaluate telemetry only when the backend explicitly declares it.

        Missing or non-finite fields from a backend that advertised the field
        are themselves safety violations. Backends that do not advertise the
        complete autonomous envelope are rejected at ``policy.start``.
        """
        if event.event != "telemetry.basic":
            return None
        safety = self._safety_capabilities()
        if not safety:
            return None

        pose_error: str | None = None
        if safety.get("joint_positions") is True:
            self._joint_pose_generation += 1
            raw_positions = event.data.get("joint_positions")
            specs = {joint.name: joint for joint in self.profile.kinematics.joints}
            if not isinstance(raw_positions, dict) or set(raw_positions) != set(specs):
                pose_error = "trusted joint_positions telemetry must contain every profile joint"
            else:
                observed: dict[str, float] = {}
                for name, raw in raw_positions.items():
                    spec = specs[name]
                    if isinstance(raw, bool) or not isinstance(raw, int | float):
                        pose_error = f"trusted joint position {name!r} is non-finite"
                        break
                    try:
                        value = float(raw)
                    except OverflowError:
                        pose_error = f"trusted joint position {name!r} is non-finite"
                        break
                    if not math.isfinite(value):
                        pose_error = f"trusted joint position {name!r} is non-finite"
                        break
                    if value < spec.lower_rad or value > spec.upper_rad:
                        pose_error = f"trusted joint position {name!r} is outside profile limits"
                        break
                    observed[name] = value
                if pose_error is None:
                    self._trusted_joint_positions = observed
                    self._resource_generation_seen = self.registry.motion_generation(
                        self.resource_id
                    )
            if pose_error is not None:
                # Once a backend that attests joint telemetry emits an invalid
                # sample, an older connect-time or commanded pose is no longer
                # a trustworthy baseline, even if motion has not been acquired.
                self._trusted_joint_positions = None

        if not self.owns_motion:
            return None
        if pose_error is not None:
            return pose_error

        values: dict[str, float] = {}
        for name in self._REQUIRED_AUTONOMOUS_TELEMETRY:
            if safety.get(name) is not True:
                continue
            raw = event.data.get(name)
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                return f"trusted telemetry field {name} is missing or non-numeric"
            try:
                value = float(raw)
            except OverflowError:
                return f"trusted telemetry field {name} is non-finite"
            if not math.isfinite(value):
                return f"trusted telemetry field {name} is non-finite"
            values[name] = value

        roll = values.get("imu_roll")
        if roll is not None and abs(roll) > self.profile.safety.fall_roll_rad:
            return f"fall roll {roll} exceeds profile limit {self.profile.safety.fall_roll_rad}"
        pitch = values.get("imu_pitch")
        if pitch is not None and abs(pitch) > self.profile.safety.fall_pitch_rad:
            return f"fall pitch {pitch} exceeds profile limit {self.profile.safety.fall_pitch_rad}"
        battery = values.get("battery_mv")
        if battery is not None and battery <= self.profile.safety.battery_low_mv:
            return (
                f"battery {battery} mV at or below profile limit "
                f"{self.profile.safety.battery_low_mv} mV"
            )

        if safety.get("joint_torques") is True:
            torques = event.data.get("joint_torques")
            if not isinstance(torques, dict):
                return "trusted telemetry field joint_torques is missing or invalid"
            profile_joints = {joint.name for joint in self.profile.kinematics.joints}
            if set(torques) != profile_joints:
                return "trusted joint_torques telemetry must contain every profile joint"
            torque_limit = float(self.profile.control.safe_torque_clip_nm)
            for name, raw in torques.items():
                if isinstance(raw, bool) or not isinstance(raw, int | float):
                    return f"joint torque {name!r} is non-numeric"
                try:
                    value = float(raw)
                except OverflowError:
                    return f"joint torque {name!r} is non-finite"
                if not math.isfinite(value) or abs(value) > torque_limit:
                    return f"joint torque {name!r}={value} exceeds profile limit {torque_limit} Nm"
        self._last_safety_telemetry_monotonic = time.monotonic()
        return None

    def _error(self, request_id: str, message: str) -> ResponseEnvelope:
        return ResponseEnvelope(
            request_id=request_id,
            timestamp=utc_now_iso(),
            ok=False,
            backend=self.backend.backend_name,
            message=message,
            data={},
        )

    @staticmethod
    def _is_stop(command: CommandEnvelope) -> bool:
        if command.command == "walk.command":
            return command.payload.get("action") in {
                "stop",
                "disable",
                "disable_control",
            }
        return (
            command.command == "asimov.mode"
            and str(command.payload.get("mode", "")).upper() == "DAMP"
        )

    @classmethod
    def is_stop_command(cls, command: CommandEnvelope) -> bool:
        """Public classification used before server rate limits and queues."""
        return cls._is_stop(command)

    @classmethod
    def is_motion_command(cls, command: CommandEnvelope) -> bool:
        """Public classification used for fail-closed server rejections."""
        return cls._is_motion(command)

    @classmethod
    def is_motion_name(cls, command_name: str) -> bool:
        """Conservatively classify a command whose envelope is malformed."""
        return command_name in cls._MOTION_COMMANDS or command_name in {
            "walk.command",
            "policy.start",
            "policy.stop",
            "policy.tick",
        }

    @classmethod
    def _is_motion(cls, command: CommandEnvelope) -> bool:
        if command.command in cls._MOTION_COMMANDS:
            return True
        return command.command == "walk.command" and not cls._is_stop(command)

    @classmethod
    def _changes_joint_pose(cls, command: CommandEnvelope) -> bool:
        if not cls._is_motion(command):
            return False
        # These commands arm the walking controller but do not themselves
        # provide a new gait target. The subsequent walk.set/velocity command
        # is the pose-changing operation.
        arms_walking_only = command.command == "walk.command" and command.payload.get("action") in {
            "start",
            "enable",
            "enable_control",
        }
        return not arms_walking_only

    def _canonical_servo_payload(
        self,
        command: CommandEnvelope,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        payload = command.payload
        has_named = "joint_positions" in payload
        has_pulses = "positions" in payload
        if has_named == has_pulses:
            raise ValueError(
                "servo.set must use exactly one target format: joint_positions or positions"
            )

        raw_duration = payload.get("duration")
        if isinstance(raw_duration, bool) or not isinstance(raw_duration, int | float):
            raise ValueError("duration must be numeric")
        try:
            duration = float(raw_duration)
        except OverflowError as exc:
            raise ValueError("duration must be finite") from exc
        if not math.isfinite(duration) or duration <= 0.0 or duration > 5.0:
            raise ValueError("duration must be finite and in (0, 5.0] seconds")

        canonical_targets: dict[str, Any]
        if has_named:
            named, invalid = _validate_policy_joint_positions(
                payload.get("joint_positions"),
                profile=self.profile,
                previous_joint_positions=self.last_joint_positions,
                duration_sec=duration,
            )
            if invalid:
                raise ValueError("; ".join(invalid))
            # AiNex hardware consumes pulse targets. Other profiles/backends can
            # retain named positions when no complete pulse mapping exists.
            from eliza_robot.bridge.isaaclab.joint_map import (
                joint_name_to_servo_id,
                pulse_to_radians,
                radians_to_pulse,
            )

            pulses: list[dict[str, int]] = []
            effective_named: dict[str, float] = {}
            try:
                for name, value in named.items():
                    servo_id = joint_name_to_servo_id(name)
                    nominal_pulse = radians_to_pulse(value, servo_id)
                    candidates: list[tuple[float, int, float]] = []
                    for pulse_candidate in {
                        nominal_pulse,
                        max(0, nominal_pulse - 1),
                        min(1000, nominal_pulse + 1),
                    }:
                        effective = pulse_to_radians(pulse_candidate, servo_id)
                        _, candidate_invalid = _validate_policy_joint_positions(
                            {name: effective},
                            profile=self.profile,
                            previous_joint_positions=self.last_joint_positions,
                            duration_sec=duration,
                        )
                        if not candidate_invalid:
                            candidates.append((abs(effective - value), pulse_candidate, effective))
                    if candidates:
                        _, pulse, effective_value = min(candidates)
                    else:
                        # Preserve the nominal effective value so the complete
                        # revalidation below emits the precise hard failure.
                        pulse = nominal_pulse
                        effective_value = pulse_to_radians(pulse, servo_id)
                    pulses.append(
                        {
                            "id": servo_id,
                            "position": pulse,
                        }
                    )
                    effective_named[name] = effective_value
            except ValueError:
                # A profile with non-AiNex joint names retains its named
                # representation. Once every name maps, however, any failure
                # in effective pulse revalidation below is a hard rejection,
                # never a reason to fall back to the pre-quantized target.
                canonical_targets = {"joint_positions": dict(named)}
            else:
                effective_named, effective_invalid = _validate_policy_joint_positions(
                    effective_named,
                    profile=self.profile,
                    previous_joint_positions=self.last_joint_positions,
                    duration_sec=duration,
                )
                if effective_invalid:
                    raise ValueError(
                        "effective pulse target rejected: " + "; ".join(effective_invalid)
                    )
                named = effective_named
                canonical_targets = {"positions": pulses}
        else:
            positions = payload.get("positions")
            if not isinstance(positions, list) or not positions:
                raise ValueError("positions must be a non-empty list")
            from eliza_robot.bridge.isaaclab.joint_map import (
                pulse_to_radians,
                servo_id_to_joint_name,
            )

            named = {}
            canonical_positions: list[dict[str, int]] = []
            seen_ids: set[int] = set()
            for index, item in enumerate(positions):
                if not isinstance(item, dict):
                    raise ValueError(f"positions[{index}] must be an object")
                raw_servo_id = item.get("id")
                raw_pulse = item.get("position")
                if isinstance(raw_servo_id, bool) or not isinstance(raw_servo_id, int):
                    raise ValueError(f"positions[{index}].id must be an integer")
                if isinstance(raw_pulse, bool) or not isinstance(raw_pulse, int):
                    raise ValueError(f"positions[{index}].position must be an integer")
                servo_id = raw_servo_id
                pulse = raw_pulse
                if servo_id in seen_ids:
                    raise ValueError(f"duplicate servo id {servo_id}")
                seen_ids.add(servo_id)
                if pulse < 0 or pulse > 1000:
                    raise ValueError(f"positions[{index}].position outside 0..1000")
                name = servo_id_to_joint_name(servo_id)
                named[name] = pulse_to_radians(pulse, servo_id)
                canonical_positions.append({"id": servo_id, "position": pulse})
            validated, invalid = _validate_policy_joint_positions(
                named,
                profile=self.profile,
                previous_joint_positions=self.last_joint_positions,
                duration_sec=duration,
            )
            if invalid:
                raise ValueError("; ".join(invalid))
            named = validated
            canonical_targets = {"positions": canonical_positions}

        return {**canonical_targets, "duration": duration}, named

    def _validate_backend_response(
        self,
        command: CommandEnvelope,
        response: object,
    ) -> ResponseEnvelope:
        if not isinstance(response, ResponseEnvelope):
            return self._error(
                command.request_id,
                "backend returned an invalid response envelope",
            )
        if response.request_id != command.request_id:
            return self._error(
                command.request_id,
                "backend response request_id did not match the motion command",
            )
        if response.ok is not True and response.ok is not False:
            return self._error(
                command.request_id,
                "backend response ok acknowledgement was not an exact boolean",
            )
        return response

    async def _backend_call(self, command: CommandEnvelope) -> ResponseEnvelope:
        async def _dispatch() -> ResponseEnvelope:
            if self._is_motion(command):
                with _supervised_motion_dispatch_authority(
                    command,
                    self.backend.physical_motion_resources(),
                ):
                    return await self.backend.handle_command(command)
            return await self.backend.handle_command(command)

        dispatch_task = asyncio.create_task(_dispatch())
        try:
            response = await asyncio.shield(dispatch_task)
        except asyncio.CancelledError:
            if not self._is_motion(command):
                dispatch_task.cancel()
                raise
            # Cancelling an asyncio wrapper does not cancel an underlying ROS
            # service call or worker thread. Keep the command fence held until
            # the real I/O settles. Stop immediately in parallel, then stop a
            # second time in case the late side effect landed after that ACK.
            self.registry.begin_stop(self.resource_id, force=True)
            immediate_stop_task = asyncio.create_task(
                self._stop_from_guarded_dispatch(
                    f"{command.request_id}-cancelled-dispatch-immediate-stop",
                    force=True,
                )
            )
            while not dispatch_task.done():
                try:
                    await asyncio.shield(dispatch_task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            while not immediate_stop_task.done():
                try:
                    await asyncio.shield(immediate_stop_task)
                except asyncio.CancelledError:
                    continue
            stop_task = asyncio.create_task(
                self._stop_from_guarded_dispatch(
                    f"{command.request_id}-cancelled-dispatch-stop",
                    force=True,
                )
            )
            while not stop_task.done():
                try:
                    await asyncio.shield(stop_task)
                except asyncio.CancelledError:
                    continue
            with contextlib.suppress(asyncio.CancelledError, Exception):
                _ = stop_task.result()
            raise
        except Exception as exc:
            return self._error(command.request_id, f"backend error: {exc}")
        return self._validate_backend_response(command, response)

    async def _out_of_band_backend_call(
        self,
        command: CommandEnvelope,
    ) -> ResponseEnvelope:
        handler = getattr(self.backend, "handle_emergency_stop", None)
        if not callable(handler):
            return self._error(
                command.request_id,
                "backend has no dedicated out-of-band stop endpoint",
            )
        try:
            response = await handler(command)
        except Exception as exc:
            return self._error(command.request_id, f"backend stop error: {exc}")
        return self._validate_backend_response(command, response)

    async def _stop_locked(
        self,
        request_id: str,
        *,
        out_of_band: bool,
        force: bool = False,
    ) -> ResponseEnvelope:
        if not force and not self.registry.stop_required(self.resource_id):
            return self._already_stopped_response(request_id)
        was_owner = self.owns_motion
        stop = CommandEnvelope(
            request_id=request_id,
            timestamp=utc_now_iso(),
            command="walk.command",
            payload={"action": "stop"},
            preempt=True,
        )
        try:
            stop_timeout_sec = max(
                0.1,
                min(float(self.profile.safety.deadman_timeout_s), 2.0),
            )
            response = await asyncio.wait_for(
                (self._out_of_band_backend_call(stop) if out_of_band else self._backend_call(stop)),
                timeout=stop_timeout_sec,
            )
        except TimeoutError:
            response = self._error(
                request_id,
                f"backend stop acknowledgement timed out after {stop_timeout_sec}s",
            )
        except asyncio.CancelledError:
            self.registry.record_stop_result(
                self.resource_id,
                self.owner_id,
                ok=False,
            )
            self._ensure_emergency_stop_retry()
            raise
        resource_generation = self.registry.record_stop_result(
            self.resource_id,
            self.owner_id,
            ok=response.ok,
        )
        if response.ok:
            self._resource_generation_seen = resource_generation
            if (
                not was_owner
                or self._safety_capabilities().get("pose_remains_trusted_after_stop") is not True
            ):
                self._trusted_joint_positions = None
        else:
            self._ensure_emergency_stop_retry()
        return response

    def _ensure_emergency_stop_retry(self) -> None:
        """Keep retrying a failed actuator stop independently of a session."""
        task = self._stop_retry_task
        if task is not None and not task.done():
            return
        task = asyncio.create_task(self._emergency_stop_retry_loop())
        self._stop_retry_task = task
        _PROCESS_STOP_RETRY_TASKS.add(task)
        task.add_done_callback(_PROCESS_STOP_RETRY_TASKS.discard)

    async def _emergency_stop_retry_loop(self) -> None:
        attempt = 1
        while self.emergency_stop_pending:
            await asyncio.sleep(0.1)
            try:
                response = await self.emergency_stop_once(
                    f"process-retry-{attempt}"
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("process-wide emergency stop retry failed: %s", exc)
            else:
                if response.ok:
                    return
            attempt += 1

    def _already_stopped_response(self, request_id: str) -> ResponseEnvelope:
        return ResponseEnvelope(
            request_id=request_id,
            timestamp=utc_now_iso(),
            ok=True,
            backend=self.backend.backend_name,
            message="stop already confirmed",
            data={},
        )

    async def _out_of_band_stop(
        self,
        request_id: str,
        *,
        force: bool = False,
    ) -> ResponseEnvelope:
        # Latch before waiting for any asyncio lock.  New acquisitions and
        # in-flight command completions can now observe the stop immediately.
        if not self.registry.begin_stop(self.resource_id, force=force):
            return self._already_stopped_response(request_id)
        async with self.registry.stop_lock(self.resource_id):
            return await self._stop_locked(
                request_id,
                out_of_band=True,
                force=force,
            )

    async def _stop_from_guarded_dispatch(
        self,
        request_id: str,
        *,
        force: bool = False,
    ) -> ResponseEnvelope:
        out_of_band = self._safety_capabilities().get("stop_out_of_band") is True and callable(
            getattr(self.backend, "handle_emergency_stop", None)
        )
        if out_of_band:
            if not force:
                return await self._out_of_band_stop(request_id)
            # A previously acknowledged stop may have raced with an older
            # actuator call that completed afterward. The confirmed-state
            # shortcut is unsafe for this case: issue another physical stop.
            self.registry.begin_stop(self.resource_id, force=True)
            async with self.registry.stop_lock(self.resource_id):
                return await self._stop_locked(
                    request_id,
                    out_of_band=True,
                    force=True,
                )
        # The caller already holds command_lock for a backend without an
        # out-of-band stop path.
        if not self.registry.begin_stop(self.resource_id, force=force):
            return self._already_stopped_response(request_id)
        async with self.registry.stop_lock(self.resource_id):
            return await self._stop_locked(
                request_id,
                out_of_band=False,
                force=force,
            )

    async def emergency_stop_once(
        self,
        reason: str,
        *,
        force_physical: bool = False,
    ) -> ResponseEnvelope:
        """Attempt one stop; failed attempts retain ownership for retry."""
        force = force_physical and bool(self.backend.physical_motion_resources())
        out_of_band = self._safety_capabilities().get("stop_out_of_band") is True and callable(
            getattr(self.backend, "handle_emergency_stop", None)
        )
        if out_of_band:
            return await self._out_of_band_stop(
                f"safety-stop-{reason}-{time.time_ns()}",
                force=force,
            )
        request_id = f"safety-stop-{reason}-{time.time_ns()}"
        # This must happen before waiting for an in-flight motion command to
        # release command_lock.  The command's completion then fails closed
        # instead of being accepted as fresh motion after the stop request.
        if not self.registry.begin_stop(self.resource_id, force=force):
            return self._already_stopped_response(request_id)
        async with (
            self.registry.command_lock(self.resource_id),
            self.registry.stop_lock(self.resource_id),
        ):
            return await self._stop_locked(
                request_id,
                out_of_band=False,
                force=force,
            )

    async def _reject_owned_motion(
        self,
        command: CommandEnvelope,
        message: str,
        *,
        reason: str,
    ) -> ResponseEnvelope:
        """Reject motion and stop when this session already owns the robot."""
        if self.owns_motion:
            stop_response = await self.emergency_stop_once(reason)
            message = f"{message}; emergency_stop_ok={stop_response.ok}"
        return self._error(command.request_id, message)

    async def guarded_dispatch(
        self,
        command: CommandEnvelope,
        *,
        require_servo_capability: bool = False,
    ) -> ResponseEnvelope:
        """Validate, serialize, dispatch, and fail closed on backend failure."""
        if self._is_motion(command) or self._is_stop(command):
            try:
                validate_command_payload(command)
            except ValueError as exc:
                return await self._reject_owned_motion(
                    command,
                    f"motion payload validation failed: {exc}",
                    reason="invalid-motion-payload",
                )
        if self._is_stop(command):
            force_physical = bool(self.backend.physical_motion_resources())
            if self._safety_capabilities().get("stop_out_of_band") is True and callable(
                getattr(self.backend, "handle_emergency_stop", None)
            ):
                return await self._out_of_band_stop(
                    command.request_id,
                    force=force_physical,
                )
            if not self.registry.begin_stop(
                self.resource_id,
                force=force_physical,
            ):
                return self._already_stopped_response(command.request_id)
            async with (
                self.registry.command_lock(self.resource_id),
                self.registry.stop_lock(self.resource_id),
            ):
                return await self._stop_locked(
                    command.request_id,
                    out_of_band=False,
                    force=force_physical,
                )

        if self._is_motion(command):
            if self.registry.stop_in_progress(self.resource_id):
                return self._error(
                    command.request_id,
                    "motion blocked while a stop is in progress",
                )
            if self.emergency_stop_pending:
                return self._error(
                    command.request_id,
                    "motion blocked while emergency stop acknowledgement is pending",
                )
            if self.motion_revoked:
                return self._error(
                    command.request_id,
                    "motion blocked because another session stopped this owner; "
                    "a new explicit command must rearm it",
                )
            profile_error = self.profile_command_error(command.command)
            if profile_error is not None:
                return await self._reject_owned_motion(
                    command,
                    profile_error,
                    reason="profile-rejected-motion",
                )
            backend_error = self.backend_command_error(command.command)
            if backend_error is not None:
                return await self._reject_owned_motion(
                    command,
                    backend_error,
                    reason="backend-capability-rejected-motion",
                )
            stop_error = self._interruptible_stop_error(command.command)
            if stop_error is not None:
                return await self._reject_owned_motion(
                    command,
                    stop_error,
                    reason="noninterruptible-motion-rejected",
                )
            physical_error = self.physical_motion_capability_error(command)
            if physical_error is not None:
                return await self._reject_owned_motion(
                    command,
                    physical_error,
                    reason="physical-hard-envelope-rejected",
                )

        joint_positions: dict[str, float] | None = None
        pose_generation_before_dispatch = self._joint_pose_generation
        dispatch_command = command
        if command.command == "servo.set":
            capability_error = self.servo_capability_error()
            if capability_error is not None:
                return await self._reject_owned_motion(
                    command,
                    capability_error,
                    reason="servo-capability-rejected",
                )
            try:
                payload, joint_positions = self._canonical_servo_payload(command)
            except (TypeError, ValueError) as exc:
                return await self._reject_owned_motion(
                    command,
                    f"servo safety guard blocked: {exc}",
                    reason="servo-guard-rejected",
                )
            dispatch_command = CommandEnvelope(
                request_id=command.request_id,
                timestamp=command.timestamp,
                command=command.command,
                payload=payload,
                preempt=command.preempt,
            )

        if self._is_motion(command) and not self.acquire_motion():
            owner = self.registry.owner(self.resource_id)
            return self._error(
                command.request_id,
                f"motion resource is exclusively owned by another session ({owner})",
            )

        async with self.registry.command_lock(self.resource_id):
            if self._is_motion(command) and not self.owns_motion:
                return self._error(command.request_id, "motion ownership was lost before dispatch")
            if self._is_motion(command) and self.registry.stop_in_progress(self.resource_id):
                return self._error(
                    command.request_id,
                    "motion blocked while a stop is in progress",
                )
            if command.command == "servo.set":
                try:
                    payload, joint_positions = self._canonical_servo_payload(command)
                except (TypeError, ValueError) as exc:
                    stop_response = await self._stop_from_guarded_dispatch(
                        f"{command.request_id}-pose-race-stop"
                    )
                    return self._error(
                        command.request_id,
                        "servo safety guard changed before dispatch: "
                        f"{exc}; emergency_stop_ok={stop_response.ok}",
                    )
                dispatch_command = CommandEnvelope(
                    request_id=command.request_id,
                    timestamp=command.timestamp,
                    command=command.command,
                    payload=payload,
                    preempt=command.preempt,
                )
                pose_generation_before_dispatch = self._joint_pose_generation
            resource_generation_before_dispatch = self.registry.motion_generation(self.resource_id)
            response = await self._backend_call(dispatch_command)
            if not response.ok and self._is_motion(command):
                stop_response = await self._stop_from_guarded_dispatch(
                    f"{command.request_id}-backend-rejected-stop"
                )
                return self._error(
                    command.request_id,
                    f"backend rejected motion: {response.message}; "
                    f"emergency_stop_ok={stop_response.ok}",
                )
            stop_generation_changed = (
                self.registry.motion_generation(self.resource_id)
                != resource_generation_before_dispatch
            )
            if (
                response.ok
                and self._is_motion(command)
                and (
                    stop_generation_changed
                    or not self.owns_motion
                    or self.registry.stop_in_progress(self.resource_id)
                )
            ):
                # An unfinished latched stop owns the cancellation path. If it
                # already completed and the owner was rearmed too early, issue
                # a second out-of-band stop before releasing command_lock.
                stop_pending = self.registry.stop_in_progress(self.resource_id)
                stop_response = (
                    None
                    if stop_pending and not stop_generation_changed
                    else await self._stop_from_guarded_dispatch(
                        f"{command.request_id}-post-dispatch-ownership-stop",
                        force=stop_generation_changed,
                    )
                )
                return self._error(
                    command.request_id,
                    "motion was cancelled while backend dispatch was in flight; "
                    + (
                        "emergency_stop_pending=True"
                        if stop_response is None
                        else f"emergency_stop_ok={stop_response.ok}"
                    ),
                )
            if response.ok and self._changes_joint_pose(command):
                resource_generation = self.registry.record_motion_success(self.resource_id)
                if (
                    joint_positions is not None
                    and self._joint_pose_generation == pose_generation_before_dispatch
                ):
                    self._trusted_joint_positions = {
                        **(self._trusted_joint_positions or {}),
                        **joint_positions,
                    }
                else:
                    # Opaque high-level motion, or telemetry racing an in-flight
                    # command, leaves no trustworthy post-command pose.
                    self._trusted_joint_positions = None
                self._resource_generation_seen = resource_generation
            if response.ok and self._is_motion(command):
                self.record_activity()
            return response
