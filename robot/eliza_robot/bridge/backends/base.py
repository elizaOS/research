"""Backend interface for the websocket bridge."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from eliza_robot.bridge.protocol import (
    CommandEnvelope,
    EventEnvelope,
    ResponseEnvelope,
    utc_now_iso,
)
from eliza_robot.bridge.types import JsonDict

PHYSICAL_RESOURCE_ID_MAX_LENGTH = 128
_PHYSICAL_RESOURCE_PREFIX = "physical:"


def canonical_physical_resource_id(raw_id: str) -> str:
    """Validate one configured actuator identity and add its namespace.

    The caller must supply a raw, stable identity rather than an inferred
    transport address.  Rejecting instead of trimming or otherwise
    normalizing prevents two configurations from silently naming different
    hardware with the same process-local ownership key.
    """
    if not isinstance(raw_id, str):
        raise ValueError("physical_resource_id must be a string")
    if not raw_id:
        raise ValueError("physical_resource_id must not be empty")
    if len(raw_id) > PHYSICAL_RESOURCE_ID_MAX_LENGTH:
        raise ValueError(
            "physical_resource_id must be at most "
            f"{PHYSICAL_RESOURCE_ID_MAX_LENGTH} characters"
        )
    if any(not 0x21 <= ord(character) <= 0x7E for character in raw_id):
        raise ValueError(
            "physical_resource_id must be already-trimmed visible ASCII"
        )
    return f"{_PHYSICAL_RESOURCE_PREFIX}{raw_id}"


@dataclass
class _MotionDispatchAuthority:
    active: bool = True
    command_fingerprint: str = ""
    consumed: bool = False
    physical_resources: frozenset[str] = frozenset()


_MOTION_DISPATCH_AUTHORITY: ContextVar[_MotionDispatchAuthority | None] = ContextVar(
    "eliza_robot_motion_dispatch_authority",
    default=None,
)


def _command_fingerprint(command: CommandEnvelope) -> str:
    return json.dumps(
        command.to_json(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@contextmanager
def _supervised_motion_dispatch_authority(
    command: CommandEnvelope,
    physical_resources: tuple[str, ...],
) -> Iterator[None]:
    """Scope physical-motion authority to one supervisor-owned backend call.

    Context propagation lets transparent/dual-target wrappers forward the call
    without receiving a reusable credential.  The shared authority object is
    invalidated on exit so a wrapper-created background task cannot retain it.
    This is intentionally internal; ordinary callers must use
    ``MotionSafetySupervisor.guarded_dispatch``.
    """
    authority = _MotionDispatchAuthority(
        command_fingerprint=_command_fingerprint(command),
        physical_resources=frozenset(physical_resources),
    )
    token = _MOTION_DISPATCH_AUTHORITY.set(authority)
    try:
        yield
    finally:
        authority.active = False
        _MOTION_DISPATCH_AUTHORITY.reset(token)


def _is_direct_emergency_stop(command: CommandEnvelope) -> bool:
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


def _physical_motion_authority_error(
    backend_name: str,
    physical_resource: str,
    command: CommandEnvelope,
) -> ResponseEnvelope | None:
    """Reject unsupervised physical motion while always permitting a stop."""
    if _is_direct_emergency_stop(command):
        return None
    if command.command in {"profile.describe", "camera.snapshot", "policy.status"}:
        return None
    authority = _MOTION_DISPATCH_AUTHORITY.get()
    if (
        authority is not None
        and authority.active
        and not authority.consumed
        and authority.command_fingerprint == _command_fingerprint(command)
        and physical_resource in authority.physical_resources
    ):
        # A supervisor dispatch authorizes exactly one physical leaf. Child
        # tasks inherit ContextVars, so consuming the shared mutable token
        # prevents a wrapper from duplicating or retaining that authority.
        authority.consumed = True
        return None
    return ResponseEnvelope(
        request_id=command.request_id,
        timestamp=utc_now_iso(),
        ok=False,
        backend=backend_name,
        message=(
            "physical motion rejected: dispatch requires "
            "MotionSafetySupervisor.guarded_dispatch"
        ),
        data={"required_boundary": "MotionSafetySupervisor"},
    )


class BridgeBackend(ABC):
    """Abstract backend contract used by websocket server."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier used in responses/events."""

    @abstractmethod
    async def connect(self) -> None:
        """Initialize backend resources."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release backend resources."""

    @abstractmethod
    async def handle_command(self, cmd: CommandEnvelope) -> ResponseEnvelope:
        """Execute one command envelope."""

    @abstractmethod
    async def poll_events(self) -> list[EventEnvelope]:
        """Return any pending events that should be pushed to clients."""

    @abstractmethod
    def capabilities(self) -> JsonDict:
        """Return backend capabilities in JSON-serializable form."""

    def snapshot_camera(self, _camera: str = "head") -> np.ndarray | None:
        """Return the current camera frame as (H, W, 3) uint8 RGB, or None
        when the backend does not expose camera frames yet.

        The server-level `camera.snapshot` handler encodes the frame as PNG
        and ships it as base64. Subclasses (mujoco, mock, ros_real) override
        this to return real pixels.
        """
        return None

    def physical_motion_resources(self) -> tuple[str, ...]:
        """Return physical actuator identities intentionally owned by this backend.

        Nonphysical backends return no identities. Physical leaves and wrappers
        that deliberately forward to them must opt in explicitly.
        """
        return ()
