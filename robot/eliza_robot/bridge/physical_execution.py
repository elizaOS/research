"""Fail-closed guard for legacy direct-hardware entry points.

Physical motion belongs behind the authenticated unified bridge and its
``MotionSafetySupervisor``.  Legacy calibration/evidence programs that create a
physical transport and call ``handle_command`` directly cannot establish that
boundary.  They must fail before connecting or creating evidence artifacts
until they are rewritten as authenticated command-envelope clients.
"""

from __future__ import annotations

import ipaddress
from typing import NoReturn
from urllib.parse import urlparse

from eliza_robot.bridge.protocol import ResponseEnvelope


class UnsupervisedPhysicalControlError(RuntimeError):
    """Raised before a legacy entry point can contact physical actuators."""


def reject_unsupervised_physical_motion(entrypoint: str) -> NoReturn:
    """Reject a direct physical transport path with actionable guidance."""
    raise UnsupervisedPhysicalControlError(
        f"{entrypoint} is quarantined because it calls a physical backend directly. "
        "Start the authenticated, loopback-only unified endpoint with "
        "`python -m eliza_robot.bridge.launch --target real --envelope` and use "
        "a supervised command-envelope client. This legacy entry point will not "
        "connect to hardware or emit physical-run evidence."
    )


def require_exact_command_ack(response: object, *, command: str) -> ResponseEnvelope:
    """Prevent a rejected/invalid dispatch from becoming apparent evidence."""
    if not isinstance(response, ResponseEnvelope):
        raise RuntimeError(f"{command} returned an invalid response envelope")
    if response.ok is not True:
        raise RuntimeError(f"{command} was rejected: {response.message}")
    return response


def require_loopback_simulation_uri(uri: str, *, entrypoint: str) -> None:
    """Keep ROSBridge compatibility motion tools away from network hardware."""
    parsed = urlparse(uri)
    host = parsed.hostname
    try:
        loopback = host is not None and (
            host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
        )
    except ValueError:
        loopback = False
    if parsed.scheme not in {"ws", "wss"} or not loopback:
        reject_unsupervised_physical_motion(f"{entrypoint} target {uri!r}")
