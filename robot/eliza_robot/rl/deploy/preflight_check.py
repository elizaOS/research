"""Pre-flight checks before deploying a policy to the real robot.

Verifies that the bridge server is reachable, the backend reports servo
positions and IMU data, and the reported battery level is adequate.

Usage::

    ELIZA_ROBOT_BRIDGE_AUTH_TOKEN=<secret> \
        python -m eliza_robot.rl.deploy.preflight_check \
        --bridge ws://127.0.0.1:9100 --profile hiwonder-ainex

The physical bridge must be reached through its local loopback endpoint (or
the loopback end of a secure tunnel). Use ``--simulation`` only for a
nonphysical bridge; simulation mode deliberately sends no physical credential.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

BRIDGE_AUTH_TOKEN_ENV = "ELIZA_ROBOT_BRIDGE_AUTH_TOKEN"
_MIN_PHYSICAL_BEARER_LENGTH = 32
_MAX_PHYSICAL_BEARER_LENGTH = 4_096


class BridgeTransportSecurityError(RuntimeError):
    """Raised before an unsafe or unauthenticated physical connection."""


def redacted_bridge_url(bridge_url: str) -> str:
    """Return a log-safe endpoint with credentials, query, and fragment removed."""
    if not isinstance(bridge_url, str):
        return "<invalid bridge endpoint>"
    try:
        parsed = urlsplit(bridge_url)
        host = parsed.hostname
        if host is None:
            return "<invalid bridge endpoint>"
        rendered_host = f"[{host}]" if ":" in host else host
        try:
            port = parsed.port
        except ValueError:
            port = None
        netloc = rendered_host if port is None else f"{rendered_host}:{port}"
        return urlunsplit((parsed.scheme, netloc, "", "", ""))
    except (TypeError, ValueError):
        return "<invalid bridge endpoint>"


def _parse_bridge_url(bridge_url: str) -> SplitResult:
    if not isinstance(bridge_url, str) or not bridge_url:
        raise BridgeTransportSecurityError("bridge URL must be a non-empty string")
    try:
        parsed = urlsplit(bridge_url)
        _ = parsed.port
    except ValueError as exc:
        raise BridgeTransportSecurityError("bridge URL is malformed") from exc
    if parsed.scheme not in {"ws", "wss"} or parsed.hostname is None:
        raise BridgeTransportSecurityError("bridge URL must use ws:// or wss:// with a host")
    if parsed.username is not None or parsed.password is not None:
        raise BridgeTransportSecurityError(
            "bridge URL must not contain credentials; use the bearer-token environment secret"
        )
    if parsed.query or parsed.fragment:
        raise BridgeTransportSecurityError(
            "bridge URL must not contain a query or fragment; secrets belong in environment "
            "configuration, never in the URL"
        )
    return parsed


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _physical_bearer_token() -> str:
    token = os.environ.get(BRIDGE_AUTH_TOKEN_ENV, "")
    if not token:
        raise BridgeTransportSecurityError(
            f"physical bridge requires {BRIDGE_AUTH_TOKEN_ENV}"
        )
    if len(token) < _MIN_PHYSICAL_BEARER_LENGTH:
        raise BridgeTransportSecurityError(
            f"{BRIDGE_AUTH_TOKEN_ENV} must contain at least "
            f"{_MIN_PHYSICAL_BEARER_LENGTH} characters"
        )
    if len(token) > _MAX_PHYSICAL_BEARER_LENGTH:
        raise BridgeTransportSecurityError(
            f"{BRIDGE_AUTH_TOKEN_ENV} must contain at most "
            f"{_MAX_PHYSICAL_BEARER_LENGTH} characters"
        )
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in token):
        raise BridgeTransportSecurityError(
            f"{BRIDGE_AUTH_TOKEN_ENV} must contain only visible ASCII without whitespace"
        )
    return token


def bridge_connect_options(bridge_url: str, *, physical: bool) -> dict[str, object]:
    """Validate a unified-bridge target and build non-URL authentication options."""
    parsed = _parse_bridge_url(bridge_url)
    if not physical:
        # Never forward a hardware credential to a simulation endpoint. This
        # preserves the existing unauthenticated simulation connection path.
        return {}
    if not _is_loopback_host(parsed.hostname or ""):
        raise BridgeTransportSecurityError(
            "physical bridge connections must target loopback; use the local end "
            f"of an authenticated secure tunnel (got {redacted_bridge_url(bridge_url)})"
        )
    if parsed.path not in {"", "/"}:
        raise BridgeTransportSecurityError(
            "physical bridge URL must use the root websocket path"
        )
    token = _physical_bearer_token()
    return {"additional_headers": {"Authorization": f"Bearer {token}"}}


def policy_tick_joint_payload(
    joint_positions: Mapping[str, float],
    *,
    duration_sec: float,
) -> dict[str, object]:
    """Build the nested, seconds-based joint action required by policy.tick."""
    if not joint_positions:
        raise ValueError("policy.tick joint_positions must not be empty")
    if isinstance(duration_sec, bool) or not isinstance(duration_sec, int | float):
        raise ValueError("policy.tick duration_sec must be finite and in (0, 5]")
    try:
        duration = float(duration_sec)
    except OverflowError as exc:
        raise ValueError("policy.tick duration_sec must be finite and in (0, 5]") from exc
    if not math.isfinite(duration) or duration <= 0.0 or duration > 5.0:
        raise ValueError("policy.tick duration_sec must be finite and in (0, 5]")
    normalized: dict[str, float] = {}
    for name, value in joint_positions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("policy.tick joint names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"policy.tick joint {name!r} must be finite")
        try:
            normalized_value = float(value)
        except OverflowError as exc:
            raise ValueError(f"policy.tick joint {name!r} must be finite") from exc
        if not math.isfinite(normalized_value):
            raise ValueError(f"policy.tick joint {name!r} must be finite")
        normalized[name] = normalized_value
    return {
        "action": {
            "joint_positions": normalized,
            "duration": duration,
        }
    }


async def _receive_bridge_event(
    websocket: object,
    event_name: str,
    *,
    deadline: float,
) -> dict[str, object]:
    recv = getattr(websocket, "recv", None)
    if not callable(recv):
        raise RuntimeError("bridge websocket does not expose recv()")
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0.0:
            raise TimeoutError(f"timed out waiting for {event_name}")
        raw = await asyncio.wait_for(recv(), timeout=remaining)
        frame = json.loads(raw)
        if not isinstance(frame, dict):
            raise RuntimeError("bridge emitted a non-object frame")
        if frame.get("type") == "event" and frame.get("event") == event_name:
            return frame


async def check_bridge(
    bridge_url: str,
    timeout: float = 5.0,
    *,
    physical: bool = True,
) -> dict[str, Any]:
    """Run all pre-flight checks and return a results dict."""
    import websockets

    results: dict[str, Any] = {
        "bridge_reachable": False,
        "backend_type": "unknown",
        "servo_count": 0,
        "imu_data": False,
        "battery_mv": 0,
        "battery_ok": False,
        "all_ok": False,
        "errors": [],
    }

    safe_endpoint = redacted_bridge_url(bridge_url)
    try:
        connect_options = bridge_connect_options(bridge_url, physical=physical)
        async with websockets.connect(
            bridge_url,
            open_timeout=timeout,
            **connect_options,
        ) as ws:
            results["bridge_reachable"] = True
            deadline = asyncio.get_running_loop().time() + timeout
            hello = await _receive_bridge_event(ws, "session.hello", deadline=deadline)
            backend_name = hello.get("backend")
            if isinstance(backend_name, str):
                results["backend_type"] = backend_name
            telemetry = await _receive_bridge_event(
                ws,
                "telemetry.basic",
                deadline=deadline,
            )
            data = telemetry.get("data", {})
            if not isinstance(data, dict):
                raise RuntimeError("telemetry.basic data must be an object")
            results["battery_mv"] = data.get("battery_mv", 0)
            results["battery_ok"] = results["battery_mv"] >= 6500 or (
                not physical and results["battery_mv"] == 0
            )
            if data.get("imu_roll") is not None:
                results["imu_data"] = True

            if "joint_positions" in data and isinstance(data["joint_positions"], dict):
                results["servo_count"] = len(data["joint_positions"])

            if not results["battery_ok"]:
                results["errors"].append(
                    f"Battery low: {results['battery_mv']}mV (minimum 6500mV)"
                )
            if not results["imu_data"]:
                results["errors"].append("No IMU data in telemetry.basic")
            if results["servo_count"] == 0:
                results["errors"].append("No servo position data in telemetry.basic")

    except BridgeTransportSecurityError as exc:
        results["errors"].append(str(exc))
    except TimeoutError:
        results["errors"].append(f"Connection timeout after {timeout}s")
    except ConnectionRefusedError:
        results["errors"].append(f"Connection refused at {safe_endpoint}")
    except Exception as e:  # noqa: BLE001 — surface unexpected failures in results
        detail = type(e).__name__ if physical else str(e)
        results["errors"].append(f"Connection error: {detail}")

    results["all_ok"] = (
        results["bridge_reachable"]
        and results["battery_ok"]
        and len(results["errors"]) == 0
    )
    return results


def print_results(results: dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("PRE-FLIGHT CHECK RESULTS")
    print("=" * 50)

    checks = [
        ("Bridge reachable", results["bridge_reachable"]),
        ("Backend type", results["backend_type"]),
        ("Servo count", results["servo_count"]),
        ("IMU data flowing", results["imu_data"]),
        ("Battery voltage", f"{results['battery_mv']}mV"),
        ("Battery OK", results["battery_ok"]),
    ]

    for name, value in checks:
        icon = ("PASS" if value else "FAIL") if isinstance(value, bool) else str(value)
        print(f"  {name:25s}: {icon}")

    if results["errors"]:
        print("\nERRORS:")
        for err in results["errors"]:
            print(f"  - {err}")

    print()
    if results["all_ok"]:
        print(
            "OBSERVATIONAL CHECKS PASSED — this does not authorize physical motion. "
            "The bridge safety supervisor remains authoritative."
        )
    else:
        print("CHECKS FAILED — do NOT deploy until issues are resolved.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight checks for robot deployment")
    parser.add_argument("--bridge", default="ws://localhost:9100", help="Bridge WebSocket URL")
    parser.add_argument("--timeout", type=float, default=5.0, help="Connection timeout (seconds)")
    parser.add_argument(
        "--profile",
        default="hiwonder-ainex",
        help="Robot profile id (reserved for future per-profile checks).",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Connect to a nonphysical bridge without sending the hardware bearer token",
    )
    args = parser.parse_args()

    results = asyncio.run(
        check_bridge(
            args.bridge,
            args.timeout,
            physical=not args.simulation,
        )
    )
    results["profile"] = args.profile
    print_results(results)
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
