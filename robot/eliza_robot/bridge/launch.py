"""Unified bridge launcher with target-based configuration.

Single entrypoint that selects backend (real/sim/isaac/mock) and starts
both ROSBridge-compatible and command-envelope websocket servers.

Usage:
    python -m eliza_robot.bridge.launch --target isaac
    python -m eliza_robot.bridge.launch --target real
    python -m eliza_robot.bridge.launch --target mock

Environment variable overrides:
    AINEX_BRIDGE_HOST, AINEX_ROSBRIDGE_PORT, AINEX_ENVELOPE_PORT,
    AINEX_PUBLISH_HZ, AINEX_MAX_CMD_SEC, AINEX_DEADMAN_SEC,
    ASIMOV_LIVEKIT_URL, ASIMOV_LIVEKIT_TOKEN,
    ELIZA_ROBOT_BRIDGE_AUTH_TOKEN, ELIZA_ROBOT_PHYSICAL_RESOURCE_ID
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from eliza_robot.bridge.backends.base import canonical_physical_resource_id
from eliza_robot.bridge.types import JsonDict

CONFIG_PATH = Path(__file__).parent / "config" / "bridge_targets.json"
logger = logging.getLogger(__name__)
_COMPATIBILITY_BACKENDS = frozenset({"mock", "ros_sim", "isaac"})
_PHYSICAL_BACKENDS = frozenset(
    {"ros", "ros_real", "ainex_remote", "ros_remote", "asimov_remote"}
)


@dataclass
class TargetConfig:
    """Resolved configuration for a bridge target."""

    name: str
    description: str
    backend: str
    host: str
    rosbridge_port: int
    envelope_port: int
    publish_hz: float
    max_commands_per_sec: int
    deadman_timeout_sec: float
    requires_ros: bool
    camera_url: str
    profile_id: str = "hiwonder-ainex"
    asimov_livekit_url: str = ""
    asimov_livekit_token: str = field(default="", repr=False)
    auth_token: str = field(default="", repr=False)
    physical_resource_id: str = ""


def _load_targets() -> dict[str, JsonDict]:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    targets = raw.get("targets", {})
    if not isinstance(targets, dict):
        raise ValueError("bridge_targets.json must have a 'targets' object")
    return targets


def _apply_env_overrides(cfg: dict[str, object]) -> dict[str, object]:
    """Apply environment variable overrides to target config."""
    env_map = {
        "AINEX_BRIDGE_HOST": ("host", str),
        "AINEX_ROSBRIDGE_PORT": ("rosbridge_port", int),
        "AINEX_ENVELOPE_PORT": ("envelope_port", int),
        "AINEX_PUBLISH_HZ": ("publish_hz", float),
        "AINEX_MAX_CMD_SEC": ("max_commands_per_sec", int),
        "AINEX_DEADMAN_SEC": ("deadman_timeout_sec", float),
        "AINEX_CAMERA_URL": ("camera_url", str),
        "ASIMOV_LIVEKIT_URL": ("asimov_livekit_url", str),
        "ASIMOV_LIVEKIT_TOKEN": ("asimov_livekit_token", str),
        "ELIZA_ROBOT_BRIDGE_AUTH_TOKEN": ("auth_token", str),
        "ELIZA_ROBOT_PHYSICAL_RESOURCE_ID": ("physical_resource_id", str),
    }
    for env_var, (config_field, cast) in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            cfg[config_field] = cast(value)
    return cfg


def resolve_target(name: str) -> TargetConfig:
    """Resolve a target name to a fully-configured TargetConfig."""
    targets = _load_targets()
    if name not in targets:
        available = ", ".join(sorted(targets.keys()))
        raise ValueError(f"unknown target '{name}'. Available: {available}")

    raw = dict(targets[name])
    raw = _apply_env_overrides(raw)

    return TargetConfig(
        name=name,
        description=str(raw.get("description", "")),
        backend=str(raw.get("backend", name)),
        host=str(raw.get("host", "0.0.0.0")),
        rosbridge_port=int(raw.get("rosbridge_port", 9090)),
        envelope_port=int(raw.get("envelope_port", 9100)),
        publish_hz=float(raw.get("publish_hz", 15.0)),
        max_commands_per_sec=int(raw.get("max_commands_per_sec", 30)),
        deadman_timeout_sec=float(raw.get("deadman_timeout_sec", 5.0)),
        requires_ros=bool(raw.get("requires_ros", False)),
        camera_url=str(raw.get("camera_url", "")),
        profile_id=str(raw.get("profile_id", "hiwonder-ainex")),
        asimov_livekit_url=str(raw.get("asimov_livekit_url", "")),
        asimov_livekit_token=str(raw.get("asimov_livekit_token", "")),
        auth_token=str(raw.get("auth_token", "")),
        physical_resource_id=str(raw.get("physical_resource_id", "")),
    )


def _resolve_server_modes(
    target: TargetConfig,
    *,
    rosbridge: bool | None,
    envelope: bool | None,
) -> tuple[bool, bool]:
    """Resolve safe defaults and reject compatibility control of hardware."""
    resolved_rosbridge = (
        target.backend in _COMPATIBILITY_BACKENDS if rosbridge is None else rosbridge
    )
    resolved_envelope = (
        target.backend not in _COMPATIBILITY_BACKENDS if envelope is None else envelope
    )
    if target.backend in _PHYSICAL_BACKENDS and resolved_rosbridge:
        raise ValueError(
            "physical targets cannot use the ROSBridge-compatible server; "
            "use the authenticated unified envelope server"
        )
    if target.backend in _PHYSICAL_BACKENDS and not resolved_envelope:
        raise ValueError("physical targets require the unified envelope server")
    if target.backend in _PHYSICAL_BACKENDS:
        if not 32 <= len(target.auth_token) <= 4_096 or any(
            not 0x21 <= ord(character) <= 0x7E for character in target.auth_token
        ):
            raise ValueError(
                "physical targets require ELIZA_ROBOT_BRIDGE_AUTH_TOKEN "
                "with 32..4096 visible ASCII characters"
            )
        try:
            canonical_physical_resource_id(target.physical_resource_id)
        except ValueError as exc:
            raise ValueError(
                "physical targets require an explicit canonical "
                "ELIZA_ROBOT_PHYSICAL_RESOURCE_ID"
            ) from exc
    if not resolved_rosbridge and not resolved_envelope:
        raise ValueError("no server selected")
    return resolved_rosbridge, resolved_envelope


async def _run(target: TargetConfig, rosbridge: bool, envelope: bool) -> None:
    rosbridge, envelope = _resolve_server_modes(
        target,
        rosbridge=rosbridge,
        envelope=envelope,
    )
    tasks: list[asyncio.Task[None]] = []

    if rosbridge:
        from eliza_robot.bridge.rosbridge_server import RuntimeConfig as RBConfig
        from eliza_robot.bridge.rosbridge_server import _run_server as rb_run

        rb_config = RBConfig(
            publish_hz=target.publish_hz,
            max_commands_per_sec=target.max_commands_per_sec,
            deadman_timeout_sec=target.deadman_timeout_sec,
            camera_url=target.camera_url,
        )
        tasks.append(
            asyncio.create_task(
                rb_run(target.host, target.rosbridge_port, target.backend, rb_config)
            )
        )

    if envelope:
        from eliza_robot.bridge.server import RuntimeConfig as EnvConfig
        from eliza_robot.bridge.server import _run_server as env_run

        env_config = EnvConfig(
            queue_size=256,
            max_commands_per_sec=target.max_commands_per_sec,
            deadman_timeout_sec=target.deadman_timeout_sec,
            trace_log_path="",
            profile_id=target.profile_id,
            asimov_livekit_url=target.asimov_livekit_url,
            asimov_livekit_token=target.asimov_livekit_token,
            auth_token=target.auth_token,
            physical_resource_id=target.physical_resource_id,
        )
        tasks.append(
            asyncio.create_task(
                env_run(target.host, target.envelope_port, target.backend, env_config)
            )
        )

    # Wait until interrupted.
    await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified AiNex bridge launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Targets:\n"
            "  real   - Real AiNex robot via ROS1\n"
            "  sim    - Gazebo simulation via ROS1\n"
            "  isaac  - IsaacLab simulation\n"
            "  mock   - In-memory mock for development\n"
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default="mock",
        help="target backend to launch (default: mock)",
    )
    parser.add_argument(
        "--rosbridge",
        action="store_true",
        default=None,
        help="start ROSBridge-compatible server (simulation targets only)",
    )
    parser.add_argument(
        "--no-rosbridge",
        action="store_false",
        dest="rosbridge",
        help="disable ROSBridge-compatible server",
    )
    parser.add_argument(
        "--envelope",
        action="store_true",
        default=None,
        help="start the unified command-envelope server",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="list available targets and exit",
    )
    args = parser.parse_args()

    if args.list_targets:
        targets = _load_targets()
        for name, cfg in sorted(targets.items()):
            desc = cfg.get("description", "")
            backend = cfg.get("backend", name)
            ros = "yes" if cfg.get("requires_ros", False) else "no"
            print(f"  {name:8s}  backend={backend:10s}  ros={ros:3s}  {desc}")
        return

    target = resolve_target(args.target)
    try:
        rosbridge, envelope = _resolve_server_modes(
            target,
            rosbridge=args.rosbridge,
            envelope=args.envelope,
        )
    except ValueError as exc:
        parser.error(str(exc))

    logger.info("Target: %s (%s)", target.name, target.description)
    logger.info("Backend: %s", target.backend)
    if rosbridge:
        logger.info("ROSBridge: ws://%s:%d", target.host, target.rosbridge_port)
    if envelope:
        logger.info("Envelope: ws://%s:%d", target.host, target.envelope_port)
    logger.info(
        "Safety: rate_limit=%d/s deadman=%ss",
        target.max_commands_per_sec,
        target.deadman_timeout_sec,
    )

    asyncio.run(_run(target, rosbridge=rosbridge, envelope=envelope))


if __name__ == "__main__":
    main()
