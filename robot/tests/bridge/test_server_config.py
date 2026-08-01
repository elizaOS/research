"""Tests for direct bridge server runtime configuration."""

from __future__ import annotations

import argparse
import os
import unittest
from unittest import mock

from eliza_robot.bridge.server import RuntimeConfig, _coerce_runtime_config


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "queue_size": 256,
        "max_commands_per_sec": 30,
        "deadman_timeout_sec": 1.0,
        "trace_log_path": "",
        "profile": "asimov-1",
        "mujoco_target_x": 2.0,
        "mujoco_target_y": 0.0,
        "mujoco_target_z": 0.05,
        "camera_device": -1,
        "camera_width": 640,
        "camera_height": 480,
        "rosbridge_host": "192.168.1.218",
        "rosbridge_port": 9090,
        "asimov_livekit_url": "",
        "asimov_livekit_token": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ServerConfigTests(unittest.TestCase):
    def test_asimov_livekit_env_fills_direct_server_config(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ASIMOV_LIVEKIT_URL": "wss://asimov.example.invalid",
                "ASIMOV_LIVEKIT_TOKEN": "token-123",
            },
        ):
            config = _coerce_runtime_config(_args(), {})

        self.assertEqual(config.asimov_livekit_url, "wss://asimov.example.invalid")
        self.assertEqual(config.asimov_livekit_token, "token-123")

    def test_asimov_livekit_cli_args_override_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ASIMOV_LIVEKIT_URL": "wss://env.example.invalid",
                "ASIMOV_LIVEKIT_TOKEN": "env-token",
            },
        ):
            config = _coerce_runtime_config(
                _args(
                    asimov_livekit_url="wss://cli.example.invalid",
                    asimov_livekit_token="cli-token",
                ),
                {},
            )

        self.assertEqual(config.asimov_livekit_url, "wss://cli.example.invalid")
        self.assertEqual(config.asimov_livekit_token, "cli-token")

    def test_physical_bridge_credentials_are_read_from_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ELIZA_ROBOT_BRIDGE_AUTH_TOKEN": "a" * 32,
                "ELIZA_ROBOT_PHYSICAL_RESOURCE_ID": "lab-ainex-01",
            },
            clear=True,
        ):
            config = _coerce_runtime_config(_args(), {})

        self.assertEqual(config.auth_token, "a" * 32)
        self.assertEqual(config.physical_resource_id, "lab-ainex-01")
        self.assertNotIn(config.auth_token, repr(config))

    def test_nonempty_physical_resource_id_must_be_canonical_raw_text(self) -> None:
        for invalid in (" ", " leading", "trailing ", "line\nbreak", "x" * 129):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "physical_resource_id"),
            ):
                RuntimeConfig(
                    queue_size=8,
                    max_commands_per_sec=30,
                    deadman_timeout_sec=1.0,
                    trace_log_path="",
                    physical_resource_id=invalid,
                )


if __name__ == "__main__":
    unittest.main()
