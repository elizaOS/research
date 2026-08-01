"""Tests for policy motion-bound safety checks."""

from __future__ import annotations

import time
import unittest

from eliza_robot.bridge.safety import (
    PolicyHeartbeatMonitor,
    check_policy_motion_bounds,
)
from eliza_robot.profiles.schema import load_profile


class PolicyMotionBoundsTests(unittest.TestCase):
    """Test policy action clamping and safety gating."""

    def test_within_bounds_passes(self) -> None:
        action = {
            "walk_x": 0.02,
            "walk_y": -0.01,
            "walk_yaw": 5.0,
            "walk_height": 0.036,
            "walk_speed": 2,
        }
        result = check_policy_motion_bounds(action)
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "")
        self.assertAlmostEqual(result.clamped["walk_x"], 0.02)

    def test_nan_action_rejected(self) -> None:
        # A diverged policy emitting NaN must be rejected, not silently passed
        # through (abs(nan) > MAX is False, so NaN would otherwise slip past).
        result = check_policy_motion_bounds({"walk_x": float("nan"), "walk_y": 0.0, "walk_yaw": 0.0})
        self.assertFalse(result.allowed)
        self.assertIn("walk_x", result.reason)
        # the clamped payload is still finite/neutral
        self.assertEqual(result.clamped["walk_x"], 0.0)

    def test_inf_action_rejected(self) -> None:
        result = check_policy_motion_bounds({"walk_x": 0.0, "walk_y": float("inf"), "walk_yaw": 0.0})
        self.assertFalse(result.allowed)
        self.assertEqual(result.clamped["walk_y"], 0.0)

    def test_non_numeric_action_rejected(self) -> None:
        result = check_policy_motion_bounds({"walk_x": "fast", "walk_y": 0.0, "walk_yaw": 0.0})
        self.assertFalse(result.allowed)
        self.assertEqual(result.clamped["walk_x"], 0.0)

    def test_huge_action_is_rejected_without_float_overflow(self) -> None:
        result = check_policy_motion_bounds({"walk_x": 10**1000})
        self.assertFalse(result.allowed)
        self.assertEqual(result.clamped["walk_x"], 0.0)

    def test_nan_head_rejected(self) -> None:
        result = check_policy_motion_bounds(
            {"walk_x": 0.0, "walk_y": 0.0, "walk_yaw": 0.0, "head_tilt": float("nan")}
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.clamped["head_tilt"], 0.0)

    def test_walk_x_clamped(self) -> None:
        action = {"walk_x": 0.1, "walk_y": 0.0, "walk_yaw": 0.0}
        result = check_policy_motion_bounds(action)
        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.clamped["walk_x"], 0.05)
        self.assertIn("walk_x clamped", result.reason)

    def test_walk_y_clamped(self) -> None:
        action = {"walk_x": 0.0, "walk_y": -0.2, "walk_yaw": 0.0}
        result = check_policy_motion_bounds(action)
        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.clamped["walk_y"], -0.05)

    def test_walk_yaw_clamped(self) -> None:
        action = {"walk_x": 0.0, "walk_y": 0.0, "walk_yaw": 25.0}
        result = check_policy_motion_bounds(action)
        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.clamped["walk_yaw"], 10.0)

    def test_walk_height_clamped(self) -> None:
        action = {"walk_x": 0.0, "walk_y": 0.0, "walk_yaw": 0.0, "walk_height": 0.001}
        result = check_policy_motion_bounds(action)
        self.assertAlmostEqual(result.clamped["walk_height"], 0.015)
        self.assertIn("walk_height clamped", result.reason)

    def test_walk_speed_clamped(self) -> None:
        action = {"walk_x": 0.0, "walk_y": 0.0, "walk_yaw": 0.0, "walk_speed": 10}
        result = check_policy_motion_bounds(action)
        self.assertEqual(result.clamped["walk_speed"], 4)

    def test_head_pan_clamped(self) -> None:
        action = {"walk_x": 0.0, "walk_y": 0.0, "walk_yaw": 0.0, "head_pan": 3.0}
        result = check_policy_motion_bounds(action)
        self.assertAlmostEqual(result.clamped["head_pan"], 1.5)

    def test_head_tilt_clamped(self) -> None:
        action = {"walk_x": 0.0, "walk_y": 0.0, "walk_yaw": 0.0, "head_tilt": -2.0}
        result = check_policy_motion_bounds(action)
        self.assertAlmostEqual(result.clamped["head_tilt"], -1.0)

    def test_defaults_used_for_missing_fields(self) -> None:
        action = {}
        result = check_policy_motion_bounds(action)
        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.clamped["walk_x"], 0.0)
        self.assertAlmostEqual(result.clamped["walk_y"], 0.0)
        self.assertEqual(result.clamped["walk_speed"], 2)

    def test_multiple_fields_clamped(self) -> None:
        action = {
            "walk_x": 0.1,
            "walk_y": -0.2,
            "walk_yaw": 50.0,
            "walk_height": 0.001,
            "walk_speed": 0,
        }
        result = check_policy_motion_bounds(action)
        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.clamped["walk_x"], 0.05)
        self.assertAlmostEqual(result.clamped["walk_y"], -0.05)
        self.assertAlmostEqual(result.clamped["walk_yaw"], 10.0)
        self.assertAlmostEqual(result.clamped["walk_height"], 0.015)
        self.assertEqual(result.clamped["walk_speed"], 1)


class PolicyJointMotionBoundsTests(unittest.TestCase):
    """Direct joint targets must satisfy the active profile's hard bounds."""

    def setUp(self) -> None:
        self.profile = load_profile("hiwonder-ainex")
        self.home = {
            joint.name: float(joint.home_rad)
            for joint in self.profile.kinematics.joints
        }

    def _check(self, joint_positions, *, previous=None, duration=None):
        action = {"joint_positions": joint_positions}
        if duration is not None:
            action["duration"] = duration
        return check_policy_motion_bounds(
            action,
            profile=self.profile,
            previous_joint_positions=self.home if previous is None else previous,
        )

    def test_named_joint_targets_within_limit_and_delta_pass(self) -> None:
        result = self._check(
            {
                "r_hip_pitch": -0.2,
                "l_hip_pitch": 0.1,
            }
        )

        self.assertTrue(result.allowed)
        self.assertEqual(
            result.clamped["joint_positions"],
            {
                "r_hip_pitch": -0.2,
                "l_hip_pitch": 0.1,
            },
        )
        self.assertEqual(result.clamped["duration"], 0.1)

    def test_servo_list_shape_is_rejected_as_a_bypass(self) -> None:
        result = self._check([{"id": 8, "position": 1000}])

        self.assertFalse(result.allowed)
        self.assertEqual(result.clamped["joint_positions"], {})
        self.assertIn("non-empty object keyed by joint name", result.reason)

    def test_unknown_joint_is_rejected(self) -> None:
        result = self._check({"r_hip_pitch_typo": 0.0})

        self.assertFalse(result.allowed)
        self.assertEqual(result.clamped["joint_positions"], {})
        self.assertIn("unknown joint", result.reason)

    def test_nonfinite_and_non_numeric_joint_targets_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf"), True, "0.1"):
            with self.subTest(value=value):
                result = self._check({"r_hip_pitch": value})
                self.assertFalse(result.allowed)
                self.assertEqual(result.clamped["joint_positions"], {})

    def test_profile_position_limit_is_rejected(self) -> None:
        result = self._check({"r_hip_pitch": 2.5})

        self.assertFalse(result.allowed)
        self.assertEqual(result.clamped["joint_positions"], {})
        self.assertIn("outside profile range", result.reason)

    def test_profile_per_step_delta_is_rejected(self) -> None:
        previous = dict(self.home)
        previous["r_hip_pitch"] = 0.2

        result = self._check({"r_hip_pitch": 0.51}, previous=previous)

        self.assertFalse(result.allowed)
        self.assertEqual(result.clamped["joint_positions"], {})
        self.assertIn("delta", result.reason)
        self.assertIn("profile max 0.3", result.reason)

    def test_joint_duration_must_be_finite_numeric_and_bounded(self) -> None:
        for duration in (
            True,
            "0.1",
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            -0.1,
            5.1,
        ):
            with self.subTest(duration=duration):
                result = self._check(
                    {"r_hip_pitch": 0.1},
                    duration=duration,
                )
                self.assertFalse(result.allowed)
                self.assertEqual(result.clamped["joint_positions"], {})
                self.assertNotIn("duration", result.clamped)

        result = self._check({"r_hip_pitch": 0.1}, duration=5)
        self.assertTrue(result.allowed)
        self.assertEqual(result.clamped["duration"], 5.0)

    def test_short_duration_cannot_bypass_joint_velocity_and_control_rate(self) -> None:
        rejected = self._check(
            {"r_hip_pitch": 0.02},
            duration=0.001,
        )
        self.assertFalse(rejected.allowed)
        self.assertIn("velocity", rejected.reason)
        self.assertIn("control limit", rejected.reason)

        accepted = self._check(
            {"r_hip_pitch": 0.02},
            duration=0.002,
        )
        self.assertTrue(accepted.allowed)

    def test_policy_discrete_speed_rejects_bool_and_fractional_aliases(self) -> None:
        for speed in (True, False, 1.9):
            with self.subTest(speed=speed):
                result = check_policy_motion_bounds({"walk_speed": speed})
                self.assertFalse(result.allowed)
                self.assertIn("walk_speed=non-integer", result.reason)


class PolicyHeartbeatTests(unittest.TestCase):
    """Test policy heartbeat monitoring."""

    def test_not_stale_initially(self) -> None:
        monitor = PolicyHeartbeatMonitor(timeout_sec=1.0)
        self.assertFalse(monitor.is_stale())

    def test_not_stale_after_tick(self) -> None:
        monitor = PolicyHeartbeatMonitor(timeout_sec=1.0)
        monitor.record_tick()
        self.assertFalse(monitor.is_stale())

    def test_stale_after_timeout(self) -> None:
        monitor = PolicyHeartbeatMonitor(timeout_sec=0.01)
        monitor.record_tick()
        time.sleep(0.02)
        self.assertTrue(monitor.is_stale())

    def test_age_sec(self) -> None:
        monitor = PolicyHeartbeatMonitor(timeout_sec=1.0)
        self.assertAlmostEqual(monitor.age_sec(), 0.0)
        monitor.record_tick()
        time.sleep(0.05)
        self.assertGreater(monitor.age_sec(), 0.04)


if __name__ == "__main__":
    unittest.main()
