"""Server-side text-conditioned inference loop.

Closes the loop between the trained policy and a `BridgeBackend`:

    text task ─→ TextConditionedPolicy.act(text, proprio) ─→
        24-D joint targets ─→ bridge.servo.set ─→
        backend (real AiNex and/or MuJoCo) ─→
        new proprio ─→ next tick

Designed to be invoked either:
  - directly from a script (`run_inference(backend, ckpt, text)`),
  - or by the bridge server itself on `policy.start{task=…}` when the
    `--policy-checkpoint` flag is set (server-side autonomous policy).

The loop honours `max_steps` and `hz`, and always issues an explicit
`walk.command:stop` on exit. All joint targets pass through the same guarded
dispatch supervisor used by websocket commands.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eliza_robot.bridge.backends.base import BridgeBackend
from eliza_robot.bridge.protocol import CommandEnvelope, EventEnvelope, utc_now_iso
from eliza_robot.bridge.safety import MotionSafetySupervisor
from eliza_robot.bridge.types import JsonDict
from eliza_robot.profiles.schema import RobotProfile, load_profile
from eliza_robot.rl.text_conditioned.policy import TextConditionedPolicy

logger = logging.getLogger(__name__)


@dataclass
class InferenceLoopConfig:
    hz: float = 10.0
    max_steps: int = 500
    action_scale: float = 0.3  # rad per step around home pose
    safety_clip_rad: float = 1.0  # never command farther than this from home
    profile_id: str = "hiwonder-ainex"


def _proprio_from_telemetry(
    latest: dict | None,
    profile: RobotProfile,
    *,
    proprio_dim: int,
    last_action: np.ndarray | None = None,
    velocity_command: np.ndarray | None = None,
) -> np.ndarray:
    """Convert telemetry.basic into the profile-env proprio layout.

    Layout matches `TextConditionedProfileEnv._build_obs` exactly::

        gyro(3), gravity(3), velocity_command(3), root_linvel(3),
        foot_telemetry(8), joint_qpos(n), joint_qvel(n), last_action(n)

    where ``n`` is the number of LEG joints. Fields the real backend does
    not supply (commanded velocity, base linear velocity, foot contact /
    slip / gait phase) are left zero — the policy was trained with
    observation noise + domain randomization, so a zero-filled boundary is
    tolerated, but the joint positions/velocities MUST land at the indices
    the policy expects or the deployed behaviour is garbage.
    """

    proprio: np.ndarray = np.zeros(proprio_dim, dtype=np.float32)
    if latest is None:
        return proprio

    # gyro(3) — angular velocity proxy from IMU.
    if proprio_dim >= 1:
        proprio[0] = float(latest.get("imu_roll_rate", latest.get("imu_roll", 0.0)))
    if proprio_dim >= 2:
        proprio[1] = float(latest.get("imu_pitch_rate", latest.get("imu_pitch", 0.0)))
    if proprio_dim >= 3:
        proprio[2] = float(latest.get("imu_yaw_rate", 0.0))
    # gravity(3) at [3:6] — world up in the body frame; default upright.
    if proprio_dim >= 6:
        proprio[3:6] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # velocity_command(3) at [6:9] — the matched task's commanded
    # [vx, vy, vyaw]. The policy is conditioned on this during training
    # (TextConditionedProfileEnv._build_obs writes target_velocity_* here), so
    # zeroing it is out-of-distribution on the primary locomotion signal and
    # makes the deployed policy under-track. Inject the real command.
    if velocity_command is not None and proprio_dim >= 9:
        proprio[6:9] = np.asarray(velocity_command, dtype=np.float32).reshape(-1)[:3]
    # root_linvel(3) at [9:12] and foot_telemetry(8) at [12:20] are not
    # available from telemetry.basic and remain zero-filled.

    action_joints = [j.name for j in profile.kinematics.joints if j.group == "LEG"]
    joint_positions = latest.get("joint_positions") or {}
    joint_velocities = latest.get("joint_velocities") or {}
    if not isinstance(joint_positions, dict):
        joint_positions = {}
    if not isinstance(joint_velocities, dict):
        joint_velocities = {}

    # joint_qpos / joint_qvel begin after gyro+gravity+vel_cmd+root_linvel
    # +foot_telemetry = 3+3+3+3+8 = 20 (see TextConditionedProfileEnv).
    qpos_start = 20
    qvel_start = qpos_start + len(action_joints)
    last_action_start = qvel_start + len(action_joints)
    for i, name in enumerate(action_joints):
        qpos_idx = qpos_start + i
        qvel_idx = qvel_start + i
        if qpos_idx < proprio_dim:
            proprio[qpos_idx] = float(joint_positions.get(name, 0.0))
        if qvel_idx < proprio_dim:
            proprio[qvel_idx] = float(joint_velocities.get(name, 0.0))
    # last_action(n): the policy was trained with its own previous normalized
    # action as the final proprio block. Feeding zeros here is an out-of-
    # distribution input every step, so we thread the prior step's leg action
    # back in (zeros only on the first tick).
    if last_action is not None:
        la = np.asarray(last_action, dtype=np.float32).reshape(-1)
        for i in range(len(action_joints)):
            idx = last_action_start + i
            if idx < proprio_dim and i < la.shape[0]:
                proprio[idx] = float(la[i])
    return proprio


async def _read_proprio(
    backend: BridgeBackend,
    profile: RobotProfile,
    *,
    proprio_dim: int,
    last_action: np.ndarray | None = None,
    velocity_command: np.ndarray | None = None,
    supervisor: MotionSafetySupervisor | None = None,
    telemetry_reader: Callable[[], Awaitable[list[EventEnvelope]]] | None = None,
) -> np.ndarray:
    """Pull the latest telemetry.basic and convert to a proprio vector
    that's roughly compatible with the profile-driven text-conditioned env.
    We zero-pad when the real backend doesn't supply all fields.
    """
    events = (
        await telemetry_reader() if telemetry_reader is not None else await backend.poll_events()
    )
    latest = None
    for e in events:
        if supervisor is not None:
            violation = supervisor.telemetry_violation(e)
            if violation is not None:
                stop_response = await supervisor.emergency_stop_once("inference-telemetry")
                raise RuntimeError(
                    f"telemetry safety violation: {violation}; emergency_stop_ok={stop_response.ok}"
                )
        if e.event == "telemetry.basic":
            latest = e.data
    proprio = _proprio_from_telemetry(
        latest,
        profile,
        proprio_dim=proprio_dim,
        last_action=last_action,
        velocity_command=velocity_command,
    )
    if not np.all(np.isfinite(proprio)):
        raise RuntimeError("non-finite proprioception rejected")
    return proprio


def _finite_bounded(value: object, upper: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        parsed = float(value)
    except OverflowError:
        return None
    return parsed if np.isfinite(parsed) and 0.0 < parsed <= upper else None


def _validate_inference_config(config: InferenceLoopConfig) -> float:
    hz_value = _finite_bounded(config.hz, 1_000.0)
    if hz_value is None:
        raise ValueError("inference hz must be finite and in (0, 1000]")
    if (
        isinstance(config.max_steps, bool)
        or not isinstance(config.max_steps, int)
        or config.max_steps < 1
        or config.max_steps > 100_000
    ):
        raise ValueError("inference max_steps must be an integer in 1..100000")
    for name, value in (
        ("action_scale", config.action_scale),
        ("safety_clip_rad", config.safety_clip_rad),
    ):
        if _finite_bounded(value, 2 * math.pi) is None:
            raise ValueError(f"inference {name} must be finite and in (0, 2π]")
    return hz_value


def _validated_policy_action(action: object, joint_count: int) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.shape != (joint_count,):
        raise RuntimeError(
            "policy action shape mismatch: "
            f"expected {(joint_count,)}, got {action_array.shape}"
        )
    if not np.all(np.isfinite(action_array)):
        raise RuntimeError("non-finite policy action rejected")
    return action_array


@dataclass(frozen=True)
class PreparedInference:
    """Loaded and warmed policy state that has no motion authority."""

    config: InferenceLoopConfig
    profile: RobotProfile
    policy: TextConditionedPolicy
    joint_names: tuple[str, ...]
    home_rad: np.ndarray
    matched_task: str
    similarity: float
    velocity_command: np.ndarray
    action_scale: float
    leg_indices: tuple[int, ...]
    proprio_dim: int
    period: float


def prepare_inference(
    checkpoint_dir: str | Path,
    text: str,
    *,
    config: InferenceLoopConfig | None = None,
) -> PreparedInference:
    """Load, validate, and warm a policy without acquiring motion authority."""
    config = config or InferenceLoopConfig()
    hz_value = _validate_inference_config(config)
    profile = load_profile(config.profile_id)
    joint_names = tuple(joint.name for joint in profile.kinematics.joints)
    home_rad = np.array(
        [joint.home_rad for joint in profile.kinematics.joints],
        dtype=np.float32,
    )

    policy = TextConditionedPolicy(Path(checkpoint_dir), strict_manifest=True)
    if policy.manifest.profile_id != config.profile_id:
        raise ValueError(
            "checkpoint profile mismatch: "
            f"manifest profile_id={policy.manifest.profile_id!r}, "
            f"inference profile_id={config.profile_id!r}"
        )
    if int(policy.manifest.output_dim) != len(joint_names):
        raise ValueError(
            "checkpoint output_dim mismatch: "
            f"manifest output_dim={policy.manifest.output_dim}, "
            f"profile {config.profile_id!r} has {len(joint_names)} joints"
        )
    matched_task, _, similarity = policy.resolve_task(text)

    # The policy is conditioned on the task's commanded velocity (the env writes
    # target_velocity_* into proprio[6:9]). Resolve it before motion acquisition.
    from eliza_robot.curriculum.loader import load_curriculum

    velocity_command: np.ndarray = np.zeros(3, dtype=np.float32)
    for task in load_curriculum().tasks:
        if task.id == matched_task:
            reward = getattr(task, "reward", {}) or {}
            velocity_command = np.array(
                [
                    float(reward.get("target_velocity_x_m_s", 0.0)),
                    float(reward.get("target_velocity_y_m_s", 0.0)),
                    float(reward.get("target_yaw_rate_rad_s", 0.0)),
                ],
                dtype=np.float32,
            )
            break

    raw_action_scale = (
        policy.manifest.action_scale
        if policy.manifest.action_scale is not None
        else config.action_scale
    )
    action_scale = _finite_bounded(raw_action_scale, 2 * math.pi)
    if action_scale is None:
        raise ValueError("checkpoint action_scale must be finite and in (0, 2π]")

    leg_indices = tuple(
        index
        for index, joint in enumerate(profile.kinematics.joints)
        if joint.group == "LEG"
    )
    proprio_dim = int(policy.manifest.proprio_dim or 45)

    # Force lazy/JIT policy initialization now, while this request owns no
    # actuator resource. The synthetic action is discarded and never reaches a
    # backend; the first real action is recomputed from freshly revalidated
    # telemetry after atomic acquisition.
    warmup_proprio = _proprio_from_telemetry(
        None,
        profile,
        proprio_dim=proprio_dim,
        last_action=np.zeros(len(leg_indices), dtype=np.float32),
        velocity_command=velocity_command,
    )
    warmup_action, _ = policy.act(
        text,
        warmup_proprio,
        deterministic=True,
        output_dim=len(joint_names),
    )
    _validated_policy_action(warmup_action, len(joint_names))

    return PreparedInference(
        config=config,
        profile=profile,
        policy=policy,
        joint_names=joint_names,
        home_rad=home_rad,
        matched_task=matched_task,
        similarity=float(similarity),
        velocity_command=velocity_command,
        action_scale=action_scale,
        leg_indices=leg_indices,
        proprio_dim=proprio_dim,
        period=1.0 / hz_value,
    )


async def run_inference(
    backend: BridgeBackend,
    checkpoint_dir: str | Path,
    text: str,
    *,
    config: InferenceLoopConfig | None = None,
    supervisor: MotionSafetySupervisor | None = None,
    telemetry_reader: Callable[[], Awaitable[list[EventEnvelope]]] | None = None,
    expected_motion_generation: int | None = None,
) -> dict:
    """Prepare without authority, then atomically acquire and run one episode."""
    config = config or InferenceLoopConfig()
    _validate_inference_config(config)
    profile = load_profile(config.profile_id)
    supervisor = supervisor or MotionSafetySupervisor(
        backend,
        profile,
        owner_id=f"direct-inference-{id(backend)}-{time.time_ns()}",
    )
    if supervisor.emergency_stop_pending:
        raise RuntimeError("inference start blocked: emergency stop acknowledgement pending")
    capability_error = supervisor.policy_start_capability_error()
    if capability_error is not None:
        raise RuntimeError(f"inference start blocked: {capability_error}")

    # Capture freshness before potentially slow checkpoint/JIT work. A stop or
    # competing motion during preparation must invalidate this request rather
    # than silently turning old intent into a new acquisition.
    motion_generation_at_start = (
        supervisor.motion_generation
        if expected_motion_generation is None
        else expected_motion_generation
    )
    prepared = await asyncio.to_thread(
        prepare_inference,
        checkpoint_dir,
        text,
        config=config,
    )

    # Fetch once without authority so slow telemetry availability cannot start
    # the deadman. This also refreshes the trusted joint-pose baseline.
    prev_action_legs = np.zeros(len(prepared.leg_indices), dtype=np.float32)
    await _read_proprio(
        backend,
        prepared.profile,
        proprio_dim=prepared.proprio_dim,
        last_action=prev_action_legs,
        velocity_command=prepared.velocity_command,
        supervisor=supervisor,
        telemetry_reader=telemetry_reader,
    )

    # Capabilities and generation are revalidated immediately before atomic
    # acquisition. Physical policies remain blocked here by the hard-envelope
    # guard even if backend declarations changed while the checkpoint loaded.
    if supervisor.emergency_stop_pending:
        raise RuntimeError("inference start blocked: emergency stop acknowledgement pending")
    capability_error = supervisor.policy_start_capability_error()
    if capability_error is not None:
        raise RuntimeError(f"inference start blocked: {capability_error}")
    if not supervisor.accept_fresh_motion(motion_generation_at_start):
        raise RuntimeError(
            "inference start blocked: intent is stale, a stop is pending, "
            "or the motion resource is already owned"
        )

    logger.info(
        "inference loop start: text=%r → task=%s (sim=%.2f), %d steps @ %.1f Hz",
        text,
        prepared.matched_task,
        prepared.similarity,
        config.max_steps,
        config.hz,
    )

    steps = 0
    try:
        while steps < config.max_steps:
            t_start = time.time()
            # This second read happens under ownership: all advertised safety
            # telemetry is revalidated after acquisition and before motion.
            proprio = await _read_proprio(
                backend,
                prepared.profile,
                proprio_dim=prepared.proprio_dim,
                last_action=prev_action_legs,
                velocity_command=prepared.velocity_command,
                supervisor=supervisor,
                telemetry_reader=telemetry_reader,
            )
            action, _ = await asyncio.to_thread(
                prepared.policy.act,
                text,
                proprio,
                deterministic=True,
                output_dim=len(prepared.joint_names),
            )
            action_array = _validated_policy_action(action, len(prepared.joint_names))
            action_clipped = np.clip(action_array, -1.0, 1.0)
            targets = prepared.home_rad + action_clipped * prepared.action_scale
            targets = np.clip(
                targets,
                prepared.home_rad - config.safety_clip_rad,
                prepared.home_rad + config.safety_clip_rad,
            )
            joint_positions = {
                prepared.joint_names[index]: float(targets[index])
                for index in range(len(prepared.joint_names))
            }
            servo_duration = max(0.02, min(prepared.period, 0.06))
            servo_payload: JsonDict = {
                "duration": float(servo_duration),
                "joint_positions": joint_positions,
            }
            response = await supervisor.guarded_dispatch(
                CommandEnvelope(
                    request_id=f"infer-servo.set-{time.time_ns()}",
                    timestamp=utc_now_iso(),
                    command="servo.set",
                    payload=servo_payload,
                ),
                require_servo_capability=True,
            )
            if not response.ok:
                raise RuntimeError(f"servo.set rejected: {response.message}")
            if prepared.leg_indices:
                prev_action_legs = action_clipped[list(prepared.leg_indices)]
            steps += 1
            elapsed = time.time() - t_start
            await asyncio.sleep(max(0.0, prepared.period - elapsed))
    finally:
        stop_response = await supervisor.emergency_stop_once("inference-exit")
        if not stop_response.ok:
            raise RuntimeError(
                "inference exit emergency stop failed; motion ownership retained: "
                f"{stop_response.message}"
            )

    return {
        "text": text,
        "matched_task_id": prepared.matched_task,
        "similarity": prepared.similarity,
        "steps_completed": steps,
        "checkpoint": str(checkpoint_dir),
    }
