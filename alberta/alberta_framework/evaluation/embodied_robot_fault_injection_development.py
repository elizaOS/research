# mypy: disable-error-code="attr-defined,call-arg,arg-type,type-var"
"""Strict development-only synthetic robot fault-injection audit.

One frozen continuing schedule drives :class:`EmbodiedSafetyEnvelope` through
observation and wear drift, timing faults, delayed untrusted reward metadata,
sensor failures, bridge loss, unsafe candidates, emergency stop,
authority-token-bound reset/rollback, and exact checkpoint recovery.  This
kernel does not authenticate the caller; external caller authentication is a
required deployment responsibility.  The opaque caller
controller witness is carried unchanged for the entire life: this evaluator
does not own or reset learner state.

Only commands for which the envelope returns ``action_available`` are marked
as executed.  Execution is a simulated accounting fact, never physical
dispatch.  The lane has no comparator because a no-candidate arm would remove
the very intervention opportunities being audited and would not be matched.

The telemetry and command injections are a synthetic audit schedule, not a
dynamics simulator or geometry proof.  Recovery delays describe only when the
envelope next made a command available; learner adaptation is not tested.

This module has no RNG, seed, evaluator acceptance threshold, artifact writer, output path,
deployment authority, physical-safety claim, efficacy claim, or promotion
entry point.  Every assessment remains ``not_assessed``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.core.embodied_safety_envelope import (
    AuthorityBoundEnvelopeHandshake,
    EmbodiedCommand,
    EmbodiedEnvelopeDecision,
    EmbodiedSafetyEnvelope,
    EmbodiedSafetyEnvelopeConfig,
    EmbodiedSafetyEnvelopeState,
    EmbodiedTelemetry,
)

SCHEMA = "alberta.embodied-robot-fault-injection-development.v1"
CHECKPOINT_SCHEMA = "alberta.embodied-robot-fault-injection-development.checkpoint.v1"
ANCHOR_SCHEMA = "alberta.embodied-robot-fault-injection-development.anchor.v1"
PROTOCOL_NAMESPACE = "embodied-robot-continuing-fault-injection-v1"
ASSESSMENT = "not_assessed"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES_ALLOWED = False
ARTIFACT_WRITER_AVAILABLE = False
PHYSICAL_DISPATCHES = 0
PHYSICAL_SAFETY_CLAIM = False
DEPLOYMENT_AUTHORITY = False
EFFICACY_CLAIM = False
CALLER_AUTHENTICATION_PERFORMED = False
EXTERNAL_CALLER_AUTHENTICATION_REQUIRED = True
SYNTHETIC_TELEMETRY_AUDIT_SCHEDULE = True
DYNAMICS_SIMULATION_PERFORMED = False
GEOMETRY_PROOF = False
LEARNER_ADAPTATION_LATENCY_AVAILABLE = False
RECOVERY_DELAYS_ARE_ENVELOPE_ACTION_AVAILABILITY_ONLY = True
SIMULATED_COMMAND_EXECUTION_IS_ACCOUNTING_ONLY = True
SHADOW_SUCCESS_INPUT_IS_ACTION_AVAILABILITY_PROXY = True
RNG_DRAWS = 0
EVIDENCE_SEEDS: tuple[int, ...] = ()
ACCEPTANCE_THRESHOLDS: tuple[float, ...] = ()
COMPARISON_MODE = "single_strict_audit_lane"
NO_CANDIDATE_ARM_EXECUTED = False
NO_CANDIDATE_ARM_REASON = (
    "A no-candidate arm removes candidate-specific intervention opportunities "
    "and therefore is not a matched safety-envelope comparison."
)
FIXED_CHECKPOINT_SPLIT = 22

Operation = Literal["evaluate", "reset", "rollback"]

_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).parents[1] / "core" / "embodied_safety_envelope.py",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_safe(value: object) -> Any:
    return json.loads(_canonical_json_bytes(value))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _words(value: int) -> Array:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("identity must be a strict uint64-compatible integer")
    return jnp.asarray((value >> 32, value & 0xFFFFFFFF), dtype=jnp.uint32)


def _word_tuple(value: Array) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(jax.device_get(value)).reshape((-1,)))


def _digest_words(label: str) -> Array:
    raw = hashlib.sha256(label.encode("utf-8")).digest()
    return jnp.asarray(
        tuple(int.from_bytes(raw[offset : offset + 4], "little") for offset in range(0, 32, 4)),
        dtype=jnp.uint32,
    )


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).parents[2].resolve()
    result: dict[str, str] = {}
    for path in _SOURCE_PATHS:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(root).as_posix()
        except ValueError:
            label = resolved.name
        result[label] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def _runtime_manifest() -> dict[str, object]:
    devices = jax.devices()
    first = devices[0] if devices else None

    def package_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "unavailable"

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "byteorder": sys.byteorder,
        "jax": str(jax.__version__),
        "jaxlib": package_version("jaxlib"),
        "numpy": str(np.__version__),
        "chex": package_version("chex"),
        "default_backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kind": None if first is None else str(first.device_kind),
    }


@dataclasses.dataclass(frozen=True, slots=True)
class RobotFaultInjectionConfig:
    """Frozen evaluator dimensions; no field is an acceptance threshold."""

    num_events: int = 30
    checkpoint_split: int = FIXED_CHECKPOINT_SPLIT
    controller_revision: int = 73
    controller_state_identity: str = (
        "sha256:3c2fd7b19f547e84eaa9a850af00b726e0ff472df2a55dd87d83bbaa641d08ea"
    )
    shadow_calibration_error: float = 0.05

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "num_events": 30,
            "checkpoint_split": FIXED_CHECKPOINT_SPLIT,
            "controller_revision": 73,
            "controller_state_identity": (
                "sha256:3c2fd7b19f547e84eaa9a850af00b726e0ff472df2a55dd87d83bbaa641d08ea"
            ),
            "shadow_calibration_error": 0.05,
        }
        for name, expected_value in expected.items():
            value = getattr(self, name)
            if type(value) is not type(expected_value) or value != expected_value:
                raise ValueError(f"{name} is frozen at {expected_value!r} for {SCHEMA}")

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_safe(dataclasses.asdict(self)))


CONFIG = RobotFaultInjectionConfig()


@dataclasses.dataclass(frozen=True, slots=True)
class DeclaredHeldOutChangeFamily:
    """Machine-readable declaration only; this lane never executes it."""

    name: str
    changes: tuple[str, ...]
    declared: bool
    executed: bool
    event_count: int
    assessment: str
    evidence_seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_safe(dataclasses.asdict(self)))


HELD_OUT_CHANGE_FAMILY = DeclaredHeldOutChangeFamily(
    name="coupled-backlash-burst-loss-profile-change",
    changes=(
        "unseen asymmetric joint backlash and wear coupling",
        "bursty telemetry loss with delayed reconnect",
        "robot-profile workspace and torque-bound change",
    ),
    declared=True,
    executed=False,
    event_count=0,
    assessment=ASSESSMENT,
    evidence_seeds=(),
)


@dataclasses.dataclass(frozen=True, slots=True)
class FaultEvent:
    """One fully specified event in the continuing deterministic schedule."""

    index: int
    phase: str
    fault: str
    operation: Operation
    observation_drift: float
    wear_level: float
    telemetry_mode: str
    bridge_connected: bool
    emergency_stop: bool
    candidate_mode: str
    telemetry_age_ticks: int
    deadline_ahead_ticks: int
    reward_source_index: int
    reward_delay_events: int

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_safe(dataclasses.asdict(self)))


def build_fault_schedule(
    config: RobotFaultInjectionConfig = CONFIG,
) -> tuple[FaultEvent, ...]:
    """Build the sole frozen continuing development schedule."""

    specifications: tuple[
        tuple[str, str, Operation, float, float, str, bool, bool, str, int, int, int, int],
        ...,
    ] = (
        ("baseline", "none", "evaluate", 0.0, 0.0, "nominal", True, False, "safe", 1, 2, 0, 0),
        ("baseline", "none", "evaluate", 0.0, 0.0, "nominal", True, False, "safe", 1, 2, 1, 0),
        ("baseline", "none", "evaluate", 0.0, 0.0, "nominal", True, False, "safe", 1, 2, 2, 0),
        (
            "observation_drift",
            "positive_drift",
            "evaluate",
            0.10,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            3,
            0,
        ),
        (
            "observation_drift",
            "larger_positive_drift",
            "evaluate",
            0.20,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            4,
            0,
        ),
        (
            "observation_drift",
            "negative_drift",
            "evaluate",
            -0.20,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            5,
            0,
        ),
        (
            "dynamics_wear_drift",
            "low_wear",
            "evaluate",
            0.0,
            0.15,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            6,
            0,
        ),
        (
            "dynamics_wear_drift",
            "medium_wear",
            "evaluate",
            0.0,
            0.30,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            7,
            0,
        ),
        (
            "dynamics_wear_drift",
            "high_wear",
            "evaluate",
            0.0,
            0.45,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            8,
            0,
        ),
        (
            "timing_faults",
            "stale_telemetry",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            4,
            2,
            9,
            0,
        ),
        (
            "timing_faults",
            "deadline_miss",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            3,
            10,
            0,
        ),
        (
            "reward_delay",
            "delayed_untrusted_reward",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            9,
            2,
        ),
        (
            "sensor_faults",
            "sensor_nan",
            "evaluate",
            0.0,
            0.0,
            "nan",
            True,
            False,
            "safe",
            1,
            2,
            12,
            0,
        ),
        (
            "sensor_faults",
            "sensor_out_of_bounds",
            "evaluate",
            0.0,
            0.0,
            "out_of_bounds",
            True,
            False,
            "safe",
            1,
            2,
            13,
            0,
        ),
        (
            "sensor_faults",
            "sensor_failure",
            "evaluate",
            0.0,
            0.0,
            "failure",
            True,
            False,
            "safe",
            1,
            2,
            14,
            0,
        ),
        (
            "bridge_faults",
            "bridge_disconnect",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            False,
            False,
            "safe",
            1,
            2,
            15,
            0,
        ),
        (
            "bridge_faults",
            "bridge_reconnect",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            16,
            0,
        ),
        (
            "unsafe_candidates",
            "unsafe_position",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "unsafe_position",
            1,
            2,
            17,
            0,
        ),
        (
            "unsafe_candidates",
            "unsafe_torque",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "unsafe_torque",
            1,
            2,
            18,
            0,
        ),
        (
            "unsafe_candidates",
            "unsafe_clearance",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "unsafe_clearance",
            1,
            2,
            19,
            0,
        ),
        (
            "emergency_stop",
            "assert_stop",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            True,
            "safe",
            1,
            2,
            20,
            0,
        ),
        (
            "emergency_stop",
            "latched_stop",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            21,
            0,
        ),
        (
            "emergency_stop",
            "stationary_reset",
            "reset",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "none",
            1,
            2,
            -1,
            0,
        ),
        (
            "checkpoint_recovery",
            "post_restore_recovery",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            23,
            0,
        ),
        (
            "dynamics_wear_drift",
            "continued_high_wear",
            "evaluate",
            0.0,
            0.50,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            24,
            0,
        ),
        (
            "authority_control",
            "authority_rollback",
            "rollback",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "none",
            1,
            2,
            -1,
            0,
        ),
        (
            "authority_control",
            "suspended_after_rollback",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            26,
            0,
        ),
        (
            "authority_control",
            "stationary_reset_after_rollback",
            "reset",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "none",
            1,
            2,
            -1,
            0,
        ),
        (
            "final_recovery",
            "first_action_after_reset",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            28,
            0,
        ),
        (
            "final_recovery",
            "continuing_nominal",
            "evaluate",
            0.0,
            0.0,
            "nominal",
            True,
            False,
            "safe",
            1,
            2,
            29,
            0,
        ),
    )
    if len(specifications) != config.num_events:
        raise RuntimeError("frozen schedule length differs from its config")
    return tuple(FaultEvent(index, *spec) for index, spec in enumerate(specifications))


SCHEDULE = build_fault_schedule()


def _envelope_config() -> EmbodiedSafetyEnvelopeConfig:
    return EmbodiedSafetyEnvelopeConfig(
        n_joints=2,
        joint_position_lower=(-1.0, -1.0),
        joint_position_upper=(1.0, 1.0),
        max_abs_joint_velocity=(1.0, 1.0),
        max_abs_joint_torque=(2.0, 2.0),
        workspace_lower=(-1.0, -1.0, 0.0),
        workspace_upper=(1.0, 1.0, 2.0),
        min_collision_clearance=0.1,
        fallback_joint_position=(0.0, 0.0),
        fallback_joint_velocity=(0.0, 0.0),
        fallback_joint_torque=(0.0, 0.0),
        fallback_workspace_position=(0.0, 0.0, 1.0),
        fallback_collision_clearance=1.0,
        reset_stationary_velocity_tolerance=0.05,
        max_telemetry_age_ticks=3,
        max_control_deadline_ticks=2,
        shadow_window=6,
        min_shadow_samples=3,
        min_shadow_success_lcb=0.0,
        wilson_z=1.0,
        max_shadow_calibration_error=1.0,
        max_shadow_latency_ticks=8,
        max_decisions=64,
        max_committed_actions=64,
        max_shadow_records=64,
        max_handshakes_per_kind=8,
        reset_authority_digest=(1, 3, 5, 7, 9, 11, 13, 15),
        rollback_authority_digest=(2, 4, 6, 8, 10, 12, 14, 16),
    )


def _new_envelope() -> EmbodiedSafetyEnvelope:
    return EmbodiedSafetyEnvelope(_envelope_config())


def _source_digest() -> Array:
    return _digest_words(_digest(_source_manifest()))


_MODEL_VERSION = _digest_words("fault-injection-model-v1")
_OPTIMIZER_VERSION = _digest_words("fault-injection-optimizer-v1")
_LIFECYCLE_VERSION = _digest_words("fault-injection-lifecycle-v1")
_PARTNER_METADATA = _digest_words("fault-injection-partner-metadata-v1")


def _control_tick(event: FaultEvent) -> int:
    return 100 + 10 * event.index


def _reward(event: FaultEvent) -> float:
    if event.reward_source_index < 0:
        return 0.0
    return float((event.reward_source_index % 3) - 1)


def _telemetry(event: FaultEvent) -> EmbodiedTelemetry:
    now = _control_tick(event)
    position = (event.observation_drift, -0.5 * event.observation_drift)
    velocity = (event.wear_level, -0.5 * event.wear_level)
    torque = (0.5 + event.wear_level, -0.5)
    workspace = (0.5 * event.observation_drift, 0.0, 1.0)
    clearance = 0.5
    if event.telemetry_mode == "nan":
        position = (float("nan"), position[1])
    elif event.telemetry_mode == "out_of_bounds":
        position = (1.25, position[1])
    elif event.telemetry_mode == "failure":
        velocity = (float("nan"), float("nan"))
        clearance = float("nan")
    return EmbodiedTelemetry(
        joint_position=jnp.asarray(position, dtype=jnp.float32),
        joint_velocity=jnp.asarray(velocity, dtype=jnp.float32),
        joint_torque=jnp.asarray(torque, dtype=jnp.float32),
        workspace_position=jnp.asarray(workspace, dtype=jnp.float32),
        collision_clearance=jnp.asarray(clearance, dtype=jnp.float32),
        bridge_connected=jnp.asarray(event.bridge_connected, dtype=jnp.bool_),
        emergency_stop=jnp.asarray(event.emergency_stop, dtype=jnp.bool_),
        telemetry_id=_words(event.index + 1),
        sample_tick=_words(now - event.telemetry_age_ticks),
    )


def _candidate(event: FaultEvent) -> EmbodiedCommand:
    position = (0.1 + event.observation_drift, -0.1)
    velocity = (0.1 + event.wear_level, -0.1)
    torque = (0.2 + event.wear_level, -0.2)
    workspace = (0.1 + 0.5 * event.observation_drift, 0.0, 1.0)
    clearance = 0.4
    if event.candidate_mode == "unsafe_position":
        position = (1.5, position[1])
    elif event.candidate_mode == "unsafe_torque":
        torque = (2.5, torque[1])
    elif event.candidate_mode == "unsafe_clearance":
        clearance = 0.01
    return EmbodiedCommand(
        joint_position=jnp.asarray(position, dtype=jnp.float32),
        joint_velocity=jnp.asarray(velocity, dtype=jnp.float32),
        joint_torque=jnp.asarray(torque, dtype=jnp.float32),
        workspace_position=jnp.asarray(workspace, dtype=jnp.float32),
        collision_clearance=jnp.asarray(clearance, dtype=jnp.float32),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CommandBits:
    joint_position: tuple[int, ...]
    joint_velocity: tuple[int, ...]
    joint_torque: tuple[int, ...]
    workspace_position: tuple[int, ...]
    collision_clearance: int

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_safe(dataclasses.asdict(self)))


@dataclasses.dataclass(frozen=True, slots=True)
class TelemetryBits:
    joint_position: tuple[int, ...]
    joint_velocity: tuple[int, ...]
    joint_torque: tuple[int, ...]
    workspace_position: tuple[int, ...]
    collision_clearance: int
    bridge_connected: bool
    emergency_stop: bool
    telemetry_id: tuple[int, ...]
    sample_tick: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_safe(dataclasses.asdict(self)))


def _float_bits(value: Array) -> tuple[int, ...]:
    host = np.asarray(jax.device_get(value), dtype=np.float32)
    return tuple(int(item) for item in host.view(np.uint32).reshape((-1,)))


def _command_bits(command: EmbodiedCommand) -> CommandBits:
    return CommandBits(
        joint_position=_float_bits(command.joint_position),
        joint_velocity=_float_bits(command.joint_velocity),
        joint_torque=_float_bits(command.joint_torque),
        workspace_position=_float_bits(command.workspace_position),
        collision_clearance=_float_bits(command.collision_clearance)[0],
    )


def _telemetry_bits(telemetry: EmbodiedTelemetry) -> TelemetryBits:
    return TelemetryBits(
        joint_position=_float_bits(telemetry.joint_position),
        joint_velocity=_float_bits(telemetry.joint_velocity),
        joint_torque=_float_bits(telemetry.joint_torque),
        workspace_position=_float_bits(telemetry.workspace_position),
        collision_clearance=_float_bits(telemetry.collision_clearance)[0],
        bridge_connected=bool(telemetry.bridge_connected),
        emergency_stop=bool(telemetry.emergency_stop),
        telemetry_id=_word_tuple(telemetry.telemetry_id),
        sample_tick=_word_tuple(telemetry.sample_tick),
    )


def _state_to_dict(state: EmbodiedSafetyEnvelopeState) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in dataclasses.fields(EmbodiedSafetyEnvelopeState):
        host = np.asarray(jax.device_get(getattr(state, field.name)))
        result[field.name] = host.item() if host.shape == () else host.tolist()
    return cast(dict[str, object], _json_safe(result))


def _state_from_dict(
    envelope: EmbodiedSafetyEnvelope,
    payload: object,
) -> EmbodiedSafetyEnvelopeState:
    raw = _strict_mapping(payload, name="envelope state")
    template = envelope.init(source_digest=_source_digest())
    expected = {field.name for field in dataclasses.fields(EmbodiedSafetyEnvelopeState)}
    if set(raw) != expected:
        raise ValueError("serialized envelope state fields changed")
    values: dict[str, Any] = {}
    for field in dataclasses.fields(EmbodiedSafetyEnvelopeState):
        template_value = jnp.asarray(getattr(template, field.name))
        value = jnp.asarray(raw[field.name], dtype=template_value.dtype)
        if value.shape != template_value.shape:
            raise ValueError(f"serialized envelope state shape changed: {field.name}")
        values[field.name] = value
    state = EmbodiedSafetyEnvelopeState(**values)
    if not bool(jax.device_get(envelope.state_valid(state))):
        raise ValueError("serialized envelope state is invalid")
    return state


def _state_sha_hex(
    envelope: EmbodiedSafetyEnvelope,
    state: EmbodiedSafetyEnvelopeState,
) -> str:
    payload = envelope.checkpoint_payload(state)
    digest = np.asarray(jax.device_get(payload["state_digest"]), dtype=np.uint8)
    return bytes(digest.tolist()).hex()


def _tree_equal(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.array_equal(np.asarray(jax.device_get(a)), np.asarray(jax.device_get(b)))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class FaultTraceRecord:
    """Exact primitive causal facts for one schedule event."""

    event: FaultEvent
    controller_revision_before: int
    controller_revision_after: int
    controller_identity_before: str
    controller_identity_after: str
    learner_reset_count: int
    learner_state_mutations: int
    caller_authentication_performed: bool
    external_caller_authentication_required: bool
    synthetic_telemetry_audit_schedule: bool
    dynamics_simulation_performed: bool
    geometry_proof: bool
    simulated_command_execution_is_accounting_only: bool
    state_revision_before: int
    state_revision_after_operation: int
    state_revision_after_event: int
    state_checksum_before: tuple[int, ...]
    state_checksum_after: tuple[int, ...]
    telemetry: TelemetryBits | None
    proposed_command: CommandBits | None
    executed_command: CommandBits | None
    decision_id: tuple[int, ...]
    action_id: tuple[int, ...]
    control_tick: tuple[int, ...]
    deadline_tick: tuple[int, ...]
    reward_float32_bits: int
    reward_source_index: int
    reward_delay_events: int
    learned_cost_float32_bits: int
    model_version: tuple[int, ...]
    optimizer_version: tuple[int, ...]
    lifecycle_version: tuple[int, ...]
    partner_metadata_digest: tuple[int, ...]
    command_transaction_applied: bool
    action_available: bool
    command_executed_in_simulation: bool
    proposed_accepted: bool
    fallback_used: bool
    unavailable_reason: int
    persistent_state_valid: bool
    telemetry_fresh: bool
    control_deadline_valid: bool
    bridge_connected: bool
    current_envelope_safe: bool
    proposed_envelope_safe: bool
    fallback_certified: bool
    version_binding_valid: bool
    metadata_finite: bool
    hard_violation: bool
    emergency_stop_input: bool
    emergency_stop_latched_after: bool
    emergency_stop_latch_applied: bool
    reset_applied: bool
    reset_stationary_safe: bool
    rollback_applied: bool
    authority_unavailable_reason: int
    shadow_recorded: bool
    shadow_hard_violation: bool
    shadow_success_input: bool
    shadow_success_is_action_availability_proxy: bool
    shadow_action_would_be_available: bool
    shadow_sample_count: int
    shadow_hard_violation_count: int
    deployment_readout_ready: bool
    deployment_readout_authority: bool
    decision_count_after: int
    committed_action_count_after: int
    fallback_action_count_after: int
    rejected_action_count_after: int
    reset_count_after: int
    rollback_count_after: int
    checkpoint_resumed_before: bool
    checkpoint_revision: int
    checkpoint_state_digest: str
    checkpoint_restore_exact: bool
    physical_dispatches: int
    rng_draws: int

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_safe(dataclasses.asdict(self)))


def _core_evaluate(
    envelope: EmbodiedSafetyEnvelope,
    state: EmbodiedSafetyEnvelopeState,
    event: FaultEvent,
) -> EmbodiedEnvelopeDecision:
    now = _control_tick(event)
    return envelope.evaluate(
        state,
        _telemetry(event),
        _candidate(event),
        decision_id=_words(event.index + 1),
        action_id=_words(event.index + 1),
        control_tick=_words(now),
        control_deadline_tick=_words(now + event.deadline_ahead_ticks),
        model_version=_MODEL_VERSION,
        optimizer_version=_OPTIMIZER_VERSION,
        lifecycle_version=_LIFECYCLE_VERSION,
        untrusted_reward=float(_reward(event)),
        partner_metadata_digest=_PARTNER_METADATA,
        learned_cost_estimate=-1_000.0,
    )


def _restore_exact(
    envelope: EmbodiedSafetyEnvelope,
    state: EmbodiedSafetyEnvelopeState,
) -> tuple[EmbodiedSafetyEnvelopeState, int, str]:
    payload = envelope.checkpoint_payload(state)
    digest_array = jnp.asarray(payload["state_digest"], dtype=jnp.uint8)
    restored = envelope.restore_checkpoint(
        payload,
        expected_source_digest=_source_digest(),
        trusted_state_revision=state.revision,
        trusted_state_digest=digest_array,
    )
    return restored, int(state.revision), bytes(np.asarray(digest_array).tolist()).hex()


def _execute_event(
    envelope: EmbodiedSafetyEnvelope,
    state: EmbodiedSafetyEnvelopeState,
    event: FaultEvent,
    *,
    checkpoint_resumed_before: bool,
    checkpoint_revision: int,
    checkpoint_state_digest: str,
    checkpoint_restore_exact: bool,
) -> tuple[EmbodiedSafetyEnvelopeState, FaultTraceRecord]:
    before = state
    telemetry: EmbodiedTelemetry | None = None
    command: EmbodiedCommand | None = None
    executed: EmbodiedCommand | None = None
    now = _control_tick(event)
    decision_id = _words(event.index + 1) if event.operation == "evaluate" else _words(0)
    action_id = _words(event.index + 1) if event.operation == "evaluate" else _words(0)
    control_tick = _words(now) if event.operation != "rollback" else _words(0)
    deadline_tick = (
        _words(now + event.deadline_ahead_ticks) if event.operation != "rollback" else _words(0)
    )
    reward = _reward(event)

    command_transaction_applied = False
    action_available = False
    proposed_accepted = False
    fallback_used = False
    unavailable_reason = 0
    persistent_state_valid = bool(envelope.state_valid(state))
    telemetry_fresh = False
    deadline_valid = False
    bridge_connected = False
    current_safe = False
    proposed_safe = False
    fallback_certified = False
    version_binding_valid = False
    metadata_finite = False
    hard_violation = False
    emergency_stop_input = False
    emergency_stop_latched_after = bool(state.emergency_stop_latched)
    emergency_stop_latch_applied = False
    reset_applied = False
    reset_stationary_safe = False
    rollback_applied = False
    authority_unavailable_reason = 0
    shadow_recorded = False
    shadow_hard_violation = False
    shadow_success_input = False
    shadow_action_would_be_available = False

    if event.operation == "evaluate":
        telemetry = _telemetry(event)
        command = _candidate(event)
        decision_result = _core_evaluate(envelope, state, event)
        state = decision_result.state
        operation_revision = int(state.revision)
        command_transaction_applied = bool(decision_result.transaction_applied)
        action_available = bool(decision_result.action_available)
        proposed_accepted = bool(decision_result.proposed_accepted)
        fallback_used = bool(decision_result.fallback_used)
        unavailable_reason = int(decision_result.unavailable_reason)
        persistent_state_valid = bool(decision_result.persistent_state_valid)
        telemetry_fresh = bool(decision_result.telemetry_fresh)
        deadline_valid = bool(decision_result.control_deadline_valid)
        bridge_connected = bool(decision_result.bridge_connected)
        current_safe = bool(decision_result.current_envelope_safe)
        proposed_safe = bool(decision_result.proposed_envelope_safe)
        fallback_certified = bool(decision_result.fallback_certified)
        version_binding_valid = bool(decision_result.version_binding_valid)
        metadata_finite = bool(decision_result.metadata_finite)
        hard_violation = bool(decision_result.hard_violation)
        emergency_stop_input = bool(decision_result.emergency_stop_input)
        emergency_stop_latched_after = bool(decision_result.emergency_stop_latched_after)
        emergency_stop_latch_applied = bool(decision_result.emergency_stop_latch_applied)
        if action_available:
            executed = decision_result.command

        shadow = envelope.evaluate_shadow(
            state,
            telemetry,
            command,
            decision_id=_words(event.index + 1),
            control_tick=control_tick,
            control_deadline_tick=deadline_tick,
            model_version=_MODEL_VERSION,
            optimizer_version=_OPTIMIZER_VERSION,
            lifecycle_version=_LIFECYCLE_VERSION,
            observed_success=action_available,
            calibration_error=float(CONFIG.shadow_calibration_error),
            latency_ticks=event.telemetry_age_ticks,
            untrusted_reward=float(reward),
            partner_metadata_digest=_PARTNER_METADATA,
            learned_cost_estimate=-1_000.0,
        )
        recorded = envelope.record_shadow(state, shadow)
        state = recorded.state
        shadow_recorded = bool(recorded.applied)
        shadow_hard_violation = bool(shadow.hard_violation)
        shadow_success_input = bool(shadow.observed_success)
        shadow_action_would_be_available = bool(shadow.action_would_be_available)
    elif event.operation == "reset":
        telemetry = _telemetry(event)
        authority = jnp.asarray(
            envelope.config.reset_authority_digest,
            dtype=jnp.uint32,
        )
        handshake = AuthorityBoundEnvelopeHandshake(
            nonce=_words(int(state.reset_count) + 1),
            authority_digest=authority,
            source_digest=state.source_digest,
            config_digest=envelope.config_digest,
            observed_state_revision=state.revision,
            observed_state_checksum=state.state_checksum,
        )
        reset_result = envelope.authority_bound_reset(
            state,
            handshake,
            telemetry,
            control_tick=control_tick,
            control_deadline_tick=deadline_tick,
        )
        state = reset_result.state
        operation_revision = int(state.revision)
        reset_applied = bool(reset_result.applied)
        reset_stationary_safe = bool(reset_result.stationary_safe)
        authority_unavailable_reason = int(reset_result.unavailable_reason)
        emergency_stop_latched_after = bool(state.emergency_stop_latched)
    else:
        authority = jnp.asarray(
            envelope.config.rollback_authority_digest,
            dtype=jnp.uint32,
        )
        handshake = AuthorityBoundEnvelopeHandshake(
            nonce=_words(int(state.rollback_count) + 1),
            authority_digest=authority,
            source_digest=state.source_digest,
            config_digest=envelope.config_digest,
            observed_state_revision=state.revision,
            observed_state_checksum=state.state_checksum,
        )
        rollback_result = envelope.authority_bound_rollback(state, handshake)
        state = rollback_result.state
        operation_revision = int(state.revision)
        rollback_applied = bool(rollback_result.applied)
        authority_unavailable_reason = int(rollback_result.unavailable_reason)
        emergency_stop_latched_after = bool(state.emergency_stop_latched)

    gate = envelope.deployment_gate(state)
    record = FaultTraceRecord(
        event=event,
        controller_revision_before=CONFIG.controller_revision,
        controller_revision_after=CONFIG.controller_revision,
        controller_identity_before=CONFIG.controller_state_identity,
        controller_identity_after=CONFIG.controller_state_identity,
        learner_reset_count=0,
        learner_state_mutations=0,
        caller_authentication_performed=False,
        external_caller_authentication_required=True,
        synthetic_telemetry_audit_schedule=True,
        dynamics_simulation_performed=False,
        geometry_proof=False,
        simulated_command_execution_is_accounting_only=True,
        state_revision_before=int(before.revision),
        state_revision_after_operation=operation_revision,
        state_revision_after_event=int(state.revision),
        state_checksum_before=_word_tuple(before.state_checksum),
        state_checksum_after=_word_tuple(state.state_checksum),
        telemetry=None if telemetry is None else _telemetry_bits(telemetry),
        proposed_command=None if command is None else _command_bits(command),
        executed_command=None if executed is None else _command_bits(executed),
        decision_id=_word_tuple(decision_id),
        action_id=_word_tuple(action_id),
        control_tick=_word_tuple(control_tick),
        deadline_tick=_word_tuple(deadline_tick),
        reward_float32_bits=_float_bits(jnp.float32(reward))[0],
        reward_source_index=event.reward_source_index,
        reward_delay_events=event.reward_delay_events,
        learned_cost_float32_bits=_float_bits(jnp.float32(-1_000.0))[0],
        model_version=_word_tuple(_MODEL_VERSION) if event.operation == "evaluate" else (),
        optimizer_version=(
            _word_tuple(_OPTIMIZER_VERSION) if event.operation == "evaluate" else ()
        ),
        lifecycle_version=(
            _word_tuple(_LIFECYCLE_VERSION) if event.operation == "evaluate" else ()
        ),
        partner_metadata_digest=(
            _word_tuple(_PARTNER_METADATA) if event.operation == "evaluate" else ()
        ),
        command_transaction_applied=command_transaction_applied,
        action_available=action_available,
        command_executed_in_simulation=executed is not None,
        proposed_accepted=proposed_accepted,
        fallback_used=fallback_used,
        unavailable_reason=unavailable_reason,
        persistent_state_valid=persistent_state_valid,
        telemetry_fresh=telemetry_fresh,
        control_deadline_valid=deadline_valid,
        bridge_connected=bridge_connected,
        current_envelope_safe=current_safe,
        proposed_envelope_safe=proposed_safe,
        fallback_certified=fallback_certified,
        version_binding_valid=version_binding_valid,
        metadata_finite=metadata_finite,
        hard_violation=hard_violation,
        emergency_stop_input=emergency_stop_input,
        emergency_stop_latched_after=emergency_stop_latched_after,
        emergency_stop_latch_applied=emergency_stop_latch_applied,
        reset_applied=reset_applied,
        reset_stationary_safe=reset_stationary_safe,
        rollback_applied=rollback_applied,
        authority_unavailable_reason=authority_unavailable_reason,
        shadow_recorded=shadow_recorded,
        shadow_hard_violation=shadow_hard_violation,
        shadow_success_input=shadow_success_input,
        shadow_success_is_action_availability_proxy=True,
        shadow_action_would_be_available=shadow_action_would_be_available,
        shadow_sample_count=int(gate.sample_count),
        shadow_hard_violation_count=int(gate.hard_violation_count),
        deployment_readout_ready=bool(gate.deployment_ready),
        deployment_readout_authority=bool(gate.deployment_authority),
        decision_count_after=int(state.decision_count),
        committed_action_count_after=int(state.committed_action_count),
        fallback_action_count_after=int(state.fallback_action_count),
        rejected_action_count_after=int(state.rejected_action_count),
        reset_count_after=int(state.reset_count),
        rollback_count_after=int(state.rollback_count),
        checkpoint_resumed_before=checkpoint_resumed_before,
        checkpoint_revision=checkpoint_revision,
        checkpoint_state_digest=checkpoint_state_digest,
        checkpoint_restore_exact=checkpoint_restore_exact,
        physical_dispatches=0,
        rng_draws=0,
    )
    return state, record


def _run_schedule(
    *,
    start: int = 0,
    stop: int | None = None,
    state: EmbodiedSafetyEnvelopeState | None = None,
    prefix: tuple[FaultTraceRecord, ...] = (),
) -> tuple[EmbodiedSafetyEnvelope, EmbodiedSafetyEnvelopeState, tuple[FaultTraceRecord, ...]]:
    end = CONFIG.num_events if stop is None else stop
    if not 0 <= start <= end <= CONFIG.num_events:
        raise ValueError("run bounds are outside the frozen schedule")
    if len(prefix) != start:
        raise ValueError("trace prefix length must equal start")
    envelope = _new_envelope()
    current = envelope.init(source_digest=_source_digest()) if state is None else state
    if not bool(envelope.state_valid(current)):
        raise ValueError("starting envelope state is invalid")
    records = list(prefix)
    for event in SCHEDULE[start:end]:
        resumed = False
        checkpoint_revision = -1
        checkpoint_digest = ""
        restore_exact = False
        if event.index == CONFIG.checkpoint_split:
            before_restore = current
            current, checkpoint_revision, checkpoint_digest = _restore_exact(
                envelope,
                current,
            )
            resumed = True
            restore_exact = _tree_equal(before_restore, current)
            if not restore_exact:
                raise RuntimeError("fixed checkpoint probe failed exact restore")
        current, record = _execute_event(
            envelope,
            current,
            event,
            checkpoint_resumed_before=resumed,
            checkpoint_revision=checkpoint_revision,
            checkpoint_state_digest=checkpoint_digest,
            checkpoint_restore_exact=restore_exact,
        )
        records.append(record)
    return envelope, current, tuple(records)


def _kernel_parity_manifest() -> dict[str, object]:
    envelope = _new_envelope()
    initial = envelope.init(source_digest=_source_digest())
    first = SCHEDULE[0]
    eager_single = _core_evaluate(envelope, initial, first)
    jitted_single = jax.jit(lambda state: _core_evaluate(envelope, state, first))(initial)

    indices = (0, 10, 17)
    events = tuple(SCHEDULE[index] for index in indices)
    telemetry = jax.tree.map(lambda *values: jnp.stack(values), *(_telemetry(e) for e in events))
    commands = jax.tree.map(lambda *values: jnp.stack(values), *(_candidate(e) for e in events))
    decision_ids = jnp.stack(tuple(_words(e.index + 1) for e in events))
    action_ids = jnp.stack(tuple(_words(e.index + 1) for e in events))
    control_ticks = jnp.stack(tuple(_words(_control_tick(e)) for e in events))
    deadlines = jnp.stack(tuple(_words(_control_tick(e) + e.deadline_ahead_ticks) for e in events))
    rewards = jnp.asarray(tuple(_reward(e) for e in events), dtype=jnp.float32)
    costs = jnp.full((len(events),), -1_000.0, dtype=jnp.float32)
    inputs = (
        telemetry,
        commands,
        decision_ids,
        action_ids,
        control_ticks,
        deadlines,
        rewards,
        costs,
    )

    def scan_step(
        state: EmbodiedSafetyEnvelopeState,
        values: tuple[
            EmbodiedTelemetry,
            EmbodiedCommand,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[EmbodiedSafetyEnvelopeState, tuple[Array, ...]]:
        telemetry_value, command, decision, action, now, deadline, reward, cost = values
        result = envelope.evaluate(
            state,
            telemetry_value,
            command,
            decision_id=decision,
            action_id=action,
            control_tick=now,
            control_deadline_tick=deadline,
            model_version=_MODEL_VERSION,
            optimizer_version=_OPTIMIZER_VERSION,
            lifecycle_version=_LIFECYCLE_VERSION,
            untrusted_reward=reward,
            partner_metadata_digest=_PARTNER_METADATA,
            learned_cost_estimate=cost,
        )
        return result.state, (
            result.action_available,
            result.proposed_accepted,
            result.fallback_used,
            result.hard_violation,
            result.command.joint_position,
        )

    eager_state = initial
    eager_rows: list[tuple[Array, ...]] = []
    for index in range(len(events)):
        row = jax.tree.map(lambda value: value[index], inputs)
        eager_state, facts = scan_step(eager_state, row)
        eager_rows.append(facts)
    expected_rows = jax.tree.map(lambda *values: jnp.stack(values), *eager_rows)
    scanned_state, scanned_rows = jax.jit(
        lambda state, values: jax.lax.scan(scan_step, state, values)
    )(initial, inputs)
    return {
        "single_event_eager_jit_equal": _tree_equal(eager_single, jitted_single),
        "scan_final_state_equal": _tree_equal(eager_state, scanned_state),
        "scan_outputs_equal": _tree_equal(expected_rows, scanned_rows),
        "scan_event_indices": list(indices),
        "reset_and_checkpoint_host_orchestrated": True,
        "rng_draws": 0,
    }


@dataclasses.dataclass(frozen=True, slots=True)
class FaultRunSummary:
    """Descriptive accounting only; no field is an acceptance gate."""

    event_count: int
    evaluate_calls: int
    reset_calls: int
    rollback_calls: int
    shadow_evaluation_calls: int
    shadow_record_calls: int
    shadow_record_applied_count: int
    deployment_readout_calls: int
    proposed_command_count: int
    proposed_accepted_count: int
    fallback_count: int
    unavailable_count: int
    intervention_count: int
    simulated_executed_command_count: int
    physical_dispatch_count: int
    hard_violation_count_in_trace: int
    emergency_stop_assertion_count: int
    emergency_stop_latch_count: int
    successful_reset_count: int
    successful_rollback_count: int
    checkpoint_create_count: int
    checkpoint_restore_count: int
    bridge_reconnect_recovery_delay_events: int | None
    primary_reset_recovery_delay_events: int | None
    rollback_reset_recovery_delay_events: int | None
    adaptation_latency_assessed: bool
    adaptation_latency_events: int | None
    learner_adaptation_latency_unavailable_reason: str
    recovery_delays_are_envelope_action_availability_only: bool
    simulated_command_execution_is_accounting_only: bool
    shadow_success_input_is_action_availability_proxy: bool
    controller_revision_initial: int
    controller_revision_final: int
    controller_identity_initial: str
    controller_identity_final: str
    learner_reset_count: int
    learner_state_mutations: int
    final_envelope_revision: int
    final_decision_count: int
    final_committed_action_count: int
    final_fallback_action_count: int
    final_rejected_action_count: int
    final_shadow_record_count: int
    persistent_state_bytes: int
    joint_values_checked_per_evaluation: int
    workspace_values_checked_per_evaluation: int
    rng_draws: int

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_safe(dataclasses.asdict(self)))


def _first_available_delay(
    trace: tuple[FaultTraceRecord, ...],
    marker_fault: str,
) -> int | None:
    marker = next(
        (record.event.index for record in trace if record.event.fault == marker_fault), None
    )
    if marker is None:
        return None
    later = next(
        (
            record.event.index
            for record in trace
            if record.event.index >= marker and record.action_available
        ),
        None,
    )
    return None if later is None else later - marker


def _summarize(
    envelope: EmbodiedSafetyEnvelope,
    state: EmbodiedSafetyEnvelopeState,
    trace: tuple[FaultTraceRecord, ...],
) -> FaultRunSummary:
    evaluate = [record for record in trace if record.event.operation == "evaluate"]
    budget = envelope.resource_budget(state)
    return FaultRunSummary(
        event_count=len(trace),
        evaluate_calls=len(evaluate),
        reset_calls=sum(record.event.operation == "reset" for record in trace),
        rollback_calls=sum(record.event.operation == "rollback" for record in trace),
        shadow_evaluation_calls=len(evaluate),
        shadow_record_calls=len(evaluate),
        shadow_record_applied_count=sum(record.shadow_recorded for record in evaluate),
        deployment_readout_calls=len(trace),
        proposed_command_count=len(evaluate),
        proposed_accepted_count=sum(record.proposed_accepted for record in evaluate),
        fallback_count=sum(record.fallback_used for record in evaluate),
        unavailable_count=sum(not record.action_available for record in evaluate),
        intervention_count=sum(
            record.fallback_used or not record.action_available for record in evaluate
        ),
        simulated_executed_command_count=sum(
            record.command_executed_in_simulation for record in evaluate
        ),
        physical_dispatch_count=0,
        hard_violation_count_in_trace=sum(record.hard_violation for record in evaluate),
        emergency_stop_assertion_count=sum(record.emergency_stop_input for record in evaluate),
        emergency_stop_latch_count=int(state.emergency_stop_latch_count),
        successful_reset_count=sum(record.reset_applied for record in trace),
        successful_rollback_count=sum(record.rollback_applied for record in trace),
        checkpoint_create_count=sum(record.checkpoint_resumed_before for record in trace),
        checkpoint_restore_count=sum(record.checkpoint_restore_exact for record in trace),
        bridge_reconnect_recovery_delay_events=_first_available_delay(trace, "bridge_reconnect"),
        primary_reset_recovery_delay_events=_first_available_delay(trace, "stationary_reset"),
        rollback_reset_recovery_delay_events=_first_available_delay(
            trace, "stationary_reset_after_rollback"
        ),
        adaptation_latency_assessed=False,
        adaptation_latency_events=None,
        learner_adaptation_latency_unavailable_reason=(
            "The caller learner/controller witness is opaque and unchanged; "
            "learner adaptation is not exercised or measured."
        ),
        recovery_delays_are_envelope_action_availability_only=True,
        simulated_command_execution_is_accounting_only=True,
        shadow_success_input_is_action_availability_proxy=True,
        controller_revision_initial=CONFIG.controller_revision,
        controller_revision_final=CONFIG.controller_revision,
        controller_identity_initial=CONFIG.controller_state_identity,
        controller_identity_final=CONFIG.controller_state_identity,
        learner_reset_count=0,
        learner_state_mutations=0,
        final_envelope_revision=int(state.revision),
        final_decision_count=int(state.decision_count),
        final_committed_action_count=int(state.committed_action_count),
        final_fallback_action_count=int(state.fallback_action_count),
        final_rejected_action_count=int(state.rejected_action_count),
        final_shadow_record_count=int(state.shadow_record_count),
        persistent_state_bytes=budget.persistent_state_nbytes,
        joint_values_checked_per_evaluation=budget.joint_values_checked_per_evaluation,
        workspace_values_checked_per_evaluation=budget.workspace_values_checked_per_evaluation,
        rng_draws=0,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RobotFaultInjectionReport:
    """Self-bound deterministic report with no evidence or deployment authority."""

    schema: str
    namespace: str
    assessment: str
    development_only: bool
    scientific_promotion_allowed: bool
    output_writes_allowed: bool
    artifact_writer_available: bool
    physical_safety_claim: bool
    deployment_authority: bool
    efficacy_claim: bool
    caller_authentication_performed: bool
    external_caller_authentication_required: bool
    synthetic_telemetry_audit_schedule: bool
    dynamics_simulation_performed: bool
    geometry_proof: bool
    learner_adaptation_latency_available: bool
    recovery_delays_are_envelope_action_availability_only: bool
    simulated_command_execution_is_accounting_only: bool
    shadow_success_input_is_action_availability_proxy: bool
    physical_dispatches: int
    rng_draws: int
    evidence_seeds: tuple[int, ...]
    acceptance_thresholds: tuple[float, ...]
    comparison_mode: str
    no_candidate_arm_executed: bool
    no_candidate_arm_reason: str
    config: dict[str, object]
    envelope_config: dict[str, object]
    held_out_change_family: DeclaredHeldOutChangeFamily
    schedule_digest: str
    source_manifest: dict[str, str]
    runtime_manifest: dict[str, object]
    source_digest: tuple[int, ...]
    model_version: tuple[int, ...]
    optimizer_version: tuple[int, ...]
    lifecycle_version: tuple[int, ...]
    trace: tuple[FaultTraceRecord, ...]
    summary: FaultRunSummary
    kernel_parity: dict[str, object]
    final_state_digest: str
    deterministic_payload_digest: str

    def payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "namespace": self.namespace,
            "assessment": self.assessment,
            "development_only": self.development_only,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "output_writes_allowed": self.output_writes_allowed,
            "artifact_writer_available": self.artifact_writer_available,
            "physical_safety_claim": self.physical_safety_claim,
            "deployment_authority": self.deployment_authority,
            "efficacy_claim": self.efficacy_claim,
            "caller_authentication_performed": self.caller_authentication_performed,
            "external_caller_authentication_required": (
                self.external_caller_authentication_required
            ),
            "synthetic_telemetry_audit_schedule": self.synthetic_telemetry_audit_schedule,
            "dynamics_simulation_performed": self.dynamics_simulation_performed,
            "geometry_proof": self.geometry_proof,
            "learner_adaptation_latency_available": (self.learner_adaptation_latency_available),
            "recovery_delays_are_envelope_action_availability_only": (
                self.recovery_delays_are_envelope_action_availability_only
            ),
            "simulated_command_execution_is_accounting_only": (
                self.simulated_command_execution_is_accounting_only
            ),
            "shadow_success_input_is_action_availability_proxy": (
                self.shadow_success_input_is_action_availability_proxy
            ),
            "physical_dispatches": self.physical_dispatches,
            "rng_draws": self.rng_draws,
            "evidence_seeds": list(self.evidence_seeds),
            "acceptance_thresholds": list(self.acceptance_thresholds),
            "comparison_mode": self.comparison_mode,
            "no_candidate_arm_executed": self.no_candidate_arm_executed,
            "no_candidate_arm_reason": self.no_candidate_arm_reason,
            "config": self.config,
            "envelope_config": self.envelope_config,
            "held_out_change_family": self.held_out_change_family.to_dict(),
            "schedule_digest": self.schedule_digest,
            "source_manifest": self.source_manifest,
            "runtime_manifest": self.runtime_manifest,
            "source_digest": list(self.source_digest),
            "model_version": list(self.model_version),
            "optimizer_version": list(self.optimizer_version),
            "lifecycle_version": list(self.lifecycle_version),
            "trace": [record.to_dict() for record in self.trace],
            "summary": self.summary.to_dict(),
            "kernel_parity": self.kernel_parity,
            "final_state_digest": self.final_state_digest,
        }
        if include_digest:
            payload["deterministic_payload_digest"] = self.deterministic_payload_digest
        return payload


def _assemble_report(
    envelope: EmbodiedSafetyEnvelope,
    state: EmbodiedSafetyEnvelopeState,
    trace: tuple[FaultTraceRecord, ...],
) -> RobotFaultInjectionReport:
    kwargs: dict[str, object] = {
        "schema": SCHEMA,
        "namespace": PROTOCOL_NAMESPACE,
        "assessment": ASSESSMENT,
        "development_only": DEVELOPMENT_ONLY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "output_writes_allowed": OUTPUT_WRITES_ALLOWED,
        "artifact_writer_available": ARTIFACT_WRITER_AVAILABLE,
        "physical_safety_claim": PHYSICAL_SAFETY_CLAIM,
        "deployment_authority": DEPLOYMENT_AUTHORITY,
        "efficacy_claim": EFFICACY_CLAIM,
        "caller_authentication_performed": CALLER_AUTHENTICATION_PERFORMED,
        "external_caller_authentication_required": (EXTERNAL_CALLER_AUTHENTICATION_REQUIRED),
        "synthetic_telemetry_audit_schedule": SYNTHETIC_TELEMETRY_AUDIT_SCHEDULE,
        "dynamics_simulation_performed": DYNAMICS_SIMULATION_PERFORMED,
        "geometry_proof": GEOMETRY_PROOF,
        "learner_adaptation_latency_available": LEARNER_ADAPTATION_LATENCY_AVAILABLE,
        "recovery_delays_are_envelope_action_availability_only": (
            RECOVERY_DELAYS_ARE_ENVELOPE_ACTION_AVAILABILITY_ONLY
        ),
        "simulated_command_execution_is_accounting_only": (
            SIMULATED_COMMAND_EXECUTION_IS_ACCOUNTING_ONLY
        ),
        "shadow_success_input_is_action_availability_proxy": (
            SHADOW_SUCCESS_INPUT_IS_ACTION_AVAILABILITY_PROXY
        ),
        "physical_dispatches": PHYSICAL_DISPATCHES,
        "rng_draws": RNG_DRAWS,
        "evidence_seeds": EVIDENCE_SEEDS,
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "comparison_mode": COMPARISON_MODE,
        "no_candidate_arm_executed": NO_CANDIDATE_ARM_EXECUTED,
        "no_candidate_arm_reason": NO_CANDIDATE_ARM_REASON,
        "config": CONFIG.to_dict(),
        "envelope_config": cast(dict[str, object], _json_safe(envelope.to_config())),
        "held_out_change_family": HELD_OUT_CHANGE_FAMILY,
        "schedule_digest": _digest([event.to_dict() for event in SCHEDULE]),
        "source_manifest": _source_manifest(),
        "runtime_manifest": _runtime_manifest(),
        "source_digest": _word_tuple(_source_digest()),
        "model_version": _word_tuple(_MODEL_VERSION),
        "optimizer_version": _word_tuple(_OPTIMIZER_VERSION),
        "lifecycle_version": _word_tuple(_LIFECYCLE_VERSION),
        "trace": trace,
        "summary": _summarize(envelope, state, trace),
        "kernel_parity": _kernel_parity_manifest(),
        "final_state_digest": _state_sha_hex(envelope, state),
    }
    provisional = RobotFaultInjectionReport(
        **cast(dict[str, Any], kwargs),
        deterministic_payload_digest="",
    )
    return dataclasses.replace(
        provisional,
        deterministic_payload_digest=_digest(provisional.payload(include_digest=False)),
    )


def _run_unvalidated() -> RobotFaultInjectionReport:
    envelope, state, trace = _run_schedule()
    return _assemble_report(envelope, state, trace)


def run_embodied_robot_fault_injection_development() -> RobotFaultInjectionReport:
    """Execute and strictly validate the frozen nonpromoting audit lane."""

    report = _run_unvalidated()
    errors = validate_embodied_robot_fault_injection_report(report)
    if errors:
        raise RuntimeError("invalid robot fault-injection report: " + "; ".join(errors))
    return report


def validate_embodied_robot_fault_injection_report(report: object) -> tuple[str, ...]:
    """Reject policy drift, digest tamper, and any difference from causal replay."""

    if not isinstance(report, RobotFaultInjectionReport):
        return ("report has the wrong type",)
    errors: list[str] = []
    if report.schema != SCHEMA:
        errors.append("report schema changed")
    if report.namespace != PROTOCOL_NAMESPACE:
        errors.append("protocol namespace changed")
    if report.assessment != ASSESSMENT:
        errors.append("assessment must remain not_assessed")
    if not report.development_only:
        errors.append("development_only must remain true")
    if report.scientific_promotion_allowed:
        errors.append("scientific promotion is forbidden")
    if report.output_writes_allowed or report.artifact_writer_available:
        errors.append("artifact or output writes are forbidden")
    if report.physical_safety_claim or report.deployment_authority or report.efficacy_claim:
        errors.append("safety, deployment, and efficacy authority must remain false")
    if report.caller_authentication_performed:
        errors.append("this kernel must not claim caller authentication")
    if not report.external_caller_authentication_required:
        errors.append("external caller authentication must remain required")
    if (
        not report.synthetic_telemetry_audit_schedule
        or report.dynamics_simulation_performed
        or report.geometry_proof
    ):
        errors.append("synthetic audit scope must not become simulation or geometry proof")
    if (
        report.learner_adaptation_latency_available
        or not report.recovery_delays_are_envelope_action_availability_only
    ):
        errors.append("recovery descriptors must not become learner adaptation claims")
    if (
        not report.simulated_command_execution_is_accounting_only
        or not report.shadow_success_input_is_action_availability_proxy
    ):
        errors.append("simulation accounting and shadow-success proxy scope must remain explicit")
    if report.physical_dispatches != 0 or report.rng_draws != 0:
        errors.append("physical dispatch and RNG counts must remain zero")
    if report.evidence_seeds or report.acceptance_thresholds:
        errors.append("evidence seeds and acceptance thresholds are forbidden")
    if report.held_out_change_family.executed or report.held_out_change_family.event_count:
        errors.append("the declared held-out family must remain unexecuted")
    if report.no_candidate_arm_executed or report.comparison_mode != COMPARISON_MODE:
        errors.append("the unmatched no-candidate arm must remain unexecuted")
    if any(
        record.command_executed_in_simulation != record.action_available
        or record.physical_dispatches != 0
        or record.rng_draws != 0
        or record.learner_reset_count != 0
        or record.learner_state_mutations != 0
        or record.caller_authentication_performed
        or not record.external_caller_authentication_required
        or not record.synthetic_telemetry_audit_schedule
        or record.dynamics_simulation_performed
        or record.geometry_proof
        or not record.simulated_command_execution_is_accounting_only
        or not record.shadow_success_is_action_availability_proxy
        or (
            record.shadow_success_input != record.action_available
            if record.event.operation == "evaluate"
            else record.shadow_success_input
        )
        or record.controller_revision_before != CONFIG.controller_revision
        or record.controller_revision_after != CONFIG.controller_revision
        or record.controller_identity_before != CONFIG.controller_state_identity
        or record.controller_identity_after != CONFIG.controller_state_identity
        for record in report.trace
    ):
        errors.append("trace violates execution or uninterrupted-controller accounting")
    if (
        report.summary.adaptation_latency_assessed
        or report.summary.adaptation_latency_events is not None
        or not report.summary.recovery_delays_are_envelope_action_availability_only
        or not report.summary.simulated_command_execution_is_accounting_only
        or not report.summary.shadow_success_input_is_action_availability_proxy
    ):
        errors.append("summary overstates opaque-learner adaptation or recovery scope")
    actual_digest = _digest(report.payload(include_digest=False))
    if report.deterministic_payload_digest != actual_digest:
        errors.append("deterministic report digest mismatch")
    if errors:
        return tuple(errors)
    try:
        expected = _run_unvalidated()
    except Exception as exc:  # pragma: no cover - fail-closed diagnostic
        errors.append(f"causal replay failed: {type(exc).__name__}: {exc}")
        return tuple(errors)
    if report.payload() != expected.payload():
        errors.append("report differs from exact causal replay")
    return tuple(errors)


@dataclasses.dataclass(frozen=True, slots=True)
class ExternalRestoreAnchor:
    """Exact checkpoint pin that must be retained separately from the payload."""

    schema: str
    namespace: str
    next_event: int
    checkpoint_digest: str
    envelope_revision: int
    envelope_state_digest: str
    source_digest: tuple[int, ...]
    config_digest: tuple[int, ...]
    schedule_digest: str

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_safe(dataclasses.asdict(self)))

    @classmethod
    def from_object(cls, value: object) -> ExternalRestoreAnchor:
        raw = _strict_mapping(value, name="external restore anchor")
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(raw) != expected:
            raise ValueError("external restore anchor fields changed")
        source = raw["source_digest"]
        config = raw["config_digest"]
        if not isinstance(source, (list, tuple)) or len(source) != 8:
            raise ValueError("external source digest changed")
        if not isinstance(config, (list, tuple)) or len(config) != 8:
            raise ValueError("external config digest changed")
        if any(type(item) is not int or not 0 <= item <= 0xFFFFFFFF for item in source):
            raise ValueError("external source digest words changed")
        if any(type(item) is not int or not 0 <= item <= 0xFFFFFFFF for item in config):
            raise ValueError("external config digest words changed")
        next_event = raw["next_event"]
        revision = raw["envelope_revision"]
        if type(next_event) is not int or type(revision) is not int:
            raise ValueError("external anchor counters must be strict integers")
        strings = (
            "schema",
            "namespace",
            "checkpoint_digest",
            "envelope_state_digest",
            "schedule_digest",
        )
        if any(type(raw[name]) is not str for name in strings):
            raise ValueError("external anchor string field changed")
        for name in ("checkpoint_digest", "envelope_state_digest", "schedule_digest"):
            digest = cast(str, raw[name])
            if (
                len(digest) != 64
                or digest != digest.lower()
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"external anchor {name} is not canonical SHA-256 hex")
        return cls(
            schema=cast(str, raw["schema"]),
            namespace=cast(str, raw["namespace"]),
            next_event=next_event,
            checkpoint_digest=cast(str, raw["checkpoint_digest"]),
            envelope_revision=revision,
            envelope_state_digest=cast(str, raw["envelope_state_digest"]),
            source_digest=tuple(int(item) for item in source),
            config_digest=tuple(int(item) for item in config),
            schedule_digest=cast(str, raw["schedule_digest"]),
        )


def make_embodied_robot_fault_injection_checkpoint(
    next_event: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return a prefix checkpoint and a separately stored exact trust anchor."""

    if type(next_event) is not int or not 0 <= next_event <= CONFIG.num_events:
        raise ValueError("next_event must be a strict integer inside the frozen life")
    envelope, state, trace = _run_schedule(stop=next_event)
    payload: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "namespace": PROTOCOL_NAMESPACE,
        "next_event": next_event,
        "config_digest": _digest(CONFIG.to_dict()),
        "envelope_config_digest": _digest(_json_safe(envelope.to_config())),
        "schedule_digest": _digest([event.to_dict() for event in SCHEDULE]),
        "source_manifest": _source_manifest(),
        "runtime_manifest": _runtime_manifest(),
        "controller_revision": CONFIG.controller_revision,
        "controller_state_identity": CONFIG.controller_state_identity,
        "learner_reset_count": 0,
        "learner_state_mutations": 0,
        "envelope_state": _state_to_dict(state),
        "trace_prefix": [record.to_dict() for record in trace],
    }
    payload["checkpoint_digest"] = _digest(payload)
    anchor = ExternalRestoreAnchor(
        schema=ANCHOR_SCHEMA,
        namespace=PROTOCOL_NAMESPACE,
        next_event=next_event,
        checkpoint_digest=cast(str, payload["checkpoint_digest"]),
        envelope_revision=int(state.revision),
        envelope_state_digest=_state_sha_hex(envelope, state),
        source_digest=_word_tuple(state.source_digest),
        config_digest=_word_tuple(state.config_digest),
        schedule_digest=cast(str, payload["schedule_digest"]),
    )
    return payload, anchor.to_dict()


def _validate_checkpoint_and_restore(
    checkpoint: object,
    external_anchor: object,
) -> tuple[EmbodiedSafetyEnvelopeState, tuple[FaultTraceRecord, ...], int]:
    payload = _strict_mapping(checkpoint, name="checkpoint")
    expected_fields = {
        "schema",
        "namespace",
        "next_event",
        "config_digest",
        "envelope_config_digest",
        "schedule_digest",
        "source_manifest",
        "runtime_manifest",
        "controller_revision",
        "controller_state_identity",
        "learner_reset_count",
        "learner_state_mutations",
        "envelope_state",
        "trace_prefix",
        "checkpoint_digest",
    }
    if set(payload) != expected_fields:
        raise ValueError("checkpoint fields do not match the v1 schema")
    supplied_digest = payload["checkpoint_digest"]
    unsigned = dict(payload)
    unsigned.pop("checkpoint_digest")
    if supplied_digest != _digest(unsigned):
        raise ValueError("checkpoint digest mismatch")
    anchor = ExternalRestoreAnchor.from_object(external_anchor)
    if anchor.schema != ANCHOR_SCHEMA or anchor.namespace != PROTOCOL_NAMESPACE:
        raise ValueError("external restore anchor binding changed")
    if anchor.checkpoint_digest != supplied_digest:
        raise ValueError("checkpoint differs from the external trust anchor")
    next_event = payload["next_event"]
    if type(next_event) is not int or not 0 <= next_event <= CONFIG.num_events:
        raise ValueError("checkpoint next_event is invalid")
    if anchor.next_event != next_event:
        raise ValueError("checkpoint event differs from the external trust anchor")
    envelope = _new_envelope()
    expected_static = {
        "schema": CHECKPOINT_SCHEMA,
        "namespace": PROTOCOL_NAMESPACE,
        "config_digest": _digest(CONFIG.to_dict()),
        "envelope_config_digest": _digest(_json_safe(envelope.to_config())),
        "schedule_digest": _digest([event.to_dict() for event in SCHEDULE]),
        "source_manifest": _source_manifest(),
        "runtime_manifest": _runtime_manifest(),
        "controller_revision": CONFIG.controller_revision,
        "controller_state_identity": CONFIG.controller_state_identity,
        "learner_reset_count": 0,
        "learner_state_mutations": 0,
    }
    for name, expected in expected_static.items():
        if payload[name] != expected:
            raise ValueError(f"checkpoint {name} changed")
    state = _state_from_dict(envelope, payload["envelope_state"])
    if anchor.envelope_revision != int(state.revision):
        raise ValueError("checkpoint revision differs from the external trust anchor")
    if anchor.source_digest != _word_tuple(state.source_digest):
        raise ValueError("checkpoint source differs from the external trust anchor")
    if anchor.config_digest != _word_tuple(state.config_digest):
        raise ValueError("checkpoint config differs from the external trust anchor")
    if anchor.schedule_digest != payload["schedule_digest"]:
        raise ValueError("checkpoint schedule differs from the external trust anchor")
    try:
        trusted_digest = jnp.asarray(
            tuple(bytes.fromhex(anchor.envelope_state_digest)),
            dtype=jnp.uint8,
        )
    except ValueError as exc:
        raise ValueError("external state digest is not canonical hex") from exc
    core_payload = envelope.checkpoint_payload(state)
    restored = envelope.restore_checkpoint(
        core_payload,
        expected_source_digest=_source_digest(),
        trusted_state_revision=anchor.envelope_revision,
        trusted_state_digest=trusted_digest,
    )
    expected_envelope, expected_state, expected_trace = _run_schedule(stop=next_event)
    if _state_to_dict(restored) != _state_to_dict(expected_state):
        raise ValueError("checkpoint state differs from exact causal prefix replay")
    prefix_value = payload["trace_prefix"]
    if not isinstance(prefix_value, list):
        raise ValueError("checkpoint trace prefix must be a list")
    if prefix_value != [record.to_dict() for record in expected_trace]:
        raise ValueError("checkpoint trace differs from exact causal prefix replay")
    if expected_envelope.to_config() != envelope.to_config():
        raise ValueError("checkpoint envelope configuration changed")
    return restored, expected_trace, next_event


def resume_embodied_robot_fault_injection_checkpoint(
    checkpoint: object,
    external_anchor: object,
) -> RobotFaultInjectionReport:
    """Restore an externally pinned causal prefix and finish the same life."""

    state, prefix, next_event = _validate_checkpoint_and_restore(
        checkpoint,
        external_anchor,
    )
    envelope, final_state, trace = _run_schedule(
        start=next_event,
        state=state,
        prefix=prefix,
    )
    report = _assemble_report(envelope, final_state, trace)
    errors = validate_embodied_robot_fault_injection_report(report)
    if errors:
        raise RuntimeError("resumed robot fault-injection report is invalid: " + "; ".join(errors))
    return report


__all__ = [
    "ACCEPTANCE_THRESHOLDS",
    "ANCHOR_SCHEMA",
    "ARTIFACT_WRITER_AVAILABLE",
    "ASSESSMENT",
    "CALLER_AUTHENTICATION_PERFORMED",
    "CHECKPOINT_SCHEMA",
    "COMPARISON_MODE",
    "CONFIG",
    "DEVELOPMENT_ONLY",
    "DEPLOYMENT_AUTHORITY",
    "DYNAMICS_SIMULATION_PERFORMED",
    "EFFICACY_CLAIM",
    "EVIDENCE_SEEDS",
    "ExternalRestoreAnchor",
    "FaultEvent",
    "FaultRunSummary",
    "FaultTraceRecord",
    "FIXED_CHECKPOINT_SPLIT",
    "GEOMETRY_PROOF",
    "HELD_OUT_CHANGE_FAMILY",
    "LEARNER_ADAPTATION_LATENCY_AVAILABLE",
    "NO_CANDIDATE_ARM_EXECUTED",
    "NO_CANDIDATE_ARM_REASON",
    "OUTPUT_WRITES_ALLOWED",
    "PHYSICAL_DISPATCHES",
    "PHYSICAL_SAFETY_CLAIM",
    "PROTOCOL_NAMESPACE",
    "RECOVERY_DELAYS_ARE_ENVELOPE_ACTION_AVAILABILITY_ONLY",
    "RNG_DRAWS",
    "RobotFaultInjectionConfig",
    "RobotFaultInjectionReport",
    "SCHEMA",
    "SCHEDULE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SHADOW_SUCCESS_INPUT_IS_ACTION_AVAILABILITY_PROXY",
    "SIMULATED_COMMAND_EXECUTION_IS_ACCOUNTING_ONLY",
    "SYNTHETIC_TELEMETRY_AUDIT_SCHEDULE",
    "EXTERNAL_CALLER_AUTHENTICATION_REQUIRED",
    "build_fault_schedule",
    "make_embodied_robot_fault_injection_checkpoint",
    "resume_embodied_robot_fault_injection_checkpoint",
    "run_embodied_robot_fault_injection_development",
    "validate_embodied_robot_fault_injection_report",
]
