# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,no-untyped-call,type-var"
"""Strict reset-free embodied-dynamics adaptation development diagnostic.

An adaptive :class:`PrototypeAgent` and a capacity/update-call-matched frozen-
learning control each own a two-joint simulated plant and policy trajectory.
The evaluator pairs only evaluator-owned exogenous Threefry dynamics, sensor,
latency, and fault inputs.  It never pairs or overrides either policy's
endogenous action-selection randomness.

Every primitive decision becomes an :class:`EmbodiedCommand` and crosses the
non-learning :class:`EmbodiedSafetyEnvelope` before simulation.  A certified
fallback is rebound to the real Prototype credit owner through the public
cached-action replacement API.  An unavailable envelope action causes no
dynamics step and no learner transition.

This is a finite, consumed-data L0 development diagnostic.  It writes no
files, dispatches no physical action, has no untouched held-out data, and is
always ``not_assessed``.  SHA-256 fields provide accidental-corruption and
source bindings, not authentication.  The hard envelope is not a physical-
safety certificate, geometry proof, or deployment authorization.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.extend import backend as jax_backend

from alberta_framework.core.embodied_safety_envelope import (
    EmbodiedCommand,
    EmbodiedSafetyEnvelope,
    EmbodiedSafetyEnvelopeConfig,
    EmbodiedSafetyEnvelopeState,
    EmbodiedTelemetry,
)
from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
    measure_prototype_agent_state_resources,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

EMBODIED_DYNAMICS_CONFIG_SCHEMA = "alberta.embodied-dynamics-adaptation.config.v1"
EMBODIED_DYNAMICS_PROTOCOL_SCHEMA = "alberta.embodied-dynamics-adaptation.protocol.v1"
EMBODIED_DYNAMICS_REPORT_SCHEMA = "alberta.embodied-dynamics-adaptation.report.v1"
EMBODIED_DYNAMICS_CHECKPOINT_SCHEMA = (
    "alberta.embodied-dynamics-adaptation.checkpoint.v1"
)

DEVELOPMENT_STATUS = "not_assessed"
ASSESSMENT_STATUS = "not_assessed"
OUTPUT_WRITES = False
PHYSICAL_DISPATCH_COUNT = 0
PHYSICAL_DISPATCH_AUTHORITY = False
DEPLOYMENT_AUTHORITY = False
PROMOTION_AUTHORITY = False
SCIENTIFIC_PROMOTION_ALLOWED = False
CHECKPOINT_HOST_ONLY = True
ORCHESTRATION_HOST_ONLY = True

ArmName = Literal["adaptive", "frozen_learning_control"]
ARM_ORDER: tuple[ArmName, ...] = ("adaptive", "frozen_learning_control")

FAULT_NONE = 0
FAULT_PROPOSAL_CORRUPTION = 1
FAULT_BRIDGE_DISCONNECT = 2
FAULT_STALE_TELEMETRY = 3
FAULT_SENSOR_NONFINITE = 4
FAULT_DEADLINE_MISS = 5
FAULT_NAMES = {
    FAULT_NONE: "none",
    FAULT_PROPOSAL_CORRUPTION: "proposal_corruption",
    FAULT_BRIDGE_DISCONNECT: "bridge_disconnect",
    FAULT_STALE_TELEMETRY: "stale_telemetry",
    FAULT_SENSOR_NONFINITE: "sensor_nonfinite",
    FAULT_DEADLINE_MISS: "deadline_miss",
}

_UINT32_MAX = 2**32 - 1
_MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
_MAX_REPORT_BYTES = 48 * 1024 * 1024
_FIXED_PHASE_STEPS = 3
_FIXED_CHANGE_FAMILY_STEPS = 3
FIXED_CHECKPOINT_SPLIT = 6
_N_JOINTS = 2
_OBSERVATION_DIM = 8
_N_ACTIONS = 4

_SOURCE_PATHS = (
    Path("alberta_framework/core/embodied_safety_envelope.py"),
    Path("alberta_framework/core/multi_head_learner.py"),
    Path("alberta_framework/core/oak.py"),
    Path("alberta_framework/core/options.py"),
    Path("alberta_framework/core/optimizers.py"),
    Path("alberta_framework/core/prototype_agent.py"),
    Path("alberta_framework/evaluation/embodied_dynamics_adaptation_development.py"),
)

_LIMITATIONS = (
    "finite consumed development data only; every status is not_assessed",
    "the separately declared change family is consumed and never held out or promotable",
    "independent policy trajectories may diverge under paired exogenous inputs",
    "the frozen arm matches capacity and update calls, not realized experience",
    "the synthetic two-joint kernel is not a robot, geometry proof, or system ID result",
    "the hard envelope is not a physical-safety certificate",
    "simulated command accounting is not physical dispatch",
    "SHA-256 integrity and source bindings are not authentication",
    "the selected mechanism-source manifest is not a complete transitive dependency lock",
    "no result grants policy, safety, deployment, evidence, or promotion authority",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_equal(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if type(expected) in {bool, int, float, str}:
        return type(actual) is type(expected) and actual == expected
    if type(expected) is list:
        return (
            type(actual) is list
            and len(cast(list[object], actual)) == len(cast(list[object], expected))
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(
                    cast(list[object], actual),
                    cast(list[object], expected),
                    strict=True,
                )
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = _UINT32_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


def _exact_finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embodied_dynamics_source_manifest(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash selected exercised mechanism sources; exact replay remains authoritative."""

    return {path.as_posix(): _file_sha256(root / path) for path in _SOURCE_PATHS}


def embodied_dynamics_runtime_identity() -> dict[str, object]:
    """Return observable, non-secret runtime provenance."""

    devices = tuple(jax.devices())
    backend = jax_backend.get_backend()
    return {
        "identity_scope": (
            "observable-nonsecret-python-jax-xla-device-and-config-fields; "
            "exact causal replay remains authoritative"
        ),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "chex_version": version("chex"),
        "jax_version": str(jax.__version__),
        "jaxlib_version": version("jaxlib"),
        "numpy_version": str(np.__version__),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "backend": str(backend.platform),
        "backend_platform_version": str(backend.platform_version),
        "device_count": len(devices),
        "local_device_count": int(jax.local_device_count()),
        "device_platforms": [str(device.platform) for device in devices],
        "device_kinds": [str(device.device_kind) for device in devices],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
        "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_disable_jit": bool(jax.config.jax_disable_jit),
        "jax_enable_checks": bool(jax.config.jax_enable_checks),
    }


@dataclasses.dataclass(frozen=True, slots=True)
class EmbodiedDynamicsAdaptationConfig:
    """Frozen finite A/B/A plus consumed change-family development protocol."""

    seed: int = 9_173
    phase_steps: int = _FIXED_PHASE_STEPS
    change_family_steps: int = _FIXED_CHANGE_FAMILY_STEPS
    adaptive_base_step_size: float = 0.18
    adaptive_average_reward_step_size: float = 0.04
    adaptive_option_step_size: float = 0.12
    adaptive_option_model_step_size: float = 0.08
    discount: float = 1.0

    def __post_init__(self) -> None:
        _exact_int(self.seed, name="seed")
        if type(self.phase_steps) is not int or self.phase_steps != _FIXED_PHASE_STEPS:
            raise ValueError(f"phase_steps is frozen at {_FIXED_PHASE_STEPS}")
        if (
            type(self.change_family_steps) is not int
            or self.change_family_steps != _FIXED_CHANGE_FAMILY_STEPS
        ):
            raise ValueError(
                "change_family_steps is frozen at "
                f"{_FIXED_CHANGE_FAMILY_STEPS}"
            )
        frozen_floats = {
            "adaptive_base_step_size": 0.18,
            "adaptive_average_reward_step_size": 0.04,
            "adaptive_option_step_size": 0.12,
            "adaptive_option_model_step_size": 0.08,
            "discount": 1.0,
        }
        for name, expected in frozen_floats.items():
            actual = _exact_finite_float(getattr(self, name), name=name)
            if actual != expected:
                raise ValueError(f"{name} is frozen at {expected}")

    @property
    def total_events(self) -> int:
        return 3 * self.phase_steps + self.change_family_steps

    def to_config(self) -> dict[str, object]:
        return {
            "schema": EMBODIED_DYNAMICS_CONFIG_SCHEMA,
            "seed": self.seed,
            "phase_steps": self.phase_steps,
            "change_family_steps": self.change_family_steps,
            "adaptive_base_step_size": self.adaptive_base_step_size,
            "adaptive_average_reward_step_size": (
                self.adaptive_average_reward_step_size
            ),
            "adaptive_option_step_size": self.adaptive_option_step_size,
            "adaptive_option_model_step_size": (
                self.adaptive_option_model_step_size
            ),
            "discount": self.discount,
            "n_joints": _N_JOINTS,
            "observation_dim": _OBSERVATION_DIM,
            "n_actions": _N_ACTIONS,
            "assessment_status": ASSESSMENT_STATUS,
            "development_data_consumed": True,
            "untouched_held_out_data": False,
            "output_writes": False,
            "physical_dispatch_authority": False,
            "deployment_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, object],
    ) -> EmbodiedDynamicsAdaptationConfig:
        raw = _mapping(value, name="config")
        expected_fixed = cls().to_config()
        if set(raw) != set(expected_fixed):
            raise ValueError("embodied dynamics config fields differ")
        mutable_names = {
            "seed",
            "phase_steps",
            "change_family_steps",
            "adaptive_base_step_size",
            "adaptive_average_reward_step_size",
            "adaptive_option_step_size",
            "adaptive_option_model_step_size",
            "discount",
        }
        for name in set(expected_fixed) - mutable_names:
            if not _strict_json_equal(raw[name], expected_fixed[name]):
                raise ValueError(f"embodied dynamics fixed config field {name} differs")
        result = cls(**{name: raw[name] for name in mutable_names})
        if not _strict_json_equal(dict(raw), result.to_config()):
            raise ValueError("embodied dynamics config is noncanonical")
        return result


@chex.dataclass(frozen=True)
class EmbodiedDynamicsState:
    """Bounded hidden two-joint plant state with drift and command history."""

    joint_position: Array
    joint_velocity: Array
    actuator_gain: Array
    damping: Array
    coupling: Array
    wear: Array
    sensor_bias: Array
    command_history: Array
    step_count: Array
    step_words: Array


@chex.dataclass(frozen=True)
class EmbodiedExogenousEvent:
    """One typed evaluator-owned exogenous dynamics/sensor/fault event."""

    event_index: Array
    regime_code: Array
    target_position: Array
    gain_target: Array
    damping_target: Array
    coupling_target: Array
    gain_drift: Array
    damping_drift: Array
    coupling_drift: Array
    wear_drift: Array
    sensor_bias_drift: Array
    sensor_noise: Array
    latency_ticks: Array
    fault_code: Array
    key_words: Array


@chex.dataclass(frozen=True)
class EmbodiedDynamicsStepResult:
    """Atomic pure-kernel outcome; invalid input is an exact state no-op."""

    state: EmbodiedDynamicsState
    applied: Array
    delayed_target: Array
    acceleration: Array
    tracking_error: Array


@chex.dataclass(frozen=True)
class EmbodiedDynamicsScanResult:
    """Fixed-shape scan output for parity and pure simulation use."""

    state: EmbodiedDynamicsState
    applied: Array
    joint_positions: Array
    delayed_targets: Array
    tracking_errors: Array


def initial_embodied_dynamics_state() -> EmbodiedDynamicsState:
    """Return the frozen finite initial plant state."""

    return EmbodiedDynamicsState(
        joint_position=jnp.zeros((_N_JOINTS,), dtype=jnp.float32),
        joint_velocity=jnp.zeros((_N_JOINTS,), dtype=jnp.float32),
        actuator_gain=jnp.asarray((0.95, 0.90), dtype=jnp.float32),
        damping=jnp.asarray((0.28, 0.30), dtype=jnp.float32),
        coupling=jnp.asarray(0.04, dtype=jnp.float32),
        wear=jnp.zeros((_N_JOINTS,), dtype=jnp.float32),
        sensor_bias=jnp.zeros((_N_JOINTS,), dtype=jnp.float32),
        command_history=jnp.zeros((2, _N_JOINTS), dtype=jnp.float32),
        step_count=jnp.asarray(0, dtype=jnp.int32),
        step_words=jnp.zeros((2,), dtype=jnp.uint32),
    )


def _tree_finite(value: object) -> Array:
    finite = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(value):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            finite = finite & jnp.all(jnp.isfinite(array))
    return finite


def embodied_dynamics_state_valid(state: EmbodiedDynamicsState) -> Array:
    """Return the complete finite/static/range state contract."""

    static = (
        type(state) is EmbodiedDynamicsState
        and state.joint_position.shape == (_N_JOINTS,)
        and state.joint_position.dtype == jnp.float32
        and state.joint_velocity.shape == (_N_JOINTS,)
        and state.joint_velocity.dtype == jnp.float32
        and state.actuator_gain.shape == (_N_JOINTS,)
        and state.actuator_gain.dtype == jnp.float32
        and state.damping.shape == (_N_JOINTS,)
        and state.damping.dtype == jnp.float32
        and state.coupling.shape == ()
        and state.coupling.dtype == jnp.float32
        and state.wear.shape == (_N_JOINTS,)
        and state.wear.dtype == jnp.float32
        and state.sensor_bias.shape == (_N_JOINTS,)
        and state.sensor_bias.dtype == jnp.float32
        and state.command_history.shape == (2, _N_JOINTS)
        and state.command_history.dtype == jnp.float32
        and state.step_count.shape == ()
        and state.step_count.dtype == jnp.int32
        and state.step_words.shape == (2,)
        and state.step_words.dtype == jnp.uint32
    )
    exact_count = (
        state.step_words[0] == jnp.asarray(0, dtype=jnp.uint32)
    ) & (
        state.step_words[1]
        == jnp.maximum(state.step_count, jnp.asarray(0, dtype=jnp.int32)).astype(
            jnp.uint32
        )
    )
    ranges = (
        jnp.all(jnp.abs(state.joint_position) <= 0.9)
        & jnp.all(jnp.abs(state.joint_velocity) <= 0.8)
        & jnp.all((state.actuator_gain >= 0.25) & (state.actuator_gain <= 1.4))
        & jnp.all((state.damping >= 0.05) & (state.damping <= 0.8))
        & (jnp.abs(state.coupling) <= 0.25)
        & jnp.all((state.wear >= 0.0) & (state.wear <= 0.4))
        & jnp.all(jnp.abs(state.sensor_bias) <= 0.15)
        & (state.step_count >= 0)
    )
    return (
        jnp.asarray(static, dtype=jnp.bool_)
        & _tree_finite(state)
        & exact_count
        & ranges
    )


def _event_contract_valid(event: EmbodiedExogenousEvent) -> Array:
    static = (
        type(event) is EmbodiedExogenousEvent
        and event.event_index.shape == ()
        and event.event_index.dtype == jnp.int32
        and event.regime_code.shape == ()
        and event.regime_code.dtype == jnp.int32
        and event.target_position.shape == (_N_JOINTS,)
        and event.target_position.dtype == jnp.float32
        and event.gain_target.shape == (_N_JOINTS,)
        and event.gain_target.dtype == jnp.float32
        and event.damping_target.shape == (_N_JOINTS,)
        and event.damping_target.dtype == jnp.float32
        and event.coupling_target.shape == ()
        and event.coupling_target.dtype == jnp.float32
        and event.gain_drift.shape == (_N_JOINTS,)
        and event.gain_drift.dtype == jnp.float32
        and event.damping_drift.shape == (_N_JOINTS,)
        and event.damping_drift.dtype == jnp.float32
        and event.coupling_drift.shape == ()
        and event.coupling_drift.dtype == jnp.float32
        and event.wear_drift.shape == (_N_JOINTS,)
        and event.wear_drift.dtype == jnp.float32
        and event.sensor_bias_drift.shape == (_N_JOINTS,)
        and event.sensor_bias_drift.dtype == jnp.float32
        and event.sensor_noise.shape == (_N_JOINTS,)
        and event.sensor_noise.dtype == jnp.float32
        and event.latency_ticks.shape == ()
        and event.latency_ticks.dtype == jnp.int32
        and event.fault_code.shape == ()
        and event.fault_code.dtype == jnp.int32
        and event.key_words.shape == (4, 2)
        and event.key_words.dtype == jnp.uint32
    )
    return (
        jnp.asarray(static, dtype=jnp.bool_)
        & _tree_finite(event)
        & (event.event_index >= 0)
        & (event.regime_code >= 0)
        & (event.regime_code <= 2)
        & (event.latency_ticks >= 0)
        & (event.latency_ticks <= 2)
        & (event.fault_code >= FAULT_NONE)
        & (event.fault_code <= FAULT_DEADLINE_MISS)
    )


def _command_contract_valid(command: EmbodiedCommand) -> Array:
    static = (
        type(command) is EmbodiedCommand
        and command.joint_position.shape == (_N_JOINTS,)
        and command.joint_position.dtype == jnp.float32
        and command.joint_velocity.shape == (_N_JOINTS,)
        and command.joint_velocity.dtype == jnp.float32
        and command.joint_torque.shape == (_N_JOINTS,)
        and command.joint_torque.dtype == jnp.float32
        and command.workspace_position.shape == (3,)
        and command.workspace_position.dtype == jnp.float32
        and command.collision_clearance.shape == ()
        and command.collision_clearance.dtype == jnp.float32
    )
    return jnp.asarray(static, dtype=jnp.bool_) & _tree_finite(command)


def _advance_words(words: Array) -> tuple[Array, Array]:
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == 0).astype(jnp.uint32)
    high = words[0] + carry
    available = ~jnp.all(words == jnp.asarray(_UINT32_MAX, dtype=jnp.uint32))
    return jnp.stack((high, low)), available


def embodied_dynamics_step_kernel(
    state: EmbodiedDynamicsState,
    command: EmbodiedCommand,
    event: EmbodiedExogenousEvent,
) -> EmbodiedDynamicsStepResult:
    """Advance one bounded two-joint dynamics step or fail closed as a no-op."""

    proposed_words, capacity = _advance_words(state.step_words)
    valid = (
        embodied_dynamics_state_valid(state)
        & _command_contract_valid(command)
        & _event_contract_valid(event)
        & capacity
    )

    def apply(_: None) -> EmbodiedDynamicsStepResult:
        gain = jnp.clip(
            state.actuator_gain
            + jnp.float32(0.16) * (event.gain_target - state.actuator_gain)
            + event.gain_drift,
            jnp.float32(0.25),
            jnp.float32(1.4),
        )
        damping = jnp.clip(
            state.damping
            + jnp.float32(0.14) * (event.damping_target - state.damping)
            + event.damping_drift,
            jnp.float32(0.05),
            jnp.float32(0.8),
        )
        coupling = jnp.clip(
            state.coupling
            + jnp.float32(0.18) * (event.coupling_target - state.coupling)
            + event.coupling_drift,
            jnp.float32(-0.25),
            jnp.float32(0.25),
        )
        wear = jnp.clip(
            state.wear + jnp.float32(0.0025) + event.wear_drift,
            jnp.float32(0.0),
            jnp.float32(0.4),
        )
        sensor_bias = jnp.clip(
            state.sensor_bias + event.sensor_bias_drift,
            jnp.float32(-0.15),
            jnp.float32(0.15),
        )
        delayed_target = jnp.where(
            event.latency_ticks == 0,
            command.joint_position,
            jnp.where(
                event.latency_ticks == 1,
                state.command_history[0],
                state.command_history[1],
            ),
        )
        coupled_position = jnp.flip(state.joint_position)
        acceleration = (
            gain * (delayed_target - state.joint_position)
            - (damping + wear) * state.joint_velocity
            + coupling * (coupled_position - state.joint_position)
            + jnp.float32(0.04) * command.joint_torque
        )
        velocity = jnp.clip(
            state.joint_velocity + jnp.float32(0.20) * acceleration,
            jnp.float32(-0.8),
            jnp.float32(0.8),
        )
        position = jnp.clip(
            state.joint_position + jnp.float32(0.20) * velocity,
            jnp.float32(-0.9),
            jnp.float32(0.9),
        )
        next_state = EmbodiedDynamicsState(
            joint_position=position,
            joint_velocity=velocity,
            actuator_gain=gain,
            damping=damping,
            coupling=coupling,
            wear=wear,
            sensor_bias=sensor_bias,
            command_history=jnp.stack(
                (command.joint_position, state.command_history[0])
            ),
            step_count=jnp.minimum(
                state.step_count + jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(2_147_483_647, dtype=jnp.int32),
            ),
            step_words=proposed_words,
        )
        post_valid = embodied_dynamics_state_valid(next_state)
        committed = valid & post_valid
        return EmbodiedDynamicsStepResult(
            state=jax.lax.cond(committed, lambda __: next_state, lambda __: state, None),
            applied=committed,
            delayed_target=jnp.where(
                committed, delayed_target, jnp.zeros_like(delayed_target)
            ),
            acceleration=jnp.where(
                committed, acceleration, jnp.zeros_like(acceleration)
            ),
            tracking_error=jnp.where(
                committed,
                jnp.mean(jnp.square(position - event.target_position)),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
        )

    return jax.lax.cond(
        valid,
        apply,
        lambda _: EmbodiedDynamicsStepResult(
            state=state,
            applied=jnp.asarray(False, dtype=jnp.bool_),
            delayed_target=jnp.zeros((_N_JOINTS,), dtype=jnp.float32),
            acceleration=jnp.zeros((_N_JOINTS,), dtype=jnp.float32),
            tracking_error=jnp.asarray(0.0, dtype=jnp.float32),
        ),
        operand=None,
    )


def embodied_dynamics_scan_kernel(
    state: EmbodiedDynamicsState,
    commands: EmbodiedCommand,
    events: EmbodiedExogenousEvent,
) -> EmbodiedDynamicsScanResult:
    """Scan the pure dynamics kernel over leading command/event axes."""

    def step(
        carry: EmbodiedDynamicsState,
        values: tuple[EmbodiedCommand, EmbodiedExogenousEvent],
    ) -> tuple[EmbodiedDynamicsState, tuple[Array, Array, Array, Array]]:
        command, event = values
        result = embodied_dynamics_step_kernel(carry, command, event)
        return result.state, (
            result.applied,
            result.state.joint_position,
            result.delayed_target,
            result.tracking_error,
        )

    final, outputs = jax.lax.scan(step, state, (commands, events))
    return EmbodiedDynamicsScanResult(
        state=final,
        applied=outputs[0],
        joint_positions=outputs[1],
        delayed_targets=outputs[2],
        tracking_errors=outputs[3],
    )


def _phase_for_event(
    config: EmbodiedDynamicsAdaptationConfig,
    event_index: int,
) -> tuple[str, str, int]:
    _exact_int(
        event_index,
        name="event_index",
        maximum=config.total_events - 1,
    )
    if event_index < config.phase_steps:
        return "A_initial", "nominal_family_A", 0
    if event_index < 2 * config.phase_steps:
        return "B", "low_gain_high_damping_B", 1
    if event_index < 3 * config.phase_steps:
        return "A_return", "nominal_family_A", 0
    return "change_family_diagnostic", "asymmetric_coupled_family_C", 2


def _fault_for_event(event_index: int) -> int:
    return {
        1: FAULT_PROPOSAL_CORRUPTION,
        2: FAULT_BRIDGE_DISCONNECT,
        4: FAULT_STALE_TELEMETRY,
        5: FAULT_PROPOSAL_CORRUPTION,
        7: FAULT_SENSOR_NONFINITE,
        8: FAULT_PROPOSAL_CORRUPTION,
        9: FAULT_DEADLINE_MISS,
        10: FAULT_BRIDGE_DISCONNECT,
        11: FAULT_PROPOSAL_CORRUPTION,
    }.get(event_index, FAULT_NONE)


def _regime_targets(regime_code: int) -> tuple[Array, Array, Array]:
    if regime_code == 0:
        return (
            jnp.asarray((0.95, 0.90), dtype=jnp.float32),
            jnp.asarray((0.28, 0.30), dtype=jnp.float32),
            jnp.asarray(0.04, dtype=jnp.float32),
        )
    if regime_code == 1:
        return (
            jnp.asarray((0.48, 0.62), dtype=jnp.float32),
            jnp.asarray((0.46, 0.38), dtype=jnp.float32),
            jnp.asarray(-0.09, dtype=jnp.float32),
        )
    return (
        jnp.asarray((1.15, 0.35), dtype=jnp.float32),
        jnp.asarray((0.18, 0.55), dtype=jnp.float32),
        jnp.asarray(0.16, dtype=jnp.float32),
    )


def _words(value: int) -> Array:
    _exact_int(value, name="identity", maximum=2**64 - 1)
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _digest_words(label: str) -> Array:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return jnp.asarray(
        tuple(int.from_bytes(digest[index : index + 4], "big") for index in range(0, 32, 4)),
        dtype=jnp.uint32,
    )


def _array_payload(value: Array | npt.NDArray[Any]) -> dict[str, object]:
    if isinstance(value, Array) and jax.dtypes.issubdtype(
        value.dtype,
        jax.dtypes.prng_key,
    ):
        host = np.ascontiguousarray(np.asarray(jr.key_data(value), dtype=np.uint32))
        dtype = f"key<{jr.key_impl(value)}>"
    else:
        host = np.ascontiguousarray(np.asarray(jax.device_get(value)))
        if host.dtype.hasobject:
            raise TypeError("object arrays cannot enter an exact payload")
        dtype = host.dtype.str
    return {
        "dtype": dtype,
        "shape": list(host.shape),
        "data_hex": host.tobytes(order="C").hex(),
        "sha256": hashlib.sha256(host.tobytes(order="C")).hexdigest(),
    }


def _tree_payload(value: object) -> dict[str, object]:
    leaves_with_paths, tree_definition = jax.tree_util.tree_flatten_with_path(value)
    leaves: list[dict[str, object]] = []
    for path, leaf in leaves_with_paths:
        if isinstance(leaf, Array):
            payload = _array_payload(leaf)
        elif type(leaf) in {bool, int, float}:
            payload = _array_payload(np.asarray(leaf))
            payload["python_scalar_type"] = type(leaf).__name__
        else:
            raise TypeError(
                "unsupported exact tree leaf at "
                f"{jax.tree_util.keystr(path)}: {type(leaf)!r}"
            )
        leaves.append({"path": jax.tree_util.keystr(path), "array": payload})
    body: dict[str, object] = {
        "tree_definition": str(tree_definition),
        "leaves": leaves,
    }
    return {**body, "tree_sha256": _canonical_sha256(body)}


def _tree_sha256(value: object) -> str:
    return cast(str, _tree_payload(value)["tree_sha256"])


def _tree_bits_equal(left: object, right: object) -> bool:
    return _strict_json_equal(_tree_payload(left), _tree_payload(right))


def _tree_all_finite_host(value: object) -> bool:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, Array) and jax.dtypes.issubdtype(
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            continue
        array = np.asarray(jax.device_get(leaf))
        if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
            return False
    return True


def _command_payload(command: EmbodiedCommand) -> dict[str, object]:
    return {
        "joint_position": _array_payload(command.joint_position),
        "joint_velocity": _array_payload(command.joint_velocity),
        "joint_torque": _array_payload(command.joint_torque),
        "workspace_position": _array_payload(command.workspace_position),
        "collision_clearance": _array_payload(command.collision_clearance),
    }


def _telemetry_payload(telemetry: EmbodiedTelemetry) -> dict[str, object]:
    return {
        "joint_position": _array_payload(telemetry.joint_position),
        "joint_velocity": _array_payload(telemetry.joint_velocity),
        "joint_torque": _array_payload(telemetry.joint_torque),
        "workspace_position": _array_payload(telemetry.workspace_position),
        "collision_clearance": _array_payload(telemetry.collision_clearance),
        "bridge_connected": bool(np.asarray(telemetry.bridge_connected)),
        "emergency_stop": bool(np.asarray(telemetry.emergency_stop)),
        "telemetry_id": _array_payload(telemetry.telemetry_id),
        "sample_tick": _array_payload(telemetry.sample_tick),
    }


def _event_payload(
    event: EmbodiedExogenousEvent,
    *,
    phase: str,
    regime: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "event_index": int(np.asarray(event.event_index)),
        "phase": phase,
        "regime": regime,
        "regime_code": int(np.asarray(event.regime_code)),
        "target_position": _array_payload(event.target_position),
        "gain_target": _array_payload(event.gain_target),
        "damping_target": _array_payload(event.damping_target),
        "coupling_target": _array_payload(event.coupling_target),
        "gain_drift": _array_payload(event.gain_drift),
        "damping_drift": _array_payload(event.damping_drift),
        "coupling_drift": _array_payload(event.coupling_drift),
        "wear_drift": _array_payload(event.wear_drift),
        "sensor_bias_drift": _array_payload(event.sensor_bias_drift),
        "sensor_noise": _array_payload(event.sensor_noise),
        "latency_ticks": int(np.asarray(event.latency_ticks)),
        "fault_code": int(np.asarray(event.fault_code)),
        "fault_name": FAULT_NAMES[int(np.asarray(event.fault_code))],
        "key_words": _array_payload(event.key_words),
        "rng_impl": "threefry2x32",
        "paired_across_arms": True,
        "paired_scope": "exogenous_dynamics_sensor_fault_latency_only",
    }
    return {**body, "schedule_sha256": _canonical_sha256(body)}


def embodied_dynamics_protocol(
    config: EmbodiedDynamicsAdaptationConfig,
) -> dict[str, object]:
    """Return the immutable claim-limited development protocol."""

    if type(config) is not EmbodiedDynamicsAdaptationConfig:
        raise TypeError("config must be an exact EmbodiedDynamicsAdaptationConfig")
    return {
        "schema": EMBODIED_DYNAMICS_PROTOCOL_SCHEMA,
        "type": "EmbodiedDynamicsAdaptationDevelopmentProtocol",
        "arms": list(ARM_ORDER),
        "continuing": True,
        "task_labels_supplied": False,
        "learner_resets": 0,
        "environment_resets": 0,
        "a_b_a_phases": ["A_initial", "B", "A_return"],
        "change_family_diagnostic": {
            "name": "asymmetric_coupled_family_C",
            "declared_separately": True,
            "executed": True,
            "development_data_consumed": True,
            "untouched_held_out": False,
            "ever_promotable": False,
        },
        "common_randomness": {
            "rng_impl": "threefry2x32",
            "typed_keys": True,
            "paired_across_arms": True,
            "paired_scope": "exogenous_dynamics_sensor_fault_latency_only",
            "policy_action_randomness_paired": False,
            "trajectory_equality_assumed": False,
        },
        "each_arm_owns_policy_and_environment": True,
        "frozen_control_capacity_matched": True,
        "frozen_control_update_call_matched": True,
        "frozen_control_learning_rates_zero": True,
        "every_primitive_maps_to_two_joint_command": True,
        "every_potentially_executed_command_crosses_envelope": True,
        "fallback_rebinds_public_prototype_credit_owner": True,
        "unavailable_action_advances_dynamics": False,
        "unavailable_action_fabricates_transition": False,
        "simulated_commands_only": True,
        "physical_dispatch_count": 0,
        "physical_dispatch_authority": False,
        "orchestration_host_only": True,
        "pure_dynamics_kernel_jittable": True,
        "thresholds": [],
        "output_writes": False,
        "output_path": None,
        "assessment_status": ASSESSMENT_STATUS,
        "performance_claimed": False,
        "adaptation_efficacy_claimed": False,
        "safety_claimed": False,
        "deployment_authority": False,
        "evidence_claimed": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "config_sha256": _canonical_sha256(config.to_config()),
    }


def _prototype_config(
    config: EmbodiedDynamicsAdaptationConfig,
    *,
    adaptive: bool,
) -> PrototypeAgentConfig:
    if adaptive:
        base_step = config.adaptive_base_step_size
        avg_step = config.adaptive_average_reward_step_size
        option_step = config.adaptive_option_step_size
        option_model_step = config.adaptive_option_model_step_size
        model_decay = 0.82
        utility_decay = 0.90
    else:
        base_step = 0.0
        avg_step = 0.0
        option_step = 0.0
        option_model_step = 0.0
        model_decay = 1.0
        utility_decay = 1.0
    stomp = STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=0.45,
                pseudo_reward_scale=0.5,
                max_option_steps=4,
            ),
        ),
        observation_dim=_OBSERVATION_DIM,
        n_primitive_actions=_N_ACTIONS,
        base_step_size=base_step,
        base_avg_reward_step_size=avg_step,
        base_trace_decay=0.2,
        base_hidden_sizes=(),
        option_step_size=option_step,
        option_avg_reward_step_size=avg_step,
        option_trace_decay=0.1,
        option_gamma=0.98,
        option_model_decay=model_decay,
        option_model_step_size=option_model_step,
        option_planning_backups_per_step=0,
        epsilon_base=0.15,
        epsilon_option=0.15,
        option_target_epsilon=0.15,
        option_importance_clip=4.0,
    )
    return PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=stomp,
            utility_ema_decay=utility_decay,
            curation_threshold=0.0,
            min_steps_before_curation=0,
        ),
        buffer_capacity=16,
        n_dreams_per_step=0,
        auto_curate_every=0,
    )


def _envelope_config(config: EmbodiedDynamicsAdaptationConfig) -> EmbodiedSafetyEnvelopeConfig:
    capacity = config.total_events + 4
    return EmbodiedSafetyEnvelopeConfig(
        n_joints=_N_JOINTS,
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
        max_control_deadline_ticks=3,
        shadow_window=4,
        min_shadow_samples=3,
        min_shadow_success_lcb=0.0,
        wilson_z=1.0,
        max_shadow_calibration_error=1.0,
        max_shadow_latency_ticks=8,
        max_decisions=capacity,
        max_committed_actions=capacity,
        max_shadow_records=capacity,
        max_handshakes_per_kind=4,
        reset_authority_digest=(1, 3, 5, 7, 9, 11, 13, 15),
        rollback_authority_digest=(2, 4, 6, 8, 10, 12, 14, 16),
    )


def _primitive_command(action: int) -> EmbodiedCommand:
    if type(action) is not int or not 0 <= action < _N_ACTIONS:
        raise ValueError("primitive action is outside the frozen command table")
    positions = (
        (0.0, 0.0),
        (0.65, 0.35),
        (-0.65, 0.35),
        (0.0, -0.65),
    )
    position = jnp.asarray(positions[action], dtype=jnp.float32)
    return EmbodiedCommand(
        joint_position=position,
        joint_velocity=jnp.zeros((_N_JOINTS,), dtype=jnp.float32),
        joint_torque=jnp.asarray(
            (0.12 * float(position[0]), 0.12 * float(position[1])),
            dtype=jnp.float32,
        ),
        workspace_position=jnp.asarray(
            (float(position[0]), float(position[1]), 1.0),
            dtype=jnp.float32,
        ),
        collision_clearance=jnp.asarray(
            1.0 if action == 0 else 0.5,
            dtype=jnp.float32,
        ),
    )


def _corrupt_proposal(command: EmbodiedCommand) -> EmbodiedCommand:
    return cast(
        EmbodiedCommand,
        command.replace(
            joint_position=command.joint_position.at[0].set(
                jnp.asarray(1.25, dtype=jnp.float32)
            )
        ),
    )


def _observation(
    state: EmbodiedDynamicsState,
    event: EmbodiedExogenousEvent,
) -> Array:
    noise = jnp.float32(0.012) * event.sensor_noise
    position = state.joint_position + state.sensor_bias + noise
    velocity = state.joint_velocity + jnp.float32(0.35) * noise
    return jnp.concatenate(
        (
            position,
            velocity,
            event.target_position,
            jnp.asarray(
                (jnp.mean(state.actuator_gain), jnp.mean(state.wear)),
                dtype=jnp.float32,
            ),
        )
    )


def _telemetry(
    state: EmbodiedDynamicsState,
    event: EmbodiedExogenousEvent,
    *,
    control_tick: int,
) -> EmbodiedTelemetry:
    observed = _observation(state, event)
    position = observed[:2]
    velocity = observed[2:4]
    fault = int(np.asarray(event.fault_code))
    if fault == FAULT_SENSOR_NONFINITE:
        position = position.at[0].set(jnp.asarray(jnp.nan, dtype=jnp.float32))
    sample_tick = control_tick - int(np.asarray(event.latency_ticks))
    if fault == FAULT_STALE_TELEMETRY:
        sample_tick = control_tick - 10
    return EmbodiedTelemetry(
        joint_position=position,
        joint_velocity=velocity,
        joint_torque=jnp.asarray(
            (0.25 * float(state.joint_velocity[0]), 0.25 * float(state.joint_velocity[1])),
            dtype=jnp.float32,
        ),
        workspace_position=jnp.asarray(
            (float(position[0]), float(position[1]), 1.0),
            dtype=jnp.float32,
        ),
        collision_clearance=jnp.asarray(0.45, dtype=jnp.float32),
        bridge_connected=jnp.asarray(
            fault != FAULT_BRIDGE_DISCONNECT,
            dtype=jnp.bool_,
        ),
        emergency_stop=jnp.asarray(False, dtype=jnp.bool_),
        telemetry_id=_words(int(np.asarray(event.event_index)) + 1),
        sample_tick=_words(sample_tick),
    )


def _parameter_tree(state: PrototypeAgentState) -> object:
    oak = cast(OaKState, state.oak_state)
    stomp = oak.stomp_state
    learner = stomp.base_learner_state
    return (
        learner.trunk_params,
        learner.head_params,
        stomp.base_average_reward,
        stomp.option_policies.q_weights,
        stomp.option_policies.average_rewards,
        stomp.option_models.cumreward_ema,
        stomp.option_models.env_return_ema,
        stomp.option_models.duration_ema,
        stomp.option_models.baseline_mass_ema,
        stomp.option_models.discount_ema,
        stomp.option_models.next_state_weights,
        oak.utility_ema,
    )


def _parameter_payload(state: PrototypeAgentState) -> dict[str, object]:
    return _tree_payload(_parameter_tree(state))


def _parameter_sha256(state: PrototypeAgentState) -> str:
    return cast(str, _parameter_payload(state)["tree_sha256"])


def _replace_policy_rng(state: PrototypeAgentState, key: Array) -> PrototypeAgentState:
    oak = cast(OaKState, state.oak_state)
    return cast(
        PrototypeAgentState,
        state.replace(
            oak_state=oak.replace(
                stomp_state=oak.stomp_state.replace(rng_key=key)
            )
        ),
    )


def _canonicalize_host_metadata(state: PrototypeAgentState) -> PrototypeAgentState:
    oak = cast(OaKState, state.oak_state)
    learner = oak.stomp_state.base_learner_state.replace(
        birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
        uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
    )
    return cast(
        PrototypeAgentState,
        state.replace(
            oak_state=oak.replace(
                stomp_state=oak.stomp_state.replace(base_learner_state=learner)
            )
        ),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class EmbodiedDynamicsArmState:
    """One arm's owned policy, envelope, plant, and causal counters."""

    prototype_state: PrototypeAgentState
    envelope_state: EmbodiedSafetyEnvelopeState
    dynamics_state: EmbodiedDynamicsState
    learner_revision: int
    update_call_count: int
    skip_count: int
    intervention_count: int
    recovery_count: int
    previous_action_available: bool


@dataclasses.dataclass(frozen=True, slots=True)
class EmbodiedDynamicsAdaptationRunState:
    """Integrity-sealed full composite causal prefix for both owned arms."""

    event_index: int
    adaptive: EmbodiedDynamicsArmState
    frozen_learning_control: EmbodiedDynamicsArmState
    chain_heads: tuple[str, str]
    records_json: tuple[str, ...]
    integrity_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class EmbodiedDynamicsValidationReceipt:
    """Strict source/runtime-bound replay receipt with no assessment authority."""

    valid: bool
    assessment_status: str
    source_runtime_bound: bool
    exact_causal_replay: bool
    checkpoint_resume_exact: bool
    output_written: bool
    physical_dispatch_count: int
    deployment_authority: bool
    promotion_authority: bool


class EmbodiedDynamicsAdaptationEvaluator:
    """Host orchestrator around pure dynamics and existing learner/safety kernels."""

    def __init__(self, config: EmbodiedDynamicsAdaptationConfig):
        if type(config) is not EmbodiedDynamicsAdaptationConfig:
            raise TypeError("config must be an exact EmbodiedDynamicsAdaptationConfig")
        self.config = config
        self.adaptive_agent = PrototypeAgent(_prototype_config(config, adaptive=True))
        self.frozen_agent = PrototypeAgent(_prototype_config(config, adaptive=False))
        self.adaptive_envelope = EmbodiedSafetyEnvelope(_envelope_config(config))
        self.frozen_envelope = EmbodiedSafetyEnvelope(_envelope_config(config))
        self._root_key = jr.key(config.seed, impl="threefry2x32")
        self._source_digest = _digest_words(
            _canonical_sha256(embodied_dynamics_source_manifest())
        )
        self._model_version = _digest_words("bounded-two-joint-dynamics-v1")
        self._partner_metadata = _digest_words("no-partner-input-development-v1")
        self._initial_frozen_parameter_sha256: str | None = None
        self._initial_chain_heads: tuple[str, str] | None = None

    def common_event(self, event_index: int) -> EmbodiedExogenousEvent:
        """Return one typed Threefry CRN event paired only as exogenous input."""

        _, _, regime_code = _phase_for_event(self.config, event_index)
        event_key = jr.fold_in(self._root_key, np.uint32(10_000 + event_index))
        keys = jr.split(event_key, 4)
        dynamics_noise = jr.normal(keys[0], (7,), dtype=jnp.float32)
        sensor_noise = jr.normal(keys[1], (_N_JOINTS,), dtype=jnp.float32)
        bias_noise = jr.normal(keys[2], (_N_JOINTS,), dtype=jnp.float32)
        latency = jr.randint(
            keys[3],
            (),
            minval=0,
            maxval=3,
            dtype=jnp.int32,
        )
        gain_target, damping_target, coupling_target = _regime_targets(regime_code)
        phase_angle = jnp.float32(0.47 * event_index)
        target = jnp.asarray(
            (
                jnp.float32(0.58) * jnp.sin(phase_angle),
                jnp.float32(0.48) * jnp.cos(jnp.float32(0.73) * phase_angle),
            ),
            dtype=jnp.float32,
        )
        target = jnp.clip(
            target + jnp.float32(0.015) * dynamics_noise[:2],
            jnp.float32(-0.7),
            jnp.float32(0.7),
        )
        return EmbodiedExogenousEvent(
            event_index=jnp.asarray(event_index, dtype=jnp.int32),
            regime_code=jnp.asarray(regime_code, dtype=jnp.int32),
            target_position=target,
            gain_target=gain_target,
            damping_target=damping_target,
            coupling_target=coupling_target,
            gain_drift=jnp.float32(0.004) * dynamics_noise[2:4],
            damping_drift=jnp.float32(0.003) * dynamics_noise[4:6],
            coupling_drift=jnp.float32(0.002) * dynamics_noise[6],
            wear_drift=jnp.float32(0.0008)
            * jnp.abs(jnp.asarray((dynamics_noise[0], dynamics_noise[1]))),
            sensor_bias_drift=jnp.float32(0.0015) * bias_noise,
            sensor_noise=sensor_noise,
            latency_ticks=latency,
            fault_code=jnp.asarray(_fault_for_event(event_index), dtype=jnp.int32),
            key_words=jnp.stack(tuple(jr.key_data(key) for key in keys)),
        )

    def common_schedule_payload(self) -> list[dict[str, object]]:
        result = []
        for index in range(self.config.total_events):
            phase, regime, _ = _phase_for_event(self.config, index)
            result.append(
                _event_payload(self.common_event(index), phase=phase, regime=regime)
            )
        return result

    def _initial_chain_head(
        self,
        arm: ArmName,
        state: EmbodiedDynamicsArmState,
    ) -> str:
        return _canonical_sha256(
            {
                "arm": arm,
                "prototype": _tree_payload(state.prototype_state),
                "envelope": _tree_payload(state.envelope_state),
                "dynamics": _tree_payload(state.dynamics_state),
            }
        )

    def _new_prototype_state(
        self,
        *,
        agent: PrototypeAgent,
        policy_rng_tag: int,
        lifecycle: tuple[int, int],
    ) -> PrototypeAgentState:
        parameter_key = jr.fold_in(self._root_key, np.uint32(20_000))
        policy_key = jr.fold_in(self._root_key, np.uint32(policy_rng_tag))
        event = self.common_event(0)
        state = agent.init(
            parameter_key,
            lifecycle_id=jnp.asarray(lifecycle, dtype=jnp.uint32),
        )
        state = _canonicalize_host_metadata(state)
        state = _replace_policy_rng(state, policy_key)
        with jax.disable_jit():
            state = agent.start(
                state,
                _observation(initial_embodied_dynamics_state(), event),
            )
            valid = agent.validate_state(state)
        if not bool(np.asarray(valid)):
            raise ValueError("initial Prototype state is invalid")
        return state

    def init(self) -> EmbodiedDynamicsAdaptationRunState:
        """Initialize matched parameters with independent policy RNG and owned plants."""

        adaptive_prototype = self._new_prototype_state(
            agent=self.adaptive_agent,
            policy_rng_tag=21_007,
            lifecycle=(0xA11CE001, 0xA11CE002),
        )
        frozen_prototype = self._new_prototype_state(
            agent=self.frozen_agent,
            policy_rng_tag=21_011,
            lifecycle=(0xF002E001, 0xF002E002),
        )
        if _parameter_sha256(adaptive_prototype) != _parameter_sha256(
            frozen_prototype
        ):
            raise ValueError("adaptive and frozen initial learned parameters differ")
        dynamics = initial_embodied_dynamics_state()
        adaptive = EmbodiedDynamicsArmState(
            prototype_state=adaptive_prototype,
            envelope_state=self.adaptive_envelope.init(
                source_digest=self._source_digest
            ),
            dynamics_state=dynamics,
            learner_revision=0,
            update_call_count=0,
            skip_count=0,
            intervention_count=0,
            recovery_count=0,
            previous_action_available=True,
        )
        frozen = EmbodiedDynamicsArmState(
            prototype_state=frozen_prototype,
            envelope_state=self.frozen_envelope.init(
                source_digest=self._source_digest
            ),
            dynamics_state=dynamics,
            learner_revision=0,
            update_call_count=0,
            skip_count=0,
            intervention_count=0,
            recovery_count=0,
            previous_action_available=True,
        )
        chain_heads = (
            self._initial_chain_head("adaptive", adaptive),
            self._initial_chain_head("frozen_learning_control", frozen),
        )
        if self._initial_frozen_parameter_sha256 is None:
            self._initial_frozen_parameter_sha256 = _parameter_sha256(
                frozen.prototype_state
            )
        if self._initial_chain_heads is None:
            self._initial_chain_heads = chain_heads
        if (
            self._initial_frozen_parameter_sha256
            != _parameter_sha256(frozen.prototype_state)
            or self._initial_chain_heads != chain_heads
        ):
            raise ValueError("initial Prototype bindings are not deterministic")
        state = EmbodiedDynamicsAdaptationRunState(
            event_index=0,
            adaptive=adaptive,
            frozen_learning_control=frozen,
            chain_heads=chain_heads,
            records_json=(),
            integrity_sha256="",
        )
        return self._seal_state(state)

    def _arm_payload(self, arm: EmbodiedDynamicsArmState) -> dict[str, object]:
        return {
            "prototype_state": _tree_payload(arm.prototype_state),
            "envelope_state": _tree_payload(arm.envelope_state),
            "dynamics_state": _tree_payload(arm.dynamics_state),
            "learner_revision": arm.learner_revision,
            "update_call_count": arm.update_call_count,
            "skip_count": arm.skip_count,
            "intervention_count": arm.intervention_count,
            "recovery_count": arm.recovery_count,
            "previous_action_available": arm.previous_action_available,
        }

    def _state_body(
        self,
        state: EmbodiedDynamicsAdaptationRunState,
    ) -> dict[str, object]:
        return {
            "event_index": state.event_index,
            "adaptive": self._arm_payload(state.adaptive),
            "frozen_learning_control": self._arm_payload(
                state.frozen_learning_control
            ),
            "chain_heads": list(state.chain_heads),
            "records": [json.loads(record) for record in state.records_json],
        }

    def _seal_state(
        self,
        state: EmbodiedDynamicsAdaptationRunState,
    ) -> EmbodiedDynamicsAdaptationRunState:
        return dataclasses.replace(
            state,
            integrity_sha256=_canonical_sha256(self._state_body(state)),
        )

    def _arm_valid(
        self,
        arm: EmbodiedDynamicsArmState,
        *,
        agent: PrototypeAgent,
        envelope: EmbodiedSafetyEnvelope,
        frozen: bool,
    ) -> bool:
        if type(arm) is not EmbodiedDynamicsArmState:
            return False
        if any(
            type(value) is not int or value < 0
            for value in (
                arm.learner_revision,
                arm.update_call_count,
                arm.skip_count,
                arm.intervention_count,
                arm.recovery_count,
            )
        ) or type(arm.previous_action_available) is not bool:
            return False
        if arm.learner_revision != arm.update_call_count:
            return False
        if int(np.asarray(arm.prototype_state.step_count)) != arm.update_call_count:
            return False
        if type(arm.prototype_state) is not PrototypeAgentState:
            return False
        if not bool(np.asarray(envelope.state_valid(arm.envelope_state))):
            return False
        if not bool(np.asarray(embodied_dynamics_state_valid(arm.dynamics_state))):
            return False
        if int(np.asarray(arm.dynamics_state.step_count)) != arm.update_call_count:
            return False
        if not _tree_all_finite_host(
            (arm.prototype_state, arm.envelope_state, arm.dynamics_state)
        ):
            return False
        if frozen:
            if (
                self._initial_frozen_parameter_sha256 is None
                or _parameter_sha256(arm.prototype_state)
                != self._initial_frozen_parameter_sha256
            ):
                return False
        return True

    def _valid_state_structure(
        self,
        state: EmbodiedDynamicsAdaptationRunState,
        *,
        check_frozen_parameters: bool = True,
    ) -> bool:
        if type(state) is not EmbodiedDynamicsAdaptationRunState:
            return False
        if not 0 <= state.event_index <= self.config.total_events:
            return False
        if (
            type(state.integrity_sha256) is not str
            or len(state.integrity_sha256) != 64
            or state.integrity_sha256 != _canonical_sha256(self._state_body(state))
        ):
            return False
        if not self._arm_valid(
            state.adaptive,
            agent=self.adaptive_agent,
            envelope=self.adaptive_envelope,
            frozen=False,
        ):
            return False
        if check_frozen_parameters:
            if (
                self._initial_frozen_parameter_sha256 is None
                or _parameter_sha256(state.frozen_learning_control.prototype_state)
                != self._initial_frozen_parameter_sha256
            ):
                return False
        if not self._arm_valid(
            state.frozen_learning_control,
            agent=self.frozen_agent,
            envelope=self.frozen_envelope,
            frozen=False,
        ):
            return False
        return self._record_chains_valid_without_init(state)

    def _record_chains_valid_without_init(
        self,
        state: EmbodiedDynamicsAdaptationRunState,
    ) -> bool:
        if self._initial_chain_heads is None:
            return False
        heads = {
            "adaptive": self._initial_chain_heads[0],
            "frozen_learning_control": self._initial_chain_heads[1],
        }
        try:
            records = [json.loads(raw) for raw in state.records_json]
        except json.JSONDecodeError:
            return False
        if len(records) != state.event_index * len(ARM_ORDER):
            return False
        for offset, record in enumerate(records):
            if not isinstance(record, Mapping):
                return False
            event_index = offset // len(ARM_ORDER)
            arm = ARM_ORDER[offset % len(ARM_ORDER)]
            if (
                record.get("event_index") != event_index
                or record.get("arm") != arm
                or record.get("causal_parent_sha256") != heads[arm]
            ):
                return False
            body = {name: record[name] for name in record if name != "record_sha256"}
            if record.get("record_sha256") != _canonical_sha256(body):
                return False
            heads[arm] = cast(str, record["record_sha256"])
        return tuple(heads[arm] for arm in ARM_ORDER) == state.chain_heads

    def _version_bindings(
        self,
        state: PrototypeAgentState,
    ) -> tuple[Array, Array, Array]:
        optimizer = _digest_words(_parameter_sha256(state))
        lifecycle = _digest_words(
            np.asarray(state.current_decision_id, dtype=np.uint32)
            .tobytes(order="C")
            .hex()
        )
        return self._model_version, optimizer, lifecycle

    def _transition_payload(self, transition: PrototypeTransition) -> dict[str, object]:
        return {
            "observation": _array_payload(transition.observation),
            "action": int(np.asarray(transition.action)),
            "decision_id": _array_payload(transition.decision_id),
            "reward": float(np.asarray(transition.reward)),
            "discount": float(np.asarray(transition.discount)),
            "terminated": bool(np.asarray(transition.terminated)),
            "truncated": bool(np.asarray(transition.truncated)),
            "next_observation": _array_payload(transition.next_observation),
            "next_decision_observation": _array_payload(
                transition.next_decision_observation
            ),
            "horde_cumulants": None,
            "horde_discounts": None,
        }

    def _record(
        self,
        *,
        arm_name: ArmName,
        parent: str,
        phase: str,
        regime: str,
        schedule_payload: Mapping[str, object],
        decision_observation: Array,
        proposed_action: int,
        prototype_decision_id: Array,
        clean_command: EmbodiedCommand,
        envelope_proposal: EmbodiedCommand,
        telemetry: EmbodiedTelemetry,
        envelope_decision_id: Array,
        envelope_action_id: Array,
        versions: tuple[Array, Array, Array],
        action_available: bool,
        proposed_accepted: bool,
        fallback_used: bool,
        transaction_applied: bool,
        available_command: EmbodiedCommand | None,
        replacement_attempted: bool,
        replacement_committed: bool,
        replacement_changed_action: bool,
        executed_action: int | None,
        transition: PrototypeTransition | None,
        update_called: bool,
        update_applied: bool,
        learner_revision_before: int,
        learner_revision_after: int,
        parameter_sha256_before: str,
        parameter_sha256_after: str,
        prototype_step_words_before: Array,
        prototype_step_words_after: Array,
        dynamics_before: EmbodiedDynamicsState,
        dynamics_after: EmbodiedDynamicsState,
        tracking_error: float | None,
        reward: float | None,
        skip: bool,
        intervention: bool,
        recovery: bool,
    ) -> dict[str, object]:
        model_version, optimizer_version, lifecycle_version = versions
        body: dict[str, object] = {
            "schema": "alberta.embodied-dynamics-adaptation.record.v1",
            "event_index": int(cast(int, schedule_payload["event_index"])),
            "arm": arm_name,
            "phase": phase,
            "regime": regime,
            "change_family_diagnostic": phase == "change_family_diagnostic",
            "change_family_data_status": (
                "consumed_development" if phase == "change_family_diagnostic" else None
            ),
            "causal_parent_sha256": parent,
            "common_schedule_sha256": schedule_payload["schedule_sha256"],
            "common_schedule": dict(schedule_payload),
            "environment_parent_sha256": _tree_sha256(dynamics_before),
            "environment_child_sha256": _tree_sha256(dynamics_after),
            "prototype_decision_observation": _array_payload(decision_observation),
            "prototype_proposed_action": proposed_action,
            "prototype_decision_id": _array_payload(prototype_decision_id),
            "clean_primitive_command": _command_payload(clean_command),
            "envelope_proposed_command": _command_payload(envelope_proposal),
            "telemetry": _telemetry_payload(telemetry),
            "envelope_decision_id": _array_payload(envelope_decision_id),
            "envelope_action_id": _array_payload(envelope_action_id),
            "model_version": _array_payload(model_version),
            "optimizer_version": _array_payload(optimizer_version),
            "lifecycle_version": _array_payload(lifecycle_version),
            "action_available": action_available,
            "proposed_accepted": proposed_accepted,
            "fallback_used": fallback_used,
            "envelope_transaction_applied": transaction_applied,
            "fallback_command_checked_inside_envelope": True,
            "available_command": (
                None if available_command is None else _command_payload(available_command)
            ),
            "simulated_command_executed": action_available,
            "physical_command_dispatched": False,
            "cached_action_replacement_attempted": replacement_attempted,
            "cached_action_replacement_committed": replacement_committed,
            "cached_action_replacement_changed_action": replacement_changed_action,
            "executed_primitive_action": executed_action,
            "dynamics_advanced": action_available,
            "learner_transition": (
                None if transition is None else self._transition_payload(transition)
            ),
            "prototype_update_called": update_called,
            "prototype_update_applied": update_applied,
            "learner_revision_before": learner_revision_before,
            "learner_revision_after": learner_revision_after,
            "parameter_sha256_before": parameter_sha256_before,
            "parameter_sha256_after": parameter_sha256_after,
            "prototype_step_words_before": _array_payload(
                prototype_step_words_before
            ),
            "prototype_step_words_after": _array_payload(prototype_step_words_after),
            "tracking_error": tracking_error,
            "reward": reward,
            "update": update_called,
            "skip": skip,
            "intervention": intervention,
            "recovery": recovery,
            "task_label": None,
            "learner_reset": False,
            "environment_reset": False,
        }
        return {**body, "record_sha256": _canonical_sha256(body)}

    def _advance_arm(
        self,
        *,
        arm_name: ArmName,
        arm: EmbodiedDynamicsArmState,
        agent: PrototypeAgent,
        envelope: EmbodiedSafetyEnvelope,
        event: EmbodiedExogenousEvent,
        next_event: EmbodiedExogenousEvent,
        parent: str,
        phase: str,
        regime: str,
        schedule_payload: Mapping[str, object],
    ) -> tuple[EmbodiedDynamicsArmState, dict[str, object]]:
        decision = agent.decision(arm.prototype_state)
        if not bool(np.asarray(decision.armed)):
            raise ValueError("Prototype decision is not armed")
        proposed_action = int(np.asarray(decision.action))
        clean_command = _primitive_command(proposed_action)
        fault = int(np.asarray(event.fault_code))
        envelope_proposal = (
            _corrupt_proposal(clean_command)
            if fault == FAULT_PROPOSAL_CORRUPTION
            else clean_command
        )
        control_tick = 100 + 4 * int(np.asarray(event.event_index))
        telemetry = _telemetry(
            arm.dynamics_state,
            event,
            control_tick=control_tick,
        )
        envelope_decision_id = _words(int(np.asarray(event.event_index)) + 1)
        envelope_action_id = _words(int(np.asarray(event.event_index)) + 1)
        versions = self._version_bindings(arm.prototype_state)
        deadline_tick = control_tick + (
            10 if fault == FAULT_DEADLINE_MISS else 2
        )
        envelope_result = envelope.evaluate(
            arm.envelope_state,
            telemetry,
            envelope_proposal,
            decision_id=envelope_decision_id,
            action_id=envelope_action_id,
            control_tick=_words(control_tick),
            control_deadline_tick=_words(deadline_tick),
            model_version=versions[0],
            optimizer_version=versions[1],
            lifecycle_version=versions[2],
            untrusted_reward=jnp.asarray(0.0, dtype=jnp.float32),
            partner_metadata_digest=self._partner_metadata,
            learned_cost_estimate=jnp.asarray(0.0, dtype=jnp.float32),
        )
        action_available = bool(np.asarray(envelope_result.action_available))
        proposed_accepted = bool(np.asarray(envelope_result.proposed_accepted))
        fallback_used = bool(np.asarray(envelope_result.fallback_used))
        transaction_applied = bool(np.asarray(envelope_result.transaction_applied))
        intervention = not proposed_accepted
        recovery = (not arm.previous_action_available) and action_available
        parameter_before = _parameter_sha256(arm.prototype_state)
        step_words_before = arm.prototype_state.step_words

        replacement_attempted = False
        replacement_committed = False
        replacement_changed = False
        executed_action: int | None = None
        transition: PrototypeTransition | None = None
        tracking_error: float | None = None
        reward: float | None = None
        update_applied = False
        prototype_after = arm.prototype_state
        dynamics_after = arm.dynamics_state
        revision_after = arm.learner_revision

        if action_available:
            available_command = envelope_result.command
            executed_action = proposed_action
            dispatch_state = arm.prototype_state
            if fallback_used:
                replacement_attempted = True
                with jax.disable_jit():
                    replacement = agent.replace_cached_primitive_action(
                        dispatch_state,
                        decision_id=decision.decision_id,
                        decision_observation=dispatch_state.current_representation,
                        proposed_action=jnp.asarray(0, dtype=jnp.int32),
                        safety_action_mask=jnp.ones((_N_ACTIONS,), dtype=jnp.bool_),
                    )
                replacement_committed = bool(np.asarray(replacement.committed))
                replacement_changed = proposed_action != 0
                if not replacement_committed or int(np.asarray(replacement.action)) != 0:
                    raise ValueError("Prototype fallback credit-owner replacement failed")
                dispatch_state = replacement.state
                executed_action = 0
                if not _strict_json_equal(
                    _command_payload(available_command),
                    _command_payload(_primitive_command(0)),
                ):
                    raise ValueError("envelope fallback differs from primitive zero")
            dynamics_result = embodied_dynamics_step_kernel(
                arm.dynamics_state,
                available_command,
                event,
            )
            if not bool(np.asarray(dynamics_result.applied)):
                raise ValueError("certified simulated dynamics step failed closed")
            dynamics_after = dynamics_result.state
            tracking_error = float(np.asarray(dynamics_result.tracking_error))
            energy = float(
                np.mean(
                    np.square(
                        np.asarray(available_command.joint_position, dtype=np.float32)
                    )
                )
            )
            reward = float(np.float32(-tracking_error - 0.01 * energy))
            next_observation = _observation(dynamics_after, next_event)
            transition = PrototypeTransition(
                observation=dispatch_state.current_raw_observation,
                action=jnp.asarray(executed_action, dtype=jnp.int32),
                decision_id=dispatch_state.current_decision_id,
                reward=jnp.asarray(reward, dtype=jnp.float32),
                discount=jnp.asarray(self.config.discount, dtype=jnp.float32),
                terminated=jnp.asarray(False, dtype=jnp.bool_),
                truncated=jnp.asarray(False, dtype=jnp.bool_),
                next_observation=next_observation,
                next_decision_observation=next_observation,
            )
            with jax.disable_jit():
                update_result = agent.update_transition(dispatch_state, transition)
            update_applied = bool(np.asarray(update_result.transition_diagnostics.valid))
            if not update_applied:
                raise ValueError("actual executed Prototype transition was rejected")
            prototype_after = update_result.state
            revision_after += 1
        else:
            available_command = None

        parameter_after = _parameter_sha256(prototype_after)
        record = self._record(
            arm_name=arm_name,
            parent=parent,
            phase=phase,
            regime=regime,
            schedule_payload=schedule_payload,
            decision_observation=decision.observation,
            proposed_action=proposed_action,
            prototype_decision_id=decision.decision_id,
            clean_command=clean_command,
            envelope_proposal=envelope_proposal,
            telemetry=telemetry,
            envelope_decision_id=envelope_decision_id,
            envelope_action_id=envelope_action_id,
            versions=versions,
            action_available=action_available,
            proposed_accepted=proposed_accepted,
            fallback_used=fallback_used,
            transaction_applied=transaction_applied,
            available_command=available_command,
            replacement_attempted=replacement_attempted,
            replacement_committed=replacement_committed,
            replacement_changed_action=replacement_changed,
            executed_action=executed_action,
            transition=transition,
            update_called=action_available,
            update_applied=update_applied,
            learner_revision_before=arm.learner_revision,
            learner_revision_after=revision_after,
            parameter_sha256_before=parameter_before,
            parameter_sha256_after=parameter_after,
            prototype_step_words_before=step_words_before,
            prototype_step_words_after=prototype_after.step_words,
            dynamics_before=arm.dynamics_state,
            dynamics_after=dynamics_after,
            tracking_error=tracking_error,
            reward=reward,
            skip=not action_available,
            intervention=intervention,
            recovery=recovery,
        )
        next_arm = EmbodiedDynamicsArmState(
            prototype_state=prototype_after,
            envelope_state=envelope_result.state,
            dynamics_state=dynamics_after,
            learner_revision=revision_after,
            update_call_count=arm.update_call_count + int(action_available),
            skip_count=arm.skip_count + int(not action_available),
            intervention_count=arm.intervention_count + int(intervention),
            recovery_count=arm.recovery_count + int(recovery),
            previous_action_available=action_available,
        )
        return next_arm, record

    def _advance_once(
        self,
        state: EmbodiedDynamicsAdaptationRunState,
    ) -> EmbodiedDynamicsAdaptationRunState:
        if not self._valid_state_structure(state):
            raise ValueError("embodied dynamics run state structure is invalid")
        if state.event_index >= self.config.total_events:
            raise ValueError("embodied dynamics run is already complete")
        index = state.event_index
        phase, regime, _ = _phase_for_event(self.config, index)
        event = self.common_event(index)
        next_event = self.common_event(
            min(index + 1, self.config.total_events - 1)
        )
        schedule_payload = _event_payload(event, phase=phase, regime=regime)
        adaptive, adaptive_record = self._advance_arm(
            arm_name="adaptive",
            arm=state.adaptive,
            agent=self.adaptive_agent,
            envelope=self.adaptive_envelope,
            event=event,
            next_event=next_event,
            parent=state.chain_heads[0],
            phase=phase,
            regime=regime,
            schedule_payload=schedule_payload,
        )
        frozen, frozen_record = self._advance_arm(
            arm_name="frozen_learning_control",
            arm=state.frozen_learning_control,
            agent=self.frozen_agent,
            envelope=self.frozen_envelope,
            event=event,
            next_event=next_event,
            parent=state.chain_heads[1],
            phase=phase,
            regime=regime,
            schedule_payload=schedule_payload,
        )
        record_json = (
            _canonical_json_bytes(adaptive_record).decode("utf-8"),
            _canonical_json_bytes(frozen_record).decode("utf-8"),
        )
        next_state = EmbodiedDynamicsAdaptationRunState(
            event_index=index + 1,
            adaptive=adaptive,
            frozen_learning_control=frozen,
            chain_heads=(
                cast(str, adaptive_record["record_sha256"]),
                cast(str, frozen_record["record_sha256"]),
            ),
            records_json=state.records_json + record_json,
            integrity_sha256="",
        )
        sealed = self._seal_state(next_state)
        if not self._valid_state_structure(sealed):
            raise ValueError("embodied dynamics next state is invalid")
        return sealed

    def _reconstruct(self, event_index: int) -> EmbodiedDynamicsAdaptationRunState:
        _exact_int(
            event_index,
            name="event_index",
            maximum=self.config.total_events,
        )
        state = self.init()
        for _ in range(event_index):
            state = self._advance_once(state)
        return state

    def validate_state(
        self,
        state: EmbodiedDynamicsAdaptationRunState,
        *,
        causal: bool = True,
    ) -> bool:
        if not self._valid_state_structure(state):
            return False
        if not causal:
            return True
        expected = self._reconstruct(state.event_index)
        return _strict_json_equal(self._state_body(state), self._state_body(expected))

    def advance(
        self,
        state: EmbodiedDynamicsAdaptationRunState,
    ) -> EmbodiedDynamicsAdaptationRunState:
        """Advance only an exact config/source-derived causal prefix."""

        if not self.validate_state(state):
            raise ValueError("embodied dynamics state differs from exact causal prefix")
        return self._advance_once(state)

    def run_to_end(
        self,
        state: EmbodiedDynamicsAdaptationRunState | None = None,
    ) -> EmbodiedDynamicsAdaptationRunState:
        current = self.init() if state is None else state
        if not self.validate_state(current):
            raise ValueError("cannot resume from an invalid embodied dynamics state")
        while current.event_index < self.config.total_events:
            current = self._advance_once(current)
        return current

    def initial_snapshot_payload(self) -> dict[str, object]:
        state = self.init()
        return {
            "adaptive": self._arm_payload(state.adaptive),
            "frozen_learning_control": self._arm_payload(
                state.frozen_learning_control
            ),
            "initial_learned_parameters_equal": (
                _parameter_sha256(state.adaptive.prototype_state)
                == _parameter_sha256(
                    state.frozen_learning_control.prototype_state
                )
            ),
            "policy_rng_states_independent": (
                not _tree_bits_equal(
                    cast(OaKState, state.adaptive.prototype_state.oak_state)
                    .stomp_state.rng_key,
                    cast(
                        OaKState,
                        state.frozen_learning_control.prototype_state.oak_state,
                    ).stomp_state.rng_key,
                )
            ),
        }

    def checkpoint_payload(
        self,
        state: EmbodiedDynamicsAdaptationRunState,
    ) -> dict[str, object]:
        """Return the full source/runtime/config-bound composite host checkpoint."""

        if not self.validate_state(state):
            raise ValueError("cannot checkpoint an invalid embodied dynamics state")
        return self._checkpoint_payload_from_causal_state(state)

    def _checkpoint_payload_from_causal_state(
        self,
        state: EmbodiedDynamicsAdaptationRunState,
    ) -> dict[str, object]:
        """Serialize a state already produced by this evaluator's causal loop."""

        source = embodied_dynamics_source_manifest()
        runtime = embodied_dynamics_runtime_identity()
        protocol = embodied_dynamics_protocol(self.config)
        schedules = self.common_schedule_payload()
        body: dict[str, object] = {
            "schema": EMBODIED_DYNAMICS_CHECKPOINT_SCHEMA,
            "type": "EmbodiedDynamicsAdaptationCheckpoint",
            "host_only": True,
            "full_composite_state": True,
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(protocol),
            "source_manifest_sha256": _canonical_sha256(source),
            "runtime_sha256": _canonical_sha256(runtime),
            "initial_snapshot_sha256": _canonical_sha256(
                self.initial_snapshot_payload()
            ),
            "common_schedule_sha256": _canonical_sha256(schedules),
            "common_schedule_prefix_sha256": _canonical_sha256(
                schedules[: state.event_index]
            ),
            "event_index": state.event_index,
            "composite_state": self._state_body(state),
            "state_integrity_sha256": state.integrity_sha256,
            "assessment_status": ASSESSMENT_STATUS,
            "output_path": None,
            "physical_dispatch_count": 0,
            "deployment_authority": False,
            "promotion_authority": False,
        }
        payload = {**body, "checkpoint_sha256": _canonical_sha256(body)}
        if len(_canonical_json_bytes(payload)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("embodied dynamics checkpoint exceeds byte cap")
        return payload

    def restore_checkpoint(
        self,
        payload: object,
    ) -> EmbodiedDynamicsAdaptationRunState:
        """Restore only the exact causally reconstructed composite prefix."""

        raw = _mapping(payload, name="checkpoint")
        if len(_canonical_json_bytes(raw)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("embodied dynamics checkpoint exceeds byte cap")
        expected_fields = {
            "schema",
            "type",
            "host_only",
            "full_composite_state",
            "config_sha256",
            "protocol_sha256",
            "source_manifest_sha256",
            "runtime_sha256",
            "initial_snapshot_sha256",
            "common_schedule_sha256",
            "common_schedule_prefix_sha256",
            "event_index",
            "composite_state",
            "state_integrity_sha256",
            "assessment_status",
            "output_path",
            "physical_dispatch_count",
            "deployment_authority",
            "promotion_authority",
            "checkpoint_sha256",
        }
        if set(raw) != expected_fields:
            raise ValueError("embodied dynamics checkpoint fields differ")
        if (
            raw.get("schema") != EMBODIED_DYNAMICS_CHECKPOINT_SCHEMA
            or raw.get("type") != "EmbodiedDynamicsAdaptationCheckpoint"
            or raw.get("host_only") is not True
            or raw.get("full_composite_state") is not True
        ):
            raise ValueError("embodied dynamics checkpoint schema/type differs")
        if (
            raw.get("assessment_status") != ASSESSMENT_STATUS
            or raw.get("output_path") is not None
            or raw.get("physical_dispatch_count") != 0
            or raw.get("deployment_authority") is not False
            or raw.get("promotion_authority") is not False
        ):
            raise ValueError("embodied dynamics checkpoint authority fields differ")
        body = {name: raw[name] for name in raw if name != "checkpoint_sha256"}
        if raw.get("checkpoint_sha256") != _canonical_sha256(body):
            raise ValueError("embodied dynamics checkpoint digest integrity failed")
        event_index = _exact_int(
            raw.get("event_index"),
            name="event_index",
            maximum=self.config.total_events,
        )
        schedules = self.common_schedule_payload()
        bindings = {
            "config_sha256": _canonical_sha256(self.config.to_config()),
            "protocol_sha256": _canonical_sha256(
                embodied_dynamics_protocol(self.config)
            ),
            "source_manifest_sha256": _canonical_sha256(
                embodied_dynamics_source_manifest()
            ),
            "runtime_sha256": _canonical_sha256(
                embodied_dynamics_runtime_identity()
            ),
            "initial_snapshot_sha256": _canonical_sha256(
                self.initial_snapshot_payload()
            ),
            "common_schedule_sha256": _canonical_sha256(schedules),
            "common_schedule_prefix_sha256": _canonical_sha256(
                schedules[:event_index]
            ),
        }
        if any(raw.get(name) != expected for name, expected in bindings.items()):
            raise ValueError("embodied dynamics checkpoint source/runtime binding differs")
        expected = self._reconstruct(event_index)
        expected_payload = self._checkpoint_payload_from_causal_state(expected)
        if not _strict_json_equal(dict(raw), expected_payload):
            raise ValueError("embodied dynamics checkpoint differs from exact causal prefix")
        return expected


def _stack_pytrees(values: Sequence[object]) -> object:
    if not values:
        raise ValueError("cannot stack an empty PyTree sequence")
    return jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *values,
    )


def _tree_float32_parity(
    left: object,
    right: object,
    *,
    atol: float = 1.0e-7,
) -> tuple[bool, float]:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if str(left_tree) != str(right_tree) or len(left_leaves) != len(right_leaves):
        return False, float("inf")
    maximum = 0.0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(jax.device_get(left_leaf))
        right_array = np.asarray(jax.device_get(right_leaf))
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return False, float("inf")
        if np.issubdtype(left_array.dtype, np.inexact):
            difference = float(
                np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64)))
            )
            maximum = max(maximum, difference)
            if not np.allclose(left_array, right_array, rtol=0.0, atol=atol):
                return False, maximum
        elif not np.array_equal(left_array, right_array):
            return False, float("inf")
    return True, maximum


def _dynamics_kernel_parity(
    evaluator: EmbodiedDynamicsAdaptationEvaluator,
) -> dict[str, object]:
    events = tuple(evaluator.common_event(index) for index in range(4))
    commands = tuple(_primitive_command(index % _N_ACTIONS) for index in range(4))
    initial = initial_embodied_dynamics_state()
    with jax.disable_jit():
        eager_single = embodied_dynamics_step_kernel(initial, commands[0], events[0])
    compiled_single = jax.jit(embodied_dynamics_step_kernel)(
        initial,
        commands[0],
        events[0],
    )
    iterative_state = initial
    iterative_applied = []
    iterative_positions = []
    iterative_targets = []
    iterative_errors = []
    with jax.disable_jit():
        for command, event in zip(commands, events, strict=True):
            result = embodied_dynamics_step_kernel(iterative_state, command, event)
            iterative_state = result.state
            iterative_applied.append(result.applied)
            iterative_positions.append(result.state.joint_position)
            iterative_targets.append(result.delayed_target)
            iterative_errors.append(result.tracking_error)
    batched_commands = cast(EmbodiedCommand, _stack_pytrees(commands))
    batched_events = cast(EmbodiedExogenousEvent, _stack_pytrees(events))
    scan = embodied_dynamics_scan_kernel(initial, batched_commands, batched_events)
    compiled_scan = jax.jit(embodied_dynamics_scan_kernel)(
        initial,
        batched_commands,
        batched_events,
    )
    expected_scan = EmbodiedDynamicsScanResult(
        state=iterative_state,
        applied=jnp.stack(iterative_applied),
        joint_positions=jnp.stack(iterative_positions),
        delayed_targets=jnp.stack(iterative_targets),
        tracking_errors=jnp.stack(iterative_errors),
    )
    iterative_scan_parity, _ = _tree_float32_parity(
        expected_scan,
        scan,
    )
    return {
        "pure_nonexecuting_parity_probe": True,
        "single_step_eager_jit_exact": _tree_bits_equal(
            eager_single, compiled_single
        ),
        "iterative_scan_bit_exact_claimed": False,
        "iterative_scan_float32_parity": iterative_scan_parity,
        "iterative_scan_absolute_tolerance": 1.0e-7,
        "scan_jit_exact": _tree_bits_equal(scan, compiled_scan),
        "dynamics_kernel_jittable": True,
        "full_orchestration_host_only": True,
    }


def _resource_declaration(
    evaluator: EmbodiedDynamicsAdaptationEvaluator,
) -> dict[str, object]:
    initial = evaluator.init()
    adaptive_prototype = measure_prototype_agent_state_resources(
        initial.adaptive.prototype_state
    ).to_config()
    frozen_prototype = measure_prototype_agent_state_resources(
        initial.frozen_learning_control.prototype_state
    ).to_config()
    adaptive_envelope = dataclasses.asdict(
        evaluator.adaptive_envelope.resource_budget(initial.adaptive.envelope_state)
    )
    frozen_envelope = dataclasses.asdict(
        evaluator.frozen_envelope.resource_budget(
            initial.frozen_learning_control.envelope_state
        )
    )
    dynamics_nbytes = sum(
        int(np.asarray(leaf).nbytes)
        for leaf in jax.tree.leaves(initial.adaptive.dynamics_state)
        if isinstance(leaf, Array)
    )
    return {
        "prototype_adaptive": adaptive_prototype,
        "prototype_frozen_control": frozen_prototype,
        "prototype_capacity_exactly_matched": (
            adaptive_prototype == frozen_prototype
        ),
        "envelope_adaptive": adaptive_envelope,
        "envelope_frozen_control": frozen_envelope,
        "envelope_capacity_exactly_matched": (
            adaptive_envelope == frozen_envelope
        ),
        "dynamics_state_nbytes_per_arm": dynamics_nbytes,
        "dynamics_state_shape_fixed": True,
        "command_history_slots_per_arm": 2,
        "maximum_events": evaluator.config.total_events,
        "maximum_prototype_updates_per_arm": evaluator.config.total_events,
        "maximum_envelope_decisions_per_arm": evaluator.config.total_events + 4,
        "output_bytes_written": 0,
        "physical_dispatches": 0,
    }


def _records_from_state(
    state: EmbodiedDynamicsAdaptationRunState,
) -> list[dict[str, object]]:
    return [cast(dict[str, object], json.loads(raw)) for raw in state.records_json]


def _arm_records(
    records: Sequence[Mapping[str, object]],
    arm: ArmName,
) -> list[Mapping[str, object]]:
    return [record for record in records if record["arm"] == arm]


def _diagnostics(
    evaluator: EmbodiedDynamicsAdaptationEvaluator,
    final: EmbodiedDynamicsAdaptationRunState,
) -> dict[str, object]:
    records = _records_from_state(final)
    adaptive_records = _arm_records(records, "adaptive")
    frozen_records = _arm_records(records, "frozen_learning_control")
    initial = evaluator.init()
    adaptive_actions = [record["executed_primitive_action"] for record in adaptive_records]
    frozen_actions = [record["executed_primitive_action"] for record in frozen_records]
    adaptive_environment = [record["environment_child_sha256"] for record in adaptive_records]
    frozen_environment = [record["environment_child_sha256"] for record in frozen_records]
    common_schedule_paired = all(
        adaptive["common_schedule_sha256"] == frozen["common_schedule_sha256"]
        for adaptive, frozen in zip(adaptive_records, frozen_records, strict=True)
    )
    blocked_contract = all(
        (
            record["available_command"] is not None
            and record["dynamics_advanced"] is True
            and record["learner_transition"] is not None
            and record["prototype_update_called"] is True
        )
        if record["action_available"] is True
        else (
            record["available_command"] is None
            and record["dynamics_advanced"] is False
            and record["learner_transition"] is None
            and record["prototype_update_called"] is False
        )
        for record in records
    )
    fallback_records = [record for record in records if record["fallback_used"] is True]
    changed_fallbacks = [
        record
        for record in fallback_records
        if record["cached_action_replacement_changed_action"] is True
    ]
    phase_names = {cast(str, record["phase"]) for record in records}
    return {
        "common_schedule_paired_exogenous_only": common_schedule_paired,
        "policy_action_randomness_paired": False,
        "trajectory_equality_assumed": False,
        "adaptive_and_frozen_trajectory_diverged": (
            adaptive_actions != frozen_actions
            or adaptive_environment != frozen_environment
        ),
        "initial_learned_parameters_equal": (
            _parameter_sha256(initial.adaptive.prototype_state)
            == _parameter_sha256(
                initial.frozen_learning_control.prototype_state
            )
        ),
        "adaptive_parameters_changed": (
            _parameter_sha256(final.adaptive.prototype_state)
            != _parameter_sha256(initial.adaptive.prototype_state)
        ),
        "frozen_parameters_unchanged": (
            _parameter_sha256(final.frozen_learning_control.prototype_state)
            == _parameter_sha256(
                initial.frozen_learning_control.prototype_state
            )
        ),
        "update_calls_matched": (
            final.adaptive.update_call_count
            == final.frozen_learning_control.update_call_count
        ),
        "update_call_opportunities_matched": (
            len(adaptive_records)
            == len(frozen_records)
            == evaluator.config.total_events
        ),
        "update_call_opportunities_per_arm": evaluator.config.total_events,
        "available_update_calls_matched": (
            final.adaptive.update_call_count
            == final.frozen_learning_control.update_call_count
        ),
        "safety_availability_can_diverge_in_principle": True,
        "safety_availability_diverged_in_this_trace": (
            [record["action_available"] for record in adaptive_records]
            != [record["action_available"] for record in frozen_records]
        ),
        "frozen_parameter_witness_scope": (
            "base action heads and biases, base reward rate, option action values "
            "and reward rates, option-model value statistics/weights, and OaK utility; "
            "eligibility traces, optimizer state, option completion/execution counters, "
            "RNG, execution ownership, and clocks excluded. Those excluded values cannot "
            "change this frozen policy's learned action-value surfaces because every "
            "learning step is zero, option-model decay is one, planning is zero, and "
            "automatic curation is disabled. Temporal option ownership and RNG may still "
            "advance realized behavior; the other exclusions are capacity/work state."
        ),
        "adaptive_update_calls": final.adaptive.update_call_count,
        "frozen_update_calls": final.frozen_learning_control.update_call_count,
        "adaptive_skips": final.adaptive.skip_count,
        "frozen_skips": final.frozen_learning_control.skip_count,
        "adaptive_interventions": final.adaptive.intervention_count,
        "frozen_interventions": final.frozen_learning_control.intervention_count,
        "adaptive_recoveries": final.adaptive.recovery_count,
        "frozen_recoveries": final.frozen_learning_control.recovery_count,
        "no_action_no_command_or_transition_contract": blocked_contract,
        "fallback_events": len(fallback_records),
        "fallback_action_changes": len(changed_fallbacks),
        "every_changed_fallback_rebound_public_credit_owner": all(
            record["cached_action_replacement_attempted"] is True
            and record["cached_action_replacement_committed"] is True
            and record["executed_primitive_action"] == 0
            for record in changed_fallbacks
        ),
        "a_b_a_observed": {
            "A_initial",
            "B",
            "A_return",
        }.issubset(phase_names),
        "change_family_observed": "change_family_diagnostic" in phase_names,
        "change_family_data_consumed": True,
        "untouched_held_out_data": False,
        "learner_resets": 0,
        "environment_resets": 0,
        "task_labels_supplied": False,
        "physical_dispatch_count": 0,
    }


def _run_with_prefix(
    evaluator: EmbodiedDynamicsAdaptationEvaluator,
) -> tuple[
    EmbodiedDynamicsAdaptationRunState,
    EmbodiedDynamicsAdaptationRunState,
]:
    state = evaluator.init()
    prefix: EmbodiedDynamicsAdaptationRunState | None = None
    while state.event_index < evaluator.config.total_events:
        state = evaluator._advance_once(state)
        if state.event_index == FIXED_CHECKPOINT_SPLIT:
            prefix = state
    if prefix is None:
        raise RuntimeError("fixed checkpoint split was not reached")
    return state, prefix


def _assemble_report(
    evaluator: EmbodiedDynamicsAdaptationEvaluator,
    final: EmbodiedDynamicsAdaptationRunState,
    prefix: EmbodiedDynamicsAdaptationRunState,
    *,
    checkpoint_resume_exact: bool,
) -> dict[str, object]:
    config = evaluator.config
    checkpoint = evaluator._checkpoint_payload_from_causal_state(prefix)
    records = _records_from_state(final)
    source = embodied_dynamics_source_manifest()
    runtime = embodied_dynamics_runtime_identity()
    protocol = embodied_dynamics_protocol(config)
    schedules = evaluator.common_schedule_payload()
    resources = _resource_declaration(evaluator)
    parity = _dynamics_kernel_parity(evaluator)
    body: dict[str, object] = {
        "schema": EMBODIED_DYNAMICS_REPORT_SCHEMA,
        "type": "EmbodiedDynamicsAdaptationDevelopmentReport",
        "assessment_status": ASSESSMENT_STATUS,
        "development_status": DEVELOPMENT_STATUS,
        "config": config.to_config(),
        "protocol": protocol,
        "source_manifest": source,
        "runtime_identity": runtime,
        "common_schedule": schedules,
        "common_schedule_sha256": _canonical_sha256(schedules),
        "initial_snapshot": evaluator.initial_snapshot_payload(),
        "records": records,
        "records_sha256": _canonical_sha256(records),
        "final_composite_state": evaluator._state_body(final),
        "final_state_integrity_sha256": final.integrity_sha256,
        "diagnostics": _diagnostics(evaluator, final),
        "resources": resources,
        "kernel_parity": parity,
        "checkpoint": {
            "schema": EMBODIED_DYNAMICS_CHECKPOINT_SCHEMA,
            "split_event_index": FIXED_CHECKPOINT_SPLIT,
            "host_only": True,
            "full_composite_state_in_payload": True,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "resume_exact": checkpoint_resume_exact,
        },
        "exact_causal_replay_required": True,
        "full_orchestration_host_only": True,
        "dynamics_kernel_pure_jax": True,
        "simulated_command_execution_is_accounting_only": True,
        "physical_dispatch_count": 0,
        "physical_dispatch_authority": False,
        "output_written": False,
        "output_path": None,
        "artifact_writer_available": False,
        "thresholds": [],
        "performance_claimed": False,
        "adaptation_efficacy_claimed": False,
        "safety_claimed": False,
        "physical_safety_certificate": False,
        "geometry_proof": False,
        "deployment_authority": False,
        "evidence_claimed": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "sha256_authentication_claimed": False,
        "caller_authentication_performed": False,
        "limitations": list(_LIMITATIONS),
    }
    report = {**body, "report_sha256": _canonical_sha256(body)}
    if len(_canonical_json_bytes(report)) > _MAX_REPORT_BYTES:
        raise ValueError("embodied dynamics report exceeds byte cap")
    return report


def _build_report_unvalidated(
    config: EmbodiedDynamicsAdaptationConfig,
) -> dict[str, object]:
    evaluator = EmbodiedDynamicsAdaptationEvaluator(config)
    uninterrupted, prefix = _run_with_prefix(evaluator)
    checkpoint = evaluator._checkpoint_payload_from_causal_state(prefix)
    restored = evaluator.restore_checkpoint(checkpoint)
    resumed = restored
    while resumed.event_index < config.total_events:
        resumed = evaluator._advance_once(resumed)
    checkpoint_resume_exact = _strict_json_equal(
        evaluator._state_body(uninterrupted),
        evaluator._state_body(resumed),
    )
    if not checkpoint_resume_exact:
        raise ValueError("embodied dynamics checkpoint/resume differs")
    return _assemble_report(
        evaluator,
        uninterrupted,
        prefix,
        checkpoint_resume_exact=True,
    )


def build_embodied_dynamics_adaptation_report(
    config: EmbodiedDynamicsAdaptationConfig | None = None,
) -> dict[str, object]:
    """Build the deterministic non-assessing report without writing files."""

    selected = EmbodiedDynamicsAdaptationConfig() if config is None else config
    if type(selected) is not EmbodiedDynamicsAdaptationConfig:
        raise TypeError("config must be an exact EmbodiedDynamicsAdaptationConfig")
    try:
        return _build_report_unvalidated(selected)
    finally:
        jax.clear_caches()
        gc.collect()


def run_embodied_dynamics_adaptation_development() -> dict[str, object]:
    """Run the one frozen development protocol without writing an artifact."""

    return build_embodied_dynamics_adaptation_report(
        EmbodiedDynamicsAdaptationConfig()
    )


def validate_embodied_dynamics_adaptation_report(
    report: object,
) -> EmbodiedDynamicsValidationReceipt:
    """Strictly reconstruct every schedule, arm, transition, and report field."""

    raw = _mapping(report, name="report")
    if len(_canonical_json_bytes(raw)) > _MAX_REPORT_BYTES:
        raise ValueError("embodied dynamics report exceeds byte cap")
    expected_fields = {
        "schema",
        "type",
        "assessment_status",
        "development_status",
        "config",
        "protocol",
        "source_manifest",
        "runtime_identity",
        "common_schedule",
        "common_schedule_sha256",
        "initial_snapshot",
        "records",
        "records_sha256",
        "final_composite_state",
        "final_state_integrity_sha256",
        "diagnostics",
        "resources",
        "kernel_parity",
        "checkpoint",
        "exact_causal_replay_required",
        "full_orchestration_host_only",
        "dynamics_kernel_pure_jax",
        "simulated_command_execution_is_accounting_only",
        "physical_dispatch_count",
        "physical_dispatch_authority",
        "output_written",
        "output_path",
        "artifact_writer_available",
        "thresholds",
        "performance_claimed",
        "adaptation_efficacy_claimed",
        "safety_claimed",
        "physical_safety_certificate",
        "geometry_proof",
        "deployment_authority",
        "evidence_claimed",
        "promotion_authority",
        "scientific_promotion_allowed",
        "sha256_authentication_claimed",
        "caller_authentication_performed",
        "limitations",
        "report_sha256",
    }
    if set(raw) != expected_fields:
        raise ValueError("embodied dynamics report fields differ")
    if (
        raw.get("schema") != EMBODIED_DYNAMICS_REPORT_SCHEMA
        or raw.get("type") != "EmbodiedDynamicsAdaptationDevelopmentReport"
    ):
        raise ValueError("embodied dynamics report schema/type differs")
    fixed = {
        "assessment_status": ASSESSMENT_STATUS,
        "development_status": DEVELOPMENT_STATUS,
        "exact_causal_replay_required": True,
        "full_orchestration_host_only": True,
        "dynamics_kernel_pure_jax": True,
        "simulated_command_execution_is_accounting_only": True,
        "physical_dispatch_count": 0,
        "physical_dispatch_authority": False,
        "output_written": False,
        "output_path": None,
        "artifact_writer_available": False,
        "thresholds": [],
        "performance_claimed": False,
        "adaptation_efficacy_claimed": False,
        "safety_claimed": False,
        "physical_safety_certificate": False,
        "geometry_proof": False,
        "deployment_authority": False,
        "evidence_claimed": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "sha256_authentication_claimed": False,
        "caller_authentication_performed": False,
        "limitations": list(_LIMITATIONS),
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(raw.get(name), expected):
            raise ValueError(f"embodied dynamics fixed report field {name} differs")
    body = {name: raw[name] for name in raw if name != "report_sha256"}
    if raw.get("report_sha256") != _canonical_sha256(body):
        raise ValueError("embodied dynamics report digest integrity failed")
    config = EmbodiedDynamicsAdaptationConfig.from_config(
        _mapping(raw.get("config"), name="report.config")
    )
    if not _strict_json_equal(
        raw.get("source_manifest"),
        embodied_dynamics_source_manifest(),
    ):
        raise ValueError("embodied dynamics source manifest differs")
    if not _strict_json_equal(
        raw.get("runtime_identity"),
        embodied_dynamics_runtime_identity(),
    ):
        raise ValueError("embodied dynamics runtime identity differs")
    evaluator = EmbodiedDynamicsAdaptationEvaluator(config)
    try:
        final, prefix = _run_with_prefix(evaluator)
        expected = _assemble_report(
            evaluator,
            final,
            prefix,
            checkpoint_resume_exact=True,
        )
        if not _strict_json_equal(dict(raw), expected):
            raise ValueError("embodied dynamics report differs from exact causal replay")
    finally:
        jax.clear_caches()
        gc.collect()
    return EmbodiedDynamicsValidationReceipt(
        valid=True,
        assessment_status=ASSESSMENT_STATUS,
        source_runtime_bound=True,
        exact_causal_replay=True,
        checkpoint_resume_exact=True,
        output_written=False,
        physical_dispatch_count=0,
        deployment_authority=False,
        promotion_authority=False,
    )


__all__ = [
    "ARM_ORDER",
    "ASSESSMENT_STATUS",
    "CHECKPOINT_HOST_ONLY",
    "DEVELOPMENT_STATUS",
    "DEPLOYMENT_AUTHORITY",
    "EMBODIED_DYNAMICS_CHECKPOINT_SCHEMA",
    "EMBODIED_DYNAMICS_CONFIG_SCHEMA",
    "EMBODIED_DYNAMICS_PROTOCOL_SCHEMA",
    "EMBODIED_DYNAMICS_REPORT_SCHEMA",
    "EmbodiedDynamicsAdaptationConfig",
    "EmbodiedDynamicsAdaptationEvaluator",
    "EmbodiedDynamicsAdaptationRunState",
    "EmbodiedDynamicsArmState",
    "EmbodiedDynamicsScanResult",
    "EmbodiedDynamicsState",
    "EmbodiedDynamicsStepResult",
    "EmbodiedDynamicsValidationReceipt",
    "EmbodiedExogenousEvent",
    "FIXED_CHECKPOINT_SPLIT",
    "ORCHESTRATION_HOST_ONLY",
    "OUTPUT_WRITES",
    "PHYSICAL_DISPATCH_AUTHORITY",
    "PHYSICAL_DISPATCH_COUNT",
    "PROMOTION_AUTHORITY",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "build_embodied_dynamics_adaptation_report",
    "embodied_dynamics_protocol",
    "embodied_dynamics_runtime_identity",
    "embodied_dynamics_scan_kernel",
    "embodied_dynamics_source_manifest",
    "embodied_dynamics_state_valid",
    "embodied_dynamics_step_kernel",
    "initial_embodied_dynamics_state",
    "run_embodied_dynamics_adaptation_development",
    "validate_embodied_dynamics_adaptation_report",
]
