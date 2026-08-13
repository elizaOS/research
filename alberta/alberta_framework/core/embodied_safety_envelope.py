# mypy: disable-error-code="attr-defined,call-arg,arg-type,type-var"
"""Fixed-resource L0 hard envelope and deployment-shadow instrumentation.

The envelope is deterministic, non-learning, and has no physical dispatch
surface.  It independently checks measured and proposed joint/workspace
quantities, collision clearance, timing, bridge, emergency-stop, identity,
version, and persistent-state contracts.  It returns a proposed command or a
statically certified fallback for a caller to dispatch; it never dispatches.

Shadow evaluation is pure and separately recorded into a fixed recent ring.
The deployment gate is a conservative readiness readout only: every retained
hard-violation bit must be zero, the Wilson success lower bound must pass, and
calibration and latency must be bounded.  Rewards, partner metadata, and a
learned-cost estimate are untrusted logged inputs and have no override path.

This is simulation/L0 infrastructure.  It makes no physical-safety,
deployment, efficacy, scientific-evidence, or promotion claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

EMBODIED_SAFETY_ENVELOPE_CONFIG_SCHEMA = "alberta.embodied-safety-envelope.config.v1"
EMBODIED_SAFETY_ENVELOPE_CHECKPOINT_SCHEMA = "alberta.embodied-safety-envelope.state.v1"
EMBODIED_SAFETY_ENVELOPE_PHYSICAL_SAFETY_CLAIM = False
EMBODIED_SAFETY_ENVELOPE_ACTION_DISPATCH_AUTHORITY = False
EMBODIED_SAFETY_ENVELOPE_LEARNING_MUTATION_AUTHORITY = False
EMBODIED_SAFETY_ENVELOPE_DEPLOYMENT_AUTHORITY = False
EMBODIED_SAFETY_ENVELOPE_PROMOTION_AUTHORITY = False
EMBODIED_SAFETY_ENVELOPE_LEARNED_COST_OVERRIDE_AUTHORITY = False
EMBODIED_SAFETY_ENVELOPE_SCIENTIFIC_PROMOTION_ALLOWED = False
EMBODIED_SAFETY_ENVELOPE_CALLER_AUTHENTICATION = False

ENVELOPE_REASON_AVAILABLE = 0
ENVELOPE_REASON_PERSISTENT_STATE = 1
ENVELOPE_REASON_CAPACITY = 2
ENVELOPE_REASON_DECISION_IDENTITY = 3
ENVELOPE_REASON_TELEMETRY_IDENTITY = 4
ENVELOPE_REASON_TIME_MONOTONICITY = 5
ENVELOPE_REASON_BRIDGE_DISCONNECTED = 6
ENVELOPE_REASON_TELEMETRY_STALE = 7
ENVELOPE_REASON_CONTROL_DEADLINE = 8
ENVELOPE_REASON_EMERGENCY_STOP = 9
ENVELOPE_REASON_DEPLOYMENT_SUSPENDED = 10
ENVELOPE_REASON_CURRENT_ENVELOPE = 11
ENVELOPE_REASON_VERSION_BINDING = 12
ENVELOPE_REASON_METADATA = 13
ENVELOPE_REASON_ACTION_IDENTITY = 14
ENVELOPE_REASON_ACTION_CAPACITY = 15
ENVELOPE_REASON_FALLBACK_UNAVAILABLE = 16

HANDSHAKE_REASON_AVAILABLE = 0
HANDSHAKE_REASON_PERSISTENT_STATE = 1
HANDSHAKE_REASON_BINDING = 2
HANDSHAKE_REASON_AUTHORITY = 3
HANDSHAKE_REASON_NONCE_REPLAY = 4
HANDSHAKE_REASON_CAPACITY = 5
HANDSHAKE_REASON_NOT_REQUESTED = 6
HANDSHAKE_REASON_NOT_STATIONARY_SAFE = 7

_DIGEST_WORDS = 8
_IDENTITY_WORDS = 2
_WORKSPACE_DIM = 3
_INT32_MAX = 2**31 - 1
_MAX_JOINTS = 4_096
_MAX_SHADOW_WINDOW = 4_096


def _positive_int(value: object, *, name: str, ceiling: int = _INT32_MAX) -> int:
    if type(value) is not int or value < 1 or value > ceiling:
        raise ValueError(f"{name} must be a positive exact Python int <= {ceiling}")
    return value


def _nonnegative_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite exact Python float")
    represented = float(np.float32(value))
    if not math.isfinite(represented) or represented < 0.0:
        raise ValueError(f"{name} must remain non-negative and finite in float32")
    return value


def _positive_float(value: object, *, name: str) -> float:
    result = _nonnegative_float(value, name=name)
    if result <= 0.0 or float(np.float32(result)) <= 0.0:
        raise ValueError(f"{name} must remain positive in float32")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _nonnegative_float(value, name=name)
    if result > 1.0 or float(np.float32(result)) > 1.0:
        raise ValueError(f"{name} must remain in [0, 1] in float32")
    return result


def _float_tuple(value: object, *, name: str, length: int) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise ValueError(f"{name} must be an exact tuple of length {length}")
    result: list[float] = []
    for index, item in enumerate(value):
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"{name}[{index}] must be a finite exact Python float")
        represented = float(np.float32(item))
        if not math.isfinite(represented):
            raise ValueError(f"{name}[{index}] must remain finite in float32")
        result.append(item)
    return tuple(result)


def _digest_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != _DIGEST_WORDS:
        raise ValueError(f"{name} must be an exact {_DIGEST_WORDS}-word tuple")
    result: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or not 0 <= item <= 0xFFFFFFFF:
            raise ValueError(f"{name}[{index}] must be uint32 compatible")
        result.append(item)
    if not any(result):
        raise ValueError(f"{name} must be nonzero")
    return tuple(result)


def _require_array(value: Any, *, name: str, shape: tuple[int, ...], dtype: Any) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose exact array shape and dtype")
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _float32_scalar(value: float | Array, *, name: str) -> Array:
    if type(value) is float:
        return jnp.asarray(value, dtype=jnp.float32)
    return _require_array(value, name=name, shape=(), dtype=jnp.float32)


def _int32_scalar(value: int | Array, *, name: str) -> Array:
    if type(value) is int:
        if not -(2**31) <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be signed-int32 compatible")
        return jnp.asarray(value, dtype=jnp.int32)
    return _require_array(value, name=name, shape=(), dtype=jnp.int32)


def _bool_scalar(value: bool | Array, *, name: str) -> Array:
    if type(value) is bool:
        return jnp.asarray(value, dtype=jnp.bool_)
    return _require_array(value, name=name, shape=(), dtype=jnp.bool_)


def _words_greater(left: Array, right: Array) -> Array:
    return (left[0] > right[0]) | ((left[0] == right[0]) & (left[1] > right[1]))


def _words_greater_equal(left: Array, right: Array) -> Array:
    return jnp.array_equal(left, right) | _words_greater(left, right)


def _words_advance(words: Array, amount: int) -> tuple[Array, Array]:
    delta = jnp.uint32(amount)
    low = words[1] + delta
    carry = (low < words[1]).astype(jnp.uint32)
    high = words[0] + carry
    available = ~((carry != 0) & (high == 0))
    return jnp.stack((high, low), dtype=jnp.uint32), available


def _within_forward_delta(earlier: Array, later: Array, maximum: int) -> Array:
    limit, capacity = _words_advance(earlier, maximum)
    return (
        capacity
        & _words_greater_equal(later, earlier)
        & _words_greater_equal(limit, later)
    )


def _checksum_arrays(arrays: tuple[Array, ...]) -> Array:
    acc0 = jnp.uint32(0x9E3779B9)
    acc1 = jnp.uint32(0x85EBCA6B)
    offset = 1
    for value in arrays:
        array = jnp.asarray(value)
        if array.dtype == jnp.float32:
            words = jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
        elif array.dtype == jnp.uint32:
            words = array.reshape((-1,))
        else:
            words = array.astype(jnp.uint32).reshape((-1,))
        if words.shape[0] == 0:
            continue
        indices = jnp.arange(offset, offset + words.shape[0], dtype=jnp.uint32)
        acc0 = acc0 + jnp.sum(words * (indices * jnp.uint32(0x27D4EB2D) + 1))
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(words ^ (indices * jnp.uint32(0x165667B1)))
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), dtype=jnp.uint32)


def _canonical_digest(value: object) -> Array:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).digest()
    return jnp.asarray(
        tuple(
            int.from_bytes(digest[offset : offset + 4], "little")
            for offset in range(0, 32, 4)
        ),
        dtype=jnp.uint32,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class EmbodiedSafetyEnvelopeConfig:
    """Static hard bounds, fallback, shadow gates, and authority bindings."""

    n_joints: int
    joint_position_lower: tuple[float, ...]
    joint_position_upper: tuple[float, ...]
    max_abs_joint_velocity: tuple[float, ...]
    max_abs_joint_torque: tuple[float, ...]
    workspace_lower: tuple[float, float, float]
    workspace_upper: tuple[float, float, float]
    min_collision_clearance: float
    fallback_joint_position: tuple[float, ...]
    fallback_joint_velocity: tuple[float, ...]
    fallback_joint_torque: tuple[float, ...]
    fallback_workspace_position: tuple[float, float, float]
    fallback_collision_clearance: float
    reset_stationary_velocity_tolerance: float
    max_telemetry_age_ticks: int
    max_control_deadline_ticks: int
    shadow_window: int
    min_shadow_samples: int
    min_shadow_success_lcb: float
    wilson_z: float
    max_shadow_calibration_error: float
    max_shadow_latency_ticks: int
    max_decisions: int
    max_committed_actions: int
    max_shadow_records: int
    max_handshakes_per_kind: int
    reset_authority_digest: tuple[int, ...]
    rollback_authority_digest: tuple[int, ...]

    SCHEMA_VERSION: ClassVar[str] = EMBODIED_SAFETY_ENVELOPE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        n = _positive_int(self.n_joints, name="n_joints", ceiling=_MAX_JOINTS)
        lower = _float_tuple(self.joint_position_lower, name="joint_position_lower", length=n)
        upper = _float_tuple(self.joint_position_upper, name="joint_position_upper", length=n)
        velocity = _float_tuple(
            self.max_abs_joint_velocity,
            name="max_abs_joint_velocity",
            length=n,
        )
        torque = _float_tuple(
            self.max_abs_joint_torque,
            name="max_abs_joint_torque",
            length=n,
        )
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("every joint position lower bound must be below its upper bound")
        if any(value < 0.0 for value in (*velocity, *torque)):
            raise ValueError("joint velocity and torque magnitudes must be non-negative")
        workspace_lower = _float_tuple(
            self.workspace_lower,
            name="workspace_lower",
            length=_WORKSPACE_DIM,
        )
        workspace_upper = _float_tuple(
            self.workspace_upper,
            name="workspace_upper",
            length=_WORKSPACE_DIM,
        )
        if any(lo >= hi for lo, hi in zip(workspace_lower, workspace_upper, strict=True)):
            raise ValueError("every workspace lower bound must be below its upper bound")
        clearance = _nonnegative_float(
            self.min_collision_clearance,
            name="min_collision_clearance",
        )
        fallback_position = _float_tuple(
            self.fallback_joint_position,
            name="fallback_joint_position",
            length=n,
        )
        fallback_velocity = _float_tuple(
            self.fallback_joint_velocity,
            name="fallback_joint_velocity",
            length=n,
        )
        fallback_torque = _float_tuple(
            self.fallback_joint_torque,
            name="fallback_joint_torque",
            length=n,
        )
        fallback_workspace = _float_tuple(
            self.fallback_workspace_position,
            name="fallback_workspace_position",
            length=_WORKSPACE_DIM,
        )
        fallback_clearance = _nonnegative_float(
            self.fallback_collision_clearance,
            name="fallback_collision_clearance",
        )
        if any(
            not lo <= value <= hi
            for value, lo, hi in zip(fallback_position, lower, upper, strict=True)
        ):
            raise ValueError("fallback joint position is outside the hard envelope")
        if any(
            abs(value) > limit
            for value, limit in zip(fallback_velocity, velocity, strict=True)
        ):
            raise ValueError("fallback joint velocity is outside the hard envelope")
        if any(
            abs(value) > limit
            for value, limit in zip(fallback_torque, torque, strict=True)
        ):
            raise ValueError("fallback joint torque is outside the hard envelope")
        if any(
            not lo <= value <= hi
            for value, lo, hi in zip(
                fallback_workspace,
                workspace_lower,
                workspace_upper,
                strict=True,
            )
        ):
            raise ValueError("fallback workspace position is outside the hard envelope")
        if fallback_clearance < clearance:
            raise ValueError("fallback collision clearance is below the hard minimum")
        tolerance = _nonnegative_float(
            self.reset_stationary_velocity_tolerance,
            name="reset_stationary_velocity_tolerance",
        )
        if any(tolerance > limit for limit in velocity):
            raise ValueError("reset stationary tolerance exceeds a joint velocity limit")
        _positive_int(self.max_telemetry_age_ticks, name="max_telemetry_age_ticks")
        _positive_int(self.max_control_deadline_ticks, name="max_control_deadline_ticks")
        window = _positive_int(
            self.shadow_window,
            name="shadow_window",
            ceiling=_MAX_SHADOW_WINDOW,
        )
        samples = _positive_int(self.min_shadow_samples, name="min_shadow_samples")
        if samples > window:
            raise ValueError("min_shadow_samples exceeds shadow_window")
        _probability(self.min_shadow_success_lcb, name="min_shadow_success_lcb")
        wilson_z = _positive_float(self.wilson_z, name="wilson_z")
        with np.errstate(over="ignore", invalid="ignore"):
            wilson_z_squared = np.float32(wilson_z) * np.float32(wilson_z)
        if not bool(np.isfinite(wilson_z_squared)):
            raise ValueError("wilson_z squared must remain finite in float32")
        _nonnegative_float(
            self.max_shadow_calibration_error,
            name="max_shadow_calibration_error",
        )
        _positive_int(self.max_shadow_latency_ticks, name="max_shadow_latency_ticks")
        decisions = _positive_int(self.max_decisions, name="max_decisions")
        actions = _positive_int(self.max_committed_actions, name="max_committed_actions")
        shadow_records = _positive_int(self.max_shadow_records, name="max_shadow_records")
        handshakes = _positive_int(
            self.max_handshakes_per_kind,
            name="max_handshakes_per_kind",
        )
        if actions > decisions:
            raise ValueError("max_committed_actions exceeds max_decisions")
        if shadow_records < window:
            raise ValueError("max_shadow_records must cover the full shadow ring")
        # One fresh stop latch may follow every successful reset, including
        # the initial unlatched state.  Latching is deliberately independent
        # of the ordinary decision budget, so reserve its revision headroom.
        if decisions + shadow_records + 3 * handshakes + 1 > _INT32_MAX:
            raise ValueError("combined persistent revision capacity exceeds signed-int32")
        reset_authority = _digest_tuple(
            self.reset_authority_digest,
            name="reset_authority_digest",
        )
        rollback_authority = _digest_tuple(
            self.rollback_authority_digest,
            name="rollback_authority_digest",
        )
        if reset_authority == rollback_authority:
            raise ValueError("reset and rollback authority digests must be distinct")

    def to_config(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload.update(
            {
                "schema_version": self.SCHEMA_VERSION,
                "physical_safety_claim": False,
                "action_dispatch_authority": False,
                "learning_mutation_authority": False,
                "deployment_authority": False,
                "promotion_authority": False,
                "learned_cost_override_authority": False,
                "scientific_promotion_allowed": False,
                "caller_authentication": False,
                "rng_draws": 0,
            }
        )
        return cast(dict[str, object], payload)

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> EmbodiedSafetyEnvelopeConfig:
        if type(value) is not dict:
            raise ValueError("envelope config must be an exact dict")
        raw = dict(value)
        fixed: dict[str, object] = {
            "schema_version": cls.SCHEMA_VERSION,
            "physical_safety_claim": False,
            "action_dispatch_authority": False,
            "learning_mutation_authority": False,
            "deployment_authority": False,
            "promotion_authority": False,
            "learned_cost_override_authority": False,
            "scientific_promotion_allowed": False,
            "caller_authentication": False,
            "rng_draws": 0,
        }
        expected = {field.name for field in dataclasses.fields(cls)} | set(fixed)
        if set(raw) != expected:
            raise ValueError("envelope config keys differ from schema v1")
        for name, expected_value in fixed.items():
            if type(raw[name]) is not type(expected_value) or raw[name] != expected_value:
                raise ValueError(f"envelope config fixed field {name} differs")
            raw.pop(name)
        result = cls(**cast(dict[str, Any], raw))
        if result.to_config() != value:
            raise ValueError("envelope config is noncanonical")
        return result


@chex.dataclass(frozen=True)
class EmbodiedCommand:
    """One joint command and its declared 3D workspace projection."""

    joint_position: Float[Array, " n_joints"]
    joint_velocity: Float[Array, " n_joints"]
    joint_torque: Float[Array, " n_joints"]
    workspace_position: Float[Array, " 3"]
    collision_clearance: Float[Array, ""]


@chex.dataclass(frozen=True)
class EmbodiedTelemetry:
    """Measured robot state, connectivity, stop signal, and exact sample clock."""

    joint_position: Float[Array, " n_joints"]
    joint_velocity: Float[Array, " n_joints"]
    joint_torque: Float[Array, " n_joints"]
    workspace_position: Float[Array, " 3"]
    collision_clearance: Float[Array, ""]
    bridge_connected: Bool[Array, ""]
    emergency_stop: Bool[Array, ""]
    telemetry_id: UInt[Array, " 2"]
    sample_tick: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class AuthorityBoundEnvelopeHandshake:
    """State-bound authority-token request with a monotonic replay nonce.

    The authority digest is a declared, non-secret policy token.  Equality to
    it scopes reset/rollback calls but is not cryptographic authentication.
    Deployments must authenticate callers outside this pure kernel.
    """

    nonce: UInt[Array, " 2"]
    authority_digest: UInt[Array, " 8"]
    source_digest: UInt[Array, " 8"]
    config_digest: UInt[Array, " 8"]
    observed_state_revision: Int[Array, ""]
    observed_state_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class EmbodiedSafetyEnvelopeState:
    """Checksummed envelope, audit counters, action log, and shadow ring.

    The checksum is an accidental-corruption sentinel, not an authentication
    tag.  Adversarial persistence requires an external trusted boundary.
    """

    source_digest: UInt[Array, " 8"]
    config_digest: UInt[Array, " 8"]
    revision: Int[Array, ""]
    emergency_stop_latched: Bool[Array, ""]
    emergency_stop_latch_count: Int[Array, ""]
    has_emergency_stop_sample: Bool[Array, ""]
    last_emergency_stop_telemetry_id: UInt[Array, " 2"]
    last_emergency_stop_sample_tick: UInt[Array, " 2"]
    deployment_suspended: Bool[Array, ""]
    has_telemetry: Bool[Array, ""]
    last_telemetry_id: UInt[Array, " 2"]
    last_sample_tick: UInt[Array, " 2"]
    last_control_tick: UInt[Array, " 2"]
    has_decision: Bool[Array, ""]
    last_decision_id: UInt[Array, " 2"]
    has_action: Bool[Array, ""]
    last_action_id: UInt[Array, " 2"]
    decision_count: Int[Array, ""]
    committed_action_count: Int[Array, ""]
    fallback_action_count: Int[Array, ""]
    rejected_action_count: Int[Array, ""]
    hard_violation_count: Int[Array, ""]
    last_committed_decision_id: UInt[Array, " 2"]
    last_model_version: UInt[Array, " 8"]
    last_optimizer_version: UInt[Array, " 8"]
    last_lifecycle_version: UInt[Array, " 8"]
    last_logged_config_digest: UInt[Array, " 8"]
    last_partner_metadata_digest: UInt[Array, " 8"]
    last_untrusted_reward: Float[Array, ""]
    last_learned_cost_estimate: Float[Array, ""]
    last_action_was_fallback: Bool[Array, ""]
    last_command_joint_position: Float[Array, " n_joints"]
    last_command_joint_velocity: Float[Array, " n_joints"]
    last_command_joint_torque: Float[Array, " n_joints"]
    last_command_workspace_position: Float[Array, " 3"]
    last_command_collision_clearance: Float[Array, ""]
    has_reset_nonce: Bool[Array, ""]
    last_reset_nonce: UInt[Array, " 2"]
    reset_count: Int[Array, ""]
    has_rollback_nonce: Bool[Array, ""]
    last_rollback_nonce: UInt[Array, " 2"]
    rollback_count: Int[Array, ""]
    shadow_valid: Bool[Array, " shadow_window"]
    shadow_hard_violation: Bool[Array, " shadow_window"]
    shadow_success: Bool[Array, " shadow_window"]
    shadow_calibration_error: Float[Array, " shadow_window"]
    shadow_latency_ticks: Int[Array, " shadow_window"]
    shadow_decision_ids: UInt[Array, "shadow_window 2"]
    shadow_model_versions: UInt[Array, "shadow_window 8"]
    shadow_optimizer_versions: UInt[Array, "shadow_window 8"]
    shadow_lifecycle_versions: UInt[Array, "shadow_window 8"]
    shadow_partner_metadata_digests: UInt[Array, "shadow_window 8"]
    shadow_untrusted_rewards: Float[Array, " shadow_window"]
    shadow_learned_cost_estimates: Float[Array, " shadow_window"]
    shadow_size: Int[Array, ""]
    shadow_write_index: Int[Array, ""]
    shadow_record_count: Int[Array, ""]
    has_shadow: Bool[Array, ""]
    last_shadow_decision_id: UInt[Array, " 2"]
    state_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class EmbodiedEnvelopeDecision:
    """Certified command availability and every independent hard gate."""

    state: EmbodiedSafetyEnvelopeState
    command: EmbodiedCommand
    action_available: Bool[Array, ""]
    proposed_accepted: Bool[Array, ""]
    fallback_used: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    emergency_stop_latch_applied: Bool[Array, ""]
    persistent_state_valid: Bool[Array, ""]
    decision_capacity_available: Bool[Array, ""]
    action_capacity_available: Bool[Array, ""]
    decision_identity_valid: Bool[Array, ""]
    telemetry_identity_valid: Bool[Array, ""]
    action_identity_valid: Bool[Array, ""]
    control_time_monotonic: Bool[Array, ""]
    telemetry_fresh: Bool[Array, ""]
    control_deadline_valid: Bool[Array, ""]
    bridge_connected: Bool[Array, ""]
    emergency_stop_input: Bool[Array, ""]
    emergency_stop_latched_before: Bool[Array, ""]
    emergency_stop_latched_after: Bool[Array, ""]
    deployment_not_suspended: Bool[Array, ""]
    current_position_finite: Bool[Array, ""]
    current_position_in_bounds: Bool[Array, ""]
    current_velocity_finite: Bool[Array, ""]
    current_velocity_in_bounds: Bool[Array, ""]
    current_torque_finite: Bool[Array, ""]
    current_torque_in_bounds: Bool[Array, ""]
    current_workspace_finite: Bool[Array, ""]
    current_workspace_in_bounds: Bool[Array, ""]
    current_clearance_finite: Bool[Array, ""]
    current_clearance_in_bounds: Bool[Array, ""]
    proposed_position_finite: Bool[Array, ""]
    proposed_position_in_bounds: Bool[Array, ""]
    proposed_velocity_finite: Bool[Array, ""]
    proposed_velocity_in_bounds: Bool[Array, ""]
    proposed_torque_finite: Bool[Array, ""]
    proposed_torque_in_bounds: Bool[Array, ""]
    proposed_workspace_finite: Bool[Array, ""]
    proposed_workspace_in_bounds: Bool[Array, ""]
    proposed_clearance_finite: Bool[Array, ""]
    proposed_clearance_in_bounds: Bool[Array, ""]
    version_binding_valid: Bool[Array, ""]
    metadata_finite: Bool[Array, ""]
    current_envelope_safe: Bool[Array, ""]
    proposed_envelope_safe: Bool[Array, ""]
    fallback_certified: Bool[Array, ""]
    hard_violation: Bool[Array, ""]
    unavailable_reason: Int[Array, ""]
    learned_cost_override_used: Bool[Array, ""]
    action_dispatch_authority: Bool[Array, ""]
    physical_safety_claim: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EmbodiedShadowEvaluation:
    """Pure checksummed shadow outcome; no write, dispatch, or authentication."""

    valid: Bool[Array, ""]
    state_revision: Int[Array, ""]
    state_checksum: UInt[Array, " 2"]
    source_digest: UInt[Array, " 8"]
    config_digest: UInt[Array, " 8"]
    decision_id: UInt[Array, " 2"]
    model_version: UInt[Array, " 8"]
    optimizer_version: UInt[Array, " 8"]
    lifecycle_version: UInt[Array, " 8"]
    partner_metadata_digest: UInt[Array, " 8"]
    untrusted_reward: Float[Array, ""]
    learned_cost_estimate: Float[Array, ""]
    hard_violation: Bool[Array, ""]
    observed_success: Bool[Array, ""]
    calibration_error: Float[Array, ""]
    latency_ticks: Int[Array, ""]
    action_would_be_available: Bool[Array, ""]
    evaluation_checksum: UInt[Array, " 2"]
    dispatches: Int[Array, ""]
    learning_state_mutations: Int[Array, ""]
    deployment_authority: Bool[Array, ""]
    physical_safety_claim: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EmbodiedShadowRecordResult:
    """Atomic fixed-ring write result."""

    state: EmbodiedSafetyEnvelopeState
    transaction_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    replay_rejected: Bool[Array, ""]
    capacity_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EmbodiedDeploymentGate:
    """Proposal-only conservative readiness calculation over the recent ring."""

    state_valid: Bool[Array, ""]
    sample_count: Int[Array, ""]
    success_count: Int[Array, ""]
    hard_violation_count: Int[Array, ""]
    hard_zero: Bool[Array, ""]
    performance_success_lcb: Float[Array, ""]
    success_lcb_ready: Bool[Array, ""]
    max_calibration_error: Float[Array, ""]
    calibration_ready: Bool[Array, ""]
    max_latency_ticks: Int[Array, ""]
    latency_ready: Bool[Array, ""]
    enough_samples: Bool[Array, ""]
    deployment_ready: Bool[Array, ""]
    learned_cost_override_used: Bool[Array, ""]
    deployment_authority: Bool[Array, ""]
    promotion_authority: Bool[Array, ""]
    scientific_promotion_allowed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EmbodiedHandshakeResult:
    """Authority-token-bound monotonic reset/rollback result."""

    state: EmbodiedSafetyEnvelopeState
    transaction_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    replay_rejected: Bool[Array, ""]
    stationary_safe: Bool[Array, ""]
    unavailable_reason: Int[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class EmbodiedSafetyEnvelopeResourceBudget:
    """Exact persistent allocation, finite caps, work, and zero authority."""

    persistent_state_nbytes: int
    n_joints: int
    workspace_dimensions: int
    shadow_window: int
    shadow_ring_cells: int
    max_decisions: int
    max_committed_actions: int
    max_shadow_records: int
    max_handshakes_per_kind: int
    joint_values_checked_per_evaluation: int
    workspace_values_checked_per_evaluation: int
    wilson_square_roots_per_deployment_gate: int
    random_generator_calls_per_operation: int
    action_dispatches_per_operation: int
    learning_state_mutations_per_operation: int
    learned_cost_override_authority: bool
    deployment_authority: bool
    promotion_authority: bool
    physical_safety_claim: bool
    scientific_promotion_allowed: bool
    caller_authentication: bool
    checkpoint_schema: str


class EmbodiedSafetyEnvelope:
    """Deterministic hard-command filter and non-authoritative shadow ledger."""

    def __init__(self, config: EmbodiedSafetyEnvelopeConfig) -> None:
        if type(config) is not EmbodiedSafetyEnvelopeConfig:
            raise TypeError("config must be an exact EmbodiedSafetyEnvelopeConfig")
        self._config = config
        self._config_digest = _canonical_digest(config.to_config())
        self._joint_lower = jnp.asarray(config.joint_position_lower, dtype=jnp.float32)
        self._joint_upper = jnp.asarray(config.joint_position_upper, dtype=jnp.float32)
        self._velocity_limit = jnp.asarray(
            config.max_abs_joint_velocity,
            dtype=jnp.float32,
        )
        self._torque_limit = jnp.asarray(config.max_abs_joint_torque, dtype=jnp.float32)
        self._workspace_lower = jnp.asarray(config.workspace_lower, dtype=jnp.float32)
        self._workspace_upper = jnp.asarray(config.workspace_upper, dtype=jnp.float32)
        self._reset_authority = jnp.asarray(
            config.reset_authority_digest,
            dtype=jnp.uint32,
        )
        self._rollback_authority = jnp.asarray(
            config.rollback_authority_digest,
            dtype=jnp.uint32,
        )
        self._fallback = EmbodiedCommand(
            joint_position=jnp.asarray(
                config.fallback_joint_position,
                dtype=jnp.float32,
            ),
            joint_velocity=jnp.asarray(
                config.fallback_joint_velocity,
                dtype=jnp.float32,
            ),
            joint_torque=jnp.asarray(
                config.fallback_joint_torque,
                dtype=jnp.float32,
            ),
            workspace_position=jnp.asarray(
                config.fallback_workspace_position,
                dtype=jnp.float32,
            ),
            collision_clearance=jnp.asarray(
                config.fallback_collision_clearance,
                dtype=jnp.float32,
            ),
        )

    @property
    def config(self) -> EmbodiedSafetyEnvelopeConfig:
        return self._config

    @property
    def config_digest(self) -> Array:
        return self._config_digest

    @property
    def fallback_command(self) -> EmbodiedCommand:
        return self._fallback

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    def _check_command_contract(self, command: EmbodiedCommand, *, name: str) -> None:
        if type(command) is not EmbodiedCommand:
            raise TypeError(f"{name} must be an exact EmbodiedCommand")
        n = self._config.n_joints
        _require_array(
            command.joint_position,
            name=f"{name}.joint_position",
            shape=(n,),
            dtype=jnp.float32,
        )
        _require_array(
            command.joint_velocity,
            name=f"{name}.joint_velocity",
            shape=(n,),
            dtype=jnp.float32,
        )
        _require_array(
            command.joint_torque,
            name=f"{name}.joint_torque",
            shape=(n,),
            dtype=jnp.float32,
        )
        _require_array(
            command.workspace_position,
            name=f"{name}.workspace_position",
            shape=(_WORKSPACE_DIM,),
            dtype=jnp.float32,
        )
        _require_array(
            command.collision_clearance,
            name=f"{name}.collision_clearance",
            shape=(),
            dtype=jnp.float32,
        )

    def _check_telemetry_contract(self, telemetry: EmbodiedTelemetry) -> None:
        if type(telemetry) is not EmbodiedTelemetry:
            raise TypeError("telemetry must be an exact EmbodiedTelemetry")
        n = self._config.n_joints
        for field_name in ("joint_position", "joint_velocity", "joint_torque"):
            _require_array(
                getattr(telemetry, field_name),
                name=f"telemetry.{field_name}",
                shape=(n,),
                dtype=jnp.float32,
            )
        _require_array(
            telemetry.workspace_position,
            name="telemetry.workspace_position",
            shape=(_WORKSPACE_DIM,),
            dtype=jnp.float32,
        )
        _require_array(
            telemetry.collision_clearance,
            name="telemetry.collision_clearance",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            telemetry.bridge_connected,
            name="telemetry.bridge_connected",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            telemetry.emergency_stop,
            name="telemetry.emergency_stop",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            telemetry.telemetry_id,
            name="telemetry.telemetry_id",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            telemetry.sample_tick,
            name="telemetry.sample_tick",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )

    def _check_handshake_contract(self, handshake: AuthorityBoundEnvelopeHandshake) -> None:
        if type(handshake) is not AuthorityBoundEnvelopeHandshake:
            raise TypeError("handshake must be an exact AuthorityBoundEnvelopeHandshake")
        for field_name, shape, dtype in (
            ("nonce", (_IDENTITY_WORDS,), jnp.uint32),
            ("authority_digest", (_DIGEST_WORDS,), jnp.uint32),
            ("source_digest", (_DIGEST_WORDS,), jnp.uint32),
            ("config_digest", (_DIGEST_WORDS,), jnp.uint32),
            ("observed_state_revision", (), jnp.int32),
            ("observed_state_checksum", (_IDENTITY_WORDS,), jnp.uint32),
        ):
            _require_array(
                getattr(handshake, field_name),
                name=f"handshake.{field_name}",
                shape=shape,
                dtype=dtype,
            )

    def _state_payload(self, state: EmbodiedSafetyEnvelopeState) -> tuple[Array, ...]:
        return tuple(
            cast(Array, getattr(state, field.name))
            for field in dataclasses.fields(EmbodiedSafetyEnvelopeState)
            if field.name != "state_checksum"
        )

    def _with_checksum(
        self,
        state: EmbodiedSafetyEnvelopeState,
    ) -> EmbodiedSafetyEnvelopeState:
        return dataclasses.replace(
            state,
            state_checksum=_checksum_arrays(self._state_payload(state)),
        )

    def _check_state_contract(self, state: EmbodiedSafetyEnvelopeState) -> None:
        if type(state) is not EmbodiedSafetyEnvelopeState:
            raise TypeError("state must be an exact EmbodiedSafetyEnvelopeState")
        n = self._config.n_joints
        window = self._config.shadow_window
        contracts: dict[str, tuple[tuple[int, ...], Any]] = {
            "source_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "config_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "revision": ((), jnp.int32),
            "emergency_stop_latched": ((), jnp.bool_),
            "emergency_stop_latch_count": ((), jnp.int32),
            "has_emergency_stop_sample": ((), jnp.bool_),
            "last_emergency_stop_telemetry_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "last_emergency_stop_sample_tick": ((_IDENTITY_WORDS,), jnp.uint32),
            "deployment_suspended": ((), jnp.bool_),
            "has_telemetry": ((), jnp.bool_),
            "last_telemetry_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "last_sample_tick": ((_IDENTITY_WORDS,), jnp.uint32),
            "last_control_tick": ((_IDENTITY_WORDS,), jnp.uint32),
            "has_decision": ((), jnp.bool_),
            "last_decision_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "has_action": ((), jnp.bool_),
            "last_action_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "decision_count": ((), jnp.int32),
            "committed_action_count": ((), jnp.int32),
            "fallback_action_count": ((), jnp.int32),
            "rejected_action_count": ((), jnp.int32),
            "hard_violation_count": ((), jnp.int32),
            "last_committed_decision_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "last_model_version": ((_DIGEST_WORDS,), jnp.uint32),
            "last_optimizer_version": ((_DIGEST_WORDS,), jnp.uint32),
            "last_lifecycle_version": ((_DIGEST_WORDS,), jnp.uint32),
            "last_logged_config_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "last_partner_metadata_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "last_untrusted_reward": ((), jnp.float32),
            "last_learned_cost_estimate": ((), jnp.float32),
            "last_action_was_fallback": ((), jnp.bool_),
            "last_command_joint_position": ((n,), jnp.float32),
            "last_command_joint_velocity": ((n,), jnp.float32),
            "last_command_joint_torque": ((n,), jnp.float32),
            "last_command_workspace_position": ((_WORKSPACE_DIM,), jnp.float32),
            "last_command_collision_clearance": ((), jnp.float32),
            "has_reset_nonce": ((), jnp.bool_),
            "last_reset_nonce": ((_IDENTITY_WORDS,), jnp.uint32),
            "reset_count": ((), jnp.int32),
            "has_rollback_nonce": ((), jnp.bool_),
            "last_rollback_nonce": ((_IDENTITY_WORDS,), jnp.uint32),
            "rollback_count": ((), jnp.int32),
            "shadow_valid": ((window,), jnp.bool_),
            "shadow_hard_violation": ((window,), jnp.bool_),
            "shadow_success": ((window,), jnp.bool_),
            "shadow_calibration_error": ((window,), jnp.float32),
            "shadow_latency_ticks": ((window,), jnp.int32),
            "shadow_decision_ids": ((window, _IDENTITY_WORDS), jnp.uint32),
            "shadow_model_versions": ((window, _DIGEST_WORDS), jnp.uint32),
            "shadow_optimizer_versions": ((window, _DIGEST_WORDS), jnp.uint32),
            "shadow_lifecycle_versions": ((window, _DIGEST_WORDS), jnp.uint32),
            "shadow_partner_metadata_digests": (
                (window, _DIGEST_WORDS),
                jnp.uint32,
            ),
            "shadow_untrusted_rewards": ((window,), jnp.float32),
            "shadow_learned_cost_estimates": ((window,), jnp.float32),
            "shadow_size": ((), jnp.int32),
            "shadow_write_index": ((), jnp.int32),
            "shadow_record_count": ((), jnp.int32),
            "has_shadow": ((), jnp.bool_),
            "last_shadow_decision_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "state_checksum": ((_IDENTITY_WORDS,), jnp.uint32),
        }
        for field_name, (shape, dtype) in contracts.items():
            _require_array(
                getattr(state, field_name),
                name=f"state.{field_name}",
                shape=shape,
                dtype=dtype,
            )

    @staticmethod
    def _tree_nbytes(value: Any) -> int:
        return sum(
            int(np.prod(np.shape(leaf), dtype=np.int64)) * int(np.dtype(leaf.dtype).itemsize)
            for leaf in jax.tree_util.tree_leaves(value)
            if hasattr(leaf, "dtype")
        )

    def _command_gates(self, command: EmbodiedCommand) -> tuple[Array, ...]:
        position_finite = jnp.all(jnp.isfinite(command.joint_position))
        position_in_bounds = position_finite & jnp.all(
            (command.joint_position >= self._joint_lower)
            & (command.joint_position <= self._joint_upper)
        )
        velocity_finite = jnp.all(jnp.isfinite(command.joint_velocity))
        velocity_in_bounds = velocity_finite & jnp.all(
            jnp.abs(command.joint_velocity) <= self._velocity_limit
        )
        torque_finite = jnp.all(jnp.isfinite(command.joint_torque))
        torque_in_bounds = torque_finite & jnp.all(
            jnp.abs(command.joint_torque) <= self._torque_limit
        )
        workspace_finite = jnp.all(jnp.isfinite(command.workspace_position))
        workspace_in_bounds = workspace_finite & jnp.all(
            (command.workspace_position >= self._workspace_lower)
            & (command.workspace_position <= self._workspace_upper)
        )
        clearance_finite = jnp.isfinite(command.collision_clearance)
        clearance_in_bounds = clearance_finite & (
            command.collision_clearance >= self._config.min_collision_clearance
        )
        safe = (
            position_in_bounds
            & velocity_in_bounds
            & torque_in_bounds
            & workspace_in_bounds
            & clearance_in_bounds
        )
        return (
            position_finite,
            position_in_bounds,
            velocity_finite,
            velocity_in_bounds,
            torque_finite,
            torque_in_bounds,
            workspace_finite,
            workspace_in_bounds,
            clearance_finite,
            clearance_in_bounds,
            safe,
        )

    def _telemetry_gates(self, telemetry: EmbodiedTelemetry) -> tuple[Array, ...]:
        measured = EmbodiedCommand(
            joint_position=telemetry.joint_position,
            joint_velocity=telemetry.joint_velocity,
            joint_torque=telemetry.joint_torque,
            workspace_position=telemetry.workspace_position,
            collision_clearance=telemetry.collision_clearance,
        )
        return self._command_gates(measured)

    @staticmethod
    def _digest_nonzero(value: Array) -> Array:
        return jnp.any(value != 0)

    def init(self, *, source_digest: Array) -> EmbodiedSafetyEnvelopeState:
        """Initialize an empty, exact, zero-RNG persistent envelope state."""

        source = _require_array(
            source_digest,
            name="source_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        if not bool(jax.device_get(self._digest_nonzero(source))):
            raise ValueError("source_digest must be nonzero")
        n = self._config.n_joints
        window = self._config.shadow_window
        z2 = jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32)
        z8 = jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
        state = EmbodiedSafetyEnvelopeState(
            source_digest=source,
            config_digest=self._config_digest,
            revision=jnp.int32(0),
            emergency_stop_latched=jnp.asarray(False, dtype=jnp.bool_),
            emergency_stop_latch_count=jnp.int32(0),
            has_emergency_stop_sample=jnp.asarray(False, dtype=jnp.bool_),
            last_emergency_stop_telemetry_id=z2,
            last_emergency_stop_sample_tick=z2,
            deployment_suspended=jnp.asarray(False, dtype=jnp.bool_),
            has_telemetry=jnp.asarray(False, dtype=jnp.bool_),
            last_telemetry_id=z2,
            last_sample_tick=z2,
            last_control_tick=z2,
            has_decision=jnp.asarray(False, dtype=jnp.bool_),
            last_decision_id=z2,
            has_action=jnp.asarray(False, dtype=jnp.bool_),
            last_action_id=z2,
            decision_count=jnp.int32(0),
            committed_action_count=jnp.int32(0),
            fallback_action_count=jnp.int32(0),
            rejected_action_count=jnp.int32(0),
            hard_violation_count=jnp.int32(0),
            last_committed_decision_id=z2,
            last_model_version=z8,
            last_optimizer_version=z8,
            last_lifecycle_version=z8,
            last_logged_config_digest=z8,
            last_partner_metadata_digest=z8,
            last_untrusted_reward=jnp.float32(0.0),
            last_learned_cost_estimate=jnp.float32(0.0),
            last_action_was_fallback=jnp.asarray(False, dtype=jnp.bool_),
            last_command_joint_position=jnp.zeros((n,), dtype=jnp.float32),
            last_command_joint_velocity=jnp.zeros((n,), dtype=jnp.float32),
            last_command_joint_torque=jnp.zeros((n,), dtype=jnp.float32),
            last_command_workspace_position=jnp.zeros(
                (_WORKSPACE_DIM,),
                dtype=jnp.float32,
            ),
            last_command_collision_clearance=jnp.float32(0.0),
            has_reset_nonce=jnp.asarray(False, dtype=jnp.bool_),
            last_reset_nonce=z2,
            reset_count=jnp.int32(0),
            has_rollback_nonce=jnp.asarray(False, dtype=jnp.bool_),
            last_rollback_nonce=z2,
            rollback_count=jnp.int32(0),
            shadow_valid=jnp.zeros((window,), dtype=jnp.bool_),
            shadow_hard_violation=jnp.zeros((window,), dtype=jnp.bool_),
            shadow_success=jnp.zeros((window,), dtype=jnp.bool_),
            shadow_calibration_error=jnp.zeros((window,), dtype=jnp.float32),
            shadow_latency_ticks=jnp.zeros((window,), dtype=jnp.int32),
            shadow_decision_ids=jnp.zeros(
                (window, _IDENTITY_WORDS),
                dtype=jnp.uint32,
            ),
            shadow_model_versions=jnp.zeros(
                (window, _DIGEST_WORDS),
                dtype=jnp.uint32,
            ),
            shadow_optimizer_versions=jnp.zeros(
                (window, _DIGEST_WORDS),
                dtype=jnp.uint32,
            ),
            shadow_lifecycle_versions=jnp.zeros(
                (window, _DIGEST_WORDS),
                dtype=jnp.uint32,
            ),
            shadow_partner_metadata_digests=jnp.zeros(
                (window, _DIGEST_WORDS),
                dtype=jnp.uint32,
            ),
            shadow_untrusted_rewards=jnp.zeros((window,), dtype=jnp.float32),
            shadow_learned_cost_estimates=jnp.zeros((window,), dtype=jnp.float32),
            shadow_size=jnp.int32(0),
            shadow_write_index=jnp.int32(0),
            shadow_record_count=jnp.int32(0),
            has_shadow=jnp.asarray(False, dtype=jnp.bool_),
            last_shadow_decision_id=z2,
            state_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        return self._with_checksum(state)

    def state_valid(self, state: EmbodiedSafetyEnvelopeState) -> Bool[Array, ""]:
        """Return the complete fixed-shape, value, counter, and checksum gate."""

        self._check_state_contract(state)
        cfg = self._config
        counter_sum = (
            state.decision_count
            + state.shadow_record_count
            + state.reset_count
            + state.rollback_count
            + state.emergency_stop_latch_count
        )
        counters_valid = (
            (state.decision_count >= 0)
            & (state.decision_count <= cfg.max_decisions)
            & (state.committed_action_count >= 0)
            & (state.committed_action_count <= cfg.max_committed_actions)
            & (state.committed_action_count <= state.decision_count)
            & (state.fallback_action_count >= 0)
            & (state.fallback_action_count <= state.committed_action_count)
            & (state.rejected_action_count >= 0)
            & (
                state.rejected_action_count
                == state.decision_count - state.committed_action_count
            )
            & (state.hard_violation_count >= 0)
            & (state.hard_violation_count <= state.decision_count)
            & (state.emergency_stop_latch_count >= 0)
            & (
                state.emergency_stop_latch_count
                == state.reset_count + state.emergency_stop_latched.astype(jnp.int32)
            )
            & ((~state.deployment_suspended) | state.emergency_stop_latched)
            & (state.reset_count >= 0)
            & (state.reset_count <= cfg.max_handshakes_per_kind)
            & (state.rollback_count >= 0)
            & (state.rollback_count <= cfg.max_handshakes_per_kind)
            & (state.shadow_record_count >= 0)
            & (state.shadow_record_count <= cfg.max_shadow_records)
            & (state.revision == counter_sum)
            & (state.revision >= 0)
        )
        identity_valid = (
            (state.has_decision == (state.decision_count > 0))
            & (state.has_action == (state.committed_action_count > 0))
            & (state.has_telemetry == ((state.decision_count + state.reset_count) > 0))
            & (state.has_reset_nonce == (state.reset_count > 0))
            & (state.has_rollback_nonce == (state.rollback_count > 0))
            & (state.has_shadow == (state.shadow_record_count > 0))
            & (
                state.has_decision
                == self._digest_nonzero(state.last_decision_id)
            )
            & (state.has_action == self._digest_nonzero(state.last_action_id))
            & (
                state.has_telemetry
                == self._digest_nonzero(state.last_telemetry_id)
            )
            & (
                state.has_reset_nonce
                == self._digest_nonzero(state.last_reset_nonce)
            )
            & (
                state.has_rollback_nonce
                == self._digest_nonzero(state.last_rollback_nonce)
            )
            & (
                state.has_shadow
                == self._digest_nonzero(state.last_shadow_decision_id)
            )
            & ((~state.has_emergency_stop_sample) | (state.emergency_stop_latch_count > 0))
            & (
                state.has_emergency_stop_sample
                | ~self._digest_nonzero(state.last_emergency_stop_telemetry_id)
            )
            & (
                state.has_emergency_stop_sample
                | ~self._digest_nonzero(state.last_emergency_stop_sample_tick)
            )
        )
        action_log_finite = (
            jnp.isfinite(state.last_untrusted_reward)
            & jnp.isfinite(state.last_learned_cost_estimate)
            & jnp.all(jnp.isfinite(state.last_command_joint_position))
            & jnp.all(jnp.isfinite(state.last_command_joint_velocity))
            & jnp.all(jnp.isfinite(state.last_command_joint_torque))
            & jnp.all(jnp.isfinite(state.last_command_workspace_position))
            & jnp.isfinite(state.last_command_collision_clearance)
        )
        logged_command = EmbodiedCommand(
            joint_position=state.last_command_joint_position,
            joint_velocity=state.last_command_joint_velocity,
            joint_torque=state.last_command_joint_torque,
            workspace_position=state.last_command_workspace_position,
            collision_clearance=state.last_command_collision_clearance,
        )
        logged_command_safe = self._command_gates(logged_command)[-1]
        action_log_present = (
            self._digest_nonzero(state.last_committed_decision_id)
            & self._digest_nonzero(state.last_model_version)
            & self._digest_nonzero(state.last_optimizer_version)
            & self._digest_nonzero(state.last_lifecycle_version)
            & jnp.array_equal(state.last_logged_config_digest, self._config_digest)
            & action_log_finite
            & logged_command_safe
        )
        action_log_empty = (
            ~self._digest_nonzero(state.last_committed_decision_id)
            & ~self._digest_nonzero(state.last_model_version)
            & ~self._digest_nonzero(state.last_optimizer_version)
            & ~self._digest_nonzero(state.last_lifecycle_version)
            & ~self._digest_nonzero(state.last_logged_config_digest)
            & ~self._digest_nonzero(state.last_partner_metadata_digest)
            & (state.last_untrusted_reward == 0.0)
            & (state.last_learned_cost_estimate == 0.0)
            & (~state.last_action_was_fallback)
            & jnp.all(state.last_command_joint_position == 0.0)
            & jnp.all(state.last_command_joint_velocity == 0.0)
            & jnp.all(state.last_command_joint_torque == 0.0)
            & jnp.all(state.last_command_workspace_position == 0.0)
            & (state.last_command_collision_clearance == 0.0)
        )
        action_log_valid = action_log_finite & jnp.where(
            state.has_action,
            action_log_present,
            action_log_empty,
        )
        expected_shadow_size = jnp.minimum(
            state.shadow_record_count,
            jnp.int32(cfg.shadow_window),
        )
        expected_shadow_valid = jnp.arange(cfg.shadow_window, dtype=jnp.int32) < (
            expected_shadow_size
        )
        shadow_layout_valid = (
            (state.shadow_size == expected_shadow_size)
            & (
                state.shadow_write_index
                == jnp.mod(state.shadow_record_count, cfg.shadow_window)
            )
            & jnp.array_equal(state.shadow_valid, expected_shadow_valid)
        )
        valid_mask = state.shadow_valid
        invalid_mask = ~valid_mask
        shadow_values_valid = (
            jnp.all(jnp.isfinite(state.shadow_calibration_error))
            & jnp.all(jnp.isfinite(state.shadow_untrusted_rewards))
            & jnp.all(jnp.isfinite(state.shadow_learned_cost_estimates))
            & jnp.all(state.shadow_calibration_error >= 0.0)
            & jnp.all(state.shadow_latency_ticks >= 0)
            & jnp.all(
                (~valid_mask)
                | jnp.any(state.shadow_decision_ids != 0, axis=1)
            )
            & jnp.all(
                (~valid_mask)
                | jnp.any(state.shadow_model_versions != 0, axis=1)
            )
            & jnp.all(
                (~valid_mask)
                | jnp.any(state.shadow_optimizer_versions != 0, axis=1)
            )
            & jnp.all(
                (~valid_mask)
                | jnp.any(state.shadow_lifecycle_versions != 0, axis=1)
            )
            & jnp.all((~invalid_mask) | (~state.shadow_hard_violation))
            & jnp.all((~invalid_mask) | (~state.shadow_success))
            & jnp.all((~invalid_mask) | (state.shadow_calibration_error == 0.0))
            & jnp.all((~invalid_mask) | (state.shadow_latency_ticks == 0))
            & jnp.all((~invalid_mask[:, None]) | (state.shadow_decision_ids == 0))
            & jnp.all((~invalid_mask[:, None]) | (state.shadow_model_versions == 0))
            & jnp.all(
                (~invalid_mask[:, None]) | (state.shadow_optimizer_versions == 0)
            )
            & jnp.all(
                (~invalid_mask[:, None]) | (state.shadow_lifecycle_versions == 0)
            )
            & jnp.all(
                (~invalid_mask[:, None])
                | (state.shadow_partner_metadata_digests == 0)
            )
            & jnp.all((~invalid_mask) | (state.shadow_untrusted_rewards == 0.0))
            & jnp.all(
                (~invalid_mask) | (state.shadow_learned_cost_estimates == 0.0)
            )
        )
        latest_index = jnp.mod(state.shadow_write_index - jnp.int32(1), cfg.shadow_window)
        latest_shadow_valid = (~state.has_shadow) | jnp.array_equal(
            state.last_shadow_decision_id,
            state.shadow_decision_ids[latest_index],
        )
        return (
            self._digest_nonzero(state.source_digest)
            & jnp.array_equal(state.config_digest, self._config_digest)
            & counters_valid
            & identity_valid
            & action_log_valid
            & shadow_layout_valid
            & shadow_values_valid
            & latest_shadow_valid
            & jnp.array_equal(
                state.state_checksum,
                _checksum_arrays(self._state_payload(state)),
            )
        )

    def evaluate(
        self,
        state: EmbodiedSafetyEnvelopeState,
        telemetry: EmbodiedTelemetry,
        proposed_command: EmbodiedCommand,
        *,
        decision_id: Array,
        action_id: Array,
        control_tick: Array,
        control_deadline_tick: Array,
        model_version: Array,
        optimizer_version: Array,
        lifecycle_version: Array,
        untrusted_reward: float | Array,
        partner_metadata_digest: Array,
        learned_cost_estimate: float | Array,
    ) -> EmbodiedEnvelopeDecision:
        """Evaluate one command, logging a rejection or one certified command."""

        self._check_state_contract(state)
        self._check_telemetry_contract(telemetry)
        self._check_command_contract(proposed_command, name="proposed_command")
        decision = _require_array(
            decision_id,
            name="decision_id",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        action = _require_array(
            action_id,
            name="action_id",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        now = _require_array(
            control_tick,
            name="control_tick",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        deadline = _require_array(
            control_deadline_tick,
            name="control_deadline_tick",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        model = _require_array(
            model_version,
            name="model_version",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        optimizer = _require_array(
            optimizer_version,
            name="optimizer_version",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        lifecycle = _require_array(
            lifecycle_version,
            name="lifecycle_version",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        partner = _require_array(
            partner_metadata_digest,
            name="partner_metadata_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        reward = _float32_scalar(untrusted_reward, name="untrusted_reward")
        learned_cost = _float32_scalar(
            learned_cost_estimate,
            name="learned_cost_estimate",
        )
        persistent_valid = self.state_valid(state)
        decision_capacity = (
            (state.decision_count < self._config.max_decisions)
            & (state.revision < _INT32_MAX)
        )
        action_capacity = state.committed_action_count < (
            self._config.max_committed_actions
        )
        decision_identity_valid = self._digest_nonzero(decision) & (
            (~state.has_decision) | _words_greater(decision, state.last_decision_id)
        )
        telemetry_identity_valid = self._digest_nonzero(telemetry.telemetry_id) & (
            (~state.has_telemetry)
            | _words_greater(telemetry.telemetry_id, state.last_telemetry_id)
        )
        action_identity_valid = self._digest_nonzero(action) & (
            (~state.has_action) | _words_greater(action, state.last_action_id)
        )
        sample_monotonic = (~state.has_telemetry) | _words_greater_equal(
            telemetry.sample_tick,
            state.last_sample_tick,
        )
        control_monotonic = (~state.has_telemetry) | _words_greater_equal(
            now,
            state.last_control_tick,
        )
        sample_not_future = _words_greater_equal(now, telemetry.sample_tick)
        control_time_monotonic = (
            sample_monotonic & control_monotonic & sample_not_future
        )
        telemetry_fresh = _within_forward_delta(
            telemetry.sample_tick,
            now,
            self._config.max_telemetry_age_ticks,
        )
        control_deadline_valid = _within_forward_delta(
            now,
            deadline,
            self._config.max_control_deadline_ticks,
        )
        current_gates = self._telemetry_gates(telemetry)
        proposed_gates = self._command_gates(proposed_command)
        current_safe = current_gates[-1]
        proposed_safe = proposed_gates[-1]
        version_binding_valid = (
            self._digest_nonzero(model)
            & self._digest_nonzero(optimizer)
            & self._digest_nonzero(lifecycle)
        )
        metadata_finite = jnp.isfinite(reward) & jnp.isfinite(learned_cost)
        base_transaction = (
            persistent_valid
            & decision_capacity
            & decision_identity_valid
            & telemetry_identity_valid
            & control_time_monotonic
        )
        fresh_stop_latch = telemetry.emergency_stop & (~state.emergency_stop_latched)
        latched_after_request = state.emergency_stop_latched | telemetry.emergency_stop
        common_command_gate = (
            base_transaction
            & action_capacity
            & action_identity_valid
            & telemetry.bridge_connected
            & telemetry_fresh
            & control_deadline_valid
            & (~latched_after_request)
            & (~state.deployment_suspended)
            & current_safe
            & version_binding_valid
            & metadata_finite
        )
        proposed_accepted_pre = common_command_gate & proposed_safe
        fallback_certified_pre = common_command_gate & (~proposed_safe)
        action_available_pre = proposed_accepted_pre | fallback_certified_pre
        fallback_used_pre = fallback_certified_pre & action_available_pre
        hard_violation = (
            (~telemetry.bridge_connected)
            | (~telemetry_fresh)
            | (~control_deadline_valid)
            | telemetry.emergency_stop
            | state.emergency_stop_latched
            | state.deployment_suspended
            | (~current_safe)
            | (~proposed_safe)
        )
        selected_position = jnp.where(
            proposed_accepted_pre,
            proposed_command.joint_position,
            self._fallback.joint_position,
        )
        selected_velocity = jnp.where(
            proposed_accepted_pre,
            proposed_command.joint_velocity,
            self._fallback.joint_velocity,
        )
        selected_torque = jnp.where(
            proposed_accepted_pre,
            proposed_command.joint_torque,
            self._fallback.joint_torque,
        )
        selected_workspace = jnp.where(
            proposed_accepted_pre,
            proposed_command.workspace_position,
            self._fallback.workspace_position,
        )
        selected_clearance = jnp.where(
            proposed_accepted_pre,
            proposed_command.collision_clearance,
            self._fallback.collision_clearance,
        )
        selected = EmbodiedCommand(
            joint_position=selected_position,
            joint_velocity=selected_velocity,
            joint_torque=selected_torque,
            workspace_position=selected_workspace,
            collision_clearance=selected_clearance,
        )
        proposed_state = state.replace(
            revision=(
                state.revision
                + jnp.int32(1)
                + fresh_stop_latch.astype(jnp.int32)
            ),
            emergency_stop_latched=latched_after_request,
            emergency_stop_latch_count=(
                state.emergency_stop_latch_count
                + fresh_stop_latch.astype(jnp.int32)
            ),
            has_emergency_stop_sample=(
                state.has_emergency_stop_sample | telemetry.emergency_stop
            ),
            last_emergency_stop_telemetry_id=jnp.where(
                telemetry.emergency_stop,
                telemetry.telemetry_id,
                state.last_emergency_stop_telemetry_id,
            ),
            last_emergency_stop_sample_tick=jnp.where(
                telemetry.emergency_stop,
                telemetry.sample_tick,
                state.last_emergency_stop_sample_tick,
            ),
            has_telemetry=jnp.asarray(True, dtype=jnp.bool_),
            last_telemetry_id=telemetry.telemetry_id,
            last_sample_tick=telemetry.sample_tick,
            last_control_tick=now,
            has_decision=jnp.asarray(True, dtype=jnp.bool_),
            last_decision_id=decision,
            has_action=state.has_action | action_available_pre,
            last_action_id=jnp.where(
                action_available_pre,
                action,
                state.last_action_id,
            ),
            decision_count=state.decision_count + jnp.int32(1),
            committed_action_count=(
                state.committed_action_count + action_available_pre.astype(jnp.int32)
            ),
            fallback_action_count=(
                state.fallback_action_count + fallback_used_pre.astype(jnp.int32)
            ),
            rejected_action_count=(
                state.rejected_action_count + (~action_available_pre).astype(jnp.int32)
            ),
            hard_violation_count=(
                state.hard_violation_count + hard_violation.astype(jnp.int32)
            ),
            last_committed_decision_id=jnp.where(
                action_available_pre,
                decision,
                state.last_committed_decision_id,
            ),
            last_model_version=jnp.where(
                action_available_pre,
                model,
                state.last_model_version,
            ),
            last_optimizer_version=jnp.where(
                action_available_pre,
                optimizer,
                state.last_optimizer_version,
            ),
            last_lifecycle_version=jnp.where(
                action_available_pre,
                lifecycle,
                state.last_lifecycle_version,
            ),
            last_logged_config_digest=jnp.where(
                action_available_pre,
                self._config_digest,
                state.last_logged_config_digest,
            ),
            last_partner_metadata_digest=jnp.where(
                action_available_pre,
                partner,
                state.last_partner_metadata_digest,
            ),
            last_untrusted_reward=jnp.where(
                action_available_pre,
                reward,
                state.last_untrusted_reward,
            ),
            last_learned_cost_estimate=jnp.where(
                action_available_pre,
                learned_cost,
                state.last_learned_cost_estimate,
            ),
            last_action_was_fallback=jnp.where(
                action_available_pre,
                fallback_used_pre,
                state.last_action_was_fallback,
            ),
            last_command_joint_position=jnp.where(
                action_available_pre,
                selected.joint_position,
                state.last_command_joint_position,
            ),
            last_command_joint_velocity=jnp.where(
                action_available_pre,
                selected.joint_velocity,
                state.last_command_joint_velocity,
            ),
            last_command_joint_torque=jnp.where(
                action_available_pre,
                selected.joint_torque,
                state.last_command_joint_torque,
            ),
            last_command_workspace_position=jnp.where(
                action_available_pre,
                selected.workspace_position,
                state.last_command_workspace_position,
            ),
            last_command_collision_clearance=jnp.where(
                action_available_pre,
                selected.collision_clearance,
                state.last_command_collision_clearance,
            ),
            state_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        proposed_state = self._with_checksum(proposed_state)
        post_valid = self.state_valid(proposed_state)
        transaction_applied = base_transaction & post_valid
        action_available = action_available_pre & transaction_applied
        proposed_accepted = proposed_accepted_pre & transaction_applied
        fallback_used = fallback_used_pre & transaction_applied
        # A structurally valid asserted emergency stop is a separate safety
        # transition.  Replay, stale clocks, exhausted decision capacity, and
        # invalid optional metadata may reject the command transaction, but
        # none of them may suppress the persistent stop latch.
        stop_only_state = state.replace(
            revision=state.revision + jnp.int32(1),
            emergency_stop_latched=jnp.asarray(True, dtype=jnp.bool_),
            emergency_stop_latch_count=state.emergency_stop_latch_count + jnp.int32(1),
            has_emergency_stop_sample=jnp.asarray(True, dtype=jnp.bool_),
            last_emergency_stop_telemetry_id=telemetry.telemetry_id,
            last_emergency_stop_sample_tick=telemetry.sample_tick,
            state_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        stop_only_state = self._with_checksum(stop_only_state)
        stop_only_applied = (
            persistent_valid & fresh_stop_latch & self.state_valid(stop_only_state)
        )
        next_state = jax.lax.cond(
            transaction_applied,
            lambda _: proposed_state,
            lambda _: jax.lax.cond(
                stop_only_applied,
                lambda __: stop_only_state,
                lambda __: state,
                None,
            ),
            None,
        )
        zero_command = EmbodiedCommand(
            joint_position=jnp.zeros_like(selected.joint_position),
            joint_velocity=jnp.zeros_like(selected.joint_velocity),
            joint_torque=jnp.zeros_like(selected.joint_torque),
            workspace_position=jnp.zeros_like(selected.workspace_position),
            collision_clearance=jnp.float32(0.0),
        )
        returned_command = jax.tree.map(
            lambda selected_value, zero_value: jnp.where(
                action_available,
                selected_value,
                zero_value,
            ),
            selected,
            zero_command,
        )
        reason = jnp.where(
            ~persistent_valid,
            ENVELOPE_REASON_PERSISTENT_STATE,
            jnp.where(
                ~decision_capacity,
                ENVELOPE_REASON_CAPACITY,
                jnp.where(
                    ~decision_identity_valid,
                    ENVELOPE_REASON_DECISION_IDENTITY,
                    jnp.where(
                        ~telemetry_identity_valid,
                        ENVELOPE_REASON_TELEMETRY_IDENTITY,
                        jnp.where(
                            ~control_time_monotonic,
                            ENVELOPE_REASON_TIME_MONOTONICITY,
                            jnp.where(
                                ~telemetry.bridge_connected,
                                ENVELOPE_REASON_BRIDGE_DISCONNECTED,
                                jnp.where(
                                    ~telemetry_fresh,
                                    ENVELOPE_REASON_TELEMETRY_STALE,
                                    jnp.where(
                                        ~control_deadline_valid,
                                        ENVELOPE_REASON_CONTROL_DEADLINE,
                                        jnp.where(
                                            latched_after_request,
                                            ENVELOPE_REASON_EMERGENCY_STOP,
                                            jnp.where(
                                                state.deployment_suspended,
                                                ENVELOPE_REASON_DEPLOYMENT_SUSPENDED,
                                                jnp.where(
                                                    ~current_safe,
                                                    ENVELOPE_REASON_CURRENT_ENVELOPE,
                                                    jnp.where(
                                                        ~version_binding_valid,
                                                        ENVELOPE_REASON_VERSION_BINDING,
                                                        jnp.where(
                                                            ~metadata_finite,
                                                            ENVELOPE_REASON_METADATA,
                                                            jnp.where(
                                                                ~action_identity_valid,
                                                                ENVELOPE_REASON_ACTION_IDENTITY,
                                                                jnp.where(
                                                                    ~action_capacity,
                                                                    ENVELOPE_REASON_ACTION_CAPACITY,
                                                                    ENVELOPE_REASON_FALLBACK_UNAVAILABLE,
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        reason = jnp.where(action_available, ENVELOPE_REASON_AVAILABLE, reason).astype(
            jnp.int32
        )
        return EmbodiedEnvelopeDecision(
            state=next_state,
            command=cast(EmbodiedCommand, returned_command),
            action_available=action_available,
            proposed_accepted=proposed_accepted,
            fallback_used=fallback_used,
            transaction_applied=transaction_applied,
            emergency_stop_latch_applied=(
                fresh_stop_latch & (transaction_applied | stop_only_applied)
            ),
            persistent_state_valid=persistent_valid,
            decision_capacity_available=decision_capacity,
            action_capacity_available=action_capacity,
            decision_identity_valid=decision_identity_valid,
            telemetry_identity_valid=telemetry_identity_valid,
            action_identity_valid=action_identity_valid,
            control_time_monotonic=control_time_monotonic,
            telemetry_fresh=telemetry_fresh,
            control_deadline_valid=control_deadline_valid,
            bridge_connected=telemetry.bridge_connected,
            emergency_stop_input=telemetry.emergency_stop,
            emergency_stop_latched_before=state.emergency_stop_latched,
            emergency_stop_latched_after=next_state.emergency_stop_latched,
            deployment_not_suspended=~state.deployment_suspended,
            current_position_finite=current_gates[0],
            current_position_in_bounds=current_gates[1],
            current_velocity_finite=current_gates[2],
            current_velocity_in_bounds=current_gates[3],
            current_torque_finite=current_gates[4],
            current_torque_in_bounds=current_gates[5],
            current_workspace_finite=current_gates[6],
            current_workspace_in_bounds=current_gates[7],
            current_clearance_finite=current_gates[8],
            current_clearance_in_bounds=current_gates[9],
            proposed_position_finite=proposed_gates[0],
            proposed_position_in_bounds=proposed_gates[1],
            proposed_velocity_finite=proposed_gates[2],
            proposed_velocity_in_bounds=proposed_gates[3],
            proposed_torque_finite=proposed_gates[4],
            proposed_torque_in_bounds=proposed_gates[5],
            proposed_workspace_finite=proposed_gates[6],
            proposed_workspace_in_bounds=proposed_gates[7],
            proposed_clearance_finite=proposed_gates[8],
            proposed_clearance_in_bounds=proposed_gates[9],
            version_binding_valid=version_binding_valid,
            metadata_finite=metadata_finite,
            current_envelope_safe=current_safe,
            proposed_envelope_safe=proposed_safe,
            fallback_certified=fallback_certified_pre & transaction_applied,
            hard_violation=hard_violation,
            unavailable_reason=reason,
            learned_cost_override_used=jnp.asarray(False, dtype=jnp.bool_),
            action_dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
            physical_safety_claim=jnp.asarray(False, dtype=jnp.bool_),
        )

    def evaluate_shadow(
        self,
        state: EmbodiedSafetyEnvelopeState,
        telemetry: EmbodiedTelemetry,
        proposed_command: EmbodiedCommand,
        *,
        decision_id: Array,
        control_tick: Array,
        control_deadline_tick: Array,
        model_version: Array,
        optimizer_version: Array,
        lifecycle_version: Array,
        observed_success: bool | Array,
        calibration_error: float | Array,
        latency_ticks: int | Array,
        untrusted_reward: float | Array,
        partner_metadata_digest: Array,
        learned_cost_estimate: float | Array,
    ) -> EmbodiedShadowEvaluation:
        """Purely evaluate one shadow outcome without changing any state."""

        self._check_state_contract(state)
        self._check_telemetry_contract(telemetry)
        self._check_command_contract(proposed_command, name="proposed_command")
        decision = _require_array(
            decision_id,
            name="decision_id",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        now = _require_array(
            control_tick,
            name="control_tick",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        deadline = _require_array(
            control_deadline_tick,
            name="control_deadline_tick",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        model = _require_array(
            model_version,
            name="model_version",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        optimizer = _require_array(
            optimizer_version,
            name="optimizer_version",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        lifecycle = _require_array(
            lifecycle_version,
            name="lifecycle_version",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        partner = _require_array(
            partner_metadata_digest,
            name="partner_metadata_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        success = _bool_scalar(observed_success, name="observed_success")
        calibration = _float32_scalar(calibration_error, name="calibration_error")
        latency = _int32_scalar(latency_ticks, name="latency_ticks")
        reward = _float32_scalar(untrusted_reward, name="untrusted_reward")
        learned_cost = _float32_scalar(
            learned_cost_estimate,
            name="learned_cost_estimate",
        )
        state_is_valid = self.state_valid(state)
        current_safe = self._telemetry_gates(telemetry)[-1]
        proposed_safe = self._command_gates(proposed_command)[-1]
        telemetry_fresh = _within_forward_delta(
            telemetry.sample_tick,
            now,
            self._config.max_telemetry_age_ticks,
        )
        deadline_valid = _within_forward_delta(
            now,
            deadline,
            self._config.max_control_deadline_ticks,
        )
        hard_violation = (
            (~telemetry.bridge_connected)
            | (~telemetry_fresh)
            | (~deadline_valid)
            | telemetry.emergency_stop
            | state.emergency_stop_latched
            | state.deployment_suspended
            | (~current_safe)
            | (~proposed_safe)
        )
        action_would_be_available = (
            telemetry.bridge_connected
            & telemetry_fresh
            & deadline_valid
            & (~telemetry.emergency_stop)
            & (~state.emergency_stop_latched)
            & (~state.deployment_suspended)
            & current_safe
        )
        metadata_finite = (
            jnp.isfinite(calibration)
            & (calibration >= 0.0)
            & (latency >= 0)
            & jnp.isfinite(reward)
            & jnp.isfinite(learned_cost)
        )
        version_valid = (
            self._digest_nonzero(model)
            & self._digest_nonzero(optimizer)
            & self._digest_nonzero(lifecycle)
        )
        valid = (
            state_is_valid
            & self._digest_nonzero(decision)
            & version_valid
            & metadata_finite
        )
        outcome = EmbodiedShadowEvaluation(
            valid=valid,
            state_revision=state.revision,
            state_checksum=state.state_checksum,
            source_digest=state.source_digest,
            config_digest=self._config_digest,
            decision_id=decision,
            model_version=model,
            optimizer_version=optimizer,
            lifecycle_version=lifecycle,
            partner_metadata_digest=partner,
            untrusted_reward=jnp.where(valid, reward, 0.0).astype(jnp.float32),
            learned_cost_estimate=jnp.where(valid, learned_cost, 0.0).astype(
                jnp.float32
            ),
            hard_violation=valid & hard_violation,
            observed_success=valid & success,
            calibration_error=jnp.where(valid, calibration, 0.0).astype(jnp.float32),
            latency_ticks=jnp.where(valid, latency, 0).astype(jnp.int32),
            action_would_be_available=valid & action_would_be_available,
            evaluation_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            dispatches=jnp.int32(0),
            learning_state_mutations=jnp.int32(0),
            deployment_authority=jnp.asarray(False, dtype=jnp.bool_),
            physical_safety_claim=jnp.asarray(False, dtype=jnp.bool_),
        )
        return dataclasses.replace(
            outcome,
            evaluation_checksum=_checksum_arrays(self._shadow_payload(outcome)),
        )

    @staticmethod
    def _shadow_payload(outcome: EmbodiedShadowEvaluation) -> tuple[Array, ...]:
        return tuple(
            cast(Array, getattr(outcome, field.name))
            for field in dataclasses.fields(EmbodiedShadowEvaluation)
            if field.name != "evaluation_checksum"
        )

    def _check_shadow_contract(self, outcome: EmbodiedShadowEvaluation) -> None:
        if type(outcome) is not EmbodiedShadowEvaluation:
            raise TypeError("outcome must be an exact EmbodiedShadowEvaluation")
        contracts: dict[str, tuple[tuple[int, ...], Any]] = {
            "valid": ((), jnp.bool_),
            "state_revision": ((), jnp.int32),
            "state_checksum": ((_IDENTITY_WORDS,), jnp.uint32),
            "source_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "config_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "decision_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "model_version": ((_DIGEST_WORDS,), jnp.uint32),
            "optimizer_version": ((_DIGEST_WORDS,), jnp.uint32),
            "lifecycle_version": ((_DIGEST_WORDS,), jnp.uint32),
            "partner_metadata_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "untrusted_reward": ((), jnp.float32),
            "learned_cost_estimate": ((), jnp.float32),
            "hard_violation": ((), jnp.bool_),
            "observed_success": ((), jnp.bool_),
            "calibration_error": ((), jnp.float32),
            "latency_ticks": ((), jnp.int32),
            "action_would_be_available": ((), jnp.bool_),
            "evaluation_checksum": ((_IDENTITY_WORDS,), jnp.uint32),
            "dispatches": ((), jnp.int32),
            "learning_state_mutations": ((), jnp.int32),
            "deployment_authority": ((), jnp.bool_),
            "physical_safety_claim": ((), jnp.bool_),
        }
        for field_name, (shape, dtype) in contracts.items():
            _require_array(
                getattr(outcome, field_name),
                name=f"outcome.{field_name}",
                shape=shape,
                dtype=dtype,
            )

    def record_shadow(
        self,
        state: EmbodiedSafetyEnvelopeState,
        outcome: EmbodiedShadowEvaluation,
    ) -> EmbodiedShadowRecordResult:
        """Atomically record one exact state-bound shadow outcome."""

        self._check_state_contract(state)
        self._check_shadow_contract(outcome)
        persistent_valid = self.state_valid(state)
        binding_valid = (
            (outcome.state_revision == state.revision)
            & jnp.array_equal(outcome.state_checksum, state.state_checksum)
            & jnp.array_equal(outcome.source_digest, state.source_digest)
            & jnp.array_equal(outcome.config_digest, self._config_digest)
            & (outcome.dispatches == 0)
            & (outcome.learning_state_mutations == 0)
            & (~outcome.deployment_authority)
            & (~outcome.physical_safety_claim)
            & jnp.array_equal(
                outcome.evaluation_checksum,
                _checksum_arrays(self._shadow_payload(outcome)),
            )
        )
        identity_valid = self._digest_nonzero(outcome.decision_id) & (
            (~state.has_shadow)
            | _words_greater(outcome.decision_id, state.last_shadow_decision_id)
        )
        capacity = (
            (state.shadow_record_count < self._config.max_shadow_records)
            & (state.revision < _INT32_MAX)
        )
        transaction_valid = persistent_valid & binding_valid
        applied_pre = transaction_valid & outcome.valid & identity_valid & capacity
        index = state.shadow_write_index
        next_record_count = state.shadow_record_count + jnp.int32(1)
        proposed = state.replace(
            revision=state.revision + jnp.int32(1),
            shadow_valid=state.shadow_valid.at[index].set(True),
            shadow_hard_violation=state.shadow_hard_violation.at[index].set(
                outcome.hard_violation
            ),
            shadow_success=state.shadow_success.at[index].set(outcome.observed_success),
            shadow_calibration_error=state.shadow_calibration_error.at[index].set(
                outcome.calibration_error
            ),
            shadow_latency_ticks=state.shadow_latency_ticks.at[index].set(
                outcome.latency_ticks
            ),
            shadow_decision_ids=state.shadow_decision_ids.at[index].set(
                outcome.decision_id
            ),
            shadow_model_versions=state.shadow_model_versions.at[index].set(
                outcome.model_version
            ),
            shadow_optimizer_versions=state.shadow_optimizer_versions.at[index].set(
                outcome.optimizer_version
            ),
            shadow_lifecycle_versions=state.shadow_lifecycle_versions.at[index].set(
                outcome.lifecycle_version
            ),
            shadow_partner_metadata_digests=(
                state.shadow_partner_metadata_digests.at[index].set(
                    outcome.partner_metadata_digest
                )
            ),
            shadow_untrusted_rewards=state.shadow_untrusted_rewards.at[index].set(
                outcome.untrusted_reward
            ),
            shadow_learned_cost_estimates=(
                state.shadow_learned_cost_estimates.at[index].set(
                    outcome.learned_cost_estimate
                )
            ),
            shadow_size=jnp.minimum(
                state.shadow_size + jnp.int32(1),
                jnp.int32(self._config.shadow_window),
            ),
            shadow_write_index=jnp.mod(
                next_record_count,
                self._config.shadow_window,
            ).astype(jnp.int32),
            shadow_record_count=next_record_count,
            has_shadow=jnp.asarray(True, dtype=jnp.bool_),
            last_shadow_decision_id=outcome.decision_id,
            state_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        applied = applied_pre & self.state_valid(proposed)
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        return EmbodiedShadowRecordResult(
            state=next_state,
            transaction_valid=transaction_valid,
            applied=applied,
            replay_rejected=transaction_valid & outcome.valid & (~identity_valid),
            capacity_available=capacity,
        )

    def deployment_gate(
        self,
        state: EmbodiedSafetyEnvelopeState,
    ) -> EmbodiedDeploymentGate:
        """Read conservative readiness from the exact fixed recent ring."""

        self._check_state_contract(state)
        state_is_valid = self.state_valid(state)
        valid = state.shadow_valid
        sample_count = jnp.sum(valid.astype(jnp.int32))
        success_count = jnp.sum((valid & state.shadow_success).astype(jnp.int32))
        hard_count = jnp.sum(
            (valid & state.shadow_hard_violation).astype(jnp.int32)
        )
        enough = sample_count >= self._config.min_shadow_samples
        safe_n = jnp.maximum(sample_count, 1).astype(jnp.float32)
        p_hat = success_count.astype(jnp.float32) / safe_n
        z = jnp.asarray(self._config.wilson_z, dtype=jnp.float32)
        z2 = z * z
        denominator = 1.0 + z2 / safe_n
        center = p_hat + z2 / (2.0 * safe_n)
        margin = z * jnp.sqrt(
            jnp.maximum(
                p_hat * (1.0 - p_hat) / safe_n + z2 / (4.0 * safe_n * safe_n),
                0.0,
            )
        )
        success_lcb = jnp.clip((center - margin) / denominator, 0.0, 1.0)
        success_ready = enough & (
            success_lcb >= self._config.min_shadow_success_lcb
        )
        max_calibration = jnp.max(
            jnp.where(valid, state.shadow_calibration_error, 0.0),
            initial=jnp.float32(0.0),
        )
        calibration_ready = enough & (
            max_calibration <= self._config.max_shadow_calibration_error
        )
        max_latency = jnp.max(
            jnp.where(valid, state.shadow_latency_ticks, 0),
            initial=jnp.int32(0),
        )
        latency_ready = enough & (
            max_latency <= self._config.max_shadow_latency_ticks
        )
        hard_zero = enough & (hard_count == 0)
        ready = (
            state_is_valid
            & enough
            & hard_zero
            & success_ready
            & calibration_ready
            & latency_ready
        )
        return EmbodiedDeploymentGate(
            state_valid=state_is_valid,
            sample_count=sample_count,
            success_count=success_count,
            hard_violation_count=hard_count,
            hard_zero=hard_zero,
            performance_success_lcb=jnp.where(state_is_valid, success_lcb, 0.0),
            success_lcb_ready=state_is_valid & success_ready,
            max_calibration_error=jnp.where(state_is_valid, max_calibration, 0.0),
            calibration_ready=state_is_valid & calibration_ready,
            max_latency_ticks=jnp.where(state_is_valid, max_latency, 0).astype(jnp.int32),
            latency_ready=state_is_valid & latency_ready,
            enough_samples=state_is_valid & enough,
            deployment_ready=ready,
            learned_cost_override_used=jnp.asarray(False, dtype=jnp.bool_),
            deployment_authority=jnp.asarray(False, dtype=jnp.bool_),
            promotion_authority=jnp.asarray(False, dtype=jnp.bool_),
            scientific_promotion_allowed=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _handshake_common(
        self,
        state: EmbodiedSafetyEnvelopeState,
        handshake: AuthorityBoundEnvelopeHandshake,
        authority: Array,
        *,
        last_nonce: Array,
        has_nonce: Array,
        count: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        persistent_valid = self.state_valid(state)
        binding_valid = (
            jnp.array_equal(handshake.source_digest, state.source_digest)
            & jnp.array_equal(handshake.config_digest, self._config_digest)
            & (handshake.observed_state_revision == state.revision)
            & jnp.array_equal(handshake.observed_state_checksum, state.state_checksum)
        )
        authority_valid = jnp.array_equal(handshake.authority_digest, authority)
        nonce_valid = self._digest_nonzero(handshake.nonce) & (
            (~has_nonce) | _words_greater(handshake.nonce, last_nonce)
        )
        capacity = (
            (count < self._config.max_handshakes_per_kind)
            & (state.revision < _INT32_MAX)
        )
        return (
            persistent_valid,
            binding_valid,
            authority_valid,
            nonce_valid,
            capacity,
        )

    @staticmethod
    def _handshake_reason(
        *,
        persistent_valid: Array,
        binding_valid: Array,
        authority_valid: Array,
        nonce_valid: Array,
        capacity: Array,
        requested: Array,
        stationary_safe: Array,
        applied: Array,
    ) -> Array:
        reason = jnp.where(
            ~persistent_valid,
            HANDSHAKE_REASON_PERSISTENT_STATE,
            jnp.where(
                ~binding_valid,
                HANDSHAKE_REASON_BINDING,
                jnp.where(
                    ~authority_valid,
                    HANDSHAKE_REASON_AUTHORITY,
                    jnp.where(
                        ~nonce_valid,
                        HANDSHAKE_REASON_NONCE_REPLAY,
                        jnp.where(
                            ~capacity,
                            HANDSHAKE_REASON_CAPACITY,
                            jnp.where(
                                ~requested,
                                HANDSHAKE_REASON_NOT_REQUESTED,
                                HANDSHAKE_REASON_NOT_STATIONARY_SAFE,
                            ),
                        ),
                    ),
                ),
            ),
        )
        return jnp.where(applied, HANDSHAKE_REASON_AVAILABLE, reason).astype(jnp.int32)

    def authority_bound_rollback(
        self,
        state: EmbodiedSafetyEnvelopeState,
        handshake: AuthorityBoundEnvelopeHandshake,
    ) -> EmbodiedHandshakeResult:
        """Token-bind a suspension without reverting identities or diagnostics.

        This pure equality gate is not caller authentication.  A deployment
        boundary must authenticate the caller before constructing the token.
        """

        self._check_state_contract(state)
        self._check_handshake_contract(handshake)
        (
            persistent_valid,
            binding_valid,
            authority_valid,
            nonce_valid,
            capacity,
        ) = self._handshake_common(
            state,
            handshake,
            self._rollback_authority,
            last_nonce=state.last_rollback_nonce,
            has_nonce=state.has_rollback_nonce,
            count=state.rollback_count,
        )
        transaction_valid = persistent_valid & binding_valid & authority_valid
        applied_pre = transaction_valid & nonce_valid & capacity
        fresh_stop_latch = ~state.emergency_stop_latched
        proposed = state.replace(
            revision=(
                state.revision
                + jnp.int32(1)
                + fresh_stop_latch.astype(jnp.int32)
            ),
            emergency_stop_latched=jnp.asarray(True, dtype=jnp.bool_),
            emergency_stop_latch_count=(
                state.emergency_stop_latch_count
                + fresh_stop_latch.astype(jnp.int32)
            ),
            deployment_suspended=jnp.asarray(True, dtype=jnp.bool_),
            has_rollback_nonce=jnp.asarray(True, dtype=jnp.bool_),
            last_rollback_nonce=handshake.nonce,
            rollback_count=state.rollback_count + jnp.int32(1),
            state_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        applied = applied_pre & self.state_valid(proposed)
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        reason = self._handshake_reason(
            persistent_valid=persistent_valid,
            binding_valid=binding_valid,
            authority_valid=authority_valid,
            nonce_valid=nonce_valid,
            capacity=capacity,
            requested=jnp.asarray(True, dtype=jnp.bool_),
            stationary_safe=jnp.asarray(True, dtype=jnp.bool_),
            applied=applied,
        )
        return EmbodiedHandshakeResult(
            state=next_state,
            transaction_valid=transaction_valid,
            applied=applied,
            replay_rejected=(
                persistent_valid
                & authority_valid
                & self._digest_nonzero(handshake.nonce)
                & state.has_rollback_nonce
                & (~_words_greater(handshake.nonce, state.last_rollback_nonce))
            ),
            stationary_safe=jnp.asarray(True, dtype=jnp.bool_),
            unavailable_reason=reason,
        )

    def authority_bound_reset(
        self,
        state: EmbodiedSafetyEnvelopeState,
        handshake: AuthorityBoundEnvelopeHandshake,
        telemetry: EmbodiedTelemetry,
        *,
        control_tick: Array,
        control_deadline_tick: Array,
    ) -> EmbodiedHandshakeResult:
        """Token-bind reset to a stationary-safe sample after external authentication."""

        self._check_state_contract(state)
        self._check_handshake_contract(handshake)
        self._check_telemetry_contract(telemetry)
        now = _require_array(
            control_tick,
            name="control_tick",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        deadline = _require_array(
            control_deadline_tick,
            name="control_deadline_tick",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        (
            persistent_valid,
            binding_valid,
            authority_valid,
            nonce_valid,
            capacity,
        ) = self._handshake_common(
            state,
            handshake,
            self._reset_authority,
            last_nonce=state.last_reset_nonce,
            has_nonce=state.has_reset_nonce,
            count=state.reset_count,
        )
        requested = state.emergency_stop_latched | state.deployment_suspended
        telemetry_identity_valid = self._digest_nonzero(telemetry.telemetry_id) & (
            (~state.has_telemetry)
            | _words_greater(telemetry.telemetry_id, state.last_telemetry_id)
        )
        control_monotonic = (
            ((~state.has_telemetry) | _words_greater_equal(now, state.last_control_tick))
            & ((~state.has_telemetry) | _words_greater_equal(
                telemetry.sample_tick,
                state.last_sample_tick,
            ))
            & _words_greater_equal(now, telemetry.sample_tick)
        )
        fresh = _within_forward_delta(
            telemetry.sample_tick,
            now,
            self._config.max_telemetry_age_ticks,
        )
        deadline_valid = _within_forward_delta(
            now,
            deadline,
            self._config.max_control_deadline_ticks,
        )
        current_safe = self._telemetry_gates(telemetry)[-1]
        stationary = jnp.all(
            jnp.abs(telemetry.joint_velocity)
            <= self._config.reset_stationary_velocity_tolerance
        )
        newer_than_stop_sample = (~state.has_emergency_stop_sample) | (
            _words_greater(
                telemetry.telemetry_id,
                state.last_emergency_stop_telemetry_id,
            )
            & _words_greater(
                telemetry.sample_tick,
                state.last_emergency_stop_sample_tick,
            )
        )
        stationary_safe = (
            telemetry.bridge_connected
            & (~telemetry.emergency_stop)
            & telemetry_identity_valid
            & control_monotonic
            & fresh
            & deadline_valid
            & current_safe
            & stationary
            & newer_than_stop_sample
        )
        transaction_valid = persistent_valid & binding_valid & authority_valid
        applied_pre = (
            transaction_valid
            & nonce_valid
            & capacity
            & requested
            & stationary_safe
        )
        proposed = state.replace(
            revision=state.revision + jnp.int32(1),
            emergency_stop_latched=jnp.asarray(False, dtype=jnp.bool_),
            deployment_suspended=jnp.asarray(False, dtype=jnp.bool_),
            has_telemetry=jnp.asarray(True, dtype=jnp.bool_),
            last_telemetry_id=telemetry.telemetry_id,
            last_sample_tick=telemetry.sample_tick,
            last_control_tick=now,
            has_reset_nonce=jnp.asarray(True, dtype=jnp.bool_),
            last_reset_nonce=handshake.nonce,
            reset_count=state.reset_count + jnp.int32(1),
            state_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        applied = applied_pre & self.state_valid(proposed)
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        reason = self._handshake_reason(
            persistent_valid=persistent_valid,
            binding_valid=binding_valid,
            authority_valid=authority_valid,
            nonce_valid=nonce_valid,
            capacity=capacity,
            requested=requested,
            stationary_safe=stationary_safe,
            applied=applied,
        )
        return EmbodiedHandshakeResult(
            state=next_state,
            transaction_valid=transaction_valid,
            applied=applied,
            replay_rejected=(
                persistent_valid
                & authority_valid
                & self._digest_nonzero(handshake.nonce)
                & state.has_reset_nonce
                & (~_words_greater(handshake.nonce, state.last_reset_nonce))
            ),
            stationary_safe=stationary_safe,
            unavailable_reason=reason,
        )

    @staticmethod
    def _cryptographic_state_digest(state: EmbodiedSafetyEnvelopeState) -> Array:
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(state):
            host = np.asarray(jax.device_get(jnp.asarray(leaf)))
            digest.update(host.dtype.str.encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
        return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)

    def checkpoint_payload(
        self,
        state: EmbodiedSafetyEnvelopeState,
    ) -> dict[str, object]:
        """Return an exact source/config-bound atomic checkpoint payload.

        ``state_digest`` is an unkeyed SHA-256 integrity value, not a MAC or
        authentication claim.  A caller must retain its exact revision and
        digest in a separate trusted store and supply them during restore.
        """

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid embodied envelope state")
        return {
            "schema_version": EMBODIED_SAFETY_ENVELOPE_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "source_digest": state.source_digest,
            "state": state,
            "state_digest": self._cryptographic_state_digest(state),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        expected_source_digest: Array,
        trusted_state_revision: int | Array,
        trusted_state_digest: Array,
    ) -> EmbodiedSafetyEnvelopeState:
        """Restore only the exact checkpoint pinned by an external trust anchor.

        The mandatory revision and SHA-256 digest must come from storage that
        is independent of ``payload``.  Copying either value from the payload
        defeats rollback protection and is explicitly outside this kernel's
        authority.
        """

        if type(payload) is not dict:
            raise ValueError("embodied envelope checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        expected_keys = {
            "schema_version",
            "config",
            "source_digest",
            "state",
            "state_digest",
        }
        if set(raw) != expected_keys:
            raise ValueError("embodied envelope checkpoint keys differ from schema v1")
        if raw["schema_version"] != EMBODIED_SAFETY_ENVELOPE_CHECKPOINT_SCHEMA:
            raise ValueError("embodied envelope checkpoint schema_version differs")
        if raw["config"] != self.to_config():
            raise ValueError("embodied envelope checkpoint config differs")
        restored = raw["state"]
        if type(restored) is not EmbodiedSafetyEnvelopeState:
            raise ValueError("embodied envelope checkpoint state type differs")
        state = restored
        source = _require_array(
            expected_source_digest,
            name="expected_source_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        trusted_revision = _int32_scalar(
            trusted_state_revision,
            name="trusted_state_revision",
        )
        trusted_digest = _require_array(
            trusted_state_digest,
            name="trusted_state_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        persisted_source = _require_array(
            raw["source_digest"],
            name="checkpoint.source_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        persisted_digest = _require_array(
            raw["state_digest"],
            name="checkpoint.state_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        valid = (
            jnp.array_equal(source, persisted_source)
            & jnp.array_equal(source, state.source_digest)
            & jnp.array_equal(state.config_digest, self._config_digest)
            & (state.revision == trusted_revision)
            & jnp.array_equal(persisted_digest, trusted_digest)
            & jnp.array_equal(
                persisted_digest,
                self._cryptographic_state_digest(state),
            )
            & self.state_valid(state)
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("embodied envelope checkpoint is invalid, stale, or tampered")
        return state

    def resource_budget(
        self,
        state: EmbodiedSafetyEnvelopeState,
    ) -> EmbodiedSafetyEnvelopeResourceBudget:
        """Return exact persistent bytes, finite caps, work, and zero authority."""

        self._check_state_contract(state)
        cfg = self._config
        return EmbodiedSafetyEnvelopeResourceBudget(
            persistent_state_nbytes=self._tree_nbytes(state),
            n_joints=cfg.n_joints,
            workspace_dimensions=_WORKSPACE_DIM,
            shadow_window=cfg.shadow_window,
            shadow_ring_cells=cfg.shadow_window
            * (
                3
                + 2
                + 2
                + 4 * _DIGEST_WORDS
                + 2
            ),
            max_decisions=cfg.max_decisions,
            max_committed_actions=cfg.max_committed_actions,
            max_shadow_records=cfg.max_shadow_records,
            max_handshakes_per_kind=cfg.max_handshakes_per_kind,
            joint_values_checked_per_evaluation=2 * cfg.n_joints * 3,
            workspace_values_checked_per_evaluation=2 * _WORKSPACE_DIM,
            wilson_square_roots_per_deployment_gate=1,
            random_generator_calls_per_operation=0,
            action_dispatches_per_operation=0,
            learning_state_mutations_per_operation=0,
            learned_cost_override_authority=False,
            deployment_authority=False,
            promotion_authority=False,
            physical_safety_claim=False,
            scientific_promotion_allowed=False,
            caller_authentication=False,
            checkpoint_schema=EMBODIED_SAFETY_ENVELOPE_CHECKPOINT_SCHEMA,
        )


__all__ = [
    "EMBODIED_SAFETY_ENVELOPE_ACTION_DISPATCH_AUTHORITY",
    "EMBODIED_SAFETY_ENVELOPE_CALLER_AUTHENTICATION",
    "EMBODIED_SAFETY_ENVELOPE_CHECKPOINT_SCHEMA",
    "EMBODIED_SAFETY_ENVELOPE_CONFIG_SCHEMA",
    "EMBODIED_SAFETY_ENVELOPE_DEPLOYMENT_AUTHORITY",
    "EMBODIED_SAFETY_ENVELOPE_LEARNED_COST_OVERRIDE_AUTHORITY",
    "EMBODIED_SAFETY_ENVELOPE_LEARNING_MUTATION_AUTHORITY",
    "EMBODIED_SAFETY_ENVELOPE_PHYSICAL_SAFETY_CLAIM",
    "EMBODIED_SAFETY_ENVELOPE_PROMOTION_AUTHORITY",
    "EMBODIED_SAFETY_ENVELOPE_SCIENTIFIC_PROMOTION_ALLOWED",
    "ENVELOPE_REASON_ACTION_CAPACITY",
    "ENVELOPE_REASON_ACTION_IDENTITY",
    "ENVELOPE_REASON_AVAILABLE",
    "ENVELOPE_REASON_BRIDGE_DISCONNECTED",
    "ENVELOPE_REASON_CAPACITY",
    "ENVELOPE_REASON_CONTROL_DEADLINE",
    "ENVELOPE_REASON_CURRENT_ENVELOPE",
    "ENVELOPE_REASON_DECISION_IDENTITY",
    "ENVELOPE_REASON_DEPLOYMENT_SUSPENDED",
    "ENVELOPE_REASON_EMERGENCY_STOP",
    "ENVELOPE_REASON_FALLBACK_UNAVAILABLE",
    "ENVELOPE_REASON_METADATA",
    "ENVELOPE_REASON_PERSISTENT_STATE",
    "ENVELOPE_REASON_TELEMETRY_IDENTITY",
    "ENVELOPE_REASON_TELEMETRY_STALE",
    "ENVELOPE_REASON_TIME_MONOTONICITY",
    "ENVELOPE_REASON_VERSION_BINDING",
    "HANDSHAKE_REASON_AUTHORITY",
    "HANDSHAKE_REASON_AVAILABLE",
    "HANDSHAKE_REASON_BINDING",
    "HANDSHAKE_REASON_CAPACITY",
    "HANDSHAKE_REASON_NONCE_REPLAY",
    "HANDSHAKE_REASON_NOT_REQUESTED",
    "HANDSHAKE_REASON_NOT_STATIONARY_SAFE",
    "HANDSHAKE_REASON_PERSISTENT_STATE",
    "AuthorityBoundEnvelopeHandshake",
    "EmbodiedCommand",
    "EmbodiedDeploymentGate",
    "EmbodiedEnvelopeDecision",
    "EmbodiedHandshakeResult",
    "EmbodiedSafetyEnvelope",
    "EmbodiedSafetyEnvelopeConfig",
    "EmbodiedSafetyEnvelopeResourceBudget",
    "EmbodiedSafetyEnvelopeState",
    "EmbodiedShadowEvaluation",
    "EmbodiedShadowRecordResult",
    "EmbodiedTelemetry",
]
