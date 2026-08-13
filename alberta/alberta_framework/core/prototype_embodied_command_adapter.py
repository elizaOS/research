# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Source-bound discrete primitives for the embodied safety envelope.

This module closes one narrow mechanical edge between
:class:`PrototypeConsolidatedSemanticMemoryAgent` and
:class:`EmbodiedSafetyEnvelope`.  A fixed finite bank maps each discrete
Prototype action to one exact float32 command payload.  ``prepare`` snapshots
the current semantic dispatch owner and one complete envelope evaluation
request into a non-wrapping receipt.  ``settle`` recomputes the envelope result
from that snapshot, bit-compares the complete caller-supplied result, maps the
returned command back to exactly one admitted primitive, and only then adopts
the permitted atomic transition. Accepted/fallback commands adopt both child
candidates and close the semantic decision. No-action and stop-only outcomes
adopt the exact envelope state, close only that attempt receipt, and preserve
the semantic owner so a fresh envelope attempt can be prepared.

The command bank is an identity adapter, not a kinematics model or proof that
its workspace projection is geometrically correct.  The envelope remains the
only component here that evaluates its configured hard bounds.  This adapter
does no physical dispatch, caller authentication, learning, evidence writes,
RNG use, deployment decision, or scientific promotion.  Its checksums and
checkpoint digest detect accidental corruption; they are unkeyed and provide
no protection against a caller able to rewrite state and integrity fields.
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
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.embodied_safety_envelope import (
    EmbodiedCommand,
    EmbodiedEnvelopeDecision,
    EmbodiedSafetyEnvelope,
    EmbodiedSafetyEnvelopeConfig,
    EmbodiedSafetyEnvelopeResourceBudget,
    EmbodiedSafetyEnvelopeState,
    EmbodiedTelemetry,
)
from alberta_framework.core.prototype_consolidated_memory import (
    PrototypeConsolidatedMemoryDispatchSettlementInput,
)
from alberta_framework.core.prototype_consolidated_semantic_memory import (
    PrototypeConsolidatedSemanticMemoryAgent,
    PrototypeConsolidatedSemanticMemoryConfig,
    PrototypeConsolidatedSemanticMemoryDispatchSettlementResult,
    PrototypeConsolidatedSemanticMemoryResourceBudget,
    PrototypeConsolidatedSemanticMemoryState,
)

PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CONFIG_SCHEMA = (
    "alberta.prototype-embodied-command-adapter.config.v1"
)
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CHECKPOINT_SCHEMA = (
    "alberta.prototype-embodied-command-adapter.state.v1"
)
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_ASSESSMENT = "not_assessed"
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CHECKPOINT_HOST_ONLY = True
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_EAGER_SUPPORTED = True
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_JIT_PREPARE_SUPPORTED = True
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_JIT_SETTLE_SUPPORTED = True
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_PHYSICAL_DISPATCH_AUTHORITY = False
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CALLER_AUTHENTICATION = False
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_LEARNING_AUTHORITY = False
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_EVIDENCE_AUTHORITY = False
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_SAFETY_AUTHORITY = False
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_PROMOTION_AUTHORITY = False
PROTOTYPE_EMBODIED_COMMAND_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED = False

_DECISION_WORDS = 4
_IDENTITY_WORDS = 2
_DIGEST_WORDS = 8
_DIGEST_BYTES = 32
_WORKSPACE_DIM = 3
_UINT32_MAX = 2**32 - 1


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose array shape and dtype")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    expected = jnp.dtype(dtype)
    if jnp.dtype(array.dtype) != expected:
        raise TypeError(f"{name} must have dtype {expected}; got {array.dtype}")
    return array


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


def _float_scalar(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact Python float")
    if not math.isfinite(float(np.float32(value))):
        raise ValueError(f"{name} must remain finite in float32")
    return value


def _canonical_digest(value: object) -> Array:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raw = hashlib.sha256(encoded.encode("utf-8")).digest()
    return jnp.asarray(
        tuple(
            int.from_bytes(raw[offset : offset + 4], "little")
            for offset in range(0, _DIGEST_BYTES, 4)
        ),
        dtype=jnp.uint32,
    )


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.size) * int(array.dtype.itemsize)
    return total


def _float_bits_equal(left: Array, right: Array) -> Array:
    return jnp.array_equal(
        jax.lax.bitcast_convert_type(left, jnp.uint32),
        jax.lax.bitcast_convert_type(right, jnp.uint32),
    )


def _tree_array_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return jnp.asarray(False, dtype=jnp.bool_)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            valid = valid & jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.float32:
            valid = valid & _float_bits_equal(left_array, right_array)
        else:
            valid = valid & jnp.array_equal(left_array, right_array)
    return valid


def _checksum_arrays(arrays: tuple[Array, ...]) -> Array:
    acc0 = jnp.uint32(0x9E3779B9)
    acc1 = jnp.uint32(0x85EBCA6B)
    offset = 1
    for value in arrays:
        array = jnp.asarray(value)
        if array.dtype in (jnp.float32, jnp.int32):
            words = jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
        elif array.dtype == jnp.uint32:
            words = array.reshape((-1,))
        else:
            words = array.astype(jnp.uint32).reshape((-1,))
        if words.shape[0] == 0:
            continue
        indices = jnp.arange(offset, offset + words.shape[0], dtype=jnp.uint32)
        acc0 = acc0 + jnp.sum(words * (indices * jnp.uint32(0x27D4EB2D) + 1))
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(
            words ^ (indices * jnp.uint32(0x165667B1))
        )
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), dtype=jnp.uint32)


def _increment_words(words: Array) -> tuple[Array, Array]:
    low = words[1] + jnp.uint32(1)
    carry = (low == 0).astype(jnp.uint32)
    high = words[0] + carry
    available = ~jnp.all(words == jnp.uint32(_UINT32_MAX))
    candidate = jnp.stack((high, low), dtype=jnp.uint32)
    return jnp.where(available, candidate, words), available


def _words_nonzero(words: Array) -> Array:
    return jnp.any(words != jnp.uint32(0))


def _words_greater(left: Array, right: Array) -> Array:
    result = jnp.asarray(False, dtype=jnp.bool_)
    equal_prefix = jnp.asarray(True, dtype=jnp.bool_)
    for index in range(left.shape[0]):
        result = result | (equal_prefix & (left[index] > right[index]))
        equal_prefix = equal_prefix & (left[index] == right[index])
    return result


def _tree_sha256(value: object) -> Array:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        host = np.asarray(jax.device_get(array))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


@dataclasses.dataclass(frozen=True, slots=True)
class DiscreteEmbodiedPrimitiveCommand:
    """One exact float32 payload template; not a geometry certificate."""

    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    joint_torque: tuple[float, ...]
    workspace_position: tuple[float, float, float]
    collision_clearance: float

    def validate(self, *, n_joints: int, name: str) -> None:
        _float_tuple(self.joint_position, name=f"{name}.joint_position", length=n_joints)
        _float_tuple(self.joint_velocity, name=f"{name}.joint_velocity", length=n_joints)
        _float_tuple(self.joint_torque, name=f"{name}.joint_torque", length=n_joints)
        _float_tuple(
            self.workspace_position,
            name=f"{name}.workspace_position",
            length=_WORKSPACE_DIM,
        )
        _float_scalar(self.collision_clearance, name=f"{name}.collision_clearance")

    def to_config(self) -> dict[str, object]:
        return {
            "joint_position": list(self.joint_position),
            "joint_velocity": list(self.joint_velocity),
            "joint_torque": list(self.joint_torque),
            "workspace_position": list(self.workspace_position),
            "collision_clearance": self.collision_clearance,
            "geometry_certificate": False,
        }

    @classmethod
    def from_config(
        cls,
        value: object,
        *,
        n_joints: int,
    ) -> DiscreteEmbodiedPrimitiveCommand:
        if type(value) is not dict:
            raise ValueError("primitive command config must be an exact dict")
        raw = cast(dict[object, object], value)
        expected = {
            "joint_position",
            "joint_velocity",
            "joint_torque",
            "workspace_position",
            "collision_clearance",
            "geometry_certificate",
        }
        if set(raw) != expected or raw["geometry_certificate"] is not False:
            raise ValueError("primitive command config fields differ")

        def exact_float_tuple(name: str, length: int) -> tuple[float, ...]:
            field = raw[name]
            if type(field) is not list or len(field) != length:
                raise ValueError(f"primitive {name} must be a list of length {length}")
            if any(type(item) is not float for item in field):
                raise ValueError(f"primitive {name} entries must be exact floats")
            return tuple(cast(list[float], field))

        clearance = raw["collision_clearance"]
        if type(clearance) is not float:
            raise ValueError("primitive collision_clearance must be an exact float")
        result = cls(
            joint_position=exact_float_tuple("joint_position", n_joints),
            joint_velocity=exact_float_tuple("joint_velocity", n_joints),
            joint_torque=exact_float_tuple("joint_torque", n_joints),
            workspace_position=cast(
                tuple[float, float, float],
                exact_float_tuple("workspace_position", _WORKSPACE_DIM),
            ),
            collision_clearance=clearance,
        )
        result.validate(n_joints=n_joints, name="primitive")
        if result.to_config() != value:
            raise ValueError("primitive command config is noncanonical")
        return result

    def float32_identity(self) -> bytes:
        pieces = (
            self.joint_position,
            self.joint_velocity,
            self.joint_torque,
            self.workspace_position,
            (self.collision_clearance,),
        )
        return b"".join(np.asarray(piece, dtype=np.float32).tobytes() for piece in pieces)


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeEmbodiedCommandAdapterConfig:
    """Static child configurations and one unique primitive command bank."""

    semantic: PrototypeConsolidatedSemanticMemoryConfig
    envelope: EmbodiedSafetyEnvelopeConfig
    command_bank: tuple[DiscreteEmbodiedPrimitiveCommand, ...]

    SCHEMA_VERSION: ClassVar[str] = PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if type(self.semantic) is not PrototypeConsolidatedSemanticMemoryConfig:
            raise TypeError("semantic must be an exact semantic-memory config")
        if type(self.envelope) is not EmbodiedSafetyEnvelopeConfig:
            raise TypeError("envelope must be an exact embodied-envelope config")
        if type(self.command_bank) is not tuple:
            raise TypeError("command_bank must be an exact tuple")
        n_actions = self.semantic.composition.controller.policy.n_actions
        if len(self.command_bank) != n_actions:
            raise ValueError("command_bank length must equal the semantic policy n_actions")
        identities: set[bytes] = set()
        for index, command in enumerate(self.command_bank):
            if type(command) is not DiscreteEmbodiedPrimitiveCommand:
                raise TypeError("every command_bank item must be an exact primitive command")
            command.validate(n_joints=self.envelope.n_joints, name=f"command_bank[{index}]")
            identity = command.float32_identity()
            if identity in identities:
                raise ValueError("command_bank payloads must have unique bit-exact identities")
            identities.add(identity)

    @property
    def n_actions(self) -> int:
        return len(self.command_bank)

    @property
    def n_joints(self) -> int:
        return self.envelope.n_joints

    def to_config(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA_VERSION,
            "assessment": PROTOTYPE_EMBODIED_COMMAND_ADAPTER_ASSESSMENT,
            "semantic": self.semantic.to_config(),
            "envelope": self.envelope.to_config(),
            "command_bank": [command.to_config() for command in self.command_bank],
            "command_identity": "all_float32_payload_bits",
            "command_geometry_certificate": False,
            "settlement_binding": "full_envelope_result_recomputed_and_bit_compared",
            "no_action_semantics": (
                "adopt_envelope_close_attempt_preserve_semantic_owner"
            ),
            "stop_only_semantics": "persist_exact_fresh_stop_latch",
            "receipt_clock": "uint64_words_nonwrapping",
            "checkpoint_host_only": True,
            "eager_prepare_and_settle": True,
            "jit_prepare_and_settle": True,
            "physical_dispatch_authority": False,
            "caller_authentication": False,
            "learning_authority": False,
            "evidence_authority": False,
            "safety_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "rng_draws": 0,
        }

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, object],
    ) -> PrototypeEmbodiedCommandAdapterConfig:
        if type(value) is not dict:
            raise ValueError("command adapter config must be an exact dict")
        raw = dict(value)
        fixed: dict[str, object] = {
            "schema": cls.SCHEMA_VERSION,
            "assessment": PROTOTYPE_EMBODIED_COMMAND_ADAPTER_ASSESSMENT,
            "command_identity": "all_float32_payload_bits",
            "command_geometry_certificate": False,
            "settlement_binding": "full_envelope_result_recomputed_and_bit_compared",
            "no_action_semantics": (
                "adopt_envelope_close_attempt_preserve_semantic_owner"
            ),
            "stop_only_semantics": "persist_exact_fresh_stop_latch",
            "receipt_clock": "uint64_words_nonwrapping",
            "checkpoint_host_only": True,
            "eager_prepare_and_settle": True,
            "jit_prepare_and_settle": True,
            "physical_dispatch_authority": False,
            "caller_authentication": False,
            "learning_authority": False,
            "evidence_authority": False,
            "safety_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
            "rng_draws": 0,
        }
        expected = {"semantic", "envelope", "command_bank", *fixed}
        if set(raw) != expected:
            raise ValueError("command adapter config fields differ from schema v1")
        for name, expected_value in fixed.items():
            if type(raw[name]) is not type(expected_value) or raw.pop(name) != expected_value:
                raise ValueError(f"command adapter fixed field {name} differs")
        semantic = PrototypeConsolidatedSemanticMemoryConfig.from_config(raw["semantic"])
        envelope = EmbodiedSafetyEnvelopeConfig.from_config(
            cast(Mapping[str, object], raw["envelope"])
        )
        bank_raw = raw["command_bank"]
        if type(bank_raw) is not list:
            raise ValueError("command_bank must be an exact list")
        result = cls(
            semantic=semantic,
            envelope=envelope,
            command_bank=tuple(
                DiscreteEmbodiedPrimitiveCommand.from_config(
                    item,
                    n_joints=envelope.n_joints,
                )
                for item in bank_raw
            ),
        )
        if result.to_config() != value:
            raise ValueError("command adapter config is noncanonical")
        return result


@chex.dataclass(frozen=True)
class PrototypeEmbodiedCommandPreparationInput:
    """Complete fixed-shape input for exactly one envelope evaluation."""

    telemetry: EmbodiedTelemetry
    envelope_decision_id: UInt[Array, " 2"]
    envelope_action_id: UInt[Array, " 2"]
    control_tick: UInt[Array, " 2"]
    control_deadline_tick: UInt[Array, " 2"]
    model_version: UInt[Array, " 8"]
    optimizer_version: UInt[Array, " 8"]
    lifecycle_version: UInt[Array, " 8"]
    untrusted_reward: Float[Array, ""]
    partner_metadata_digest: UInt[Array, " 8"]
    learned_cost_estimate: Float[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedCommandReceiptState:
    """One full source-bound pending evaluation; checksum is not authentication."""

    available: Bool[Array, ""]
    receipt_words: UInt[Array, " 2"]
    prototype_decision_id: UInt[Array, " 4"]
    selected_action: Int[Array, ""]
    hard_safety_action_mask: Bool[Array, " n_actions"]
    envelope_source_revision: Int[Array, ""]
    envelope_source_checksum: UInt[Array, " 2"]
    envelope_source_digest: UInt[Array, " 8"]
    envelope_config_digest: UInt[Array, " 8"]
    adapter_config_digest: UInt[Array, " 8"]
    telemetry: EmbodiedTelemetry
    envelope_decision_id: UInt[Array, " 2"]
    envelope_action_id: UInt[Array, " 2"]
    control_tick: UInt[Array, " 2"]
    control_deadline_tick: UInt[Array, " 2"]
    model_version: UInt[Array, " 8"]
    optimizer_version: UInt[Array, " 8"]
    lifecycle_version: UInt[Array, " 8"]
    untrusted_reward: Float[Array, ""]
    partner_metadata_digest: UInt[Array, " 8"]
    learned_cost_estimate: Float[Array, ""]
    checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedCommandAdapterState:
    """The two child states and one exact receipt/consumption ledger."""

    semantic: PrototypeConsolidatedSemanticMemoryState
    envelope: EmbodiedSafetyEnvelopeState
    adapter_config_digest: UInt[Array, " 8"]
    receipt_clock_words: UInt[Array, " 2"]
    has_settled_prototype_decision: Bool[Array, ""]
    last_settled_prototype_decision_id: UInt[Array, " 4"]
    pending: PrototypeEmbodiedCommandReceiptState
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedCommandPreparationDiagnostics:
    source_state_valid: Bool[Array, ""]
    receipt_slot_available: Bool[Array, ""]
    receipt_clock_available: Bool[Array, ""]
    dispatch_owner_available: Bool[Array, ""]
    owner_decision_matches_current: Bool[Array, ""]
    owner_action_matches_current: Bool[Array, ""]
    owner_not_already_settled: Bool[Array, ""]
    selected_action_contract_valid: Bool[Array, ""]
    selected_action_admitted_by_bound_mask: Bool[Array, ""]
    envelope_decision_identity_fresh: Bool[Array, ""]
    envelope_action_identity_fresh: Bool[Array, ""]
    telemetry_identity_fresh: Bool[Array, ""]
    versions_nonzero: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    prepared: Bool[Array, ""]
    command_geometry_certificate: Bool[Array, ""]
    physical_dispatch_authority: Bool[Array, ""]
    caller_authentication: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedCommandPreparationResult:
    state: PrototypeEmbodiedCommandAdapterState
    command: EmbodiedCommand
    receipt_words: UInt[Array, " 2"]
    prototype_decision_id: UInt[Array, " 4"]
    selected_action: Int[Array, ""]
    hard_safety_action_mask: Bool[Array, " n_actions"]
    diagnostics: PrototypeEmbodiedCommandPreparationDiagnostics


@chex.dataclass(frozen=True)
class PrototypeEmbodiedCommandMappingResult:
    """Bit-exact command-bank identity only; carries no safety authority."""

    command_contract_valid: Bool[Array, ""]
    match_count: Int[Array, ""]
    maps_exactly_one_primitive: Bool[Array, ""]
    action: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedCommandSettlementDiagnostics:
    source_state_valid: Bool[Array, ""]
    pending_receipt_available: Bool[Array, ""]
    envelope_result_exact: Bool[Array, ""]
    envelope_transaction_applied: Bool[Array, ""]
    envelope_action_available: Bool[Array, ""]
    envelope_proposed_accepted: Bool[Array, ""]
    envelope_fallback_used: Bool[Array, ""]
    command_match_count: Int[Array, ""]
    command_maps_exactly_one_primitive: Bool[Array, ""]
    mapped_action: Int[Array, ""]
    mapped_action_matches_selected_proposal: Bool[Array, ""]
    mapped_action_admitted_by_bound_mask: Bool[Array, ""]
    envelope_outcome_structurally_valid: Bool[Array, ""]
    semantic_settlement_committed: Bool[Array, ""]
    semantic_settlement_action_matches: Bool[Array, ""]
    semantic_owner_retry_preserved: Bool[Array, ""]
    envelope_only_state_committed: Bool[Array, ""]
    stop_only_latch_committed: Bool[Array, ""]
    attempt_receipt_closed: Bool[Array, ""]
    receipt_consumed: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    transaction_committed: Bool[Array, ""]
    learning_applied: Bool[Array, ""]
    evidence_written: Bool[Array, ""]
    random_generator_consumed: Bool[Array, ""]
    physical_dispatch_authority: Bool[Array, ""]
    caller_authentication: Bool[Array, ""]
    safety_authority: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeEmbodiedCommandSettlementResult:
    state: PrototypeEmbodiedCommandAdapterState
    action: Int[Array, ""]
    receipt_words: UInt[Array, " 2"]
    envelope: EmbodiedEnvelopeDecision
    semantic: PrototypeConsolidatedSemanticMemoryDispatchSettlementResult
    diagnostics: PrototypeEmbodiedCommandSettlementDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeEmbodiedCommandAdapterResourceBudget:
    persistent_state_nbytes: int
    pending_receipt_nbytes: int
    static_command_bank_nbytes: int
    n_actions: int
    n_joints: int
    primitive_payload_float32_cells: int
    maximum_pending_receipts: int
    nonwrapping_receipt_clock_words: int
    envelope_recomputations_per_settlement: int
    envelope_state_commits_per_exact_no_action: int
    stop_latch_preservations_per_exact_stop_only_result: int
    maximum_bit_exact_command_comparisons_per_settlement: int
    semantic_settlement_delegations_per_settlement: int
    physical_dispatches_per_operation: int
    learning_state_mutations_per_operation: int
    evidence_writes_per_operation: int
    random_generator_calls_per_operation: int
    persistent_growth_per_operation_bytes: int
    checkpoint_host_only: bool
    eager_prepare_and_settle: bool
    jit_prepare_and_settle: bool
    command_geometry_certificate: bool
    physical_dispatch_authority: bool
    caller_authentication: bool
    learning_authority: bool
    evidence_authority: bool
    safety_authority: bool
    promotion_authority: bool
    scientific_promotion_allowed: bool
    semantic: PrototypeConsolidatedSemanticMemoryResourceBudget
    envelope: EmbodiedSafetyEnvelopeResourceBudget


class PrototypeEmbodiedCommandAdapter:
    """Two-phase identity adapter and exact atomic settlement bridge."""

    def __init__(self, config: PrototypeEmbodiedCommandAdapterConfig) -> None:
        if type(config) is not PrototypeEmbodiedCommandAdapterConfig:
            raise TypeError("config must be an exact PrototypeEmbodiedCommandAdapterConfig")
        self._config = config
        self._semantic = PrototypeConsolidatedSemanticMemoryAgent(config.semantic)
        self._envelope = EmbodiedSafetyEnvelope(config.envelope)
        self._config_digest = _canonical_digest(config.to_config())
        self._bank_joint_position = jnp.asarray(
            tuple(command.joint_position for command in config.command_bank),
            dtype=jnp.float32,
        )
        self._bank_joint_velocity = jnp.asarray(
            tuple(command.joint_velocity for command in config.command_bank),
            dtype=jnp.float32,
        )
        self._bank_joint_torque = jnp.asarray(
            tuple(command.joint_torque for command in config.command_bank),
            dtype=jnp.float32,
        )
        self._bank_workspace_position = jnp.asarray(
            tuple(command.workspace_position for command in config.command_bank),
            dtype=jnp.float32,
        )
        self._bank_collision_clearance = jnp.asarray(
            tuple(command.collision_clearance for command in config.command_bank),
            dtype=jnp.float32,
        )

    @property
    def config(self) -> PrototypeEmbodiedCommandAdapterConfig:
        return self._config

    @property
    def semantic(self) -> PrototypeConsolidatedSemanticMemoryAgent:
        return self._semantic

    @property
    def envelope(self) -> EmbodiedSafetyEnvelope:
        return self._envelope

    @property
    def config_digest(self) -> Array:
        return self._config_digest

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> PrototypeEmbodiedCommandAdapter:
        return cls(PrototypeEmbodiedCommandAdapterConfig.from_config(payload))

    def _zero_command(self) -> EmbodiedCommand:
        return EmbodiedCommand(
            joint_position=jnp.zeros((self._config.n_joints,), dtype=jnp.float32),
            joint_velocity=jnp.zeros((self._config.n_joints,), dtype=jnp.float32),
            joint_torque=jnp.zeros((self._config.n_joints,), dtype=jnp.float32),
            workspace_position=jnp.zeros((_WORKSPACE_DIM,), dtype=jnp.float32),
            collision_clearance=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _command(self, action: Array) -> EmbodiedCommand:
        safe_action = jnp.clip(action, 0, self._config.n_actions - 1)
        return EmbodiedCommand(
            joint_position=self._bank_joint_position[safe_action],
            joint_velocity=self._bank_joint_velocity[safe_action],
            joint_torque=self._bank_joint_torque[safe_action],
            workspace_position=self._bank_workspace_position[safe_action],
            collision_clearance=self._bank_collision_clearance[safe_action],
        )

    def _blank_telemetry(self) -> EmbodiedTelemetry:
        return EmbodiedTelemetry(
            joint_position=jnp.zeros((self._config.n_joints,), dtype=jnp.float32),
            joint_velocity=jnp.zeros((self._config.n_joints,), dtype=jnp.float32),
            joint_torque=jnp.zeros((self._config.n_joints,), dtype=jnp.float32),
            workspace_position=jnp.zeros((_WORKSPACE_DIM,), dtype=jnp.float32),
            collision_clearance=jnp.asarray(0.0, dtype=jnp.float32),
            bridge_connected=jnp.asarray(False, dtype=jnp.bool_),
            emergency_stop=jnp.asarray(False, dtype=jnp.bool_),
            telemetry_id=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            sample_tick=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )

    def _receipt_payload(
        self,
        receipt: PrototypeEmbodiedCommandReceiptState,
    ) -> tuple[Array, ...]:
        leaves: list[Array] = []
        for field in dataclasses.fields(PrototypeEmbodiedCommandReceiptState):
            if field.name == "checksum":
                continue
            leaves.extend(
                jnp.asarray(leaf)
                for leaf in jax.tree_util.tree_leaves(getattr(receipt, field.name))
            )
        return tuple(leaves)

    def _receipt_record(
        self,
        *,
        available: Array,
        receipt_words: Array,
        prototype_decision_id: Array,
        selected_action: Array,
        hard_safety_action_mask: Array,
        envelope_source_revision: Array,
        envelope_source_checksum: Array,
        envelope_source_digest: Array,
        envelope_config_digest: Array,
        adapter_config_digest: Array,
        preparation: PrototypeEmbodiedCommandPreparationInput,
    ) -> PrototypeEmbodiedCommandReceiptState:
        canonical = PrototypeEmbodiedCommandReceiptState(
            available=available,
            receipt_words=receipt_words,
            prototype_decision_id=prototype_decision_id,
            selected_action=selected_action,
            hard_safety_action_mask=hard_safety_action_mask,
            envelope_source_revision=envelope_source_revision,
            envelope_source_checksum=envelope_source_checksum,
            envelope_source_digest=envelope_source_digest,
            envelope_config_digest=envelope_config_digest,
            adapter_config_digest=adapter_config_digest,
            telemetry=preparation.telemetry,
            envelope_decision_id=preparation.envelope_decision_id,
            envelope_action_id=preparation.envelope_action_id,
            control_tick=preparation.control_tick,
            control_deadline_tick=preparation.control_deadline_tick,
            model_version=preparation.model_version,
            optimizer_version=preparation.optimizer_version,
            lifecycle_version=preparation.lifecycle_version,
            untrusted_reward=preparation.untrusted_reward,
            partner_metadata_digest=preparation.partner_metadata_digest,
            learned_cost_estimate=preparation.learned_cost_estimate,
            checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        return canonical.replace(checksum=_checksum_arrays(self._receipt_payload(canonical)))

    def _blank_receipt(self) -> PrototypeEmbodiedCommandReceiptState:
        zeros2 = jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32)
        zeros8 = jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
        preparation = PrototypeEmbodiedCommandPreparationInput(
            telemetry=self._blank_telemetry(),
            envelope_decision_id=zeros2,
            envelope_action_id=zeros2,
            control_tick=zeros2,
            control_deadline_tick=zeros2,
            model_version=zeros8,
            optimizer_version=zeros8,
            lifecycle_version=zeros8,
            untrusted_reward=jnp.asarray(0.0, dtype=jnp.float32),
            partner_metadata_digest=zeros8,
            learned_cost_estimate=jnp.asarray(0.0, dtype=jnp.float32),
        )
        return self._receipt_record(
            available=jnp.asarray(False, dtype=jnp.bool_),
            receipt_words=zeros2,
            prototype_decision_id=jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            selected_action=jnp.asarray(-1, dtype=jnp.int32),
            hard_safety_action_mask=jnp.zeros(
                (self._config.n_actions,), dtype=jnp.bool_
            ),
            envelope_source_revision=jnp.asarray(-1, dtype=jnp.int32),
            envelope_source_checksum=zeros2,
            envelope_source_digest=zeros8,
            envelope_config_digest=zeros8,
            adapter_config_digest=zeros8,
            preparation=preparation,
        )

    def _binding_payload(
        self,
        state: PrototypeEmbodiedCommandAdapterState,
    ) -> tuple[Array, ...]:
        return (
            state.adapter_config_digest,
            state.receipt_clock_words,
            state.has_settled_prototype_decision,
            state.last_settled_prototype_decision_id,
            state.pending.checksum,
        )

    def _with_binding_checksum(
        self,
        state: PrototypeEmbodiedCommandAdapterState,
    ) -> PrototypeEmbodiedCommandAdapterState:
        return state.replace(binding_checksum=_checksum_arrays(self._binding_payload(state)))

    def _check_telemetry_contract(self, telemetry: EmbodiedTelemetry) -> None:
        if type(telemetry) is not EmbodiedTelemetry:
            raise TypeError("preparation.telemetry must be an exact EmbodiedTelemetry")
        n = self._config.n_joints
        for name in ("joint_position", "joint_velocity", "joint_torque"):
            _require_array(
                getattr(telemetry, name),
                name=f"preparation.telemetry.{name}",
                shape=(n,),
                dtype=jnp.float32,
            )
        _require_array(
            telemetry.workspace_position,
            name="preparation.telemetry.workspace_position",
            shape=(_WORKSPACE_DIM,),
            dtype=jnp.float32,
        )
        _require_array(
            telemetry.collision_clearance,
            name="preparation.telemetry.collision_clearance",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            telemetry.bridge_connected,
            name="preparation.telemetry.bridge_connected",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            telemetry.emergency_stop,
            name="preparation.telemetry.emergency_stop",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            telemetry.telemetry_id,
            name="preparation.telemetry.telemetry_id",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            telemetry.sample_tick,
            name="preparation.telemetry.sample_tick",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )

    def _check_preparation_contract(
        self,
        preparation: PrototypeEmbodiedCommandPreparationInput,
    ) -> None:
        if type(preparation) is not PrototypeEmbodiedCommandPreparationInput:
            raise TypeError(
                "preparation must be an exact PrototypeEmbodiedCommandPreparationInput"
            )
        self._check_telemetry_contract(preparation.telemetry)
        contracts = (
            (preparation.envelope_decision_id, (_IDENTITY_WORDS,), jnp.uint32),
            (preparation.envelope_action_id, (_IDENTITY_WORDS,), jnp.uint32),
            (preparation.control_tick, (_IDENTITY_WORDS,), jnp.uint32),
            (preparation.control_deadline_tick, (_IDENTITY_WORDS,), jnp.uint32),
            (preparation.model_version, (_DIGEST_WORDS,), jnp.uint32),
            (preparation.optimizer_version, (_DIGEST_WORDS,), jnp.uint32),
            (preparation.lifecycle_version, (_DIGEST_WORDS,), jnp.uint32),
            (preparation.untrusted_reward, (), jnp.float32),
            (preparation.partner_metadata_digest, (_DIGEST_WORDS,), jnp.uint32),
            (preparation.learned_cost_estimate, (), jnp.float32),
        )
        names = (
            "envelope_decision_id",
            "envelope_action_id",
            "control_tick",
            "control_deadline_tick",
            "model_version",
            "optimizer_version",
            "lifecycle_version",
            "untrusted_reward",
            "partner_metadata_digest",
            "learned_cost_estimate",
        )
        for name, (value, shape, dtype) in zip(names, contracts, strict=True):
            _require_array(value, name=f"preparation.{name}", shape=shape, dtype=dtype)

    def _check_receipt_contract(
        self,
        receipt: PrototypeEmbodiedCommandReceiptState,
    ) -> None:
        if type(receipt) is not PrototypeEmbodiedCommandReceiptState:
            raise TypeError("pending must be an exact PrototypeEmbodiedCommandReceiptState")
        self._check_telemetry_contract(receipt.telemetry)
        contracts = {
            "available": ((), jnp.bool_),
            "receipt_words": ((_IDENTITY_WORDS,), jnp.uint32),
            "prototype_decision_id": ((_DECISION_WORDS,), jnp.uint32),
            "selected_action": ((), jnp.int32),
            "hard_safety_action_mask": ((self._config.n_actions,), jnp.bool_),
            "envelope_source_revision": ((), jnp.int32),
            "envelope_source_checksum": ((_IDENTITY_WORDS,), jnp.uint32),
            "envelope_source_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "envelope_config_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "adapter_config_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "envelope_decision_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "envelope_action_id": ((_IDENTITY_WORDS,), jnp.uint32),
            "control_tick": ((_IDENTITY_WORDS,), jnp.uint32),
            "control_deadline_tick": ((_IDENTITY_WORDS,), jnp.uint32),
            "model_version": ((_DIGEST_WORDS,), jnp.uint32),
            "optimizer_version": ((_DIGEST_WORDS,), jnp.uint32),
            "lifecycle_version": ((_DIGEST_WORDS,), jnp.uint32),
            "untrusted_reward": ((), jnp.float32),
            "partner_metadata_digest": ((_DIGEST_WORDS,), jnp.uint32),
            "learned_cost_estimate": ((), jnp.float32),
            "checksum": ((_IDENTITY_WORDS,), jnp.uint32),
        }
        for name, (shape, dtype) in contracts.items():
            _require_array(
                getattr(receipt, name),
                name=f"pending.{name}",
                shape=shape,
                dtype=dtype,
            )

    def _check_state_contract(self, state: PrototypeEmbodiedCommandAdapterState) -> None:
        if type(state) is not PrototypeEmbodiedCommandAdapterState:
            raise TypeError("state must be an exact PrototypeEmbodiedCommandAdapterState")
        if type(state.semantic) is not PrototypeConsolidatedSemanticMemoryState:
            raise TypeError("state.semantic has the wrong exact type")
        if type(state.envelope) is not EmbodiedSafetyEnvelopeState:
            raise TypeError("state.envelope has the wrong exact type")
        _require_array(
            state.adapter_config_digest,
            name="state.adapter_config_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.receipt_clock_words,
            name="state.receipt_clock_words",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        _require_array(
            state.has_settled_prototype_decision,
            name="state.has_settled_prototype_decision",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            state.last_settled_prototype_decision_id,
            name="state.last_settled_prototype_decision_id",
            shape=(_DECISION_WORDS,),
            dtype=jnp.uint32,
        )
        self._check_receipt_contract(state.pending)
        _require_array(
            state.binding_checksum,
            name="state.binding_checksum",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )

    def _receipt_valid(
        self,
        state: PrototypeEmbodiedCommandAdapterState,
    ) -> Array:
        pending = state.pending
        owner = state.semantic.composition.dispatch_owner
        prototype = state.semantic.composition.prototype
        selected_index_valid = (
            (pending.selected_action >= 0)
            & (pending.selected_action < self._config.n_actions)
        )
        safe_selected = jnp.clip(pending.selected_action, 0, self._config.n_actions - 1)
        already_settled = state.has_settled_prototype_decision & (
            ~_words_greater(
                pending.prototype_decision_id,
                state.last_settled_prototype_decision_id,
            )
        )
        fresh_decision = _words_nonzero(pending.envelope_decision_id) & (
            (~state.envelope.has_decision)
            | _words_greater(pending.envelope_decision_id, state.envelope.last_decision_id)
        )
        fresh_action = _words_nonzero(pending.envelope_action_id) & (
            (~state.envelope.has_action)
            | _words_greater(pending.envelope_action_id, state.envelope.last_action_id)
        )
        fresh_telemetry = _words_nonzero(pending.telemetry.telemetry_id) & (
            (~state.envelope.has_telemetry)
            | _words_greater(pending.telemetry.telemetry_id, state.envelope.last_telemetry_id)
        )
        versions_nonzero = (
            _words_nonzero(pending.model_version)
            & _words_nonzero(pending.optimizer_version)
            & _words_nonzero(pending.lifecycle_version)
        )
        return (
            pending.available
            & _words_nonzero(pending.receipt_words)
            & jnp.array_equal(pending.receipt_words, state.receipt_clock_words)
            & jnp.array_equal(pending.checksum, _checksum_arrays(self._receipt_payload(pending)))
            & jnp.array_equal(pending.adapter_config_digest, self._config_digest)
            & (pending.envelope_source_revision == state.envelope.revision)
            & jnp.array_equal(pending.envelope_source_checksum, state.envelope.state_checksum)
            & jnp.array_equal(pending.envelope_source_digest, state.envelope.source_digest)
            & jnp.array_equal(pending.envelope_config_digest, self._envelope.config_digest)
            & owner.available
            & jnp.array_equal(owner.prototype_decision_id, pending.prototype_decision_id)
            & (owner.selected_action == pending.selected_action)
            & jnp.array_equal(
                owner.hard_safety_action_mask,
                pending.hard_safety_action_mask,
            )
            & prototype.started
            & jnp.array_equal(
                prototype.current_decision_id,
                pending.prototype_decision_id,
            )
            & (prototype.current_action == pending.selected_action)
            & (~already_settled)
            & selected_index_valid
            & pending.hard_safety_action_mask[safe_selected]
            & fresh_decision
            & fresh_action
            & fresh_telemetry
            & versions_nonzero
        )

    def state_valid(self, state: PrototypeEmbodiedCommandAdapterState) -> Array:
        self._check_state_contract(state)
        settled_layout = jnp.where(
            state.has_settled_prototype_decision,
            _words_nonzero(state.last_settled_prototype_decision_id),
            jnp.all(state.last_settled_prototype_decision_id == jnp.uint32(0)),
        )
        pending_layout = jnp.where(
            state.pending.available,
            self._receipt_valid(state),
            _tree_array_equal(state.pending, self._blank_receipt()),
        )
        return (
            self._semantic.validate_state(state.semantic)
            & self._envelope.state_valid(state.envelope)
            & jnp.array_equal(state.adapter_config_digest, self._config_digest)
            & settled_layout
            & pending_layout
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._binding_payload(state)),
            )
        )

    def init(
        self,
        semantic_state: PrototypeConsolidatedSemanticMemoryState,
        envelope_state: EmbodiedSafetyEnvelopeState,
    ) -> PrototypeEmbodiedCommandAdapterState:
        """Bind existing exact child states without starting either child."""

        state = PrototypeEmbodiedCommandAdapterState(
            semantic=semantic_state,
            envelope=envelope_state,
            adapter_config_digest=self._config_digest,
            receipt_clock_words=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            has_settled_prototype_decision=jnp.asarray(False, dtype=jnp.bool_),
            last_settled_prototype_decision_id=jnp.zeros(
                (_DECISION_WORDS,), dtype=jnp.uint32
            ),
            pending=self._blank_receipt(),
            binding_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
        )
        state = self._with_binding_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot bind invalid semantic or envelope source state")
        return state

    def prepare(
        self,
        state: PrototypeEmbodiedCommandAdapterState,
        preparation: PrototypeEmbodiedCommandPreparationInput,
    ) -> PrototypeEmbodiedCommandPreparationResult:
        """Stage one exact primitive and complete envelope evaluation receipt."""

        self._check_state_contract(state)
        self._check_preparation_contract(preparation)
        source_valid = self.state_valid(state)
        owner = state.semantic.composition.dispatch_owner
        prototype = state.semantic.composition.prototype
        next_clock, clock_available = _increment_words(state.receipt_clock_words)
        selected_index_valid = (
            (owner.selected_action >= 0)
            & (owner.selected_action < self._config.n_actions)
        )
        safe_selected = jnp.clip(owner.selected_action, 0, self._config.n_actions - 1)
        selected_admitted = selected_index_valid & owner.hard_safety_action_mask[safe_selected]
        owner_decision_matches = owner.available & prototype.started & jnp.array_equal(
            owner.prototype_decision_id,
            prototype.current_decision_id,
        )
        owner_action_matches = owner.available & prototype.started & (
            owner.selected_action == prototype.current_action
        )
        owner_not_settled = (~state.has_settled_prototype_decision) | _words_greater(
            owner.prototype_decision_id,
            state.last_settled_prototype_decision_id,
        )
        decision_fresh = _words_nonzero(preparation.envelope_decision_id) & (
            (~state.envelope.has_decision)
            | _words_greater(
                preparation.envelope_decision_id,
                state.envelope.last_decision_id,
            )
        )
        action_fresh = _words_nonzero(preparation.envelope_action_id) & (
            (~state.envelope.has_action)
            | _words_greater(
                preparation.envelope_action_id,
                state.envelope.last_action_id,
            )
        )
        telemetry_fresh = _words_nonzero(preparation.telemetry.telemetry_id) & (
            (~state.envelope.has_telemetry)
            | _words_greater(
                preparation.telemetry.telemetry_id,
                state.envelope.last_telemetry_id,
            )
        )
        versions_nonzero = (
            _words_nonzero(preparation.model_version)
            & _words_nonzero(preparation.optimizer_version)
            & _words_nonzero(preparation.lifecycle_version)
        )
        prepare_pre = (
            source_valid
            & (~state.pending.available)
            & clock_available
            & owner.available
            & owner_decision_matches
            & owner_action_matches
            & owner_not_settled
            & selected_index_valid
            & selected_admitted
            & decision_fresh
            & action_fresh
            & telemetry_fresh
            & versions_nonzero
        )
        pending = self._receipt_record(
            available=jnp.asarray(True, dtype=jnp.bool_),
            receipt_words=next_clock,
            prototype_decision_id=owner.prototype_decision_id,
            selected_action=owner.selected_action,
            hard_safety_action_mask=owner.hard_safety_action_mask,
            envelope_source_revision=state.envelope.revision,
            envelope_source_checksum=state.envelope.state_checksum,
            envelope_source_digest=state.envelope.source_digest,
            envelope_config_digest=self._envelope.config_digest,
            adapter_config_digest=self._config_digest,
            preparation=preparation,
        )
        candidate = self._with_binding_checksum(
            state.replace(
                receipt_clock_words=next_clock,
                pending=pending,
                binding_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            )
        )
        candidate_valid = self.state_valid(candidate)
        prepared = prepare_pre & candidate_valid
        final_state = cast(
            PrototypeEmbodiedCommandAdapterState,
            jax.lax.cond(prepared, lambda _: candidate, lambda _: state, operand=None),
        )
        proposed = self._command(safe_selected)
        command = cast(
            EmbodiedCommand,
            jax.lax.cond(
                prepared,
                lambda _: proposed,
                lambda _: self._zero_command(),
                operand=None,
            ),
        )
        return PrototypeEmbodiedCommandPreparationResult(
            state=final_state,
            command=command,
            receipt_words=jnp.where(
                prepared,
                next_clock,
                jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            ),
            prototype_decision_id=jnp.where(
                prepared,
                owner.prototype_decision_id,
                jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
            ),
            selected_action=jnp.where(
                prepared,
                owner.selected_action,
                jnp.asarray(-1, dtype=jnp.int32),
            ).astype(jnp.int32),
            hard_safety_action_mask=jnp.where(
                prepared,
                owner.hard_safety_action_mask,
                jnp.zeros((self._config.n_actions,), dtype=jnp.bool_),
            ),
            diagnostics=PrototypeEmbodiedCommandPreparationDiagnostics(
                source_state_valid=source_valid,
                receipt_slot_available=~state.pending.available,
                receipt_clock_available=clock_available,
                dispatch_owner_available=owner.available,
                owner_decision_matches_current=owner_decision_matches,
                owner_action_matches_current=owner_action_matches,
                owner_not_already_settled=owner_not_settled,
                selected_action_contract_valid=selected_index_valid,
                selected_action_admitted_by_bound_mask=selected_admitted,
                envelope_decision_identity_fresh=decision_fresh,
                envelope_action_identity_fresh=action_fresh,
                telemetry_identity_fresh=telemetry_fresh,
                versions_nonzero=versions_nonzero,
                candidate_state_valid=candidate_valid,
                prepared=prepared,
                command_geometry_certificate=jnp.asarray(False, dtype=jnp.bool_),
                physical_dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
                caller_authentication=jnp.asarray(False, dtype=jnp.bool_),
            ),
        )

    def evaluate_pending(
        self,
        state: PrototypeEmbodiedCommandAdapterState,
    ) -> EmbodiedEnvelopeDecision:
        """Purely evaluate the exact pending request without adopting its state.

        This is a convenience for an orchestrator at the trust boundary.  The
        result is still non-authoritative until ``settle`` recomputes and
        verifies it.  Calling this method does not dispatch the command.
        """

        self._check_state_contract(state)
        pending = state.pending
        return self._envelope.evaluate(
            state.envelope,
            pending.telemetry,
            self._command(pending.selected_action),
            decision_id=pending.envelope_decision_id,
            action_id=pending.envelope_action_id,
            control_tick=pending.control_tick,
            control_deadline_tick=pending.control_deadline_tick,
            model_version=pending.model_version,
            optimizer_version=pending.optimizer_version,
            lifecycle_version=pending.lifecycle_version,
            untrusted_reward=pending.untrusted_reward,
            partner_metadata_digest=pending.partner_metadata_digest,
            learned_cost_estimate=pending.learned_cost_estimate,
        )

    def _command_contract_valid(self, command: object) -> bool:
        if type(command) is not EmbodiedCommand:
            return False
        exact = command
        contracts = (
            (exact.joint_position, (self._config.n_joints,), jnp.float32),
            (exact.joint_velocity, (self._config.n_joints,), jnp.float32),
            (exact.joint_torque, (self._config.n_joints,), jnp.float32),
            (exact.workspace_position, (_WORKSPACE_DIM,), jnp.float32),
            (exact.collision_clearance, (), jnp.float32),
        )
        return all(
            hasattr(value, "shape")
            and hasattr(value, "dtype")
            and tuple(value.shape) == shape
            and jnp.dtype(value.dtype) == jnp.dtype(dtype)
            for value, shape, dtype in contracts
        )

    def _command_matches(self, command: EmbodiedCommand) -> Array:
        position = jax.lax.bitcast_convert_type(command.joint_position, jnp.uint32)
        velocity = jax.lax.bitcast_convert_type(command.joint_velocity, jnp.uint32)
        torque = jax.lax.bitcast_convert_type(command.joint_torque, jnp.uint32)
        workspace = jax.lax.bitcast_convert_type(command.workspace_position, jnp.uint32)
        clearance = jax.lax.bitcast_convert_type(
            command.collision_clearance,
            jnp.uint32,
        )
        return (
            jnp.all(
                jax.lax.bitcast_convert_type(
                    self._bank_joint_position,
                    jnp.uint32,
                )
                == position[None, :],
                axis=1,
            )
            & jnp.all(
                jax.lax.bitcast_convert_type(
                    self._bank_joint_velocity,
                    jnp.uint32,
                )
                == velocity[None, :],
                axis=1,
            )
            & jnp.all(
                jax.lax.bitcast_convert_type(
                    self._bank_joint_torque,
                    jnp.uint32,
                )
                == torque[None, :],
                axis=1,
            )
            & jnp.all(
                jax.lax.bitcast_convert_type(
                    self._bank_workspace_position,
                    jnp.uint32,
                )
                == workspace[None, :],
                axis=1,
            )
            & (
                jax.lax.bitcast_convert_type(
                    self._bank_collision_clearance,
                    jnp.uint32,
                )
                == clearance
            )
        )

    def map_command(self, command: EmbodiedCommand) -> PrototypeEmbodiedCommandMappingResult:
        """Map one public command payload to one unique bank action or ``-1``.

        This is a pure bit-identity query. It neither consults a receipt mask
        nor certifies geometry or safety; settlement performs those checks.
        """

        command_contract = self._command_contract_valid(command)
        mapping_command = command if command_contract else self._zero_command()
        matches = self._command_matches(mapping_command)
        match_count = jnp.sum(matches.astype(jnp.int32), dtype=jnp.int32)
        maps_one = jnp.asarray(command_contract, dtype=jnp.bool_) & (match_count == 1)
        action = jnp.where(
            maps_one,
            jnp.argmax(matches.astype(jnp.int32)).astype(jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
        )
        return PrototypeEmbodiedCommandMappingResult(
            command_contract_valid=jnp.asarray(command_contract, dtype=jnp.bool_),
            match_count=match_count,
            maps_exactly_one_primitive=maps_one,
            action=action,
        )

    def command_for_action(self, action: Array) -> EmbodiedCommand:
        """Return the fixed payload for one int32 action identity.

        The caller must separately establish that the action is in range and
        mask-admitted. This query clips only to keep its array contract total;
        it grants no safety or dispatch authority.
        """

        _require_array(action, name="action", shape=(), dtype=jnp.int32)
        return self._command(action)

    def settle(
        self,
        state: PrototypeEmbodiedCommandAdapterState,
        envelope_result: EmbodiedEnvelopeDecision,
    ) -> PrototypeEmbodiedCommandSettlementResult:
        """Verify and atomically settle one exact pending envelope result.

        Accepted and certified-fallback outcomes consume the exact receipt,
        adopt both child candidates, and mark the Prototype decision settled.
        A valid no-action outcome adopts the envelope rejection log, closes
        only that envelope attempt, and preserves the semantic dispatch owner.
        The same semantic owner may then prepare a fresh attempt with fresh
        envelope identities and telemetry.  A fresh emergency-stop latch is
        adopted even when the ordinary envelope transaction is unavailable.
        """

        self._check_state_contract(state)
        if type(envelope_result) is not EmbodiedEnvelopeDecision:
            raise TypeError("envelope_result must be an exact EmbodiedEnvelopeDecision")
        source_valid = self.state_valid(state)
        pending = state.pending
        expected = self.evaluate_pending(state)
        result_exact = _tree_array_equal(envelope_result, expected)

        mapping = self.map_command(envelope_result.command)
        match_count = mapping.match_count
        maps_one = mapping.maps_exactly_one_primitive
        mapped = mapping.action
        safe_mapped = jnp.clip(mapped, 0, self._config.n_actions - 1)
        mapped_admitted = maps_one & pending.hard_safety_action_mask[safe_mapped]
        mapped_matches_selected = maps_one & (mapped == pending.selected_action)

        accepted_structure = (
            expected.transaction_applied
            & expected.action_available
            & expected.proposed_accepted
            & (~expected.fallback_used)
            & maps_one
            & mapped_matches_selected
            & mapped_admitted
        )
        fallback_structure = (
            expected.transaction_applied
            & expected.action_available
            & (~expected.proposed_accepted)
            & expected.fallback_used
            & expected.fallback_certified
            & maps_one
            & mapped_admitted
        )
        no_action_structure = (
            expected.transaction_applied
            & (~expected.action_available)
            & (~expected.proposed_accepted)
            & (~expected.fallback_used)
        )
        stop_only_structure = (
            (~expected.transaction_applied)
            & expected.emergency_stop_latch_applied
            & (~expected.action_available)
            & (~expected.proposed_accepted)
            & (~expected.fallback_used)
            & (~_tree_array_equal(expected.state, state.envelope))
        )
        exact_pending = source_valid & pending.available & result_exact
        envelope_available_valid = exact_pending & (
            accepted_structure | fallback_structure
        )
        envelope_only_valid = exact_pending & (
            no_action_structure | stop_only_structure
        )
        semantic_available_candidate = envelope_available_valid
        semantic_receipt_candidate = (
            envelope_available_valid | envelope_only_valid
        )

        semantic_settlement = self._semantic.settle_dispatch(
            state.semantic,
            PrototypeConsolidatedMemoryDispatchSettlementInput(
                action_available=semantic_available_candidate,
                prototype_decision_id=jnp.where(
                    semantic_receipt_candidate,
                    pending.prototype_decision_id,
                    jnp.zeros((_DECISION_WORDS,), dtype=jnp.uint32),
                ),
                selected_action=pending.selected_action,
                executed_action=jnp.where(
                    semantic_available_candidate,
                    mapped,
                    jnp.asarray(-1, dtype=jnp.int32),
                ).astype(jnp.int32),
            ),
        )
        semantic_committed = (
            semantic_settlement.composition.diagnostics.transaction_committed
        )
        semantic_action_matches = jnp.where(
            semantic_available_candidate,
            semantic_settlement.action == mapped,
            semantic_settlement.action == -1,
        )
        consume_pre = (
            envelope_available_valid
            & semantic_committed
            & semantic_action_matches
        )
        available_candidate = self._with_binding_checksum(
            state.replace(
                semantic=semantic_settlement.state,
                envelope=expected.state,
                has_settled_prototype_decision=jnp.asarray(True, dtype=jnp.bool_),
                last_settled_prototype_decision_id=pending.prototype_decision_id,
                pending=self._blank_receipt(),
                binding_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            )
        )
        available_candidate_valid = self.state_valid(available_candidate)
        receipt_consumed = consume_pre & available_candidate_valid
        semantic_owner_retry = (
            envelope_only_valid
            & semantic_committed
            & semantic_action_matches
            & _tree_array_equal(semantic_settlement.state, state.semantic)
        )
        envelope_only_candidate = self._with_binding_checksum(
            state.replace(
                envelope=expected.state,
                pending=self._blank_receipt(),
                binding_checksum=jnp.zeros((_IDENTITY_WORDS,), dtype=jnp.uint32),
            )
        )
        envelope_only_candidate_valid = self.state_valid(envelope_only_candidate)
        envelope_only_committed = semantic_owner_retry & envelope_only_candidate_valid
        attempt_receipt_closed = receipt_consumed | envelope_only_committed
        final_state = cast(
            PrototypeEmbodiedCommandAdapterState,
            jax.lax.cond(
                receipt_consumed,
                lambda _: available_candidate,
                lambda _: jax.lax.cond(
                    envelope_only_committed,
                    lambda __: envelope_only_candidate,
                    lambda __: state,
                    operand=None,
                ),
                operand=None,
            ),
        )
        final_action = jnp.where(
            receipt_consumed,
            mapped,
            jnp.asarray(-1, dtype=jnp.int32),
        ).astype(jnp.int32)
        transaction_committed = receipt_consumed | envelope_only_committed
        candidate_valid = jnp.where(
            envelope_only_valid,
            envelope_only_candidate_valid,
            available_candidate_valid,
        )
        return PrototypeEmbodiedCommandSettlementResult(
            state=final_state,
            action=final_action,
            receipt_words=pending.receipt_words,
            envelope=envelope_result,
            semantic=semantic_settlement,
            diagnostics=PrototypeEmbodiedCommandSettlementDiagnostics(
                source_state_valid=source_valid,
                pending_receipt_available=pending.available,
                envelope_result_exact=result_exact,
                envelope_transaction_applied=expected.transaction_applied,
                envelope_action_available=expected.action_available,
                envelope_proposed_accepted=expected.proposed_accepted,
                envelope_fallback_used=expected.fallback_used,
                command_match_count=match_count,
                command_maps_exactly_one_primitive=maps_one,
                mapped_action=mapped,
                mapped_action_matches_selected_proposal=mapped_matches_selected,
                mapped_action_admitted_by_bound_mask=mapped_admitted,
                envelope_outcome_structurally_valid=(
                    accepted_structure
                    | fallback_structure
                    | no_action_structure
                    | stop_only_structure
                ),
                semantic_settlement_committed=semantic_committed,
                semantic_settlement_action_matches=semantic_action_matches,
                semantic_owner_retry_preserved=(
                    envelope_only_committed
                    & _tree_array_equal(final_state.semantic, state.semantic)
                ),
                envelope_only_state_committed=envelope_only_committed,
                stop_only_latch_committed=(
                    envelope_only_committed
                    & expected.emergency_stop_latch_applied
                ),
                attempt_receipt_closed=attempt_receipt_closed,
                receipt_consumed=receipt_consumed,
                candidate_state_valid=candidate_valid,
                transaction_committed=transaction_committed,
                learning_applied=jnp.asarray(False, dtype=jnp.bool_),
                evidence_written=jnp.asarray(False, dtype=jnp.bool_),
                random_generator_consumed=jnp.asarray(False, dtype=jnp.bool_),
                physical_dispatch_authority=jnp.asarray(False, dtype=jnp.bool_),
                caller_authentication=jnp.asarray(False, dtype=jnp.bool_),
                safety_authority=jnp.asarray(False, dtype=jnp.bool_),
            ),
        )

    def checkpoint_payload(
        self,
        state: PrototypeEmbodiedCommandAdapterState,
    ) -> dict[str, object]:
        """Return a strict host-only checkpoint with an unkeyed whole-state hash."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid command adapter state")
        return {
            "schema": PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "semantic": self._semantic.checkpoint_payload(state.semantic),
            "envelope": self._envelope.checkpoint_payload(state.envelope),
            "adapter_config_digest": state.adapter_config_digest,
            "receipt_clock_words": state.receipt_clock_words,
            "has_settled_prototype_decision": state.has_settled_prototype_decision,
            "last_settled_prototype_decision_id": (
                state.last_settled_prototype_decision_id
            ),
            "pending": state.pending,
            "binding_checksum": state.binding_checksum,
            "state_sha256": _tree_sha256(state),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        semantic_source_digest: Array,
        semantic_namespace_digest: Array,
        semantic_representation_revision: int | Array,
        semantic_source_revision: int | Array,
        envelope_source_digest: Array,
        trusted_envelope_state_revision: int | Array,
        trusted_envelope_state_digest: Array,
        trusted_adapter_state_digest: Array,
    ) -> PrototypeEmbodiedCommandAdapterState:
        """Restore only child states and adapter bytes pinned by external anchors."""

        if type(payload) is not dict:
            raise ValueError("command adapter checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        expected_keys = {
            "schema",
            "config",
            "semantic",
            "envelope",
            "adapter_config_digest",
            "receipt_clock_words",
            "has_settled_prototype_decision",
            "last_settled_prototype_decision_id",
            "pending",
            "binding_checksum",
            "state_sha256",
        }
        if set(raw) != expected_keys:
            raise ValueError("command adapter checkpoint fields differ from schema v1")
        if raw["schema"] != PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CHECKPOINT_SCHEMA:
            raise ValueError("command adapter checkpoint schema differs")
        if raw["config"] != self.to_config():
            raise ValueError("command adapter checkpoint config differs")
        semantic = self._semantic.restore_checkpoint(
            raw["semantic"],
            source_digest=semantic_source_digest,
            semantic_namespace_digest=semantic_namespace_digest,
            representation_revision=semantic_representation_revision,
            source_revision=semantic_source_revision,
        )
        envelope = self._envelope.restore_checkpoint(
            raw["envelope"],
            expected_source_digest=envelope_source_digest,
            trusted_state_revision=trusted_envelope_state_revision,
            trusted_state_digest=trusted_envelope_state_digest,
        )
        adapter_digest = _require_array(
            raw["adapter_config_digest"],
            name="checkpoint.adapter_config_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        receipt_clock = _require_array(
            raw["receipt_clock_words"],
            name="checkpoint.receipt_clock_words",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        settled = _require_array(
            raw["has_settled_prototype_decision"],
            name="checkpoint.has_settled_prototype_decision",
            shape=(),
            dtype=jnp.bool_,
        )
        last_settled = _require_array(
            raw["last_settled_prototype_decision_id"],
            name="checkpoint.last_settled_prototype_decision_id",
            shape=(_DECISION_WORDS,),
            dtype=jnp.uint32,
        )
        pending = raw["pending"]
        if type(pending) is not PrototypeEmbodiedCommandReceiptState:
            raise ValueError("command adapter checkpoint pending type differs")
        self._check_receipt_contract(pending)
        binding = _require_array(
            raw["binding_checksum"],
            name="checkpoint.binding_checksum",
            shape=(_IDENTITY_WORDS,),
            dtype=jnp.uint32,
        )
        persisted_digest = _require_array(
            raw["state_sha256"],
            name="checkpoint.state_sha256",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        trusted_digest = _require_array(
            trusted_adapter_state_digest,
            name="trusted_adapter_state_digest",
            shape=(_DIGEST_BYTES,),
            dtype=jnp.uint8,
        )
        state = PrototypeEmbodiedCommandAdapterState(
            semantic=semantic,
            envelope=envelope,
            adapter_config_digest=adapter_digest,
            receipt_clock_words=receipt_clock,
            has_settled_prototype_decision=settled,
            last_settled_prototype_decision_id=last_settled,
            pending=pending,
            binding_checksum=binding,
        )
        valid = (
            jnp.array_equal(adapter_digest, self._config_digest)
            & jnp.array_equal(persisted_digest, trusted_digest)
            & jnp.array_equal(persisted_digest, _tree_sha256(state))
            & self.state_valid(state)
        )
        if not bool(jax.device_get(valid)):
            raise ValueError("command adapter checkpoint is invalid, stale, or tampered")
        return state

    def resource_budget(
        self,
        state: PrototypeEmbodiedCommandAdapterState,
    ) -> PrototypeEmbodiedCommandAdapterResourceBudget:
        """Report exact storage, bounded work, execution modes, and zero authority."""

        self._check_state_contract(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot measure an invalid command adapter state")
        cells_per_primitive = 3 * self._config.n_joints + _WORKSPACE_DIM + 1
        return PrototypeEmbodiedCommandAdapterResourceBudget(
            persistent_state_nbytes=_tree_nbytes(state),
            pending_receipt_nbytes=_tree_nbytes(state.pending),
            static_command_bank_nbytes=(
                self._config.n_actions * cells_per_primitive * 4
            ),
            n_actions=self._config.n_actions,
            n_joints=self._config.n_joints,
            primitive_payload_float32_cells=cells_per_primitive,
            maximum_pending_receipts=1,
            nonwrapping_receipt_clock_words=_IDENTITY_WORDS,
            envelope_recomputations_per_settlement=1,
            envelope_state_commits_per_exact_no_action=1,
            stop_latch_preservations_per_exact_stop_only_result=1,
            maximum_bit_exact_command_comparisons_per_settlement=(
                self._config.n_actions
            ),
            semantic_settlement_delegations_per_settlement=1,
            physical_dispatches_per_operation=0,
            learning_state_mutations_per_operation=0,
            evidence_writes_per_operation=0,
            random_generator_calls_per_operation=0,
            persistent_growth_per_operation_bytes=0,
            checkpoint_host_only=True,
            eager_prepare_and_settle=True,
            jit_prepare_and_settle=True,
            command_geometry_certificate=False,
            physical_dispatch_authority=False,
            caller_authentication=False,
            learning_authority=False,
            evidence_authority=False,
            safety_authority=False,
            promotion_authority=False,
            scientific_promotion_allowed=False,
            semantic=self._semantic.resource_budget,
            envelope=self._envelope.resource_budget(state.envelope),
        )


__all__ = [
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_ASSESSMENT",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CALLER_AUTHENTICATION",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CHECKPOINT_HOST_ONLY",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CHECKPOINT_SCHEMA",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CONFIG_SCHEMA",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_EAGER_SUPPORTED",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_EVIDENCE_AUTHORITY",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_JIT_PREPARE_SUPPORTED",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_JIT_SETTLE_SUPPORTED",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_LEARNING_AUTHORITY",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_PHYSICAL_DISPATCH_AUTHORITY",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_PROMOTION_AUTHORITY",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_SAFETY_AUTHORITY",
    "PROTOTYPE_EMBODIED_COMMAND_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED",
    "DiscreteEmbodiedPrimitiveCommand",
    "PrototypeEmbodiedCommandAdapter",
    "PrototypeEmbodiedCommandAdapterConfig",
    "PrototypeEmbodiedCommandAdapterResourceBudget",
    "PrototypeEmbodiedCommandAdapterState",
    "PrototypeEmbodiedCommandMappingResult",
    "PrototypeEmbodiedCommandPreparationDiagnostics",
    "PrototypeEmbodiedCommandPreparationInput",
    "PrototypeEmbodiedCommandPreparationResult",
    "PrototypeEmbodiedCommandReceiptState",
    "PrototypeEmbodiedCommandSettlementDiagnostics",
    "PrototypeEmbodiedCommandSettlementResult",
]
