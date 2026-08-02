"""Independent host oracle for one slot-signaling role transition.

This module deliberately re-expresses the slot learner's transition on the
host.  It does not call ``SlotSignalingAgent.update`` or its private role
update, and it does not replay an evaluator.  A trace record contains the old
state, the externally visible decision, reward and write permits, update
diagnostics, and the new state.  The oracle reconstructs the expected
decision, diagnostics, and complete new state from those primitives.

The record is strict JSON data.  It is intended to let a later evidence
artifact expose enough information for an implementation-independent state
machine audit; it is development infrastructure and does not promote a
scientific claim by itself.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from numbers import Real
from typing import Literal, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.slot_signaling_agent import (
    N_DURABLE_SLOTS,
    N_SLOT_ACTIONS,
    N_SLOT_INPUTS,
    N_SLOTS,
    SCRATCH_SLOT,
    SLOT_DURABLE,
    SLOT_SCRATCH,
    SLOT_VACANT,
    SlotRoleDecision,
    SlotRoleState,
    SlotRoleUpdate,
    SlotSignalingConfig,
)

SLOT_ROLE_TRANSITION_ORACLE_SCHEMA = "alberta.hidden-regime.slot-role-transition.v1"
ROLE_HELPER: Literal["helper"] = "helper"
ROLE_BENEFICIARY: Literal["beneficiary"] = "beneficiary"
INPUT_PRIVATE_CUE: Literal["private_cue"] = "private_cue"
INPUT_DELIVERED_MESSAGE: Literal["delivered_message"] = "delivered_message"
# Policy literals are intentionally re-declared here rather than imported from
# :mod:`~alberta_framework.core.slot_signaling_agent`: the oracle re-expresses
# the learner independently, so a producer-side literal change must disagree
# with recorded traces instead of silently teaching the oracle the same change.
DURABLE_WRITE_SELECTIVE = "selective"
DURABLE_WRITE_WRITABLE = "writable"
REPLACEMENT_TARGET_EVIDENCE = "evidence"
REPLACEMENT_TARGET_LRU = "lru"

# 2**24 — the largest float32 integer count for which ``x + 1.0`` is still
# exact.  The agent saturates its relevance-mass counter at this value (one
# step further, adding 1.0 would be a float32 rounding no-op), so the oracle
# enforces the same cap when validating and reconstructing masses.
_FLOAT32_MASS_LIMIT = np.float32(16_777_216.0)
_INT32_MAX = int(np.iinfo(np.int32).max)

FloatVector = tuple[float, ...]
FloatMatrix = tuple[FloatVector, ...]
FloatTensor3 = tuple[FloatMatrix, ...]
IntVector = tuple[int, ...]
KeyData = tuple[int, ...]
RoleName = Literal["helper", "beneficiary"]
InputStreamName = Literal["private_cue", "delivered_message"]


class SlotLifecycleOracleInputError(ValueError):
    """A transition record is malformed or outside the oracle contract."""


def _require_keys(mapping: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SlotLifecycleOracleInputError(
            f"{path} has noncanonical keys; missing={missing}, extra={extra}"
        )


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SlotLifecycleOracleInputError(f"{path} must be an object")
    if not all(type(key) is str for key in value):
        raise SlotLifecycleOracleInputError(f"{path} keys must be strings")
    return cast(Mapping[str, object], value)


def _list(value: object, path: str, length: int) -> list[object]:
    if type(value) is not list:
        raise SlotLifecycleOracleInputError(f"{path} must be a JSON list")
    result = cast(list[object], value)
    if len(result) != length:
        raise SlotLifecycleOracleInputError(f"{path} must contain exactly {length} items")
    return result


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise SlotLifecycleOracleInputError(f"{path} must be boolean")
    return value


def _integer(
    value: object,
    path: str,
    *,
    minimum: int = -_INT32_MAX - 1,
    maximum: int = _INT32_MAX,
) -> int:
    if type(value) is not int:
        raise SlotLifecycleOracleInputError(f"{path} must be an integer")
    result = value
    if not minimum <= result <= maximum:
        raise SlotLifecycleOracleInputError(
            f"{path} must lie in [{minimum}, {maximum}]"
        )
    return result


def _float(value: object, path: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise SlotLifecycleOracleInputError(f"{path} must be numeric")
    result = float(value)
    result32 = np.float32(result)
    if not math.isfinite(result) or not np.isfinite(result32):
        raise SlotLifecycleOracleInputError(f"{path} must be finite in float32")
    return float(result32)


def _string(value: object, path: str) -> str:
    if type(value) is not str:
        raise SlotLifecycleOracleInputError(f"{path} must be a string")
    return value


def _float_vector(value: object, path: str, length: int) -> FloatVector:
    items = _list(value, path, length)
    return tuple(_float(item, f"{path}[{index}]") for index, item in enumerate(items))


def _int_vector(value: object, path: str, length: int) -> IntVector:
    items = _list(value, path, length)
    return tuple(_integer(item, f"{path}[{index}]") for index, item in enumerate(items))


def _float_matrix(value: object, path: str, rows: int, columns: int) -> FloatMatrix:
    items = _list(value, path, rows)
    return tuple(
        _float_vector(item, f"{path}[{index}]", columns)
        for index, item in enumerate(items)
    )


def _float_tensor3(
    value: object,
    path: str,
    depth: int,
    rows: int,
    columns: int,
) -> FloatTensor3:
    items = _list(value, path, depth)
    return tuple(
        _float_matrix(item, f"{path}[{index}]", rows, columns)
        for index, item in enumerate(items)
    )


def _array_float_vector(array: object, shape: tuple[int, ...], path: str) -> FloatVector:
    host = np.asarray(array, dtype=np.float32)
    if host.shape != shape or not np.all(np.isfinite(host)):
        raise SlotLifecycleOracleInputError(f"{path} must be finite with shape {shape}")
    return tuple(float(value) for value in host.tolist())


def _array_int_vector(array: object, shape: tuple[int, ...], path: str) -> IntVector:
    host = np.asarray(array)
    if host.shape != shape or host.dtype.kind not in "iu":
        raise SlotLifecycleOracleInputError(f"{path} must be integral with shape {shape}")
    values = tuple(int(value) for value in host.tolist())
    if any(value < -_INT32_MAX - 1 or value > _INT32_MAX for value in values):
        raise SlotLifecycleOracleInputError(f"{path} must fit int32")
    return values


def _array_float_tensor3(
    array: object,
    shape: tuple[int, int, int],
    path: str,
) -> FloatTensor3:
    host = np.asarray(array, dtype=np.float32)
    if host.shape != shape or not np.all(np.isfinite(host)):
        raise SlotLifecycleOracleInputError(f"{path} must be finite with shape {shape}")
    return tuple(
        tuple(tuple(float(value) for value in row) for row in matrix)
        for matrix in host.tolist()
    )


def _scalar_float(value: object, path: str) -> float:
    host = np.asarray(value, dtype=np.float32)
    if host.shape != () or not np.isfinite(host):
        raise SlotLifecycleOracleInputError(f"{path} must be a finite float32 scalar")
    return float(host)


def _scalar_int(value: object, path: str) -> int:
    host = np.asarray(value)
    if host.shape != () or host.dtype.kind not in "iu":
        raise SlotLifecycleOracleInputError(f"{path} must be an integer scalar")
    return _integer(int(host), path)


def _scalar_bool(value: object, path: str) -> bool:
    host = np.asarray(value)
    if host.shape != () or host.dtype.kind != "b":
        raise SlotLifecycleOracleInputError(f"{path} must be a boolean scalar")
    return bool(host)


def _key_data(key: Array, path: str) -> KeyData:
    try:
        host = np.asarray(jr.key_data(key), dtype=np.uint32)
    except (TypeError, ValueError) as error:
        raise SlotLifecycleOracleInputError(f"{path} must be a JAX PRNG key") from error
    if host.shape != (2,):
        raise SlotLifecycleOracleInputError(f"{path} must use two-word threefry key data")
    return tuple(int(value) for value in host.tolist())


def _wrapped_key(data: KeyData) -> Array:
    return cast(
        Array,
        jr.wrap_key_data(jnp.asarray(data, dtype=jnp.uint32), impl="threefry2x32"),
    )


@dataclasses.dataclass(frozen=True)
class SlotRoleOracleConfig:
    """The static transition parameters, including resolved factorial axes."""

    learning_rate: float
    epsilon: float
    relevance_rate: float
    lease_length: int
    confirmation_steps: int
    durable_retrieval_threshold: float
    candidate_confirmation_threshold: float
    candidate_confirmation_leases: int
    scratch_training_leases_before_retest: int
    durable_write_policy: str
    replacement_target_policy: str

    @classmethod
    def from_config(cls, config: SlotSignalingConfig) -> SlotRoleOracleConfig:
        """Freeze only behaviorally effective core settings."""

        write_policy = getattr(config, "effective_durable_write_policy", None)
        replacement_policy = getattr(config, "effective_replacement_target_policy", None)
        if write_policy is None:
            write_policy = (
                DURABLE_WRITE_WRITABLE
                if config.writable_lru_ablation
                else DURABLE_WRITE_SELECTIVE
            )
        if replacement_policy is None:
            replacement_policy = (
                REPLACEMENT_TARGET_LRU
                if config.writable_lru_ablation
                else REPLACEMENT_TARGET_EVIDENCE
            )
        result = cls(
            learning_rate=float(np.float32(config.learning_rate)),
            epsilon=float(np.float32(config.epsilon)),
            relevance_rate=float(np.float32(config.relevance_rate)),
            lease_length=config.lease_length,
            confirmation_steps=config.confirmation_steps,
            durable_retrieval_threshold=float(
                np.float32(config.durable_retrieval_threshold)
            ),
            candidate_confirmation_threshold=float(
                np.float32(config.candidate_confirmation_threshold)
            ),
            candidate_confirmation_leases=config.candidate_confirmation_leases,
            scratch_training_leases_before_retest=(
                config.scratch_training_leases_before_retest
            ),
            durable_write_policy=str(write_policy),
            replacement_target_policy=str(replacement_policy),
        )
        result.validate()
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SlotRoleOracleConfig:
        """Parse a strict JSON configuration snapshot."""

        expected = {
            "learning_rate",
            "epsilon",
            "relevance_rate",
            "lease_length",
            "confirmation_steps",
            "durable_retrieval_threshold",
            "candidate_confirmation_threshold",
            "candidate_confirmation_leases",
            "scratch_training_leases_before_retest",
            "durable_write_policy",
            "replacement_target_policy",
        }
        _require_keys(raw, expected, "config")
        result = cls(
            learning_rate=_float(raw["learning_rate"], "config.learning_rate"),
            epsilon=_float(raw["epsilon"], "config.epsilon"),
            relevance_rate=_float(raw["relevance_rate"], "config.relevance_rate"),
            lease_length=_integer(raw["lease_length"], "config.lease_length", minimum=1),
            confirmation_steps=_integer(
                raw["confirmation_steps"], "config.confirmation_steps", minimum=1
            ),
            durable_retrieval_threshold=_float(
                raw["durable_retrieval_threshold"],
                "config.durable_retrieval_threshold",
            ),
            candidate_confirmation_threshold=_float(
                raw["candidate_confirmation_threshold"],
                "config.candidate_confirmation_threshold",
            ),
            candidate_confirmation_leases=_integer(
                raw["candidate_confirmation_leases"],
                "config.candidate_confirmation_leases",
                minimum=1,
            ),
            scratch_training_leases_before_retest=_integer(
                raw["scratch_training_leases_before_retest"],
                "config.scratch_training_leases_before_retest",
                minimum=1,
            ),
            durable_write_policy=_string(
                raw["durable_write_policy"], "config.durable_write_policy"
            ),
            replacement_target_policy=_string(
                raw["replacement_target_policy"], "config.replacement_target_policy"
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Reject non-core or ambiguous static settings."""

        numeric_fields = (
            self.learning_rate,
            self.epsilon,
            self.relevance_rate,
            self.durable_retrieval_threshold,
            self.candidate_confirmation_threshold,
        )
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in numeric_fields):
            raise SlotLifecycleOracleInputError("config rates and thresholds must be numeric")
        integer_fields = (
            self.lease_length,
            self.confirmation_steps,
            self.candidate_confirmation_leases,
            self.scratch_training_leases_before_retest,
        )
        if any(type(value) is not int for value in integer_fields):
            raise SlotLifecycleOracleInputError("config counts must be integers")
        if not 1 <= self.lease_length <= _INT32_MAX:
            raise SlotLifecycleOracleInputError("config.lease_length must be positive int32")
        if not 1 <= self.candidate_confirmation_leases <= _INT32_MAX:
            raise SlotLifecycleOracleInputError(
                "config.candidate_confirmation_leases must be positive int32"
            )
        if not 1 <= self.scratch_training_leases_before_retest <= _INT32_MAX:
            raise SlotLifecycleOracleInputError(
                "config.scratch_training_leases_before_retest must be positive int32"
            )
        if not 0.0 < self.learning_rate <= 1.0:
            raise SlotLifecycleOracleInputError("config.learning_rate must lie in (0, 1]")
        if not 0.0 <= self.epsilon <= 1.0:
            raise SlotLifecycleOracleInputError("config.epsilon must lie in [0, 1]")
        if not 0.0 < self.relevance_rate <= 1.0:
            raise SlotLifecycleOracleInputError("config.relevance_rate must lie in (0, 1]")
        if not 1 <= self.confirmation_steps <= self.lease_length:
            raise SlotLifecycleOracleInputError(
                "config.confirmation_steps must lie in [1, lease_length]"
            )
        if not 0.0 <= self.durable_retrieval_threshold <= 1.0:
            raise SlotLifecycleOracleInputError(
                "config.durable_retrieval_threshold must lie in [0, 1]"
            )
        if not 0.0 <= self.candidate_confirmation_threshold <= 1.0:
            raise SlotLifecycleOracleInputError(
                "config.candidate_confirmation_threshold must lie in [0, 1]"
            )
        if self.durable_retrieval_threshold >= self.candidate_confirmation_threshold:
            raise SlotLifecycleOracleInputError(
                "durable threshold must be below candidate threshold"
            )
        if self.durable_write_policy not in {
            DURABLE_WRITE_SELECTIVE,
            DURABLE_WRITE_WRITABLE,
        }:
            raise SlotLifecycleOracleInputError("unknown effective durable-write policy")
        if self.replacement_target_policy not in {
            REPLACEMENT_TARGET_EVIDENCE,
            REPLACEMENT_TARGET_LRU,
        }:
            raise SlotLifecycleOracleInputError("unknown effective replacement-target policy")

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SlotRoleStateSnapshot:
    """Strict, complete host snapshot of one persistent role state."""

    values: FloatTensor3
    relevance_mean: FloatVector
    relevance_mass: FloatVector
    failed_leases: IntVector
    idle_leases: IntVector
    status: IntVector
    generation: IntVector
    active_slot: int
    lease_offset: int
    lease_reward_sum: float
    remaining_durable_tests: int
    search_cursor: int
    candidate_successful_leases: int
    next_generation: int
    key_data: KeyData

    @classmethod
    def from_state(cls, state: SlotRoleState) -> SlotRoleStateSnapshot:
        """Copy a device state into immutable host primitives."""

        return cls(
            values=_array_float_tensor3(
                state.values,
                (N_SLOTS, N_SLOT_INPUTS, N_SLOT_ACTIONS),
                "state.values",
            ),
            relevance_mean=_array_float_vector(
                state.relevance_mean, (N_SLOTS,), "state.relevance_mean"
            ),
            relevance_mass=_array_float_vector(
                state.relevance_mass, (N_SLOTS,), "state.relevance_mass"
            ),
            failed_leases=_array_int_vector(
                state.failed_leases, (N_SLOTS,), "state.failed_leases"
            ),
            idle_leases=_array_int_vector(
                state.idle_leases, (N_SLOTS,), "state.idle_leases"
            ),
            status=_array_int_vector(state.status, (N_SLOTS,), "state.status"),
            generation=_array_int_vector(
                state.generation, (N_SLOTS,), "state.generation"
            ),
            active_slot=_scalar_int(state.active_slot, "state.active_slot"),
            lease_offset=_scalar_int(state.lease_offset, "state.lease_offset"),
            lease_reward_sum=_scalar_float(
                state.lease_reward_sum, "state.lease_reward_sum"
            ),
            remaining_durable_tests=_scalar_int(
                state.remaining_durable_tests, "state.remaining_durable_tests"
            ),
            search_cursor=_scalar_int(state.search_cursor, "state.search_cursor"),
            candidate_successful_leases=_scalar_int(
                state.candidate_successful_leases,
                "state.candidate_successful_leases",
            ),
            next_generation=_scalar_int(
                state.next_generation, "state.next_generation"
            ),
            key_data=_key_data(state.key, "state.key"),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SlotRoleStateSnapshot:
        """Parse a complete state and reject omissions or extensions."""

        expected = {
            "values",
            "relevance_mean",
            "relevance_mass",
            "failed_leases",
            "idle_leases",
            "status",
            "generation",
            "active_slot",
            "lease_offset",
            "lease_reward_sum",
            "remaining_durable_tests",
            "search_cursor",
            "candidate_successful_leases",
            "next_generation",
            "key_data",
        }
        _require_keys(raw, expected, "state")
        key_items = _list(raw["key_data"], "state.key_data", 2)
        return cls(
            values=_float_tensor3(
                raw["values"],
                "state.values",
                N_SLOTS,
                N_SLOT_INPUTS,
                N_SLOT_ACTIONS,
            ),
            relevance_mean=_float_vector(
                raw["relevance_mean"], "state.relevance_mean", N_SLOTS
            ),
            relevance_mass=_float_vector(
                raw["relevance_mass"], "state.relevance_mass", N_SLOTS
            ),
            failed_leases=_int_vector(
                raw["failed_leases"], "state.failed_leases", N_SLOTS
            ),
            idle_leases=_int_vector(raw["idle_leases"], "state.idle_leases", N_SLOTS),
            status=_int_vector(raw["status"], "state.status", N_SLOTS),
            generation=_int_vector(raw["generation"], "state.generation", N_SLOTS),
            active_slot=_integer(raw["active_slot"], "state.active_slot"),
            lease_offset=_integer(raw["lease_offset"], "state.lease_offset"),
            lease_reward_sum=_float(raw["lease_reward_sum"], "state.lease_reward_sum"),
            remaining_durable_tests=_integer(
                raw["remaining_durable_tests"], "state.remaining_durable_tests"
            ),
            search_cursor=_integer(raw["search_cursor"], "state.search_cursor"),
            candidate_successful_leases=_integer(
                raw["candidate_successful_leases"],
                "state.candidate_successful_leases",
            ),
            next_generation=_integer(raw["next_generation"], "state.next_generation"),
            key_data=tuple(
                _integer(item, f"state.key_data[{index}]", minimum=0, maximum=2**32 - 1)
                for index, item in enumerate(key_items)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON data without dtype-bearing device arrays."""

        return {
            "values": [[list(row) for row in matrix] for matrix in self.values],
            "relevance_mean": list(self.relevance_mean),
            "relevance_mass": list(self.relevance_mass),
            "failed_leases": list(self.failed_leases),
            "idle_leases": list(self.idle_leases),
            "status": list(self.status),
            "generation": list(self.generation),
            "active_slot": self.active_slot,
            "lease_offset": self.lease_offset,
            "lease_reward_sum": self.lease_reward_sum,
            "remaining_durable_tests": self.remaining_durable_tests,
            "search_cursor": self.search_cursor,
            "candidate_successful_leases": self.candidate_successful_leases,
            "next_generation": self.next_generation,
            "key_data": list(self.key_data),
        }


@dataclasses.dataclass(frozen=True)
class SlotRoleDecisionSnapshot:
    """Serializable public decision, including consumed key output."""

    slot: int
    private_input: int
    action: int
    selected_value: float
    next_key_data: KeyData

    @classmethod
    def from_decision(cls, decision: SlotRoleDecision) -> SlotRoleDecisionSnapshot:
        return cls(
            slot=_scalar_int(decision.slot, "decision.slot"),
            private_input=_scalar_int(decision.private_input, "decision.private_input"),
            action=_scalar_int(decision.action, "decision.action"),
            selected_value=_scalar_float(
                decision.selected_value, "decision.selected_value"
            ),
            next_key_data=_key_data(decision.next_key, "decision.next_key"),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SlotRoleDecisionSnapshot:
        expected = {"slot", "private_input", "action", "selected_value", "next_key_data"}
        _require_keys(raw, expected, "decision")
        key_items = _list(raw["next_key_data"], "decision.next_key_data", 2)
        return cls(
            slot=_integer(raw["slot"], "decision.slot"),
            private_input=_integer(raw["private_input"], "decision.private_input"),
            action=_integer(raw["action"], "decision.action"),
            selected_value=_float(raw["selected_value"], "decision.selected_value"),
            next_key_data=tuple(
                _integer(
                    item,
                    f"decision.next_key_data[{index}]",
                    minimum=0,
                    maximum=2**32 - 1,
                )
                for index, item in enumerate(key_items)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "private_input": self.private_input,
            "action": self.action,
            "selected_value": self.selected_value,
            "next_key_data": list(self.next_key_data),
        }


@dataclasses.dataclass(frozen=True)
class SlotRoleUpdateDiagnosticsSnapshot:
    """All scalar diagnostics exposed by ``SlotRoleUpdate``."""

    value_pre: float
    candidate_value: float
    value_post: float
    value_write: bool
    lease_boundary: bool
    lease_reward_mean: float
    relevance_ready: bool
    durable_relevant: bool
    candidate_relevant: bool
    candidate_lease_success: bool
    scratch_failed_leases_pre: int
    scratch_failed_leases_post: int
    scratch_retest_started: bool
    generation_exhausted: bool
    committed_slot: int
    committed_generation: int
    retired_slot: int
    retired_generation: int

    @classmethod
    def from_update(cls, update: SlotRoleUpdate) -> SlotRoleUpdateDiagnosticsSnapshot:
        return cls(
            value_pre=_scalar_float(update.value_pre, "diagnostics.value_pre"),
            candidate_value=_scalar_float(
                update.candidate_value, "diagnostics.candidate_value"
            ),
            value_post=_scalar_float(update.value_post, "diagnostics.value_post"),
            value_write=_scalar_bool(update.value_write, "diagnostics.value_write"),
            lease_boundary=_scalar_bool(
                update.lease_boundary, "diagnostics.lease_boundary"
            ),
            lease_reward_mean=_scalar_float(
                update.lease_reward_mean, "diagnostics.lease_reward_mean"
            ),
            relevance_ready=_scalar_bool(
                update.relevance_ready, "diagnostics.relevance_ready"
            ),
            durable_relevant=_scalar_bool(
                update.durable_relevant, "diagnostics.durable_relevant"
            ),
            candidate_relevant=_scalar_bool(
                update.candidate_relevant, "diagnostics.candidate_relevant"
            ),
            candidate_lease_success=_scalar_bool(
                update.candidate_lease_success,
                "diagnostics.candidate_lease_success",
            ),
            scratch_failed_leases_pre=_scalar_int(
                update.scratch_failed_leases_pre,
                "diagnostics.scratch_failed_leases_pre",
            ),
            scratch_failed_leases_post=_scalar_int(
                update.scratch_failed_leases_post,
                "diagnostics.scratch_failed_leases_post",
            ),
            scratch_retest_started=_scalar_bool(
                update.scratch_retest_started,
                "diagnostics.scratch_retest_started",
            ),
            generation_exhausted=_scalar_bool(
                update.generation_exhausted,
                "diagnostics.generation_exhausted",
            ),
            committed_slot=_scalar_int(
                update.committed_slot, "diagnostics.committed_slot"
            ),
            committed_generation=_scalar_int(
                update.committed_generation,
                "diagnostics.committed_generation",
            ),
            retired_slot=_scalar_int(update.retired_slot, "diagnostics.retired_slot"),
            retired_generation=_scalar_int(
                update.retired_generation, "diagnostics.retired_generation"
            ),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SlotRoleUpdateDiagnosticsSnapshot:
        expected = {field.name for field in dataclasses.fields(cls)}
        _require_keys(raw, expected, "diagnostics")
        return cls(
            value_pre=_float(raw["value_pre"], "diagnostics.value_pre"),
            candidate_value=_float(
                raw["candidate_value"], "diagnostics.candidate_value"
            ),
            value_post=_float(raw["value_post"], "diagnostics.value_post"),
            value_write=_boolean(raw["value_write"], "diagnostics.value_write"),
            lease_boundary=_boolean(
                raw["lease_boundary"], "diagnostics.lease_boundary"
            ),
            lease_reward_mean=_float(
                raw["lease_reward_mean"], "diagnostics.lease_reward_mean"
            ),
            relevance_ready=_boolean(
                raw["relevance_ready"], "diagnostics.relevance_ready"
            ),
            durable_relevant=_boolean(
                raw["durable_relevant"], "diagnostics.durable_relevant"
            ),
            candidate_relevant=_boolean(
                raw["candidate_relevant"], "diagnostics.candidate_relevant"
            ),
            candidate_lease_success=_boolean(
                raw["candidate_lease_success"],
                "diagnostics.candidate_lease_success",
            ),
            scratch_failed_leases_pre=_integer(
                raw["scratch_failed_leases_pre"],
                "diagnostics.scratch_failed_leases_pre",
            ),
            scratch_failed_leases_post=_integer(
                raw["scratch_failed_leases_post"],
                "diagnostics.scratch_failed_leases_post",
            ),
            scratch_retest_started=_boolean(
                raw["scratch_retest_started"],
                "diagnostics.scratch_retest_started",
            ),
            generation_exhausted=_boolean(
                raw["generation_exhausted"],
                "diagnostics.generation_exhausted",
            ),
            committed_slot=_integer(
                raw["committed_slot"], "diagnostics.committed_slot"
            ),
            committed_generation=_integer(
                raw["committed_generation"],
                "diagnostics.committed_generation",
            ),
            retired_slot=_integer(raw["retired_slot"], "diagnostics.retired_slot"),
            retired_generation=_integer(
                raw["retired_generation"], "diagnostics.retired_generation"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SlotRoleTransitionRecord:
    """One complete role transition observation suitable for JSON traces."""

    role: RoleName
    input_stream: InputStreamName
    config: SlotRoleOracleConfig
    old_state: SlotRoleStateSnapshot
    decision: SlotRoleDecisionSnapshot
    reward: float
    external_value_write: bool
    lifecycle_write: bool
    diagnostics: SlotRoleUpdateDiagnosticsSnapshot
    new_state: SlotRoleStateSnapshot
    schema: str = SLOT_ROLE_TRANSITION_ORACLE_SCHEMA

    @classmethod
    def from_runtime(
        cls,
        *,
        role: RoleName,
        config: SlotSignalingConfig,
        old_state: SlotRoleState,
        decision: SlotRoleDecision,
        reward: object,
        external_value_write: bool,
        lifecycle_write: bool,
        update: SlotRoleUpdate,
    ) -> SlotRoleTransitionRecord:
        """Capture public runtime objects without validating their transition."""

        input_stream: InputStreamName = (
            INPUT_PRIVATE_CUE if role == ROLE_HELPER else INPUT_DELIVERED_MESSAGE
        )
        return cls(
            role=role,
            input_stream=input_stream,
            config=SlotRoleOracleConfig.from_config(config),
            old_state=SlotRoleStateSnapshot.from_state(old_state),
            decision=SlotRoleDecisionSnapshot.from_decision(decision),
            reward=_scalar_float(reward, "reward"),
            external_value_write=external_value_write,
            lifecycle_write=lifecycle_write,
            diagnostics=SlotRoleUpdateDiagnosticsSnapshot.from_update(update),
            new_state=SlotRoleStateSnapshot.from_state(update.state),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SlotRoleTransitionRecord:
        """Parse a fail-closed JSON record."""

        expected = {
            "schema",
            "role",
            "input_stream",
            "config",
            "old_state",
            "decision",
            "reward",
            "external_value_write",
            "lifecycle_write",
            "diagnostics",
            "new_state",
        }
        _require_keys(raw, expected, "record")
        schema = _string(raw["schema"], "record.schema")
        if schema != SLOT_ROLE_TRANSITION_ORACLE_SCHEMA:
            raise SlotLifecycleOracleInputError("record.schema is not supported")
        role_raw = _string(raw["role"], "record.role")
        if role_raw not in {ROLE_HELPER, ROLE_BENEFICIARY}:
            raise SlotLifecycleOracleInputError("record.role is not supported")
        input_raw = _string(raw["input_stream"], "record.input_stream")
        if input_raw not in {INPUT_PRIVATE_CUE, INPUT_DELIVERED_MESSAGE}:
            raise SlotLifecycleOracleInputError("record.input_stream is not supported")
        role = role_raw
        input_stream = input_raw
        _validate_role_stream(role, input_stream)
        return cls(
            schema=schema,
            role=role,
            input_stream=input_stream,
            config=SlotRoleOracleConfig.from_dict(_mapping(raw["config"], "config")),
            old_state=SlotRoleStateSnapshot.from_dict(
                _mapping(raw["old_state"], "old_state")
            ),
            decision=SlotRoleDecisionSnapshot.from_dict(
                _mapping(raw["decision"], "decision")
            ),
            reward=_float(raw["reward"], "record.reward"),
            external_value_write=_boolean(
                raw["external_value_write"], "record.external_value_write"
            ),
            lifecycle_write=_boolean(
                raw["lifecycle_write"], "record.lifecycle_write"
            ),
            diagnostics=SlotRoleUpdateDiagnosticsSnapshot.from_dict(
                _mapping(raw["diagnostics"], "diagnostics")
            ),
            new_state=SlotRoleStateSnapshot.from_dict(
                _mapping(raw["new_state"], "new_state")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON data."""

        return {
            "schema": self.schema,
            "role": self.role,
            "input_stream": self.input_stream,
            "config": self.config.to_dict(),
            "old_state": self.old_state.to_dict(),
            "decision": self.decision.to_dict(),
            "reward": self.reward,
            "external_value_write": self.external_value_write,
            "lifecycle_write": self.lifecycle_write,
            "diagnostics": self.diagnostics.to_dict(),
            "new_state": self.new_state.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class SlotRoleTransitionExpectation:
    """Host-reconstructed decision, diagnostics, and complete new state."""

    decision: SlotRoleDecisionSnapshot
    diagnostics: SlotRoleUpdateDiagnosticsSnapshot
    new_state: SlotRoleStateSnapshot


@dataclasses.dataclass(frozen=True)
class SlotRoleOracleValidation:
    """Fail-closed comparison result with field-addressable discrepancies."""

    valid: bool
    mismatches: tuple[str, ...]
    expected: SlotRoleTransitionExpectation | None


@dataclasses.dataclass(frozen=True)
class SlotRoleStateValidation:
    """Standalone snapshot/continuity validation result."""

    valid: bool
    mismatches: tuple[str, ...]


def _validate_role_stream(role: RoleName, input_stream: InputStreamName) -> None:
    expected: InputStreamName = (
        INPUT_PRIVATE_CUE if role == ROLE_HELPER else INPUT_DELIVERED_MESSAGE
    )
    if input_stream != expected:
        raise SlotLifecycleOracleInputError(
            f"role {role!r} requires input stream {expected!r}"
        )


def _validate_state(snapshot: SlotRoleStateSnapshot, config: SlotRoleOracleConfig) -> None:
    if snapshot.status[SCRATCH_SLOT] != SLOT_SCRATCH:
        raise SlotLifecycleOracleInputError("state.status[0] must be scratch")
    if any(status not in {SLOT_VACANT, SLOT_DURABLE} for status in snapshot.status[1:]):
        raise SlotLifecycleOracleInputError("durable slot status must be vacant or durable")
    if not 0 <= snapshot.active_slot < N_SLOTS:
        raise SlotLifecycleOracleInputError("state.active_slot is out of range")
    if (
        snapshot.active_slot > SCRATCH_SLOT
        and snapshot.status[snapshot.active_slot] != SLOT_DURABLE
    ):
        raise SlotLifecycleOracleInputError("state.active_slot cannot name a vacant durable")
    if not 0 <= snapshot.lease_offset < config.lease_length:
        raise SlotLifecycleOracleInputError("state.lease_offset is outside the lease")
    if not 0 <= snapshot.remaining_durable_tests <= N_DURABLE_SLOTS:
        raise SlotLifecycleOracleInputError("state.remaining_durable_tests is out of range")
    if not 1 <= snapshot.search_cursor <= N_DURABLE_SLOTS:
        raise SlotLifecycleOracleInputError("state.search_cursor is out of range")
    if snapshot.candidate_successful_leases < 0:
        raise SlotLifecycleOracleInputError("candidate_successful_leases cannot be negative")
    if not 1 <= snapshot.next_generation <= _INT32_MAX:
        raise SlotLifecycleOracleInputError("next_generation must be a positive int32")
    if any(value < 0 for value in snapshot.failed_leases):
        raise SlotLifecycleOracleInputError("failed_leases cannot be negative")
    if any(value < 0 for value in snapshot.idle_leases):
        raise SlotLifecycleOracleInputError("idle_leases cannot be negative")
    if any(value < 0 or value > float(_FLOAT32_MASS_LIMIT) for value in snapshot.relevance_mass):
        raise SlotLifecycleOracleInputError("relevance_mass lies outside the float32 count range")
    if any(value != float(np.floor(np.float32(value))) for value in snapshot.relevance_mass):
        raise SlotLifecycleOracleInputError("relevance_mass must be an integral float32 count")
    if snapshot.generation[SCRATCH_SLOT] != 0:
        raise SlotLifecycleOracleInputError("scratch generation must be zero")
    for index in range(1, N_SLOTS):
        if snapshot.status[index] == SLOT_VACANT and snapshot.generation[index] != 0:
            raise SlotLifecycleOracleInputError("vacant durable generation must be zero")
        if snapshot.status[index] == SLOT_DURABLE and snapshot.generation[index] <= 0:
            raise SlotLifecycleOracleInputError("occupied durable generation must be positive")


def _sat_increment(value: int) -> int:
    return value + 1 if value < _INT32_MAX else value


def _next_durable_slot(
    status: np.ndarray,
    cursor: int,
    excluded_slot: int,
) -> tuple[int, int, int]:
    candidates = [((cursor - 1 + offset) % N_DURABLE_SLOTS) + 1 for offset in range(3)]
    valid = [
        candidate
        for candidate in candidates
        if int(status[candidate]) == SLOT_DURABLE and candidate != excluded_slot
    ]
    if not valid:
        return SCRATCH_SLOT, cursor, 0
    selected = valid[0]
    return selected, (selected % N_DURABLE_SLOTS) + 1, len(valid)


def _expected_decision(
    old: SlotRoleStateSnapshot,
    private_input: int,
    epsilon: float,
) -> SlotRoleDecisionSnapshot:
    if not 0 <= private_input < N_SLOT_INPUTS:
        raise SlotLifecycleOracleInputError("decision.private_input is out of range")
    slot = old.active_slot
    row = np.asarray(old.values[slot][private_input], dtype=np.float32)
    key = _wrapped_key(old.key_data)
    next_key, explore_key, random_action_key, tie_key = jr.split(key, 4)
    random_action = int(
        np.asarray(
            jr.randint(random_action_key, (), 0, N_SLOT_ACTIONS, dtype=jnp.int32)
        )
    )
    tie_action = int(np.asarray(jr.randint(tie_key, (), 0, N_SLOT_ACTIONS, dtype=jnp.int32)))
    maximum = np.max(row)
    tied = row == maximum
    tie_order = (np.arange(N_SLOT_ACTIONS, dtype=np.int32) - tie_action) % N_SLOT_ACTIONS
    greedy_action = int(np.argmin(np.where(tied, tie_order, N_SLOT_ACTIONS)))
    explore = bool(
        np.asarray(jr.uniform(explore_key, (), dtype=jnp.float32) < np.float32(epsilon))
    )
    action = random_action if explore else greedy_action
    return SlotRoleDecisionSnapshot(
        slot=slot,
        private_input=private_input,
        action=action,
        selected_value=float(row[action]),
        next_key_data=_key_data(next_key, "expected.next_key"),
    )


def _state_from_arrays(
    *,
    values: np.ndarray,
    relevance_mean: np.ndarray,
    relevance_mass: np.ndarray,
    failed_leases: np.ndarray,
    idle_leases: np.ndarray,
    status: np.ndarray,
    generation: np.ndarray,
    active_slot: int,
    lease_offset: int,
    lease_reward_sum: np.float32,
    remaining_durable_tests: int,
    search_cursor: int,
    candidate_successful_leases: int,
    next_generation: int,
    key_data: KeyData,
) -> SlotRoleStateSnapshot:
    return SlotRoleStateSnapshot(
        values=_array_float_tensor3(
            values, (N_SLOTS, N_SLOT_INPUTS, N_SLOT_ACTIONS), "expected.values"
        ),
        relevance_mean=_array_float_vector(
            relevance_mean, (N_SLOTS,), "expected.relevance_mean"
        ),
        relevance_mass=_array_float_vector(
            relevance_mass, (N_SLOTS,), "expected.relevance_mass"
        ),
        failed_leases=_array_int_vector(
            failed_leases, (N_SLOTS,), "expected.failed_leases"
        ),
        idle_leases=_array_int_vector(idle_leases, (N_SLOTS,), "expected.idle_leases"),
        status=_array_int_vector(status, (N_SLOTS,), "expected.status"),
        generation=_array_int_vector(generation, (N_SLOTS,), "expected.generation"),
        active_slot=active_slot,
        lease_offset=lease_offset,
        lease_reward_sum=float(lease_reward_sum),
        remaining_durable_tests=remaining_durable_tests,
        search_cursor=search_cursor,
        candidate_successful_leases=candidate_successful_leases,
        next_generation=next_generation,
        key_data=key_data,
    )


def reconstruct_slot_role_transition(
    record: SlotRoleTransitionRecord,
) -> SlotRoleTransitionExpectation:
    """Reconstruct one transition without invoking the learner implementation."""

    if record.schema != SLOT_ROLE_TRANSITION_ORACLE_SCHEMA:
        raise SlotLifecycleOracleInputError("record.schema is not supported")
    if record.role not in {ROLE_HELPER, ROLE_BENEFICIARY}:
        raise SlotLifecycleOracleInputError("record.role is not supported")
    if record.input_stream not in {INPUT_PRIVATE_CUE, INPUT_DELIVERED_MESSAGE}:
        raise SlotLifecycleOracleInputError("record.input_stream is not supported")
    _validate_role_stream(record.role, record.input_stream)
    record.config.validate()
    _validate_state(record.old_state, record.config)
    if not math.isfinite(record.reward) or not np.isfinite(np.float32(record.reward)):
        raise SlotLifecycleOracleInputError("reward must be finite in float32")
    if type(record.external_value_write) is not bool or type(record.lifecycle_write) is not bool:
        raise SlotLifecycleOracleInputError("write permits must be booleans")
    if not 0 <= record.decision.action < N_SLOT_ACTIONS:
        raise SlotLifecycleOracleInputError("decision.action is out of range")

    config = record.config
    old = record.old_state
    expected_decision = _expected_decision(old, record.decision.private_input, config.epsilon)
    slot = expected_decision.slot
    private_input = expected_decision.private_input
    action = expected_decision.action
    reward = np.float32(record.reward)

    values = np.asarray(old.values, dtype=np.float32).copy()
    relevance_mean = np.asarray(old.relevance_mean, dtype=np.float32).copy()
    relevance_mass = np.asarray(old.relevance_mass, dtype=np.float32).copy()
    failed_leases = np.asarray(old.failed_leases, dtype=np.int32).copy()
    idle_leases = np.asarray(old.idle_leases, dtype=np.int32).copy()
    status = np.asarray(old.status, dtype=np.int32).copy()
    generation = np.asarray(old.generation, dtype=np.int32).copy()

    value_pre = np.float32(values[slot, private_input, action])
    candidate_value = np.float32(
        value_pre
        + np.float32(config.learning_rate) * np.float32(reward - value_pre)
    )
    is_scratch = slot == SCRATCH_SLOT
    is_durable = int(status[slot]) == SLOT_DURABLE
    ordinary_slot_writable = is_scratch or (
        config.durable_write_policy == DURABLE_WRITE_WRITABLE and is_durable
    )
    value_write = record.external_value_write and ordinary_slot_writable
    if value_write:
        values[slot, private_input, action] = candidate_value

    active_mass = np.float32(
        min(
            float(np.float32(relevance_mass[slot] + np.float32(1.0))),
            float(_FLOAT32_MASS_LIMIT),
        )
    )
    relevance_gain = np.float32(
        max(
            float(np.float32(config.relevance_rate)),
            float(np.float32(np.float32(1.0) / active_mass)),
        )
    )
    active_relevance = np.float32(
        relevance_mean[slot]
        + relevance_gain * np.float32(reward - relevance_mean[slot])
    )
    relevance_mean[slot] = active_relevance
    relevance_mass[slot] = active_mass
    lease_sum = np.float32(np.float32(old.lease_reward_sum) + reward)
    boundary = old.lease_offset == config.lease_length - 1
    lease_mean = np.float32(lease_sum / np.float32(config.lease_length))
    relevance_ready = bool(active_mass >= np.float32(config.confirmation_steps))
    durable_relevant = relevance_ready and bool(
        active_relevance >= np.float32(config.durable_retrieval_threshold)
    )
    candidate_relevant = relevance_ready and bool(
        active_relevance >= np.float32(config.candidate_confirmation_threshold)
    )

    if boundary:
        for durable_index in range(1, N_SLOTS):
            if int(status[durable_index]) == SLOT_DURABLE:
                idle_leases[durable_index] = _sat_increment(
                    int(idle_leases[durable_index])
                )
    boundary_success = boundary and is_durable and durable_relevant
    boundary_failure = boundary and is_durable and relevance_ready and not durable_relevant
    if boundary_success:
        failed_leases[slot] = 0
        idle_leases[slot] = 0
    elif boundary_failure:
        failed_leases[slot] = _sat_increment(int(failed_leases[slot]))

    candidate_lease_success = (
        boundary
        and is_scratch
        and candidate_relevant
        and bool(lease_mean >= np.float32(config.candidate_confirmation_threshold))
    )
    scratch_failure = (
        boundary and is_scratch and relevance_ready and not candidate_lease_success
    )
    scratch_failed_leases_pre = int(old.failed_leases[SCRATCH_SLOT])
    incremented_scratch_failed = _sat_increment(scratch_failed_leases_pre)
    scratch_retest_due = (
        scratch_failure
        and incremented_scratch_failed >= config.scratch_training_leases_before_retest
    )
    if candidate_lease_success:
        scratch_failed_after_evidence = 0
    elif scratch_failure:
        scratch_failed_after_evidence = 0 if scratch_retest_due else incremented_scratch_failed
    else:
        scratch_failed_after_evidence = scratch_failed_leases_pre
    failed_leases[SCRATCH_SLOT] = scratch_failed_after_evidence

    if candidate_lease_success:
        candidate_successful_leases = _sat_increment(old.candidate_successful_leases)
    elif scratch_failure:
        candidate_successful_leases = 0
    else:
        candidate_successful_leases = old.candidate_successful_leases
    candidate_confirmed = (
        candidate_lease_success
        and candidate_successful_leases >= config.candidate_confirmation_leases
    )

    vacant_indices = [
        index for index in range(1, N_SLOTS) if int(status[index]) == SLOT_VACANT
    ]
    has_vacancy = bool(vacant_indices)
    first_vacant = vacant_indices[0] if vacant_indices else 1
    lru_slot = int(np.argmax(idle_leases[1:])) + 1
    maximum_failures = int(np.max(failed_leases[1:]))
    evidence_idle = np.where(
        failed_leases[1:] == maximum_failures,
        idle_leases[1:],
        np.int32(-1),
    )
    evidence_slot = int(np.argmax(evidence_idle)) + 1
    full_bank_target = (
        lru_slot
        if config.replacement_target_policy == REPLACEMENT_TARGET_LRU
        else evidence_slot
    )
    target = first_vacant if has_vacancy else full_bank_target
    commit_requested = candidate_confirmed and record.lifecycle_write
    generation_available = old.next_generation < _INT32_MAX
    generation_exhausted = commit_requested and not generation_available
    commit = commit_requested and generation_available
    replace = commit and not has_vacancy
    replaced_generation = int(generation[target])

    if commit:
        values[target] = values[SCRATCH_SLOT]
        values[SCRATCH_SLOT] = np.zeros((N_SLOT_INPUTS, N_SLOT_ACTIONS), dtype=np.float32)
        relevance_mean[target] = relevance_mean[SCRATCH_SLOT]
        relevance_mean[SCRATCH_SLOT] = np.float32(0.0)
        relevance_mass[target] = relevance_mass[SCRATCH_SLOT]
        relevance_mass[SCRATCH_SLOT] = np.float32(0.0)
        failed_leases[target] = 0
        failed_leases[SCRATCH_SLOT] = 0
        idle_leases[target] = 0
        idle_leases[SCRATCH_SLOT] = 0
        status[target] = SLOT_DURABLE
        status[SCRATCH_SLOT] = SLOT_SCRATCH
        generation[target] = old.next_generation
        generation[SCRATCH_SLOT] = 0
        candidate_successful_leases = 0
        next_generation = _sat_increment(old.next_generation)
    else:
        next_generation = old.next_generation

    other_slot, other_cursor, other_count = _next_durable_slot(
        status, old.search_cursor, slot
    )
    any_slot, any_cursor, any_count = _next_durable_slot(
        status, old.search_cursor, -1
    )
    continuing_search = old.remaining_durable_tests > 0
    tests_after_current = max(old.remaining_durable_tests - 1, 0)
    can_continue = tests_after_current > 0 and other_count > 0
    continued_slot = other_slot if can_continue else SCRATCH_SLOT
    continued_remaining = min(tests_after_current, other_count) if can_continue else 0
    continued_cursor = other_cursor if can_continue else old.search_cursor
    can_start_other_search = other_count > 0
    started_other_slot = other_slot if can_start_other_search else SCRATCH_SLOT
    started_other_remaining = other_count if can_start_other_search else 0
    started_other_cursor = other_cursor if can_start_other_search else old.search_cursor
    if continuing_search:
        failed_durable_slot = continued_slot
        failed_durable_remaining = continued_remaining
        failed_durable_cursor = continued_cursor
    else:
        failed_durable_slot = started_other_slot
        failed_durable_remaining = started_other_remaining
        failed_durable_cursor = started_other_cursor
    can_start_any_search = any_count > 0
    failed_scratch_slot = any_slot if can_start_any_search else SCRATCH_SLOT
    failed_scratch_remaining = any_count if can_start_any_search else 0
    failed_scratch_cursor = any_cursor if can_start_any_search else old.search_cursor
    scratch_retest_started = scratch_retest_due and can_start_any_search

    if is_durable:
        failed_slot = failed_durable_slot
        failed_remaining = failed_durable_remaining
        failed_cursor = failed_durable_cursor
    else:
        failed_slot = failed_scratch_slot
        failed_remaining = failed_scratch_remaining
        failed_cursor = failed_scratch_cursor
    insufficient_evidence = not relevance_ready
    if is_scratch:
        keep_relevant_slot = candidate_lease_success or (
            scratch_failure and not scratch_retest_due
        )
    else:
        keep_relevant_slot = durable_relevant

    if commit:
        boundary_slot = target
        boundary_remaining = 0
        boundary_cursor = (target % N_DURABLE_SLOTS) + 1
    elif insufficient_evidence:
        boundary_slot = slot
        boundary_remaining = old.remaining_durable_tests
        boundary_cursor = old.search_cursor
    elif keep_relevant_slot:
        boundary_slot = slot
        boundary_remaining = 0
        boundary_cursor = (
            (slot % N_DURABLE_SLOTS) + 1 if is_durable else old.search_cursor
        )
    else:
        boundary_slot = failed_slot
        boundary_remaining = failed_remaining
        boundary_cursor = failed_cursor

    if boundary:
        active_slot = boundary_slot
        remaining_durable_tests = boundary_remaining
        search_cursor = boundary_cursor
        lease_offset = 0
        next_lease_sum = np.float32(0.0)
    else:
        active_slot = slot
        remaining_durable_tests = old.remaining_durable_tests
        search_cursor = old.search_cursor
        lease_offset = old.lease_offset + 1
        next_lease_sum = lease_sum

    exhausted_durable_search = boundary_failure and failed_durable_slot == SCRATCH_SLOT
    reset_scratch_failed = commit or boundary_success or exhausted_durable_search
    if reset_scratch_failed:
        failed_leases[SCRATCH_SLOT] = 0

    expected_state = _state_from_arrays(
        values=values,
        relevance_mean=relevance_mean,
        relevance_mass=relevance_mass,
        failed_leases=failed_leases,
        idle_leases=idle_leases,
        status=status,
        generation=generation,
        active_slot=active_slot,
        lease_offset=lease_offset,
        lease_reward_sum=next_lease_sum,
        remaining_durable_tests=remaining_durable_tests,
        search_cursor=search_cursor,
        candidate_successful_leases=candidate_successful_leases,
        next_generation=next_generation,
        key_data=expected_decision.next_key_data,
    )
    expected_diagnostics = SlotRoleUpdateDiagnosticsSnapshot(
        value_pre=float(value_pre),
        candidate_value=float(candidate_value),
        value_post=float(np.asarray(values[slot, private_input, action], dtype=np.float32)),
        value_write=value_write,
        lease_boundary=boundary,
        lease_reward_mean=float(lease_mean),
        relevance_ready=relevance_ready,
        durable_relevant=durable_relevant,
        candidate_relevant=candidate_relevant,
        candidate_lease_success=candidate_lease_success,
        scratch_failed_leases_pre=scratch_failed_leases_pre,
        scratch_failed_leases_post=int(failed_leases[SCRATCH_SLOT]),
        scratch_retest_started=scratch_retest_started,
        generation_exhausted=generation_exhausted,
        committed_slot=target if commit else -1,
        committed_generation=old.next_generation if commit else -1,
        retired_slot=target if replace else -1,
        retired_generation=replaced_generation if replace else -1,
    )
    return SlotRoleTransitionExpectation(
        decision=expected_decision,
        diagnostics=expected_diagnostics,
        new_state=expected_state,
    )


def _float_equal(left: float, right: float) -> bool:
    left_bits = np.asarray(np.float32(left)).view(np.uint32)
    right_bits = np.asarray(np.float32(right)).view(np.uint32)
    return bool(left_bits == right_bits)


def _compare_nested(
    expected: object,
    actual: object,
    path: str,
    mismatches: list[str],
) -> None:
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(expected) != len(actual):
            mismatches.append(path)
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _compare_nested(expected_item, actual_item, f"{path}[{index}]", mismatches)
    elif isinstance(expected, float):
        if not isinstance(actual, Real) or isinstance(actual, bool) or not _float_equal(
            expected, float(actual)
        ):
            mismatches.append(path)
    elif actual != expected:
        mismatches.append(path)


def _compare_dataclass(
    expected: object,
    actual: object,
    path: str,
    mismatches: list[str],
) -> None:
    if type(expected) is not type(actual) or not dataclasses.is_dataclass(expected):
        mismatches.append(path)
        return
    for field in dataclasses.fields(expected):
        expected_value = getattr(expected, field.name)
        actual_value = getattr(actual, field.name)
        child_path = f"{path}.{field.name}"
        if dataclasses.is_dataclass(expected_value):
            _compare_dataclass(expected_value, actual_value, child_path, mismatches)
        else:
            _compare_nested(expected_value, actual_value, child_path, mismatches)


def validate_slot_role_transition(
    record: SlotRoleTransitionRecord,
) -> SlotRoleOracleValidation:
    """Validate a record and enumerate every discrepant leaf field."""

    try:
        canonical = SlotRoleTransitionRecord.from_dict(record.to_dict())
        expected = reconstruct_slot_role_transition(canonical)
        _validate_state(canonical.new_state, canonical.config)
    except (
        IndexError,
        KeyError,
        OverflowError,
        SlotLifecycleOracleInputError,
        TypeError,
        ValueError,
    ) as error:
        return SlotRoleOracleValidation(
            valid=False,
            mismatches=(f"input: {error}",),
            expected=None,
        )
    mismatches: list[str] = []
    _compare_dataclass(expected.decision, canonical.decision, "decision", mismatches)
    _compare_dataclass(expected.diagnostics, canonical.diagnostics, "diagnostics", mismatches)
    _compare_dataclass(expected.new_state, canonical.new_state, "new_state", mismatches)
    return SlotRoleOracleValidation(
        valid=not mismatches,
        mismatches=tuple(mismatches),
        expected=expected,
    )


def validate_slot_role_state_snapshot(
    snapshot: SlotRoleStateSnapshot,
    config: SlotRoleOracleConfig,
) -> SlotRoleStateValidation:
    """Validate one snapshot directly, without requiring a transition trace."""

    try:
        canonical_config = SlotRoleOracleConfig.from_dict(config.to_dict())
        canonical_snapshot = SlotRoleStateSnapshot.from_dict(snapshot.to_dict())
        _validate_state(canonical_snapshot, canonical_config)
    except (
        IndexError,
        KeyError,
        OverflowError,
        SlotLifecycleOracleInputError,
        TypeError,
        ValueError,
    ) as error:
        return SlotRoleStateValidation(valid=False, mismatches=(f"input: {error}",))
    return SlotRoleStateValidation(valid=True, mismatches=())


def validate_slot_role_state_continuity(
    previous_new_state: SlotRoleStateSnapshot,
    current_old_state: SlotRoleStateSnapshot,
) -> SlotRoleStateValidation:
    """Require bit-exact continuity between adjacent records for one named role."""

    mismatches: list[str] = []
    _compare_dataclass(
        previous_new_state,
        current_old_state,
        "continuity",
        mismatches,
    )
    return SlotRoleStateValidation(valid=not mismatches, mismatches=tuple(mismatches))
