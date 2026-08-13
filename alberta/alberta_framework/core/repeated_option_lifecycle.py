# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Bounded repetition around the one-shot authorized option transaction.

``AuthorizedOptionReplacementController`` deliberately owns exactly one
retirement followed by exactly one replacement.  This opt-in L0 coordinator
reuses that complete transaction without weakening it: after a successful
replacement it rolls the *same* canonical scheduler/installer owner into a
fresh one-shot child and retains only bounded cycle lineage alongside it.

Every retirement is bound to a fresh caller-supplied Threefry cycle key and to
the two explicit reset keys already carried by
``OptionRetirementAuthorityReceipt``.  Replacement must present the same cycle
key through a new source/preparation-bound receipt.  Wrapper cycle indices,
checksums, and globally monotone authority revisions reject accidental stale
or cross-cycle reuse.  These bindings are unkeyed integrity declarations; they
do not authenticate a caller.

Declined replacement authority adopts only the one-shot controller's ordinary
incumbent/discovery advance.  The cold slot and active cycle key remain, and a
later retry requires a newly armed transition, preparation, and receipt.  No
proposal or receipt is persisted.  Successful replacement immediately creates
a fresh child around the successor scheduler state.  Thus there is always one
and only one persistent scheduler/installation/lifecycle subtree.

The expensive prepare/commit boundary remains host-orchestrated because the
underlying v1 controller re-derives complete discovery provenance on the host.
State validation and retirement adoption remain array-only and JIT/scan-safe.
This mechanism writes no outputs and owns no discovery, retirement,
replacement, go/no-go, safety, dispatch, evidence, promotion, or autonomous
curation authority.  It is ``not_assessed`` and makes no empirical or Alberta
Plan completion claim.
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
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.authorized_option_replacement import (
    AuthorizedOptionReplacementArm,
    AuthorizedOptionReplacementController,
    AuthorizedOptionReplacementPrepared,
    AuthorizedOptionReplacementResult,
    AuthorizedOptionReplacementRetirementResult,
    AuthorizedOptionReplacementStartResult,
    AuthorizedOptionReplacementState,
    AuthorizedOptionReplacementUpdateResult,
    OptionReplacementAuthorityReceipt,
)
from alberta_framework.core.authorized_option_retirement import (
    OptionRetirementAuthorityReceipt,
)
from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionLiveInputs,
    CumulantOptionMaterialization,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionInstallationAuthorityReceipt,
    CumulantOptionRetirementHandoff,
    CumulantOptionSchedulerArmInputs,
    CumulantOptionSchedulerObservation,
)
from alberta_framework.core.cumulant_subtask_discovery import (
    CumulantSubtaskProposalBundle,
)

REPEATED_OPTION_LIFECYCLE_CONFIG_SCHEMA = "alberta.repeated-option-lifecycle.config.v1"
REPEATED_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA = "alberta.repeated-option-lifecycle.state.v1"
REPEATED_OPTION_LIFECYCLE_ASSESSMENT = "not_assessed"
REPEATED_OPTION_LIFECYCLE_OUTPUT_WRITES = False
REPEATED_OPTION_LIFECYCLE_EVIDENCE_AUTHORITY = False
REPEATED_OPTION_LIFECYCLE_PROMOTION_AUTHORITY = False
REPEATED_OPTION_LIFECYCLE_SAFETY_AUTHORITY = False
REPEATED_OPTION_LIFECYCLE_GO_NO_GO_AUTHORITY = False
REPEATED_OPTION_LIFECYCLE_RETIREMENT_AUTHORITY = False
REPEATED_OPTION_LIFECYCLE_REPLACEMENT_AUTHORITY = False
REPEATED_OPTION_LIFECYCLE_DISCOVERY_AUTHORITY = False
REPEATED_OPTION_LIFECYCLE_DISPATCH_AUTHORITY = False
REPEATED_OPTION_LIFECYCLE_AUTONOMOUS_CURATION_AUTHORITY = False
REPEATED_OPTION_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED = False

REPEATED_OPTION_LIFECYCLE_ERROR_NONE = 0
REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY = 1

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _positive_int32(value: object, *, name: str) -> int:
    if type(value) is not int or not 1 <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be a positive exact Python int32")
    return value


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    expected = jnp.dtype(dtype)
    if jnp.dtype(array.dtype) != expected:
        raise TypeError(f"{name} must have dtype {expected}; got {array.dtype}")
    return array


def _require_threefry_key(value: object, *, name: str) -> Array:
    try:
        implementation = str(jr.key_impl(value))
        data = jr.key_data(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be one typed Threefry JAX key") from exc
    array = cast(Array, value)
    if (
        not jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
        or array.shape != ()
        or implementation != "threefry2x32"
        or data.shape != (2,)
        or data.dtype != jnp.uint32
    ):
        raise TypeError(f"{name} must be one typed Threefry JAX key")
    return array


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
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(right_leaves):
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
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        if array.dtype == jnp.float32:
            words = jax.lax.bitcast_convert_type(array, jnp.uint32).reshape((-1,))
        elif array.dtype == jnp.int32:
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


def _words_less(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _words_to_int(value: Array) -> int:
    host = np.asarray(jax.device_get(value), dtype=np.uint32)
    return (int(host[0]) << 32) | int(host[1])


@dataclasses.dataclass(frozen=True, slots=True)
class RepeatedOptionLifecycleConfig:
    """Exact finite number of one-retirement/one-replacement cycles."""

    max_cycles: int = 2

    SCHEMA_VERSION: ClassVar[str] = REPEATED_OPTION_LIFECYCLE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _positive_int32(self.max_cycles, name="max_cycles")

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_cycles": self.max_cycles,
            "cycle_contract": "one_authorized_retirement_then_one_authorized_replacement",
            "one_shot_child_schema": "preserved_v1",
            "persistent_owner_count": 1,
            "pending_proposal_slots": 0,
            "receipt_persistence": "none",
            "fresh_cycle_key_required": True,
            "host_prepare": True,
            "host_commit": True,
            "assessment": REPEATED_OPTION_LIFECYCLE_ASSESSMENT,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "safety_authority": False,
            "go_no_go_authority": False,
            "retirement_authority": False,
            "replacement_authority": False,
            "discovery_authority": False,
            "dispatch_authority": False,
            "autonomous_curation_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> RepeatedOptionLifecycleConfig:
        if type(value) is not dict:
            raise ValueError("repeated lifecycle config must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "max_cycles",
            "cycle_contract",
            "one_shot_child_schema",
            "persistent_owner_count",
            "pending_proposal_slots",
            "receipt_persistence",
            "fresh_cycle_key_required",
            "host_prepare",
            "host_commit",
            "assessment",
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "safety_authority",
            "go_no_go_authority",
            "retirement_authority",
            "replacement_authority",
            "discovery_authority",
            "dispatch_authority",
            "autonomous_curation_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("repeated lifecycle config keys differ from schema v1")
        fixed: dict[str, object] = {
            "schema_version": cls.SCHEMA_VERSION,
            "cycle_contract": "one_authorized_retirement_then_one_authorized_replacement",
            "one_shot_child_schema": "preserved_v1",
            "persistent_owner_count": 1,
            "pending_proposal_slots": 0,
            "receipt_persistence": "none",
            "fresh_cycle_key_required": True,
            "host_prepare": True,
            "host_commit": True,
            "assessment": REPEATED_OPTION_LIFECYCLE_ASSESSMENT,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "safety_authority": False,
            "go_no_go_authority": False,
            "retirement_authority": False,
            "replacement_authority": False,
            "discovery_authority": False,
            "dispatch_authority": False,
            "autonomous_curation_authority": False,
            "scientific_promotion_allowed": False,
        }
        for name, expected_value in fixed.items():
            if raw.pop(name) != expected_value:
                raise ValueError(f"repeated lifecycle config {name} differs")
        return cls(max_cycles=cast(int, raw.pop("max_cycles")))


@chex.dataclass(frozen=True)
class RepeatedOptionLifecycleState:
    """One v1 child owner plus bounded global cycle lineage."""

    cycle_state: AuthorizedOptionReplacementState
    completed_cycles: Int[Array, ""]
    total_retirements: Int[Array, ""]
    total_replacements: Int[Array, ""]
    cycle_key_active: Bool[Array, ""]
    active_cycle_key_data: UInt[Array, " 2"]
    has_completed_cycle: Bool[Array, ""]
    cycle_key_history: UInt[Array, "max_cycles 2"]
    last_completed_cycle_key_data: UInt[Array, " 2"]
    last_retirement_authority_revision_words: UInt[Array, " 2"]
    last_replacement_authority_revision_words: UInt[Array, " 2"]
    revision: Int[Array, ""]
    unavailable: Bool[Array, ""]
    error: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class RepeatedOptionRetirementAuthorityReceipt:
    """Fresh wrapper binding for one exact v1 retirement receipt."""

    retirement_authority: OptionRetirementAuthorityReceipt
    cycle_index: Int[Array, ""]
    source_state_checksum: UInt[Array, " 2"]
    source_child_checksum: UInt[Array, " 2"]
    source_revision: Int[Array, ""]
    cycle_key_data: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class RepeatedOptionReplacementAuthorityReceipt:
    """Fresh wrapper binding for one exact v1 replacement preparation."""

    replacement_authority: OptionReplacementAuthorityReceipt
    cycle_index: Int[Array, ""]
    source_state_checksum: UInt[Array, " 2"]
    source_child_checksum: UInt[Array, " 2"]
    source_revision: Int[Array, ""]
    cycle_key_data: UInt[Array, " 2"]
    prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class RepeatedOptionLifecycleArm:
    """One child arm bound to the exact active wrapper cycle."""

    replacement_arm: AuthorizedOptionReplacementArm
    cycle_index: Int[Array, ""]
    source_state_checksum: UInt[Array, " 2"]
    source_revision: Int[Array, ""]
    cycle_key_data: UInt[Array, " 2"]
    available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class RepeatedOptionLifecyclePrepared:
    """Transient wrapper binding; never included in persistent state."""

    replacement_prepared: AuthorizedOptionReplacementPrepared
    cycle_index: Int[Array, ""]
    source_state_checksum: UInt[Array, " 2"]
    source_child_checksum: UInt[Array, " 2"]
    source_revision: Int[Array, ""]
    cycle_key_data: UInt[Array, " 2"]
    prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class RepeatedOptionLifecycleRetirementDiagnostics:
    """Wrapper-cycle and unchanged child retirement facts."""

    source_state_valid: Bool[Array, ""]
    receipt_binding_valid: Bool[Array, ""]
    cycle_capacity_available: Bool[Array, ""]
    fresh_cycle_key: Bool[Array, ""]
    global_authority_revision_fresh: Bool[Array, ""]
    child_transaction_applied: Bool[Array, ""]
    wrapper_transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class RepeatedOptionLifecycleRetirementResult:
    """Atomic wrapper result for one retirement half-cycle."""

    state: RepeatedOptionLifecycleState
    retirement: AuthorizedOptionReplacementRetirementResult
    diagnostics: RepeatedOptionLifecycleRetirementDiagnostics


@chex.dataclass(frozen=True)
class RepeatedOptionLifecycleCommitDiagnostics:
    """Host replacement, retry, rollover, and exhaustion facts."""

    source_state_valid: Bool[Array, ""]
    prepared_binding_valid: Bool[Array, ""]
    receipt_binding_valid: Bool[Array, ""]
    global_authority_revision_fresh: Bool[Array, ""]
    child_ordinary_advance_applied: Bool[Array, ""]
    child_replacement_applied: Bool[Array, ""]
    ordinary_advance_adopted: Bool[Array, ""]
    cycle_completed: Bool[Array, ""]
    rollover_state_valid: Bool[Array, ""]
    capacity_exhausted: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class RepeatedOptionLifecycleCommitResult:
    """Host result; ``replacement`` is absent on wrapper-integrity refusal."""

    state: RepeatedOptionLifecycleState
    replacement: AuthorizedOptionReplacementResult | None
    diagnostics: RepeatedOptionLifecycleCommitDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class RepeatedOptionLifecycleStartResult:
    """Masked control start reprojected into the sole child owner."""

    state: RepeatedOptionLifecycleState
    replacement: AuthorizedOptionReplacementStartResult | None
    applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class RepeatedOptionLifecycleUpdateResult:
    """Masked control update reprojected into the sole child owner."""

    state: RepeatedOptionLifecycleState
    replacement: AuthorizedOptionReplacementUpdateResult | None
    applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class RepeatedOptionLifecycleResourceBudget:
    """Exact persistence and finite cycle/work accounting."""

    persistent_state_nbytes: int
    child_state_nbytes: int
    scheduler_state_nbytes: int
    installation_state_nbytes: int
    wrapper_metadata_nbytes: int
    duplicated_scheduler_state_nbytes: int
    duplicated_installation_state_nbytes: int
    prepared_state_nbytes: int
    persistent_lifecycle_owner_count: int
    pending_proposal_slots: int
    persisted_receipt_count: int
    max_cycles: int
    completed_cycles: int
    remaining_cycles: int
    active_cycle: bool
    total_retirements: int
    total_replacements: int
    installation_capacity_remaining: int
    scheduler_attempt_capacity_remaining: int
    scheduler_step_capacity_remaining: int
    underlying_capacity_supports_remaining_cycles: bool
    max_retirements_per_cycle: int
    max_replacements_per_cycle: int
    max_reset_keys_per_cycle: int
    max_cycle_keys_per_cycle: int
    max_scheduler_observations_per_attempt: int
    host_prepare: bool
    host_commit: bool
    assessment: str
    output_writes: bool
    evidence_authority: bool
    promotion_authority: bool
    safety_authority: bool
    go_no_go_authority: bool
    retirement_authority: bool
    replacement_authority: bool
    discovery_authority: bool
    dispatch_authority: bool
    autonomous_curation_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str


class RepeatedOptionLifecycle:
    """Finite repeated coordinator around an unchanged v1 child transaction."""

    def __init__(
        self,
        replacement: AuthorizedOptionReplacementController,
        config: RepeatedOptionLifecycleConfig | None = None,
    ) -> None:
        if type(replacement) is not AuthorizedOptionReplacementController:
            raise TypeError("replacement must be an exact AuthorizedOptionReplacementController")
        if replacement.config.max_replacements != 1:
            raise ValueError("repeated lifecycle requires the exact one-shot v1 child")
        if replacement.retirement.config.max_retirements < 1:
            raise ValueError("repeated lifecycle requires one retirement per child")
        self._replacement = replacement
        self._config = config or RepeatedOptionLifecycleConfig()

    @property
    def config(self) -> RepeatedOptionLifecycleConfig:
        return self._config

    @property
    def replacement(self) -> AuthorizedOptionReplacementController:
        return self._replacement

    def to_config(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(
                json.dumps(
                    {
                        "schema_version": REPEATED_OPTION_LIFECYCLE_CONFIG_SCHEMA,
                        "lifecycle": self._config.to_config(),
                        "replacement": self._replacement.to_config(),
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
        )

    def _payload_arrays(self, state: RepeatedOptionLifecycleState) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                tuple(
                    getattr(state, field.name)
                    for field in dataclasses.fields(RepeatedOptionLifecycleState)
                    if field.name != "binding_checksum"
                )
            )
        )

    def _with_checksum(
        self,
        state: RepeatedOptionLifecycleState,
    ) -> RepeatedOptionLifecycleState:
        return dataclasses.replace(
            state,
            binding_checksum=_checksum_arrays(self._payload_arrays(state)),
        )

    def _check_state_contract(self, state: RepeatedOptionLifecycleState) -> None:
        if type(state) is not RepeatedOptionLifecycleState:
            raise TypeError("state must be an exact RepeatedOptionLifecycleState")
        self._replacement._check_state_contract(state.cycle_state)
        contracts = (
            (state.completed_cycles, "completed_cycles", (), jnp.int32),
            (state.total_retirements, "total_retirements", (), jnp.int32),
            (state.total_replacements, "total_replacements", (), jnp.int32),
            (state.cycle_key_active, "cycle_key_active", (), jnp.bool_),
            (state.active_cycle_key_data, "active_cycle_key_data", (2,), jnp.uint32),
            (state.has_completed_cycle, "has_completed_cycle", (), jnp.bool_),
            (
                state.cycle_key_history,
                "cycle_key_history",
                (self._config.max_cycles, 2),
                jnp.uint32,
            ),
            (
                state.last_completed_cycle_key_data,
                "last_completed_cycle_key_data",
                (2,),
                jnp.uint32,
            ),
            (
                state.last_retirement_authority_revision_words,
                "last_retirement_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (
                state.last_replacement_authority_revision_words,
                "last_replacement_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (state.revision, "revision", (), jnp.int32),
            (state.unavailable, "unavailable", (), jnp.bool_),
            (state.error, "error", (), jnp.int32),
            (state.binding_checksum, "binding_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def init(
        self,
        cycle_state: AuthorizedOptionReplacementState,
    ) -> RepeatedOptionLifecycleState:
        """Adopt one pristine v1 child without copying its owner subtree."""

        self._replacement._check_state_contract(cycle_state)
        child_valid = self._replacement.state_valid(cycle_state)
        fresh = (
            jnp.all(cycle_state.retirement_words == 0)
            & jnp.all(cycle_state.replacement_words == 0)
            & jnp.all(cycle_state.installed_slot_mask)
        )
        if not bool(jax.device_get(child_valid & fresh)):
            raise ValueError("initial repeated lifecycle child must be valid and pristine")
        state = RepeatedOptionLifecycleState(
            cycle_state=cycle_state,
            completed_cycles=jnp.asarray(0, dtype=jnp.int32),
            total_retirements=jnp.asarray(0, dtype=jnp.int32),
            total_replacements=jnp.asarray(0, dtype=jnp.int32),
            cycle_key_active=jnp.asarray(False, dtype=jnp.bool_),
            active_cycle_key_data=jnp.zeros((2,), dtype=jnp.uint32),
            has_completed_cycle=jnp.asarray(False, dtype=jnp.bool_),
            cycle_key_history=jnp.zeros((self._config.max_cycles, 2), dtype=jnp.uint32),
            last_completed_cycle_key_data=jnp.zeros((2,), dtype=jnp.uint32),
            last_retirement_authority_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            last_replacement_authority_revision_words=jnp.zeros((2,), dtype=jnp.uint32),
            revision=jnp.asarray(0, dtype=jnp.int32),
            unavailable=jnp.asarray(False, dtype=jnp.bool_),
            error=jnp.asarray(REPEATED_OPTION_LIFECYCLE_ERROR_NONE, dtype=jnp.int32),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        state = self._with_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initialized repeated lifecycle state failed its contract")
        return state

    def state_valid(self, state: RepeatedOptionLifecycleState) -> Bool[Array, ""]:
        """Validate the one-owner phase, global clocks, cap, and checksum."""

        self._check_state_contract(state)
        child = state.cycle_state
        zero_retirements = jnp.all(child.retirement_words == 0)
        one_retirement = jnp.array_equal(
            child.retirement_words,
            jnp.asarray((0, 1), dtype=jnp.uint32),
        )
        zero_replacements = jnp.all(child.replacement_words == 0)
        cold_count = jnp.sum(~child.installed_slot_mask, dtype=jnp.int32)
        fresh_phase = zero_retirements & zero_replacements & (cold_count == 0)
        retired_phase = one_retirement & zero_replacements & (cold_count == 1)
        exhausted = state.completed_cycles == self._config.max_cycles
        expected_error = jnp.where(
            exhausted,
            jnp.asarray(REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY, dtype=jnp.int32),
            jnp.asarray(REPEATED_OPTION_LIFECYCLE_ERROR_NONE, dtype=jnp.int32),
        )
        counts_valid = (
            (state.completed_cycles >= 0)
            & (state.completed_cycles <= self._config.max_cycles)
            & (state.total_replacements == state.completed_cycles)
            & (
                state.total_retirements
                == state.completed_cycles + state.cycle_key_active.astype(jnp.int32)
            )
            & (state.total_retirements <= self._config.max_cycles)
        )
        retirement_revision_bound = jnp.where(
            state.total_retirements == 0,
            jnp.all(state.last_retirement_authority_revision_words == 0),
            jnp.any(state.last_retirement_authority_revision_words != 0),
        )
        replacement_revision_bound = jnp.where(
            state.total_replacements == 0,
            jnp.all(state.last_replacement_authority_revision_words == 0),
            jnp.any(state.last_replacement_authority_revision_words != 0),
        )
        key_history_bound = jnp.where(
            state.has_completed_cycle,
            jnp.array_equal(
                state.last_completed_cycle_key_data,
                state.cycle_key_history[
                    jnp.clip(state.completed_cycles - 1, 0, self._config.max_cycles - 1)
                ],
            ),
            (state.completed_cycles == 0) & jnp.all(state.last_completed_cycle_key_data == 0),
        )
        cycle_indices = jnp.arange(self._config.max_cycles, dtype=jnp.int32)
        used_cycle_keys = cycle_indices < state.completed_cycles
        pairwise_key_equality = jnp.all(
            state.cycle_key_history[:, None, :] == state.cycle_key_history[None, :, :],
            axis=2,
        )
        duplicate_used_key = jnp.any(
            pairwise_key_equality
            & used_cycle_keys[:, None]
            & used_cycle_keys[None, :]
            & (~jnp.eye(self._config.max_cycles, dtype=jnp.bool_))
        )
        history_tail_zero = jnp.all(
            jnp.where(
                used_cycle_keys[:, None],
                jnp.zeros_like(state.cycle_key_history),
                state.cycle_key_history,
            )
            == 0
        )
        active_key_bound = jnp.where(
            state.cycle_key_active,
            retired_phase,
            fresh_phase & jnp.all(state.active_cycle_key_data == 0),
        )
        active_key_is_fresh = (~state.cycle_key_active) | (
            ~jnp.any(
                used_cycle_keys
                & jnp.all(
                    state.cycle_key_history == state.active_cycle_key_data[None, :],
                    axis=1,
                )
            )
        )
        return (
            self._replacement.state_valid(child)
            & (fresh_phase | retired_phase)
            & counts_valid
            & active_key_bound
            & active_key_is_fresh
            & key_history_bound
            & (~duplicate_used_key)
            & history_tail_zero
            & retirement_revision_bound
            & replacement_revision_bound
            & (state.has_completed_cycle == (state.completed_cycles > 0))
            & (state.revision >= state.total_retirements + state.total_replacements)
            & (state.revision >= 0)
            & (state.unavailable == exhausted)
            & (state.error == expected_error)
            & ((~exhausted) | fresh_phase)
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def _cycle_key_fresh(self, state: RepeatedOptionLifecycleState, key_data: Array) -> Array:
        used_cycle_keys = (
            jnp.arange(self._config.max_cycles, dtype=jnp.int32) < state.completed_cycles
        )
        already_used = jnp.any(
            used_cycle_keys & jnp.all(state.cycle_key_history == key_data[None, :], axis=1)
        )
        return ~already_used

    def _check_retirement_receipt_contract(
        self,
        receipt: RepeatedOptionRetirementAuthorityReceipt,
    ) -> None:
        if type(receipt) is not RepeatedOptionRetirementAuthorityReceipt:
            raise TypeError(
                "authority_receipt must be an exact RepeatedOptionRetirementAuthorityReceipt"
            )
        self._replacement.retirement._check_receipt_contract(receipt.retirement_authority)
        contracts = (
            (receipt.cycle_index, "cycle_index", (), jnp.int32),
            (receipt.source_state_checksum, "source_state_checksum", (2,), jnp.uint32),
            (receipt.source_child_checksum, "source_child_checksum", (2,), jnp.uint32),
            (receipt.source_revision, "source_revision", (), jnp.int32),
            (receipt.cycle_key_data, "cycle_key_data", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"authority_receipt.{name}", shape=shape, dtype=dtype)

    def retirement_authority_receipt(
        self,
        state: RepeatedOptionLifecycleState,
        retirement_authority: OptionRetirementAuthorityReceipt,
        cycle_key: Array,
    ) -> RepeatedOptionRetirementAuthorityReceipt:
        """Bind a caller's exact v1 retirement receipt to one fresh cycle key."""

        self._check_state_contract(state)
        self._replacement.retirement._check_receipt_contract(retirement_authority)
        key = _require_threefry_key(cycle_key, name="cycle_key")
        ready = (
            self.state_valid(state)
            & (~state.unavailable)
            & (~state.cycle_key_active)
            & self._cycle_key_fresh(state, jr.key_data(key))
            & _words_less(
                state.last_retirement_authority_revision_words,
                retirement_authority.authority_revision_words,
            )
        )
        if not bool(jax.device_get(ready)):
            raise ValueError("retirement receipt is stale, cross-cycle, or capacity-exhausted")
        return RepeatedOptionRetirementAuthorityReceipt(
            retirement_authority=retirement_authority,
            cycle_index=state.completed_cycles,
            source_state_checksum=state.binding_checksum,
            source_child_checksum=state.cycle_state.binding_checksum,
            source_revision=state.revision,
            cycle_key_data=jr.key_data(key),
        )

    def retire(
        self,
        state: RepeatedOptionLifecycleState,
        handoff: CumulantOptionRetirementHandoff,
        authority_receipt: RepeatedOptionRetirementAuthorityReceipt,
        cycle_key: Array,
        phase_one_key: Array,
        phase_two_key: Array,
    ) -> RepeatedOptionLifecycleRetirementResult:
        """Apply one fresh-key-bound child retirement or a whole-state no-op."""

        self._check_state_contract(state)
        self._check_retirement_receipt_contract(authority_receipt)
        key = _require_threefry_key(cycle_key, name="cycle_key")
        _require_threefry_key(phase_one_key, name="phase_one_key")
        _require_threefry_key(phase_two_key, name="phase_two_key")
        source_valid = self.state_valid(state)
        capacity = (~state.unavailable) & (state.completed_cycles < self._config.max_cycles)
        fresh_key = self._cycle_key_fresh(state, jr.key_data(key))
        revision_fresh = _words_less(
            state.last_retirement_authority_revision_words,
            authority_receipt.retirement_authority.authority_revision_words,
        )
        receipt_valid = (
            (authority_receipt.cycle_index == state.completed_cycles)
            & jnp.array_equal(authority_receipt.source_state_checksum, state.binding_checksum)
            & jnp.array_equal(
                authority_receipt.source_child_checksum,
                state.cycle_state.binding_checksum,
            )
            & (authority_receipt.source_revision == state.revision)
            & jnp.array_equal(authority_receipt.cycle_key_data, jr.key_data(key))
            & (~state.cycle_key_active)
        )
        child = self._replacement.retire(
            state.cycle_state,
            handoff,
            authority_receipt.retirement_authority,
            phase_one_key,
            phase_two_key,
        )
        preconditions = source_valid & capacity & fresh_key & revision_fresh & receipt_valid
        proposed = self._with_checksum(
            dataclasses.replace(
                state,
                cycle_state=child.state,
                total_retirements=state.total_retirements + jnp.int32(1),
                cycle_key_active=jnp.asarray(True, dtype=jnp.bool_),
                active_cycle_key_data=jr.key_data(key),
                last_retirement_authority_revision_words=(
                    authority_receipt.retirement_authority.authority_revision_words
                ),
                revision=state.revision + jnp.int32(1),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        applied = (
            preconditions
            & child.transaction_applied
            & (state.revision < _INT32_MAX)
            & self.state_valid(proposed)
        )
        next_state = cast(
            RepeatedOptionLifecycleState,
            jax.lax.cond(applied, lambda _: proposed, lambda _: state, None),
        )
        return RepeatedOptionLifecycleRetirementResult(
            state=next_state,
            retirement=child,
            diagnostics=RepeatedOptionLifecycleRetirementDiagnostics(
                source_state_valid=source_valid,
                receipt_binding_valid=receipt_valid,
                cycle_capacity_available=capacity,
                fresh_cycle_key=fresh_key,
                global_authority_revision_fresh=revision_fresh,
                child_transaction_applied=child.transaction_applied,
                wrapper_transaction_applied=applied,
            ),
        )

    def extended_action_mask(
        self,
        state: RepeatedOptionLifecycleState,
    ) -> Bool[Array, " n_total_actions"]:
        """Delegate the authoritative live/cold mask to the sole child."""

        self._check_state_contract(state)
        return self._replacement.extended_action_mask(state.cycle_state)

    def arm(
        self,
        state: RepeatedOptionLifecycleState,
        inputs: CumulantOptionSchedulerArmInputs,
    ) -> RepeatedOptionLifecycleArm:
        """Arm a fresh replacement attempt only in the active retired phase."""

        self._check_state_contract(state)
        child_arm = self._replacement.arm(state.cycle_state, inputs)
        available = (
            self.state_valid(state)
            & state.cycle_key_active
            & (~state.unavailable)
            & child_arm.available
        )
        child_arm = dataclasses.replace(child_arm, available=available)
        return RepeatedOptionLifecycleArm(
            replacement_arm=child_arm,
            cycle_index=state.completed_cycles,
            source_state_checksum=state.binding_checksum,
            source_revision=state.revision,
            cycle_key_data=state.active_cycle_key_data,
            available=available,
        )

    def _check_arm_contract(self, arm: RepeatedOptionLifecycleArm) -> None:
        if type(arm) is not RepeatedOptionLifecycleArm:
            raise TypeError("arm must be an exact RepeatedOptionLifecycleArm")
        if type(arm.replacement_arm) is not AuthorizedOptionReplacementArm:
            raise TypeError("arm.replacement_arm must be exact")
        contracts = (
            (arm.cycle_index, "cycle_index", (), jnp.int32),
            (arm.source_state_checksum, "source_state_checksum", (2,), jnp.uint32),
            (arm.source_revision, "source_revision", (), jnp.int32),
            (arm.cycle_key_data, "cycle_key_data", (2,), jnp.uint32),
            (arm.available, "available", (), jnp.bool_),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"arm.{name}", shape=shape, dtype=dtype)

    def _prepared_payload_arrays(
        self,
        prepared: RepeatedOptionLifecyclePrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    prepared.replacement_prepared,
                    prepared.cycle_index,
                    prepared.source_state_checksum,
                    prepared.source_child_checksum,
                    prepared.source_revision,
                    prepared.cycle_key_data,
                )
            )
        )

    def _with_prepared_checksum(
        self,
        prepared: RepeatedOptionLifecyclePrepared,
    ) -> RepeatedOptionLifecyclePrepared:
        return dataclasses.replace(
            prepared,
            prepared_checksum=_checksum_arrays(self._prepared_payload_arrays(prepared)),
        )

    def prepare(
        self,
        state: RepeatedOptionLifecycleState,
        arm: RepeatedOptionLifecycleArm,
        observation: CumulantOptionSchedulerObservation,
        live_inputs: CumulantOptionLiveInputs,
    ) -> RepeatedOptionLifecyclePrepared:
        """Produce one transient child candidate bound to the active cycle."""

        self._check_state_contract(state)
        self._check_arm_contract(arm)
        arm_valid = (
            arm.available
            & (arm.cycle_index == state.completed_cycles)
            & jnp.array_equal(arm.source_state_checksum, state.binding_checksum)
            & (arm.source_revision == state.revision)
            & jnp.array_equal(arm.cycle_key_data, state.active_cycle_key_data)
        )
        bound_child_arm = dataclasses.replace(
            arm.replacement_arm,
            available=arm.replacement_arm.available & arm_valid,
        )
        child = self._replacement.prepare(
            state.cycle_state,
            bound_child_arm,
            observation,
            live_inputs,
        )
        prepared = RepeatedOptionLifecyclePrepared(
            replacement_prepared=child,
            cycle_index=state.completed_cycles,
            source_state_checksum=state.binding_checksum,
            source_child_checksum=state.cycle_state.binding_checksum,
            source_revision=state.revision,
            cycle_key_data=state.active_cycle_key_data,
            prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_prepared_checksum(prepared)

    def _check_prepared_contract(self, prepared: RepeatedOptionLifecyclePrepared) -> None:
        if type(prepared) is not RepeatedOptionLifecyclePrepared:
            raise TypeError("prepared must be an exact RepeatedOptionLifecyclePrepared")
        if type(prepared.replacement_prepared) is not AuthorizedOptionReplacementPrepared:
            raise TypeError("prepared.replacement_prepared must be exact")
        contracts = (
            (prepared.cycle_index, "cycle_index", (), jnp.int32),
            (prepared.source_state_checksum, "source_state_checksum", (2,), jnp.uint32),
            (prepared.source_child_checksum, "source_child_checksum", (2,), jnp.uint32),
            (prepared.source_revision, "source_revision", (), jnp.int32),
            (prepared.cycle_key_data, "cycle_key_data", (2,), jnp.uint32),
            (prepared.prepared_checksum, "prepared_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"prepared.{name}", shape=shape, dtype=dtype)

    def _prepared_binding_valid(
        self,
        state: RepeatedOptionLifecycleState,
        prepared: RepeatedOptionLifecyclePrepared,
    ) -> Array:
        return (
            self.state_valid(state)
            & state.cycle_key_active
            & (~state.unavailable)
            & (prepared.cycle_index == state.completed_cycles)
            & jnp.array_equal(prepared.source_state_checksum, state.binding_checksum)
            & jnp.array_equal(
                prepared.source_child_checksum,
                state.cycle_state.binding_checksum,
            )
            & (prepared.source_revision == state.revision)
            & jnp.array_equal(prepared.cycle_key_data, state.active_cycle_key_data)
            & jnp.array_equal(
                prepared.prepared_checksum,
                _checksum_arrays(self._prepared_payload_arrays(prepared)),
            )
            & _tree_array_equal(
                prepared.replacement_prepared.source_state,
                state.cycle_state,
            )
        )

    def _check_replacement_receipt_contract(
        self,
        receipt: RepeatedOptionReplacementAuthorityReceipt,
    ) -> None:
        if type(receipt) is not RepeatedOptionReplacementAuthorityReceipt:
            raise TypeError(
                "authority_receipt must be an exact RepeatedOptionReplacementAuthorityReceipt"
            )
        self._replacement._check_receipt_contract(receipt.replacement_authority)
        contracts = (
            (receipt.cycle_index, "cycle_index", (), jnp.int32),
            (receipt.source_state_checksum, "source_state_checksum", (2,), jnp.uint32),
            (receipt.source_child_checksum, "source_child_checksum", (2,), jnp.uint32),
            (receipt.source_revision, "source_revision", (), jnp.int32),
            (receipt.cycle_key_data, "cycle_key_data", (2,), jnp.uint32),
            (receipt.prepared_checksum, "prepared_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"authority_receipt.{name}", shape=shape, dtype=dtype)

    def replacement_authority_receipt(
        self,
        state: RepeatedOptionLifecycleState,
        prepared: RepeatedOptionLifecyclePrepared,
        installation_authority: CumulantOptionInstallationAuthorityReceipt,
        cycle_key: Array,
        *,
        replacement_authorized: bool | Array,
    ) -> RepeatedOptionReplacementAuthorityReceipt:
        """Create a new preparation- and cycle-bound replacement receipt."""

        self._check_state_contract(state)
        self._check_prepared_contract(prepared)
        key = _require_threefry_key(cycle_key, name="cycle_key")
        prepared_valid = self._prepared_binding_valid(state, prepared)
        key_valid = jnp.array_equal(jr.key_data(key), state.active_cycle_key_data)
        if not bool(jax.device_get(prepared_valid & key_valid)):
            raise ValueError("replacement preparation is stale or cross-cycle")
        nested = self._replacement.authority_receipt(
            prepared.replacement_prepared,
            installation_authority,
            replacement_authorized=replacement_authorized,
        )
        revision_fresh = _words_less(
            state.last_replacement_authority_revision_words,
            nested.installation_authority.authority_revision_words,
        )
        if not bool(jax.device_get(revision_fresh)):
            raise ValueError("replacement authority revision is not globally fresh")
        return RepeatedOptionReplacementAuthorityReceipt(
            replacement_authority=nested,
            cycle_index=state.completed_cycles,
            source_state_checksum=state.binding_checksum,
            source_child_checksum=state.cycle_state.binding_checksum,
            source_revision=state.revision,
            cycle_key_data=jr.key_data(key),
            prepared_checksum=prepared.prepared_checksum,
        )

    @staticmethod
    def _false_commit_diagnostics(
        *,
        source_valid: Array,
        prepared_valid: Array,
        receipt_valid: Array,
        revision_fresh: Array,
        exhausted: Array,
    ) -> RepeatedOptionLifecycleCommitDiagnostics:
        false = jnp.asarray(False, dtype=jnp.bool_)
        return RepeatedOptionLifecycleCommitDiagnostics(
            source_state_valid=source_valid,
            prepared_binding_valid=prepared_valid,
            receipt_binding_valid=receipt_valid,
            global_authority_revision_fresh=revision_fresh,
            child_ordinary_advance_applied=false,
            child_replacement_applied=false,
            ordinary_advance_adopted=false,
            cycle_completed=false,
            rollover_state_valid=false,
            capacity_exhausted=exhausted,
        )

    def commit(
        self,
        state: RepeatedOptionLifecycleState,
        prepared: RepeatedOptionLifecyclePrepared,
        authority_receipt: RepeatedOptionReplacementAuthorityReceipt,
        cycle_key: Array,
    ) -> RepeatedOptionLifecycleCommitResult:
        """Adopt one declined ordinary advance or complete and roll the cycle."""

        self._check_state_contract(state)
        self._check_prepared_contract(prepared)
        self._check_replacement_receipt_contract(authority_receipt)
        key = _require_threefry_key(cycle_key, name="cycle_key")
        source_valid = self.state_valid(state)
        prepared_valid = self._prepared_binding_valid(state, prepared)
        revision_fresh = _words_less(
            state.last_replacement_authority_revision_words,
            authority_receipt.replacement_authority.installation_authority.authority_revision_words,
        )
        receipt_valid = (
            (authority_receipt.cycle_index == state.completed_cycles)
            & jnp.array_equal(authority_receipt.source_state_checksum, state.binding_checksum)
            & jnp.array_equal(
                authority_receipt.source_child_checksum,
                state.cycle_state.binding_checksum,
            )
            & (authority_receipt.source_revision == state.revision)
            & jnp.array_equal(authority_receipt.cycle_key_data, jr.key_data(key))
            & jnp.array_equal(authority_receipt.cycle_key_data, state.active_cycle_key_data)
            & jnp.array_equal(
                authority_receipt.prepared_checksum,
                prepared.prepared_checksum,
            )
            & _tree_array_equal(
                authority_receipt.replacement_authority,
                self._replacement.authority_receipt(
                    prepared.replacement_prepared,
                    authority_receipt.replacement_authority.installation_authority,
                    replacement_authorized=(
                        authority_receipt.replacement_authority.replacement_authorized
                    ),
                ),
            )
        )
        wrapper_valid = (
            source_valid
            & prepared_valid
            & receipt_valid
            & revision_fresh
            & state.cycle_key_active
            & (~state.unavailable)
        )
        if not bool(jax.device_get(wrapper_valid)):
            return RepeatedOptionLifecycleCommitResult(
                state=state,
                replacement=None,
                diagnostics=self._false_commit_diagnostics(
                    source_valid=source_valid,
                    prepared_valid=prepared_valid,
                    receipt_valid=receipt_valid,
                    revision_fresh=revision_fresh,
                    exhausted=state.unavailable,
                ),
            )

        child = self._replacement.commit(
            state.cycle_state,
            prepared.replacement_prepared,
            authority_receipt.replacement_authority,
        )
        ordinary = bool(jax.device_get(child.diagnostics.ordinary_advance_applied))
        replacement_applied = bool(jax.device_get(child.diagnostics.replacement_applied))
        next_state = state
        rollover_valid = jnp.asarray(False, dtype=jnp.bool_)
        ordinary_adopted = jnp.asarray(False, dtype=jnp.bool_)
        cycle_completed = jnp.asarray(False, dtype=jnp.bool_)

        if replacement_applied:
            next_completed = state.completed_cycles + jnp.int32(1)
            rolled_child = self._replacement.init(
                child.state.scheduler_state,
                retirement_authority_issuer_digest=(
                    state.cycle_state.expected_retirement_authority_issuer_digest
                ),
                controller_owner_digest=state.cycle_state.controller_owner_digest,
            )
            exhausted = next_completed == self._config.max_cycles
            candidate = self._with_checksum(
                dataclasses.replace(
                    state,
                    cycle_state=rolled_child,
                    completed_cycles=next_completed,
                    total_replacements=state.total_replacements + jnp.int32(1),
                    cycle_key_active=jnp.asarray(False, dtype=jnp.bool_),
                    active_cycle_key_data=jnp.zeros((2,), dtype=jnp.uint32),
                    has_completed_cycle=jnp.asarray(True, dtype=jnp.bool_),
                    cycle_key_history=state.cycle_key_history.at[state.completed_cycles].set(
                        state.active_cycle_key_data
                    ),
                    last_completed_cycle_key_data=state.active_cycle_key_data,
                    last_replacement_authority_revision_words=(
                        authority_receipt.replacement_authority.installation_authority.authority_revision_words
                    ),
                    revision=state.revision + jnp.int32(1),
                    unavailable=exhausted,
                    error=jnp.where(
                        exhausted,
                        jnp.asarray(
                            REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY,
                            dtype=jnp.int32,
                        ),
                        jnp.asarray(
                            REPEATED_OPTION_LIFECYCLE_ERROR_NONE,
                            dtype=jnp.int32,
                        ),
                    ),
                    binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
                )
            )
            rollover_valid = self.state_valid(candidate)
            if bool(jax.device_get(rollover_valid)):
                next_state = candidate
                cycle_completed = jnp.asarray(True, dtype=jnp.bool_)
        elif ordinary:
            candidate = self._with_checksum(
                dataclasses.replace(
                    state,
                    cycle_state=child.state,
                    revision=state.revision + jnp.int32(1),
                    binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
                )
            )
            candidate_valid = self.state_valid(candidate)
            if bool(jax.device_get(candidate_valid)):
                next_state = candidate
                ordinary_adopted = jnp.asarray(True, dtype=jnp.bool_)

        return RepeatedOptionLifecycleCommitResult(
            state=next_state,
            replacement=child,
            diagnostics=RepeatedOptionLifecycleCommitDiagnostics(
                source_state_valid=source_valid,
                prepared_binding_valid=prepared_valid,
                receipt_binding_valid=receipt_valid,
                global_authority_revision_fresh=revision_fresh,
                child_ordinary_advance_applied=child.diagnostics.ordinary_advance_applied,
                child_replacement_applied=child.diagnostics.replacement_applied,
                ordinary_advance_adopted=ordinary_adopted,
                cycle_completed=cycle_completed,
                rollover_state_valid=rollover_valid,
                capacity_exhausted=next_state.unavailable,
            ),
        )

    def _adopt_control_state(
        self,
        source: RepeatedOptionLifecycleState,
        child: AuthorizedOptionReplacementState,
    ) -> tuple[RepeatedOptionLifecycleState, bool]:
        candidate = self._with_checksum(
            dataclasses.replace(
                source,
                cycle_state=child,
                revision=source.revision + jnp.int32(1),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        applied = bool(jax.device_get((source.revision < _INT32_MAX) & self.state_valid(candidate)))
        return (candidate if applied else source), applied

    def start(
        self,
        state: RepeatedOptionLifecycleState,
        materialization: CumulantOptionMaterialization,
    ) -> RepeatedOptionLifecycleStartResult:
        """Forward masked control start and stale all prior wrapper receipts."""

        self._check_state_contract(state)
        child = self._replacement.start(state.cycle_state, materialization)
        if not child.applied:
            return RepeatedOptionLifecycleStartResult(state, None, False)
        proposed, applied = self._adopt_control_state(state, child.state)
        return RepeatedOptionLifecycleStartResult(
            proposed if applied else state,
            child if applied else None,
            applied,
        )

    def update(
        self,
        state: RepeatedOptionLifecycleState,
        materialization: CumulantOptionMaterialization,
        env_reward: float | Array,
        discount: float | Array | None = None,
        *,
        execution_boundary: bool | Array = False,
        context: int | Array = 0,
        idle_candidate_option: int | Array = 0,
        idle_initiation_eligible: bool | Array = False,
        comparator_randomized: bool | Array = False,
        treatment_propensity: float | Array = 0.0,
        enable_planning: bool = True,
    ) -> RepeatedOptionLifecycleUpdateResult:
        """Forward masked control update and stale all prior wrapper receipts."""

        self._check_state_contract(state)
        child = self._replacement.update(
            state.cycle_state,
            materialization,
            env_reward,
            discount,
            execution_boundary=execution_boundary,
            context=context,
            idle_candidate_option=idle_candidate_option,
            idle_initiation_eligible=idle_initiation_eligible,
            comparator_randomized=comparator_randomized,
            treatment_propensity=treatment_propensity,
            enable_planning=enable_planning,
        )
        if not child.applied:
            return RepeatedOptionLifecycleUpdateResult(state, None, False)
        proposed, applied = self._adopt_control_state(state, child.state)
        return RepeatedOptionLifecycleUpdateResult(
            proposed if applied else state,
            child if applied else None,
            applied,
        )

    @staticmethod
    def _encode_array(value: Array) -> dict[str, object]:
        array = jnp.asarray(value)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        host = np.asarray(jax.device_get(array))
        return {
            "dtype": host.dtype.str,
            "shape": list(host.shape),
            "bytes_hex": host.tobytes(order="C").hex(),
        }

    @staticmethod
    def _decode_array(value: object) -> Array:
        if type(value) is not dict or set(value) != {"dtype", "shape", "bytes_hex"}:
            raise ValueError("encoded repeated lifecycle array differs from schema v1")
        try:
            dtype = np.dtype(value["dtype"])
        except (TypeError, ValueError) as exc:
            raise ValueError("encoded repeated lifecycle array dtype is invalid") from exc
        if dtype.hasobject:
            raise ValueError("encoded repeated lifecycle array dtype cannot contain objects")
        shape = value["shape"]
        payload = value["bytes_hex"]
        if type(shape) is not list or any(type(cell) is not int or cell < 0 for cell in shape):
            raise ValueError("encoded repeated lifecycle array shape is invalid")
        if type(payload) is not str:
            raise ValueError("encoded repeated lifecycle array bytes must be hex")
        try:
            raw = bytes.fromhex(payload)
        except ValueError as exc:
            raise ValueError("encoded repeated lifecycle array bytes are not hex") from exc
        expected = math.prod(shape) * dtype.itemsize
        if len(raw) != expected:
            raise ValueError("encoded repeated lifecycle array byte length differs")
        return jnp.asarray(np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy())

    @staticmethod
    def _state_sha256(state: RepeatedOptionLifecycleState) -> str:
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(state):
            array = jnp.asarray(leaf)
            if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
                array = jr.key_data(array)
            host = np.asarray(jax.device_get(array))
            digest.update(host.dtype.str.encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
        return digest.hexdigest()

    def checkpoint_payload(
        self,
        state: RepeatedOptionLifecycleState,
    ) -> dict[str, object]:
        """Serialize strict v1 state without preparations, receipts, or proposals."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid repeated lifecycle state")
        fields = {
            field.name: self._encode_array(cast(Array, getattr(state, field.name)))
            for field in dataclasses.fields(RepeatedOptionLifecycleState)
            if field.name != "cycle_state"
        }
        return {
            "schema_version": REPEATED_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA,
            "state_type": "RepeatedOptionLifecycleState",
            "config": self.to_config(),
            "cycle_state": self._replacement.checkpoint_payload(state.cycle_state),
            "controller_fields": fields,
            "state_sha256": self._state_sha256(state),
            "persistent_lifecycle_owner_count": 1,
            "proposal_persisted": False,
            "receipt_persisted": False,
            "assessment": REPEATED_OPTION_LIFECYCLE_ASSESSMENT,
            "evidence_authority": False,
            "scientific_promotion_allowed": False,
        }

    def restore_checkpoint(
        self,
        value: Mapping[str, object],
        *,
        expected_semantic_generation: int | Array,
        expected_source_digest: Array,
        expected_consumer_source_digest: Array,
        expected_consumer_representation_digest: Array,
        expected_lifecycle_id: Array,
        expected_installation_authority_issuer_digest: Array,
        expected_retirement_authority_issuer_digest: Array,
        expected_controller_owner_digest: Array,
        expected_descriptor_generation: Array,
        expected_descriptor_digest: Array,
        expected_installed_bundle: CumulantSubtaskProposalBundle,
        expected_completed_cycles: int | Array,
        expected_revision: int | Array,
    ) -> RepeatedOptionLifecycleState:
        """Restore only an exact state and caller-supplied anti-rollback clocks."""

        if type(value) is not dict:
            raise ValueError("repeated lifecycle checkpoint must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "state_type",
            "config",
            "cycle_state",
            "controller_fields",
            "state_sha256",
            "persistent_lifecycle_owner_count",
            "proposal_persisted",
            "receipt_persisted",
            "assessment",
            "evidence_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("repeated lifecycle checkpoint keys differ from schema v1")
        fixed: dict[str, object] = {
            "schema_version": REPEATED_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA,
            "state_type": "RepeatedOptionLifecycleState",
            "config": self.to_config(),
            "persistent_lifecycle_owner_count": 1,
            "proposal_persisted": False,
            "receipt_persisted": False,
            "assessment": REPEATED_OPTION_LIFECYCLE_ASSESSMENT,
            "evidence_authority": False,
            "scientific_promotion_allowed": False,
        }
        for name, expected_value in fixed.items():
            if raw[name] != expected_value:
                raise ValueError(f"repeated lifecycle checkpoint {name} differs")
        fields = raw["controller_fields"]
        if type(fields) is not dict:
            raise ValueError("repeated lifecycle controller_fields must be a dict")
        expected_fields = {
            field.name
            for field in dataclasses.fields(RepeatedOptionLifecycleState)
            if field.name != "cycle_state"
        }
        if set(fields) != expected_fields:
            raise ValueError("repeated lifecycle checkpoint controller fields differ")
        child = self._replacement.restore_checkpoint(
            raw["cycle_state"],
            expected_semantic_generation=expected_semantic_generation,
            expected_source_digest=expected_source_digest,
            expected_consumer_source_digest=expected_consumer_source_digest,
            expected_consumer_representation_digest=(expected_consumer_representation_digest),
            expected_lifecycle_id=expected_lifecycle_id,
            expected_installation_authority_issuer_digest=(
                expected_installation_authority_issuer_digest
            ),
            expected_retirement_authority_issuer_digest=(
                expected_retirement_authority_issuer_digest
            ),
            expected_controller_owner_digest=expected_controller_owner_digest,
            expected_descriptor_generation=expected_descriptor_generation,
            expected_descriptor_digest=expected_descriptor_digest,
            expected_installed_bundle=expected_installed_bundle,
        )
        decoded = {name: self._decode_array(fields[name]) for name in expected_fields}
        restored = RepeatedOptionLifecycleState(
            cycle_state=child,
            **cast(dict[str, Any], decoded),
        )
        completed = jnp.asarray(expected_completed_cycles, dtype=jnp.int32)
        revision = jnp.asarray(expected_revision, dtype=jnp.int32)
        if completed.shape != () or revision.shape != ():
            raise ValueError("expected repeated lifecycle clocks must be scalars")
        digest_valid = type(raw["state_sha256"]) is str and raw[
            "state_sha256"
        ] == self._state_sha256(restored)
        bindings_valid = (restored.completed_cycles == completed) & (restored.revision == revision)
        if not digest_valid or not bool(
            jax.device_get(bindings_valid & self.state_valid(restored))
        ):
            raise ValueError("restored repeated lifecycle state is invalid, stale, or rebound")
        return restored

    def resource_budget(
        self,
        state: RepeatedOptionLifecycleState,
        prepared: RepeatedOptionLifecyclePrepared | None = None,
    ) -> RepeatedOptionLifecycleResourceBudget:
        """Measure the sole owner and all remaining hard capacities exactly."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("resource measurement requires a valid repeated lifecycle state")
        if prepared is not None and type(prepared) is not RepeatedOptionLifecyclePrepared:
            raise TypeError("prepared must be an exact preparation or None")
        child = state.cycle_state
        scheduler = child.scheduler_state
        installation = scheduler.installation_state
        persistent_nbytes = _tree_nbytes(state)
        child_nbytes = _tree_nbytes(child)
        scheduler_nbytes = _tree_nbytes(scheduler)
        installation_nbytes = _tree_nbytes(installation)
        prepared_nbytes = 0 if prepared is None else _tree_nbytes(prepared)
        completed = int(jax.device_get(state.completed_cycles))
        remaining = self._config.max_cycles - completed
        installation_remaining = max(
            0,
            self._replacement.scheduler.installation.config.max_installations
            - int(jax.device_get(installation.installation_count)),
        )
        attempt_remaining = max(
            0,
            self._replacement.scheduler.config.max_install_attempts
            - _words_to_int(scheduler.install_attempt_words),
        )
        step_remaining = max(
            0,
            self._replacement.scheduler.config.max_steps - _words_to_int(scheduler.step_words),
        )
        return RepeatedOptionLifecycleResourceBudget(
            persistent_state_nbytes=persistent_nbytes,
            child_state_nbytes=child_nbytes,
            scheduler_state_nbytes=scheduler_nbytes,
            installation_state_nbytes=installation_nbytes,
            wrapper_metadata_nbytes=persistent_nbytes - child_nbytes,
            duplicated_scheduler_state_nbytes=0,
            duplicated_installation_state_nbytes=0,
            prepared_state_nbytes=prepared_nbytes,
            persistent_lifecycle_owner_count=1,
            pending_proposal_slots=0,
            persisted_receipt_count=0,
            max_cycles=self._config.max_cycles,
            completed_cycles=completed,
            remaining_cycles=remaining,
            active_cycle=bool(jax.device_get(state.cycle_key_active)),
            total_retirements=int(jax.device_get(state.total_retirements)),
            total_replacements=int(jax.device_get(state.total_replacements)),
            installation_capacity_remaining=installation_remaining,
            scheduler_attempt_capacity_remaining=attempt_remaining,
            scheduler_step_capacity_remaining=step_remaining,
            underlying_capacity_supports_remaining_cycles=(
                min(installation_remaining, attempt_remaining, step_remaining) >= remaining
            ),
            max_retirements_per_cycle=1,
            max_replacements_per_cycle=1,
            max_reset_keys_per_cycle=2,
            max_cycle_keys_per_cycle=1,
            max_scheduler_observations_per_attempt=1,
            host_prepare=True,
            host_commit=True,
            assessment=REPEATED_OPTION_LIFECYCLE_ASSESSMENT,
            output_writes=False,
            evidence_authority=False,
            promotion_authority=False,
            safety_authority=False,
            go_no_go_authority=False,
            retirement_authority=False,
            replacement_authority=False,
            discovery_authority=False,
            dispatch_authority=False,
            autonomous_curation_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=REPEATED_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA,
        )


__all__ = [
    "REPEATED_OPTION_LIFECYCLE_ASSESSMENT",
    "REPEATED_OPTION_LIFECYCLE_AUTONOMOUS_CURATION_AUTHORITY",
    "REPEATED_OPTION_LIFECYCLE_CHECKPOINT_SCHEMA",
    "REPEATED_OPTION_LIFECYCLE_CONFIG_SCHEMA",
    "REPEATED_OPTION_LIFECYCLE_DISCOVERY_AUTHORITY",
    "REPEATED_OPTION_LIFECYCLE_DISPATCH_AUTHORITY",
    "REPEATED_OPTION_LIFECYCLE_ERROR_CAPACITY",
    "REPEATED_OPTION_LIFECYCLE_ERROR_NONE",
    "REPEATED_OPTION_LIFECYCLE_EVIDENCE_AUTHORITY",
    "REPEATED_OPTION_LIFECYCLE_GO_NO_GO_AUTHORITY",
    "REPEATED_OPTION_LIFECYCLE_OUTPUT_WRITES",
    "REPEATED_OPTION_LIFECYCLE_PROMOTION_AUTHORITY",
    "REPEATED_OPTION_LIFECYCLE_REPLACEMENT_AUTHORITY",
    "REPEATED_OPTION_LIFECYCLE_RETIREMENT_AUTHORITY",
    "REPEATED_OPTION_LIFECYCLE_SAFETY_AUTHORITY",
    "REPEATED_OPTION_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED",
    "RepeatedOptionLifecycle",
    "RepeatedOptionLifecycleArm",
    "RepeatedOptionLifecycleCommitDiagnostics",
    "RepeatedOptionLifecycleCommitResult",
    "RepeatedOptionLifecycleConfig",
    "RepeatedOptionLifecyclePrepared",
    "RepeatedOptionLifecycleResourceBudget",
    "RepeatedOptionLifecycleRetirementDiagnostics",
    "RepeatedOptionLifecycleRetirementResult",
    "RepeatedOptionLifecycleStartResult",
    "RepeatedOptionLifecycleState",
    "RepeatedOptionLifecycleUpdateResult",
    "RepeatedOptionReplacementAuthorityReceipt",
    "RepeatedOptionRetirementAuthorityReceipt",
]
