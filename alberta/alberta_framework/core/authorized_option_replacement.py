# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Two-phase, externally authorized replacement of one retired option slot.

This module closes one deliberately narrow mechanical edge between
``AuthorizedOptionRetirementController`` and ``CumulantOptionScheduler``.
It does not grant discovery, scheduling, audit, or the replacement wrapper any
go/no-go authority.

The persistent state has exactly one owner for the installation/lifecycle
subtree: ``scheduler_state.installation_state``.  Retirement metadata is stored
without a second installation copy and is projected into a transient
``AuthorizedOptionRetirementState`` only while validating or executing masked
control.  A redundant scheduler checksum mirror catches accidental internal
splices across wrapper transactions; like every checksum here it is unkeyed and
is not caller authentication or cryptographic lineage proof.

Replacement is a source-bound two-phase transaction.  ``prepare`` observes one
accepted transition through the ordinary scheduler with authority denied.  It
therefore stages only discovery, incumbent live materialization, cadence, and a
zero-payload retry.  The fresh candidate bundle exists only in the returned
transient preparation.  A caller may then issue an exact receipt binding that
candidate, its one cold destination, the expected reset mask, and all source
owners/revisions.  ``commit`` bit-authenticates the preparation and either:

* installs the exact one-slot semantic change and atomically reactivates that
  slot; or
* commits only the already-staged ordinary discovery/incumbent-materialization
  advance.  Candidate materialization, installation RNG, semantic identity,
  and the cold mask never leak into this declined path.

The public commit boundary is host-orchestrated because it deterministically
reruns ``prepare`` from the supplied source, arm, observation, and live inputs,
then bit-compares the entire transaction before host composition and a compiled
atomic whole-state adoption kernel.  This prevents an unkeyed local checksum
plus a broad installation receipt from laundering a caller-edited discovery
payload.  No proposal is checkpointed or persisted, and a later retry must
prepare a newly observed bundle.  Receipts are integrity declarations, not
authentication.
This L0 mechanism writes no outputs and owns no safety, evidence, promotion,
scientific, dispatch, retirement, discovery, or autonomous curation authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.authorized_option_retirement import (
    AUTHORIZED_OPTION_RETIREMENT_ERROR_CAPACITY,
    AUTHORIZED_OPTION_RETIREMENT_ERROR_NONE,
    AuthorizedOptionRetirementController,
    AuthorizedOptionRetirementResult,
    AuthorizedOptionRetirementStartResult,
    AuthorizedOptionRetirementState,
    AuthorizedOptionRetirementUpdateResult,
    OptionRetirementAuthorityReceipt,
)
from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionInstallationResult,
    CumulantOptionLiveInputs,
    CumulantOptionMaterialization,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionExternalBundleAdoptionAuthorityReceipt,
    CumulantOptionExternalBundleAdoptionPrepared,
    CumulantOptionExternalBundleAdoptionResult,
    CumulantOptionInstallationAuthorityReceipt,
    CumulantOptionRetirementHandoff,
    CumulantOptionScheduler,
    CumulantOptionSchedulerArm,
    CumulantOptionSchedulerArmInputs,
    CumulantOptionSchedulerBorrowResult,
    CumulantOptionSchedulerMetadataState,
    CumulantOptionSchedulerObservation,
    CumulantOptionSchedulerResult,
    CumulantOptionSchedulerState,
    adopt_cumulant_option_external_bundle,
    cumulant_option_external_bundle_adoption_authority_receipt,
    prepare_cumulant_option_external_bundle_adoption,
)
from alberta_framework.core.cumulant_subtask_discovery import (
    CumulantSubtaskDiscoveryResult,
    CumulantSubtaskProposalBundle,
)

AUTHORIZED_OPTION_REPLACEMENT_CONFIG_SCHEMA = "alberta.authorized-option-replacement.config.v1"
AUTHORIZED_OPTION_REPLACEMENT_CHECKPOINT_SCHEMA = "alberta.authorized-option-replacement.state.v1"
AUTHORIZED_OPTION_REPLACEMENT_ASSESSMENT = "not_assessed"
AUTHORIZED_OPTION_REPLACEMENT_OUTPUT_WRITES = False
AUTHORIZED_OPTION_REPLACEMENT_EVIDENCE_AUTHORITY = False
AUTHORIZED_OPTION_REPLACEMENT_PROMOTION_AUTHORITY = False
AUTHORIZED_OPTION_REPLACEMENT_SAFETY_AUTHORITY = False
AUTHORIZED_OPTION_REPLACEMENT_GO_NO_GO_AUTHORITY = False
AUTHORIZED_OPTION_REPLACEMENT_RETIREMENT_AUTHORITY = False
AUTHORIZED_OPTION_REPLACEMENT_DISCOVERY_AUTHORITY = False
AUTHORIZED_OPTION_REPLACEMENT_DISPATCH_AUTHORITY = False
AUTHORIZED_OPTION_REPLACEMENT_AUTONOMOUS_CURATION_AUTHORITY = False
AUTHORIZED_OPTION_REPLACEMENT_SCIENTIFIC_PROMOTION_ALLOWED = False

_DIGEST_WORDS = 8
_CANONICAL_DIGEST_BYTES = 32
_CLOCK_WORDS = 2
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
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


def _words(value: int) -> Array:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _increment_words(words: Array) -> tuple[Array, Array]:
    low = words[1] + jnp.uint32(1)
    carry = (low == 0).astype(jnp.uint32)
    high = words[0] + carry
    available = ~jnp.all(words == jnp.uint32(_UINT32_MAX))
    candidate = jnp.stack((high, low), dtype=jnp.uint32)
    return jnp.where(available, candidate, words), available


def _words_less(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _words_less_equal(left: Array, right: Array) -> Array:
    return jnp.array_equal(left, right) | _words_less(left, right)


def _saturating_increment(value: Array) -> Array:
    return jnp.where(value < _INT32_MAX, value + jnp.int32(1), value)


def _descriptor_digest(descriptors: Array) -> Array:
    words = jax.lax.bitcast_convert_type(descriptors, jnp.uint32).reshape((-1,))
    rows: list[Array] = []
    for digest_index in range(_DIGEST_WORDS):
        acc = jnp.uint32(0x811C9DC5 ^ ((digest_index + 1) * 0x9E3779B9 & _UINT32_MAX))
        for payload_index in range(words.shape[0]):
            acc = (acc ^ words[payload_index]) * jnp.uint32(0x01000193)
            acc = acc + jnp.uint32(
                ((payload_index + 1) * (digest_index + 3) * 0x85EB) & _UINT32_MAX
            )
        rows.append(acc)
    return jnp.stack(tuple(rows), dtype=jnp.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedOptionReplacementConfig:
    """Static one-cold-slot replacement capacity and authority declaration."""

    max_replacements: int = 1

    SCHEMA_VERSION: ClassVar[str] = AUTHORIZED_OPTION_REPLACEMENT_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _positive_int(self.max_replacements, name="max_replacements")
        if self.max_replacements != 1:
            raise ValueError("the v1 single-cold-slot contract requires max_replacements == 1")

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_replacements": self.max_replacements,
            "replacement_scope": "exactly_one_previously_retired_cold_slot",
            "proposal_persistence": "none",
            "candidate_authority": "caller_receipt_bound_after_prepare",
            "assessment": AUTHORIZED_OPTION_REPLACEMENT_ASSESSMENT,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "safety_authority": False,
            "go_no_go_authority": False,
            "retirement_authority": False,
            "discovery_authority": False,
            "dispatch_authority": False,
            "autonomous_curation_authority": False,
            "scientific_promotion_allowed": False,
            "host_prepare": True,
            "host_commit": True,
            "jit_commit": False,
            "jit_atomic_adoption_kernel": True,
        }

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> AuthorizedOptionReplacementConfig:
        if type(value) is not dict:
            raise ValueError("replacement config must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "max_replacements",
            "replacement_scope",
            "proposal_persistence",
            "candidate_authority",
            "assessment",
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "safety_authority",
            "go_no_go_authority",
            "retirement_authority",
            "discovery_authority",
            "dispatch_authority",
            "autonomous_curation_authority",
            "scientific_promotion_allowed",
            "host_prepare",
            "host_commit",
            "jit_commit",
            "jit_atomic_adoption_kernel",
        }
        if set(raw) != expected:
            raise ValueError("replacement config keys differ from schema v1")
        fixed = {
            "schema_version": cls.SCHEMA_VERSION,
            "replacement_scope": "exactly_one_previously_retired_cold_slot",
            "proposal_persistence": "none",
            "candidate_authority": "caller_receipt_bound_after_prepare",
            "assessment": AUTHORIZED_OPTION_REPLACEMENT_ASSESSMENT,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "safety_authority": False,
            "go_no_go_authority": False,
            "retirement_authority": False,
            "discovery_authority": False,
            "dispatch_authority": False,
            "autonomous_curation_authority": False,
            "scientific_promotion_allowed": False,
            "host_prepare": True,
            "host_commit": True,
            "jit_commit": False,
            "jit_atomic_adoption_kernel": True,
        }
        for name, expected_value in fixed.items():
            if raw.pop(name) != expected_value:
                raise ValueError(f"replacement config {name} differs")
        return cls(max_replacements=cast(int, raw.pop("max_replacements")))


@chex.dataclass(frozen=True)
class AuthorizedOptionReplacementState:
    """One canonical scheduler/installer plus corruption-detecting bindings."""

    scheduler_state: CumulantOptionSchedulerState
    canonical_scheduler_checksum: UInt[Array, " 2"]
    installed_slot_mask: Bool[Array, " option_budget"]
    descriptor_generation: Int[Array, ""]
    descriptor_digest: UInt[Array, " 8"]
    expected_retirement_authority_issuer_digest: UInt[Array, " 8"]
    controller_owner_digest: UInt[Array, " 8"]
    controller_revision: Int[Array, ""]
    retirement_words: UInt[Array, " 2"]
    last_retirement_authority_revision_words: UInt[Array, " 2"]
    last_retirement_scheduler_step_words: UInt[Array, " 2"]
    retirement_unavailable: Bool[Array, ""]
    retirement_error: Int[Array, ""]
    replacement_words: UInt[Array, " 2"]
    last_replacement_authority_revision_words: UInt[Array, " 2"]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class AuthorizedOptionReplacementMetadataState:
    """Replacement metadata borrowing the sole external STOMP owner."""

    scheduler_metadata: CumulantOptionSchedulerMetadataState
    canonical_scheduler_checksum: UInt[Array, " 2"]
    installed_slot_mask: Bool[Array, " option_budget"]
    descriptor_generation: Int[Array, ""]
    descriptor_digest: UInt[Array, " 8"]
    expected_retirement_authority_issuer_digest: UInt[Array, " 8"]
    controller_owner_digest: UInt[Array, " 8"]
    controller_revision: Int[Array, ""]
    retirement_words: UInt[Array, " 2"]
    last_retirement_authority_revision_words: UInt[Array, " 2"]
    last_retirement_scheduler_step_words: UInt[Array, " 2"]
    retirement_unavailable: Bool[Array, ""]
    retirement_error: Int[Array, ""]
    replacement_words: UInt[Array, " 2"]
    last_replacement_authority_revision_words: UInt[Array, " 2"]
    source_binding_checksum: UInt[Array, " 2"]
    metadata_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class AuthorizedOptionReplacementBorrowResult:
    """Fail-closed transient reconstruction around one STOMP owner."""

    state: AuthorizedOptionReplacementState
    scheduler: CumulantOptionSchedulerBorrowResult
    metadata_valid: Bool[Array, ""]
    binding_matches: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AuthorizedOptionReplacementArm:
    """Scheduler arm bound to the exact cold mask and wrapper source."""

    scheduler_arm: CumulantOptionSchedulerArm
    source_checksum: UInt[Array, " 2"]
    source_controller_revision: Int[Array, ""]
    target_slot: Int[Array, ""]
    target_mask: Bool[Array, " option_budget"]
    available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AuthorizedOptionReplacementPrepareDiagnostics:
    """Facts exposed before any caller replacement authority exists."""

    source_state_valid: Bool[Array, ""]
    arm_binding_valid: Bool[Array, ""]
    ordinary_scheduler_transaction_valid: Bool[Array, ""]
    fallback_state_valid: Bool[Array, ""]
    proposal_due: Bool[Array, ""]
    proposal_ready: Bool[Array, ""]
    proposal_binding_valid: Bool[Array, ""]
    fresh_transition: Bool[Array, ""]
    exactly_one_cold_slot: Bool[Array, ""]
    target_still_cold: Bool[Array, ""]
    exact_one_slot_semantic_change: Bool[Array, ""]
    live_slots_semantically_preserved: Bool[Array, ""]
    quiescent: Bool[Array, ""]
    scheduler_attempt_capacity_available: Bool[Array, ""]
    installer_capacity_available: Bool[Array, ""]
    candidate_ready_for_authority: Bool[Array, ""]
    transaction_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AuthorizedOptionReplacementPrepared:
    """Transient source-bound candidate; never part of persistent state."""

    source_state: AuthorizedOptionReplacementState
    arm: AuthorizedOptionReplacementArm
    observation: CumulantOptionSchedulerObservation
    live_inputs: CumulantOptionLiveInputs
    scheduler_result: CumulantOptionSchedulerResult
    fallback_state: AuthorizedOptionReplacementState
    candidate_semantic_digests: UInt[Array, "option_budget 8"]
    changed_slots: Bool[Array, " option_budget"]
    target_slot: Int[Array, ""]
    target_mask: Bool[Array, " option_budget"]
    diagnostics: AuthorizedOptionReplacementPrepareDiagnostics
    prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class OptionReplacementAuthorityReceipt:
    """Caller authority for one exact prepared candidate and cold destination."""

    installation_authority: CumulantOptionInstallationAuthorityReceipt
    replacement_authorized: Bool[Array, ""]
    controller_owner_digest: UInt[Array, " 8"]
    source_scheduler_checksum: UInt[Array, " 2"]
    source_controller_revision: Int[Array, ""]
    source_descriptor_generation: Int[Array, ""]
    source_descriptor_digest: UInt[Array, " 8"]
    source_installed_slot_mask: Bool[Array, " option_budget"]
    source_installation_revision: Int[Array, ""]
    source_lifecycle_revision: Int[Array, ""]
    source_audit_revision: Int[Array, ""]
    replacement_slot: Int[Array, ""]
    expected_reset_slots: Bool[Array, " option_budget"]
    candidate_binding_digest: UInt[Array, " 2"]
    candidate_semantic_digests: UInt[Array, "option_budget 8"]
    candidate_transition_id: UInt[Array, " 2"]
    candidate_state_observation_count: Int[Array, ""]
    prepared_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class AuthorizedOptionReplacementCommitDiagnostics:
    """Prepared integrity, external authority, and atomic adoption facts."""

    destination_state_valid: Bool[Array, ""]
    destination_matches_source: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    preparation_derivation_valid: Bool[Array, ""]
    prepared_transaction_valid: Bool[Array, ""]
    authority_valid: Bool[Array, ""]
    candidate_ready: Bool[Array, ""]
    replacement_capacity_available: Bool[Array, ""]
    installation_transaction_valid: Bool[Array, ""]
    installation_applied: Bool[Array, ""]
    exact_reset_mask: Bool[Array, ""]
    exact_preserve_mask: Bool[Array, ""]
    live_policy_rng_preserved: Bool[Array, ""]
    installed_scheduler_state_valid: Bool[Array, ""]
    installed_replacement_state_valid: Bool[Array, ""]
    ordinary_advance_applied: Bool[Array, ""]
    replacement_attempted: Bool[Array, ""]
    replacement_applied: Bool[Array, ""]
    proposal_persisted: Bool[Array, ""]
    candidate_materialization_persisted_on_decline: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AuthorizedOptionReplacementResult:
    """Committed ordinary advance plus optional exact replacement."""

    state: AuthorizedOptionReplacementState
    discovery: CumulantSubtaskDiscoveryResult
    materialization: CumulantOptionMaterialization
    retirement_handoff: CumulantOptionRetirementHandoff
    diagnostics: AuthorizedOptionReplacementCommitDiagnostics
    reset_slots: Bool[Array, " option_budget"]
    preserved_slots: Bool[Array, " option_budget"]
    extended_action_mask: Bool[Array, " n_total_actions"]
    cold_mask_active: Bool[Array, ""]
    retry_scheduled: Bool[Array, ""]
    fresh_bundle_required_on_retry: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AuthorizedOptionReplacementRetirementResult:
    """Exact retirement transaction reprojected into the canonical child."""

    state: AuthorizedOptionReplacementState
    retirement: AuthorizedOptionRetirementResult
    source_state_valid: Bool[Array, ""]
    phase_valid: Bool[Array, ""]
    canonical_scheduler_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedOptionReplacementStartResult:
    """Masked host control start while the replacement composition owns state."""

    state: AuthorizedOptionReplacementState
    retirement: AuthorizedOptionRetirementStartResult | None
    applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedOptionReplacementUpdateResult:
    """Masked host control update while the replacement composition owns state."""

    state: AuthorizedOptionReplacementState
    retirement: AuthorizedOptionRetirementUpdateResult | None
    applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedOptionReplacementResourceBudget:
    """Exact persistent/transient bytes and bounded work/authority declaration."""

    persistent_state_nbytes: int
    scheduler_state_nbytes: int
    installation_state_nbytes: int
    retirement_binding_nbytes: int
    duplicated_installation_state_nbytes: int
    prepared_state_nbytes: int
    option_slots: int
    pending_proposal_slots: int
    max_replacements: int
    prepare_scheduler_observations: int
    commit_preparation_recomputations: int
    max_installations_per_commit: int
    max_lifecycle_rebinds_per_commit: int
    max_fresh_template_initializations_per_commit: int
    max_rng_splits_per_commit: int
    proposal_persisted: bool
    candidate_materialization_persisted_on_decline: bool
    host_prepare: bool
    host_commit: bool
    jit_commit: bool
    jit_atomic_adoption_kernel: bool
    assessment: str
    output_writes: bool
    evidence_authority: bool
    promotion_authority: bool
    safety_authority: bool
    go_no_go_authority: bool
    retirement_authority: bool
    discovery_authority: bool
    dispatch_authority: bool
    autonomous_curation_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str


class AuthorizedOptionReplacementController:
    """One-canonical-state coordinator for a single cold-slot replacement."""

    def __init__(
        self,
        scheduler: CumulantOptionScheduler,
        retirement: AuthorizedOptionRetirementController,
        config: AuthorizedOptionReplacementConfig | None = None,
    ) -> None:
        if type(scheduler) is not CumulantOptionScheduler:
            raise TypeError("scheduler must be an exact CumulantOptionScheduler")
        if type(retirement) is not AuthorizedOptionRetirementController:
            raise TypeError("retirement must be an exact AuthorizedOptionRetirementController")
        if scheduler.installation is not retirement.installation:
            raise ValueError("scheduler and retirement must share one installation object")
        self._scheduler = scheduler
        self._retirement = retirement
        self._installation = scheduler.installation
        self._config = config or AuthorizedOptionReplacementConfig()
        # Cache the small atomic adoption boundary once.  Discovery provenance
        # replay and candidate installation remain host-orchestrated; only the
        # final whole-state choice is compiled, avoiding repeated compilation
        # and a multi-gigabyte lowering of the complete STOMP transaction.
        self._compiled_atomic_adoption_kernel = jax.jit(self._atomic_adoption_kernel)

    @property
    def config(self) -> AuthorizedOptionReplacementConfig:
        return self._config

    @property
    def scheduler(self) -> CumulantOptionScheduler:
        return self._scheduler

    @property
    def retirement(self) -> AuthorizedOptionRetirementController:
        return self._retirement

    def to_config(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(
                json.dumps(
                    {
                        "schema_version": AUTHORIZED_OPTION_REPLACEMENT_CONFIG_SCHEMA,
                        "replacement": self._config.to_config(),
                        "scheduler": self._scheduler.to_config(),
                        "retirement": self._retirement.to_config(),
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
        )

    def _payload_arrays(
        self,
        state: AuthorizedOptionReplacementState,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    state.scheduler_state,
                    state.canonical_scheduler_checksum,
                    state.installed_slot_mask,
                    state.descriptor_generation,
                    state.descriptor_digest,
                    state.expected_retirement_authority_issuer_digest,
                    state.controller_owner_digest,
                    state.controller_revision,
                    state.retirement_words,
                    state.last_retirement_authority_revision_words,
                    state.last_retirement_scheduler_step_words,
                    state.retirement_unavailable,
                    state.retirement_error,
                    state.replacement_words,
                    state.last_replacement_authority_revision_words,
                )
            )
        )

    def _with_checksum(
        self,
        state: AuthorizedOptionReplacementState,
    ) -> AuthorizedOptionReplacementState:
        return dataclasses.replace(
            state,
            binding_checksum=_checksum_arrays(self._payload_arrays(state)),
        )

    def _metadata_payload_arrays(
        self,
        state: AuthorizedOptionReplacementMetadataState,
    ) -> tuple[Array, ...]:
        values = tuple(
            getattr(state, field.name)
            for field in dataclasses.fields(AuthorizedOptionReplacementMetadataState)
            if field.name != "metadata_checksum"
        )
        return tuple(
            cast(Array, leaf) for leaf in jax.tree_util.tree_leaves(values)
        )

    def _with_metadata_checksum(
        self,
        state: AuthorizedOptionReplacementMetadataState,
    ) -> AuthorizedOptionReplacementMetadataState:
        return dataclasses.replace(
            state,
            metadata_checksum=_checksum_arrays(self._metadata_payload_arrays(state)),
        )

    def _check_metadata_contract(
        self,
        state: AuthorizedOptionReplacementMetadataState,
    ) -> None:
        if type(state) is not AuthorizedOptionReplacementMetadataState:
            raise TypeError(
                "state must be an exact AuthorizedOptionReplacementMetadataState"
            )
        self._scheduler._check_metadata_contract(state.scheduler_metadata)
        n = self._installation.discovery.config.option_budget
        contracts = (
            (
                state.canonical_scheduler_checksum,
                "canonical_scheduler_checksum",
                (2,),
                jnp.uint32,
            ),
            (state.installed_slot_mask, "installed_slot_mask", (n,), jnp.bool_),
            (state.descriptor_generation, "descriptor_generation", (), jnp.int32),
            (state.descriptor_digest, "descriptor_digest", (8,), jnp.uint32),
            (
                state.expected_retirement_authority_issuer_digest,
                "expected_retirement_authority_issuer_digest",
                (8,),
                jnp.uint32,
            ),
            (state.controller_owner_digest, "controller_owner_digest", (8,), jnp.uint32),
            (state.controller_revision, "controller_revision", (), jnp.int32),
            (state.retirement_words, "retirement_words", (2,), jnp.uint32),
            (
                state.last_retirement_authority_revision_words,
                "last_retirement_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (
                state.last_retirement_scheduler_step_words,
                "last_retirement_scheduler_step_words",
                (2,),
                jnp.uint32,
            ),
            (state.retirement_unavailable, "retirement_unavailable", (), jnp.bool_),
            (state.retirement_error, "retirement_error", (), jnp.int32),
            (state.replacement_words, "replacement_words", (2,), jnp.uint32),
            (
                state.last_replacement_authority_revision_words,
                "last_replacement_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (state.source_binding_checksum, "source_binding_checksum", (2,), jnp.uint32),
            (state.metadata_checksum, "metadata_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def detach_borrowed_stomp(
        self,
        state: AuthorizedOptionReplacementState,
    ) -> AuthorizedOptionReplacementMetadataState:
        """Detach replacement metadata while retaining no STOMP state."""

        self._check_state_contract(state)
        scheduler_metadata = self._scheduler.detach_borrowed_stomp(
            state.scheduler_state
        )
        values = {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(AuthorizedOptionReplacementState)
            if field.name not in {"scheduler_state", "binding_checksum"}
        }
        metadata = AuthorizedOptionReplacementMetadataState(
            scheduler_metadata=scheduler_metadata,
            **values,
            source_binding_checksum=state.binding_checksum,
            metadata_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_metadata_checksum(metadata)

    def metadata_state_valid(
        self,
        state: AuthorizedOptionReplacementMetadataState,
    ) -> Bool[Array, ""]:
        """Validate the detached single-owner replacement composition."""

        self._check_metadata_contract(state)
        installation = state.scheduler_metadata.installation_metadata
        lifecycle = installation.lifecycle_metadata
        audit = lifecycle.audit_state
        executing = lifecycle.stomp_executing_option
        executing_safe = (executing < 0) | state.installed_slot_mask[
            jnp.clip(executing, 0, self._installation.discovery.config.option_budget - 1)
        ]
        active = audit.active_option
        active_safe = (active < 0) | state.installed_slot_mask[
            jnp.clip(active, 0, self._installation.discovery.config.option_budget - 1)
        ]
        at_retirement_capacity = jnp.array_equal(
            state.retirement_words,
            _words(self._retirement.config.max_retirements),
        )
        zero_retirements = jnp.array_equal(state.retirement_words, _words(0))
        one_retirement = jnp.array_equal(state.retirement_words, _words(1))
        zero_replacements = jnp.array_equal(state.replacement_words, _words(0))
        one_replacement = jnp.array_equal(state.replacement_words, _words(1))
        retirement_clock_binding = jnp.where(
            zero_retirements,
            jnp.all(state.last_retirement_authority_revision_words == 0)
            & jnp.all(state.last_retirement_scheduler_step_words == 0),
            jnp.any(state.last_retirement_authority_revision_words != 0)
            & jnp.any(state.last_retirement_scheduler_step_words != 0),
        )
        expected_retirement_error = jnp.where(
            at_retirement_capacity,
            jnp.asarray(AUTHORIZED_OPTION_RETIREMENT_ERROR_CAPACITY, dtype=jnp.int32),
            jnp.asarray(AUTHORIZED_OPTION_RETIREMENT_ERROR_NONE, dtype=jnp.int32),
        )
        retirement_metadata_valid = (
            self._installation.metadata_state_valid(installation)
            & installation.installed
            & (
                state.descriptor_generation
                == installation.installed_bundle.semantic_generation
            )
            & jnp.array_equal(
                state.descriptor_digest,
                _descriptor_digest(installation.installed_bundle.selected_descriptors),
            )
            & jnp.any(state.expected_retirement_authority_issuer_digest != 0)
            & jnp.any(state.controller_owner_digest != 0)
            & (state.controller_revision >= 0)
            & _words_less_equal(
                state.retirement_words,
                _words(self._retirement.config.max_retirements),
            )
            & (state.retirement_words[0] == 0)
            & (
                state.controller_revision
                >= state.retirement_words[1].astype(jnp.int32)
            )
            & retirement_clock_binding
            & (state.retirement_unavailable == at_retirement_capacity)
            & (state.retirement_error == expected_retirement_error)
            & executing_safe
            & active_safe
        )
        cold_count = jnp.sum(~state.installed_slot_mask, dtype=jnp.int32)
        phase_contract = (
            (zero_retirements & zero_replacements & (cold_count == 0))
            | (one_retirement & zero_replacements & (cold_count == 1))
            | (one_retirement & one_replacement & (cold_count == 0))
        )
        authority_clock_contract = jnp.where(
            zero_replacements,
            jnp.all(state.last_replacement_authority_revision_words == 0),
            jnp.any(state.last_replacement_authority_revision_words != 0),
        )
        return (
            self._scheduler.metadata_state_valid(state.scheduler_metadata)
            & jnp.array_equal(
                state.canonical_scheduler_checksum,
                state.scheduler_metadata.source_binding_checksum,
            )
            & retirement_metadata_valid
            & (zero_retirements | one_retirement)
            & _words_less_equal(
                state.replacement_words,
                _words(self._config.max_replacements),
            )
            & (state.replacement_words[0] == 0)
            & phase_contract
            & authority_clock_contract
            & (
                state.controller_revision
                >= state.retirement_words[1].astype(jnp.int32)
                + state.replacement_words[1].astype(jnp.int32)
            )
            & jnp.array_equal(
                state.metadata_checksum,
                _checksum_arrays(self._metadata_payload_arrays(state)),
            )
        )

    def attach_borrowed_stomp(
        self,
        metadata: AuthorizedOptionReplacementMetadataState,
        stomp_state: Any,
    ) -> AuthorizedOptionReplacementBorrowResult:
        """Build a transient replacement state around the borrowed STOMP."""

        self._check_metadata_contract(metadata)
        scheduler_result = self._scheduler.attach_borrowed_stomp(
            metadata.scheduler_metadata,
            stomp_state,
        )
        values = {
            field.name: getattr(metadata, field.name)
            for field in dataclasses.fields(AuthorizedOptionReplacementMetadataState)
            if field.name
            not in {
                "scheduler_metadata",
                "source_binding_checksum",
                "metadata_checksum",
            }
        }
        candidate = AuthorizedOptionReplacementState(
            scheduler_state=scheduler_result.state,
            **values,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        candidate = self._with_checksum(candidate)
        binding_matches = jnp.array_equal(
            metadata.source_binding_checksum,
            candidate.binding_checksum,
        )
        metadata_valid = self.metadata_state_valid(metadata)
        transaction_applied = (
            metadata_valid
            & scheduler_result.transaction_applied
            & binding_matches
            & self.state_valid(candidate)
        )
        return AuthorizedOptionReplacementBorrowResult(
            state=candidate,
            scheduler=scheduler_result,
            metadata_valid=metadata_valid,
            binding_matches=binding_matches,
            transaction_applied=transaction_applied,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _check_state_contract(self, state: AuthorizedOptionReplacementState) -> None:
        if type(state) is not AuthorizedOptionReplacementState:
            raise TypeError("state must be an exact AuthorizedOptionReplacementState")
        self._scheduler._check_state_contract(state.scheduler_state)
        n = self._installation.discovery.config.option_budget
        contracts = (
            (
                state.canonical_scheduler_checksum,
                "canonical_scheduler_checksum",
                (2,),
                jnp.uint32,
            ),
            (state.installed_slot_mask, "installed_slot_mask", (n,), jnp.bool_),
            (state.descriptor_generation, "descriptor_generation", (), jnp.int32),
            (state.descriptor_digest, "descriptor_digest", (8,), jnp.uint32),
            (
                state.expected_retirement_authority_issuer_digest,
                "expected_retirement_authority_issuer_digest",
                (8,),
                jnp.uint32,
            ),
            (state.controller_owner_digest, "controller_owner_digest", (8,), jnp.uint32),
            (state.controller_revision, "controller_revision", (), jnp.int32),
            (state.retirement_words, "retirement_words", (2,), jnp.uint32),
            (
                state.last_retirement_authority_revision_words,
                "last_retirement_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (
                state.last_retirement_scheduler_step_words,
                "last_retirement_scheduler_step_words",
                (2,),
                jnp.uint32,
            ),
            (state.retirement_unavailable, "retirement_unavailable", (), jnp.bool_),
            (state.retirement_error, "retirement_error", (), jnp.int32),
            (state.replacement_words, "replacement_words", (2,), jnp.uint32),
            (
                state.last_replacement_authority_revision_words,
                "last_replacement_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (state.binding_checksum, "binding_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def _as_retirement_state(
        self,
        state: AuthorizedOptionReplacementState,
    ) -> AuthorizedOptionRetirementState:
        projected = AuthorizedOptionRetirementState(
            installation_state=state.scheduler_state.installation_state,
            installed_slot_mask=state.installed_slot_mask,
            descriptor_generation=state.descriptor_generation,
            descriptor_digest=state.descriptor_digest,
            expected_authority_issuer_digest=(state.expected_retirement_authority_issuer_digest),
            controller_owner_digest=state.controller_owner_digest,
            controller_revision=state.controller_revision,
            retirement_words=state.retirement_words,
            last_authority_revision_words=(state.last_retirement_authority_revision_words),
            last_scheduler_step_words=state.last_retirement_scheduler_step_words,
            unavailable=state.retirement_unavailable,
            error=state.retirement_error,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._retirement._with_checksum(projected)

    def _replace_from_retirement(
        self,
        source: AuthorizedOptionReplacementState,
        scheduler_state: CumulantOptionSchedulerState,
        retirement_state: AuthorizedOptionRetirementState,
    ) -> AuthorizedOptionReplacementState:
        return self._with_checksum(
            dataclasses.replace(
                source,
                scheduler_state=scheduler_state,
                canonical_scheduler_checksum=scheduler_state.binding_checksum,
                installed_slot_mask=retirement_state.installed_slot_mask,
                descriptor_generation=retirement_state.descriptor_generation,
                descriptor_digest=retirement_state.descriptor_digest,
                expected_retirement_authority_issuer_digest=(
                    retirement_state.expected_authority_issuer_digest
                ),
                controller_owner_digest=retirement_state.controller_owner_digest,
                controller_revision=retirement_state.controller_revision,
                retirement_words=retirement_state.retirement_words,
                last_retirement_authority_revision_words=(
                    retirement_state.last_authority_revision_words
                ),
                last_retirement_scheduler_step_words=(retirement_state.last_scheduler_step_words),
                retirement_unavailable=retirement_state.unavailable,
                retirement_error=retirement_state.error,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )

    def init(
        self,
        scheduler_state: CumulantOptionSchedulerState,
        *,
        retirement_authority_issuer_digest: Array,
        controller_owner_digest: Array,
    ) -> AuthorizedOptionReplacementState:
        """Bind a live scheduler cohort before the in-controller retirement."""

        self._scheduler._check_state_contract(scheduler_state)
        if not bool(jax.device_get(self._scheduler.state_valid(scheduler_state))):
            raise ValueError("scheduler_state must satisfy the complete scheduler contract")
        retirement_state = self._retirement.init(
            scheduler_state.installation_state,
            authority_issuer_digest=retirement_authority_issuer_digest,
            controller_owner_digest=controller_owner_digest,
        )
        state = AuthorizedOptionReplacementState(
            scheduler_state=scheduler_state,
            canonical_scheduler_checksum=scheduler_state.binding_checksum,
            installed_slot_mask=retirement_state.installed_slot_mask,
            descriptor_generation=retirement_state.descriptor_generation,
            descriptor_digest=retirement_state.descriptor_digest,
            expected_retirement_authority_issuer_digest=(
                retirement_state.expected_authority_issuer_digest
            ),
            controller_owner_digest=retirement_state.controller_owner_digest,
            controller_revision=retirement_state.controller_revision,
            retirement_words=retirement_state.retirement_words,
            last_retirement_authority_revision_words=(
                retirement_state.last_authority_revision_words
            ),
            last_retirement_scheduler_step_words=(retirement_state.last_scheduler_step_words),
            retirement_unavailable=retirement_state.unavailable,
            retirement_error=retirement_state.error,
            replacement_words=_words(0),
            last_replacement_authority_revision_words=_words(0),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        state = self._with_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initialized replacement state failed its exact contract")
        return state

    def state_valid(self, state: AuthorizedOptionReplacementState) -> Bool[Array, ""]:
        """Validate the single canonical child, both authority bindings, and checksum."""

        self._check_state_contract(state)
        projected = self._as_retirement_state(state)
        zero_retirements = jnp.array_equal(state.retirement_words, _words(0))
        one_retirement = jnp.array_equal(state.retirement_words, _words(1))
        zero_replacements = jnp.array_equal(state.replacement_words, _words(0))
        one_replacement = jnp.array_equal(state.replacement_words, _words(1))
        cold_count = jnp.sum(~state.installed_slot_mask, dtype=jnp.int32)
        phase_contract = (
            (zero_retirements & zero_replacements & (cold_count == 0))
            | (one_retirement & zero_replacements & (cold_count == 1))
            | (one_retirement & one_replacement & (cold_count == 0))
        )
        authority_clock_contract = jnp.where(
            zero_replacements,
            jnp.all(state.last_replacement_authority_revision_words == 0),
            jnp.any(state.last_replacement_authority_revision_words != 0),
        )
        return (
            self._scheduler.state_valid(state.scheduler_state)
            & jnp.array_equal(
                state.canonical_scheduler_checksum,
                state.scheduler_state.binding_checksum,
            )
            & self._retirement.state_valid(projected)
            & (zero_retirements | one_retirement)
            & _words_less_equal(
                state.replacement_words,
                _words(self._config.max_replacements),
            )
            & (state.replacement_words[0] == 0)
            & phase_contract
            & authority_clock_contract
            & (
                state.controller_revision
                >= state.retirement_words[1].astype(jnp.int32)
                + state.replacement_words[1].astype(jnp.int32)
            )
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def retire(
        self,
        state: AuthorizedOptionReplacementState,
        handoff: CumulantOptionRetirementHandoff,
        authority_receipt: OptionRetirementAuthorityReceipt,
        phase_one_key: Array,
        phase_two_key: Array,
    ) -> AuthorizedOptionReplacementRetirementResult:
        """Execute and atomically adopt the exact authorized retirement source."""

        self._check_state_contract(state)
        source_valid = self.state_valid(state)
        phase_valid = (
            jnp.array_equal(state.retirement_words, _words(0))
            & jnp.array_equal(state.replacement_words, _words(0))
            & jnp.all(state.installed_slot_mask)
        )
        projected = self._as_retirement_state(state)
        retired = self._retirement.retire(
            projected,
            handoff,
            authority_receipt,
            phase_one_key,
            phase_two_key,
        )
        rebased_scheduler = self._scheduler._with_checksum(
            dataclasses.replace(
                state.scheduler_state,
                installation_state=retired.state.installation_state,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        candidate = self._replace_from_retirement(
            state,
            rebased_scheduler,
            retired.state,
        )
        exactly_one_cold = jnp.sum(~candidate.installed_slot_mask, dtype=jnp.int32) == 1
        scheduler_valid = self._scheduler.state_valid(rebased_scheduler)
        candidate_valid = self.state_valid(candidate)
        applied = (
            source_valid
            & phase_valid
            & retired.transaction_applied
            & exactly_one_cold
            & scheduler_valid
            & candidate_valid
        )
        next_state = cast(
            AuthorizedOptionReplacementState,
            jax.lax.cond(applied, lambda _: candidate, lambda _: state, None),
        )
        return AuthorizedOptionReplacementRetirementResult(
            state=next_state,
            retirement=retired,
            source_state_valid=source_valid,
            phase_valid=phase_valid,
            canonical_scheduler_state_valid=scheduler_valid,
            transaction_applied=applied,
        )

    def extended_action_mask(
        self,
        state: AuthorizedOptionReplacementState,
    ) -> Bool[Array, " n_total_actions"]:
        """Return the retirement controller's authoritative behavior mask."""

        self._check_state_contract(state)
        return self._retirement.extended_action_mask(self._as_retirement_state(state))

    def arm(
        self,
        state: AuthorizedOptionReplacementState,
        inputs: CumulantOptionSchedulerArmInputs,
    ) -> AuthorizedOptionReplacementArm:
        """Arm one accepted transition while binding the unique cold destination."""

        self._check_state_contract(state)
        scheduler_arm = self._scheduler.arm(state.scheduler_state, inputs)
        cold = ~state.installed_slot_mask
        exactly_one = jnp.sum(cold, dtype=jnp.int32) == 1
        target = jnp.argmax(cold.astype(jnp.int32)).astype(jnp.int32)
        replacement_capacity = _words_less(
            state.replacement_words,
            _words(self._config.max_replacements),
        )
        available = (
            self.state_valid(state) & exactly_one & replacement_capacity & scheduler_arm.available
        )
        return AuthorizedOptionReplacementArm(
            scheduler_arm=scheduler_arm,
            source_checksum=state.binding_checksum,
            source_controller_revision=state.controller_revision,
            target_slot=target,
            target_mask=cold,
            available=available,
        )

    def _denied_installation_receipt(
        self,
        state: AuthorizedOptionReplacementState,
        live_inputs: CumulantOptionLiveInputs,
    ) -> CumulantOptionInstallationAuthorityReceipt:
        next_revision, _ = _increment_words(state.scheduler_state.last_authority_revision_words)
        return CumulantOptionInstallationAuthorityReceipt(
            go_no_go_authorized=jnp.asarray(False, dtype=jnp.bool_),
            safety_boundary_authorized=jnp.asarray(False, dtype=jnp.bool_),
            semantic_generation=live_inputs.semantic_generation,
            source_digest=live_inputs.source_digest,
            canonical_digest=live_inputs.canonical_digest,
            valid_from_step_words=state.scheduler_state.step_words,
            valid_through_step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
            issuer_digest=state.scheduler_state.expected_authority_issuer_digest,
            authority_revision_words=next_revision,
        )

    def _prepared_payload_arrays(
        self,
        prepared: AuthorizedOptionReplacementPrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    prepared.source_state,
                    prepared.arm,
                    prepared.observation,
                    prepared.live_inputs,
                    prepared.scheduler_result,
                    prepared.fallback_state,
                    prepared.candidate_semantic_digests,
                    prepared.changed_slots,
                    prepared.target_slot,
                    prepared.target_mask,
                    prepared.diagnostics,
                )
            )
        )

    def _with_prepared_checksum(
        self,
        prepared: AuthorizedOptionReplacementPrepared,
    ) -> AuthorizedOptionReplacementPrepared:
        return dataclasses.replace(
            prepared,
            prepared_checksum=_checksum_arrays(self._prepared_payload_arrays(prepared)),
        )

    def prepare(
        self,
        state: AuthorizedOptionReplacementState,
        arm: AuthorizedOptionReplacementArm,
        observation: CumulantOptionSchedulerObservation,
        live_inputs: CumulantOptionLiveInputs,
    ) -> AuthorizedOptionReplacementPrepared:
        """Stage ordinary observation plus one transient, unauthorized candidate.

        The scheduler is deliberately called with both authority bits false.
        Consequently its returned persistent candidate is always the incumbent
        live materialization, never the discovered proposal.  If a proposal was
        due and ready, the staged scheduler state records only its ordinary
        bounded retry marker.
        """

        self._check_state_contract(state)
        if type(arm) is not AuthorizedOptionReplacementArm:
            raise TypeError("arm must be an exact AuthorizedOptionReplacementArm")
        result = self._scheduler.observe(
            state.scheduler_state,
            arm.scheduler_arm,
            observation,
            live_inputs,
            self._denied_installation_receipt(state, live_inputs),
        )
        source_valid = self.state_valid(state)
        arm_binding = (
            arm.available
            & jnp.array_equal(arm.source_checksum, state.binding_checksum)
            & (arm.source_controller_revision == state.controller_revision)
            & jnp.array_equal(arm.target_mask, ~state.installed_slot_mask)
            & (arm.target_slot == jnp.argmax(arm.target_mask.astype(jnp.int32)))
        )
        revision_available = state.controller_revision < _INT32_MAX
        fallback_candidate = self._with_checksum(
            dataclasses.replace(
                state,
                scheduler_state=result.state,
                canonical_scheduler_checksum=result.state.binding_checksum,
                controller_revision=_saturating_increment(state.controller_revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        fallback_valid = (
            source_valid
            & arm_binding
            & result.transaction_valid
            & revision_available
            & self.state_valid(fallback_candidate)
        )
        fallback = cast(
            AuthorizedOptionReplacementState,
            jax.lax.cond(fallback_valid, lambda _: fallback_candidate, lambda _: state, None),
        )
        bundle = result.discovery.discovered
        candidate_semantics = self._installation.semantic_digests_for_bundle(bundle)
        source_semantics = state.scheduler_state.installation_state.installed_semantic_digests
        changed = jnp.any(candidate_semantics != source_semantics, axis=1)
        exactly_one_cold = jnp.sum(~state.installed_slot_mask, dtype=jnp.int32) == 1
        target_still_cold = (~state.installed_slot_mask)[
            jnp.clip(arm.target_slot, 0, arm.target_mask.shape[0] - 1)
        ]
        exact_change = jnp.array_equal(changed, arm.target_mask)
        live_preserved = ~jnp.any(changed & state.installed_slot_mask)
        proposal_binding = self._scheduler.discovery.validate_proposal_bundle(
            bundle,
            semantic_generation=live_inputs.semantic_generation,
            source_digest=live_inputs.source_digest,
            canonical_digest=live_inputs.canonical_digest,
            transition_id=live_inputs.transition_id,
            state_observation_count=live_inputs.state_observation_count,
        ) & (bundle.cohort_id == -1)
        installation_source = state.scheduler_state.installation_state
        fresh = _words_less(
            installation_source.last_materialization_transition_id,
            bundle.transition_id,
        ) & (
            bundle.state_observation_count
            > installation_source.last_materialization_observation_count
        )
        installer_capacity = ~installation_source.installer_unavailable & (
            installation_source.installation_count < self._installation.config.max_installations
        )
        transaction_valid = source_valid & arm_binding & result.transaction_valid & fallback_valid
        candidate_ready = (
            transaction_valid
            & result.proposal_due
            & result.proposal_ready
            & proposal_binding
            & fresh
            & exactly_one_cold
            & target_still_cold
            & exact_change
            & live_preserved
            & result.quiescent_boundary
            & result.installation_attempt_capacity_available
            & installer_capacity
        )
        diagnostics = AuthorizedOptionReplacementPrepareDiagnostics(
            source_state_valid=source_valid,
            arm_binding_valid=arm_binding,
            ordinary_scheduler_transaction_valid=result.transaction_valid,
            fallback_state_valid=fallback_valid,
            proposal_due=result.proposal_due,
            proposal_ready=result.proposal_ready,
            proposal_binding_valid=proposal_binding,
            fresh_transition=fresh,
            exactly_one_cold_slot=exactly_one_cold,
            target_still_cold=target_still_cold,
            exact_one_slot_semantic_change=exact_change,
            live_slots_semantically_preserved=live_preserved,
            quiescent=result.quiescent_boundary,
            scheduler_attempt_capacity_available=(result.installation_attempt_capacity_available),
            installer_capacity_available=installer_capacity,
            candidate_ready_for_authority=candidate_ready,
            transaction_valid=transaction_valid,
        )
        prepared = AuthorizedOptionReplacementPrepared(
            source_state=state,
            arm=arm,
            observation=observation,
            live_inputs=live_inputs,
            scheduler_result=result,
            fallback_state=fallback,
            candidate_semantic_digests=candidate_semantics,
            changed_slots=changed,
            target_slot=arm.target_slot,
            target_mask=arm.target_mask,
            diagnostics=diagnostics,
            prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_prepared_checksum(prepared)

    def _check_receipt_contract(self, receipt: OptionReplacementAuthorityReceipt) -> None:
        if type(receipt) is not OptionReplacementAuthorityReceipt:
            raise TypeError("authority_receipt must be an exact OptionReplacementAuthorityReceipt")
        self._scheduler._check_authority_contract(receipt.installation_authority)
        n = self._installation.discovery.config.option_budget
        contracts = (
            (receipt.replacement_authorized, "replacement_authorized", (), jnp.bool_),
            (receipt.controller_owner_digest, "controller_owner_digest", (8,), jnp.uint32),
            (receipt.source_scheduler_checksum, "source_scheduler_checksum", (2,), jnp.uint32),
            (receipt.source_controller_revision, "source_controller_revision", (), jnp.int32),
            (
                receipt.source_descriptor_generation,
                "source_descriptor_generation",
                (),
                jnp.int32,
            ),
            (receipt.source_descriptor_digest, "source_descriptor_digest", (8,), jnp.uint32),
            (
                receipt.source_installed_slot_mask,
                "source_installed_slot_mask",
                (n,),
                jnp.bool_,
            ),
            (
                receipt.source_installation_revision,
                "source_installation_revision",
                (),
                jnp.int32,
            ),
            (
                receipt.source_lifecycle_revision,
                "source_lifecycle_revision",
                (),
                jnp.int32,
            ),
            (receipt.source_audit_revision, "source_audit_revision", (), jnp.int32),
            (receipt.replacement_slot, "replacement_slot", (), jnp.int32),
            (receipt.expected_reset_slots, "expected_reset_slots", (n,), jnp.bool_),
            (receipt.candidate_binding_digest, "candidate_binding_digest", (2,), jnp.uint32),
            (
                receipt.candidate_semantic_digests,
                "candidate_semantic_digests",
                (n, 8),
                jnp.uint32,
            ),
            (receipt.candidate_transition_id, "candidate_transition_id", (2,), jnp.uint32),
            (
                receipt.candidate_state_observation_count,
                "candidate_state_observation_count",
                (),
                jnp.int32,
            ),
            (receipt.prepared_checksum, "prepared_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"authority_receipt.{name}", shape=shape, dtype=dtype)

    def _authority_valid(
        self,
        state: AuthorizedOptionReplacementState,
        prepared: AuthorizedOptionReplacementPrepared,
        receipt: OptionReplacementAuthorityReceipt,
    ) -> Array:
        installation = state.scheduler_state.installation_state
        lifecycle = installation.lifecycle_state
        audit = lifecycle.audit_state
        bundle = prepared.scheduler_result.discovery.discovered
        nested = receipt.installation_authority
        nested_valid = self._scheduler._authority_valid(
            state.scheduler_state,
            nested,
            prepared.live_inputs,
            prepared.scheduler_result.state.step_words,
        )
        return (
            receipt.replacement_authorized
            & nested.go_no_go_authorized
            & nested.safety_boundary_authorized
            & nested_valid
            & _words_less(
                state.last_replacement_authority_revision_words,
                nested.authority_revision_words,
            )
            & jnp.array_equal(receipt.controller_owner_digest, state.controller_owner_digest)
            & jnp.array_equal(
                receipt.source_scheduler_checksum,
                state.scheduler_state.binding_checksum,
            )
            & (receipt.source_controller_revision == state.controller_revision)
            & (receipt.source_descriptor_generation == state.descriptor_generation)
            & jnp.array_equal(receipt.source_descriptor_digest, state.descriptor_digest)
            & jnp.array_equal(
                receipt.source_installed_slot_mask,
                state.installed_slot_mask,
            )
            & (receipt.source_installation_revision == installation.revision)
            & (receipt.source_lifecycle_revision == lifecycle.revision)
            & (receipt.source_audit_revision == audit.revision)
            & (receipt.replacement_slot == prepared.target_slot)
            & jnp.array_equal(receipt.expected_reset_slots, prepared.target_mask)
            & jnp.array_equal(receipt.candidate_binding_digest, bundle.binding_digest)
            & jnp.array_equal(
                receipt.candidate_semantic_digests,
                prepared.candidate_semantic_digests,
            )
            & jnp.array_equal(receipt.candidate_transition_id, bundle.transition_id)
            & (receipt.candidate_state_observation_count == bundle.state_observation_count)
            & jnp.array_equal(receipt.prepared_checksum, prepared.prepared_checksum)
        )

    def authority_receipt(
        self,
        prepared: AuthorizedOptionReplacementPrepared,
        installation_authority: CumulantOptionInstallationAuthorityReceipt,
        *,
        replacement_authorized: bool | Array,
    ) -> OptionReplacementAuthorityReceipt:
        """Construct an exact caller declaration for a prepared candidate.

        This convenience method binds facts but does not authenticate the
        caller and does not decide either boolean authority bit.  The caller
        must supply a separately created installation receipt and explicitly
        state whether replacement is authorized.
        """

        if type(prepared) is not AuthorizedOptionReplacementPrepared:
            raise TypeError("prepared must be an exact AuthorizedOptionReplacementPrepared")
        self._scheduler._check_authority_contract(installation_authority)
        source = prepared.source_state
        installation = source.scheduler_state.installation_state
        bundle = prepared.scheduler_result.discovery.discovered
        return OptionReplacementAuthorityReceipt(
            installation_authority=installation_authority,
            replacement_authorized=jnp.asarray(replacement_authorized, dtype=jnp.bool_),
            controller_owner_digest=source.controller_owner_digest,
            source_scheduler_checksum=source.scheduler_state.binding_checksum,
            source_controller_revision=source.controller_revision,
            source_descriptor_generation=source.descriptor_generation,
            source_descriptor_digest=source.descriptor_digest,
            source_installed_slot_mask=source.installed_slot_mask,
            source_installation_revision=installation.revision,
            source_lifecycle_revision=installation.lifecycle_state.revision,
            source_audit_revision=installation.lifecycle_state.audit_state.revision,
            replacement_slot=prepared.target_slot,
            expected_reset_slots=prepared.target_mask,
            candidate_binding_digest=bundle.binding_digest,
            candidate_semantic_digests=prepared.candidate_semantic_digests,
            candidate_transition_id=bundle.transition_id,
            candidate_state_observation_count=bundle.state_observation_count,
            prepared_checksum=prepared.prepared_checksum,
        )

    def commit(
        self,
        state: AuthorizedOptionReplacementState,
        prepared: AuthorizedOptionReplacementPrepared,
        authority_receipt: OptionReplacementAuthorityReceipt,
    ) -> AuthorizedOptionReplacementResult:
        """Re-derive then commit ordinary observation and one exact replacement.

        This is intentionally a host boundary.  The caller-supplied preparation
        is not trusted merely because its local checksum and receipt agree.  The
        scheduler observation is rerun from its complete source transaction and
        every leaf of the resulting preparation is compared bit-for-bit before
        the separately compiled array kernel may advance either path.
        """

        self._check_state_contract(state)
        if type(prepared) is not AuthorizedOptionReplacementPrepared:
            raise TypeError("prepared must be an exact AuthorizedOptionReplacementPrepared")
        self._check_receipt_contract(authority_receipt)
        supplied_integrity = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._prepared_payload_arrays(prepared)),
        )
        recomputed = self.prepare(
            prepared.source_state,
            prepared.arm,
            prepared.observation,
            prepared.live_inputs,
        )
        derivation_valid = _tree_array_equal(prepared, recomputed)
        return self._commit_host_transaction(
            state,
            recomputed,
            authority_receipt,
            supplied_integrity,
            derivation_valid,
        )

    @staticmethod
    def _atomic_adoption_kernel(
        destination: AuthorizedOptionReplacementState,
        ordinary_fallback: AuthorizedOptionReplacementState,
        installed_candidate: AuthorizedOptionReplacementState,
        ordinary_advance: Array,
        replacement_applied: Array,
    ) -> AuthorizedOptionReplacementState:
        """JIT-safe whole-state selection after host-derived transaction facts."""

        fallback_or_destination = cast(
            AuthorizedOptionReplacementState,
            jax.lax.cond(
                ordinary_advance,
                lambda _: ordinary_fallback,
                lambda _: destination,
                None,
            ),
        )
        return cast(
            AuthorizedOptionReplacementState,
            jax.lax.cond(
                replacement_applied,
                lambda _: installed_candidate,
                lambda _: fallback_or_destination,
                None,
            ),
        )

    def _commit_host_transaction(
        self,
        state: AuthorizedOptionReplacementState,
        prepared: AuthorizedOptionReplacementPrepared,
        authority_receipt: OptionReplacementAuthorityReceipt,
        supplied_prepared_integrity: Array,
        preparation_derivation_valid: Array,
    ) -> AuthorizedOptionReplacementResult:
        """Host-composed transaction entered only after provenance replay.

        A valid preparation is always consumed exactly once as an ordinary
        scheduler/discovery observation.  Invalid or declined authority chooses
        ``prepared.fallback_state``: that state contains the incumbent live
        materialization and at most a retry bit, never the candidate bundle,
        candidate tail, installation key successor, or a reactivated mask.
        Stale/tampered preparations do not even commit the ordinary advance.
        """

        self._check_state_contract(state)
        if type(prepared) is not AuthorizedOptionReplacementPrepared:
            raise TypeError("prepared must be an exact AuthorizedOptionReplacementPrepared")
        self._check_receipt_contract(authority_receipt)
        source = prepared.source_state
        destination_valid = self.state_valid(state)
        destination_matches = _tree_array_equal(state, source)
        prepared_integrity = jnp.asarray(
            supplied_prepared_integrity, dtype=jnp.bool_
        ) & jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._prepared_payload_arrays(prepared)),
        )
        derivation_valid = jnp.asarray(preparation_derivation_valid, dtype=jnp.bool_)
        prepared_valid = (
            prepared.diagnostics.transaction_valid
            & self.state_valid(source)
            & self.state_valid(prepared.fallback_state)
        )
        ordinary_advance = (
            destination_valid
            & destination_matches
            & prepared_integrity
            & derivation_valid
            & prepared_valid
        )
        authority_valid = self._authority_valid(source, prepared, authority_receipt)
        replacement_capacity = _words_less(
            source.replacement_words,
            _words(self._config.max_replacements),
        )
        candidate_ready = prepared.diagnostics.candidate_ready_for_authority
        attempt = ordinary_advance & authority_valid & candidate_ready & replacement_capacity

        next_rng_key, fresh_install_key = jr.split(source.scheduler_state.installation_rng_key)
        bundle = prepared.scheduler_result.discovery.discovered
        installation_result: CumulantOptionInstallationResult = self._installation.install(
            source.scheduler_state.installation_state,
            bundle,
            fresh_install_key,
            inputs=prepared.live_inputs,
        )
        exact_reset = jnp.array_equal(
            installation_result.reset_slots,
            prepared.target_mask,
        )
        exact_preserve = jnp.array_equal(
            installation_result.preserved_slots,
            ~prepared.target_mask,
        )
        next_attempt_words, attempt_counter_capacity = _increment_words(
            source.scheduler_state.install_attempt_words
        )
        next_applied_words, applied_counter_capacity = _increment_words(
            source.scheduler_state.install_applied_words
        )
        installed_scheduler = self._scheduler._with_checksum(
            dataclasses.replace(
                prepared.scheduler_result.state,
                installation_state=installation_result.state,
                installation_rng_key=next_rng_key,
                install_attempt_words=next_attempt_words,
                install_applied_words=next_applied_words,
                last_authority_revision_words=(
                    authority_receipt.installation_authority.authority_revision_words
                ),
                retry_streak=jnp.asarray(0, dtype=jnp.int32),
                retry_due=jnp.asarray(False, dtype=jnp.bool_),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        installed_scheduler_valid = self._scheduler.state_valid(installed_scheduler)
        next_replacement_words, replacement_counter_capacity = _increment_words(
            source.replacement_words
        )
        installed_candidate = self._with_checksum(
            dataclasses.replace(
                source,
                scheduler_state=installed_scheduler,
                canonical_scheduler_checksum=installed_scheduler.binding_checksum,
                installed_slot_mask=source.installed_slot_mask | prepared.target_mask,
                descriptor_generation=bundle.semantic_generation,
                descriptor_digest=_descriptor_digest(bundle.selected_descriptors),
                controller_revision=_saturating_increment(source.controller_revision),
                replacement_words=next_replacement_words,
                last_replacement_authority_revision_words=(
                    authority_receipt.installation_authority.authority_revision_words
                ),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        installed_state_valid = self.state_valid(installed_candidate)
        replacement_applied = (
            attempt
            & installation_result.transaction_valid
            & installation_result.applied
            & exact_reset
            & exact_preserve
            & installation_result.live_policy_rng_preserved
            & attempt_counter_capacity
            & applied_counter_capacity
            & replacement_counter_capacity
            & (source.controller_revision < _INT32_MAX)
            & installed_scheduler_valid
            & installed_state_valid
        )
        next_state = cast(
            AuthorizedOptionReplacementState,
            self._compiled_atomic_adoption_kernel(
                state,
                prepared.fallback_state,
                installed_candidate,
                ordinary_advance,
                replacement_applied,
            ),
        )
        materialization = cast(
            CumulantOptionMaterialization,
            jax.lax.cond(
                replacement_applied,
                lambda _: installation_result.materialization,
                lambda _: prepared.scheduler_result.materialization,
                None,
            ),
        )
        installed_handoff = self._scheduler._retirement_handoff(
            prepared.scheduler_result.discovery.state,
            installation_result.state,
            prepared.scheduler_result.state.step_words,
            available=(
                prepared.scheduler_result.retirement_handoff.available & replacement_applied
            ),
        )
        handoff = cast(
            CumulantOptionRetirementHandoff,
            jax.lax.cond(
                replacement_applied,
                lambda _: installed_handoff,
                lambda _: prepared.scheduler_result.retirement_handoff,
                None,
            ),
        )
        return AuthorizedOptionReplacementResult(
            state=next_state,
            discovery=prepared.scheduler_result.discovery,
            materialization=materialization,
            retirement_handoff=handoff,
            diagnostics=AuthorizedOptionReplacementCommitDiagnostics(
                destination_state_valid=destination_valid,
                destination_matches_source=destination_matches,
                prepared_integrity_valid=prepared_integrity,
                preparation_derivation_valid=derivation_valid,
                prepared_transaction_valid=prepared_valid,
                authority_valid=authority_valid,
                candidate_ready=candidate_ready,
                replacement_capacity_available=replacement_capacity,
                installation_transaction_valid=(attempt & installation_result.transaction_valid),
                installation_applied=attempt & installation_result.applied,
                exact_reset_mask=attempt & exact_reset,
                exact_preserve_mask=attempt & exact_preserve,
                live_policy_rng_preserved=(attempt & installation_result.live_policy_rng_preserved),
                installed_scheduler_state_valid=attempt & installed_scheduler_valid,
                installed_replacement_state_valid=attempt & installed_state_valid,
                ordinary_advance_applied=ordinary_advance,
                replacement_attempted=attempt,
                replacement_applied=replacement_applied,
                proposal_persisted=jnp.asarray(False, dtype=jnp.bool_),
                candidate_materialization_persisted_on_decline=jnp.asarray(
                    False,
                    dtype=jnp.bool_,
                ),
            ),
            reset_slots=replacement_applied & installation_result.reset_slots,
            preserved_slots=replacement_applied & installation_result.preserved_slots,
            extended_action_mask=self.extended_action_mask(next_state),
            cold_mask_active=jnp.any(~next_state.installed_slot_mask),
            retry_scheduled=(
                ordinary_advance
                & (~replacement_applied)
                & prepared.scheduler_result.retry_scheduled
            ),
            fresh_bundle_required_on_retry=jnp.asarray(True, dtype=jnp.bool_),
        )

    def _adopt_retirement_control_state(
        self,
        source: AuthorizedOptionReplacementState,
        retirement_state: AuthorizedOptionRetirementState,
    ) -> tuple[AuthorizedOptionReplacementState, bool]:
        """Reproject one host lifecycle update into the canonical scheduler."""

        scheduler_state, scheduler_applied = self._scheduler._commit_control_state(
            source.scheduler_state,
            retirement_state.installation_state,
        )
        if not scheduler_applied:
            return source, False
        proposed = self._replace_from_retirement(
            source,
            scheduler_state,
            retirement_state,
        )
        applied = bool(jax.device_get(self.state_valid(proposed)))
        return (proposed if applied else source), applied

    def start(
        self,
        state: AuthorizedOptionReplacementState,
        materialization: CumulantOptionMaterialization,
    ) -> AuthorizedOptionReplacementStartResult:
        """Host-only masked lifecycle start through the canonical installation."""

        self._check_state_contract(state)
        child = self._retirement.start(self._as_retirement_state(state), materialization)
        if not child.applied:
            return AuthorizedOptionReplacementStartResult(state, None, False)
        proposed, applied = self._adopt_retirement_control_state(state, child.state)
        return AuthorizedOptionReplacementStartResult(
            proposed if applied else state,
            child if applied else None,
            applied,
        )

    def update(
        self,
        state: AuthorizedOptionReplacementState,
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
    ) -> AuthorizedOptionReplacementUpdateResult:
        """Host-only masked lifecycle update through the canonical installation."""

        self._check_state_contract(state)
        child = self._retirement.update(
            self._as_retirement_state(state),
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
            return AuthorizedOptionReplacementUpdateResult(state, None, False)
        proposed, applied = self._adopt_retirement_control_state(state, child.state)
        return AuthorizedOptionReplacementUpdateResult(
            proposed if applied else state,
            child if applied else None,
            applied,
        )

    def release_scheduler_state(
        self,
        state: AuthorizedOptionReplacementState,
    ) -> CumulantOptionSchedulerState:
        """Release the canonical child only after the cold slot was reactivated."""

        self._check_state_contract(state)
        releasable = (
            self.state_valid(state)
            & jnp.array_equal(state.retirement_words, _words(1))
            & jnp.array_equal(state.replacement_words, _words(1))
            & jnp.all(state.installed_slot_mask)
        )
        if not bool(jax.device_get(releasable)):
            raise ValueError("scheduler state is not releasable before exact replacement")
        return state.scheduler_state

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
            raise ValueError("encoded replacement array differs from schema v1")
        dtype = np.dtype(value["dtype"])
        shape = value["shape"]
        payload = value["bytes_hex"]
        if type(shape) is not list or any(type(cell) is not int or cell < 0 for cell in shape):
            raise ValueError("encoded replacement array shape is invalid")
        if type(payload) is not str:
            raise ValueError("encoded replacement array bytes must be hex")
        try:
            raw = bytes.fromhex(payload)
        except ValueError as exc:
            raise ValueError("encoded replacement array bytes are not hex") from exc
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected:
            raise ValueError("encoded replacement array byte length differs")
        return jnp.asarray(np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy())

    @staticmethod
    def _state_sha256(state: AuthorizedOptionReplacementState) -> str:
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
        state: AuthorizedOptionReplacementState,
    ) -> dict[str, object]:
        """Return strict v1 state only; preparations and proposals are absent."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid replacement state")
        controller_fields = {
            field.name: self._encode_array(cast(Array, getattr(state, field.name)))
            for field in dataclasses.fields(AuthorizedOptionReplacementState)
            if field.name != "scheduler_state"
        }
        return {
            "schema_version": AUTHORIZED_OPTION_REPLACEMENT_CHECKPOINT_SCHEMA,
            "state_type": "AuthorizedOptionReplacementState",
            "config": self.to_config(),
            "scheduler": self._scheduler.checkpoint_payload(state.scheduler_state),
            "controller_fields": controller_fields,
            "state_sha256": self._state_sha256(state),
            "proposal_persisted": False,
            "candidate_materialization_persisted": False,
            "assessment": AUTHORIZED_OPTION_REPLACEMENT_ASSESSMENT,
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
    ) -> AuthorizedOptionReplacementState:
        """Restore only the exact config, checksum, and independent bindings."""

        if type(value) is not dict:
            raise ValueError("replacement checkpoint must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "state_type",
            "config",
            "scheduler",
            "controller_fields",
            "state_sha256",
            "proposal_persisted",
            "candidate_materialization_persisted",
            "assessment",
            "evidence_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("replacement checkpoint keys differ from schema v1")
        fixed = {
            "schema_version": AUTHORIZED_OPTION_REPLACEMENT_CHECKPOINT_SCHEMA,
            "state_type": "AuthorizedOptionReplacementState",
            "config": self.to_config(),
            "proposal_persisted": False,
            "candidate_materialization_persisted": False,
            "assessment": AUTHORIZED_OPTION_REPLACEMENT_ASSESSMENT,
            "evidence_authority": False,
            "scientific_promotion_allowed": False,
        }
        for name, expected_value in fixed.items():
            if raw[name] != expected_value:
                raise ValueError(f"replacement checkpoint {name} differs")
        fields = raw["controller_fields"]
        if type(fields) is not dict:
            raise ValueError("replacement checkpoint controller_fields must be a dict")
        expected_fields = {
            field.name
            for field in dataclasses.fields(AuthorizedOptionReplacementState)
            if field.name != "scheduler_state"
        }
        if set(fields) != expected_fields:
            raise ValueError("replacement checkpoint controller fields differ")
        scheduler = self._scheduler.restore_checkpoint(
            raw["scheduler"],
            expected_semantic_generation=expected_semantic_generation,
            expected_source_digest=expected_source_digest,
            expected_consumer_source_digest=expected_consumer_source_digest,
            expected_consumer_representation_digest=(expected_consumer_representation_digest),
            expected_lifecycle_id=expected_lifecycle_id,
            expected_authority_issuer_digest=(expected_installation_authority_issuer_digest),
            expected_installed_bundle=expected_installed_bundle,
        )
        decoded = {name: self._decode_array(fields[name]) for name in expected_fields}
        restored = AuthorizedOptionReplacementState(
            scheduler_state=scheduler,
            **cast(dict[str, Any], decoded),
        )
        retirement_issuer = _require_array(
            expected_retirement_authority_issuer_digest,
            name="expected_retirement_authority_issuer_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        owner = _require_array(
            expected_controller_owner_digest,
            name="expected_controller_owner_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        descriptor_generation = _require_array(
            expected_descriptor_generation,
            name="expected_descriptor_generation",
            shape=(),
            dtype=jnp.int32,
        )
        descriptor_digest = _require_array(
            expected_descriptor_digest,
            name="expected_descriptor_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        external_binding_valid = (
            jnp.array_equal(
                restored.expected_retirement_authority_issuer_digest,
                retirement_issuer,
            )
            & jnp.array_equal(restored.controller_owner_digest, owner)
            & (restored.descriptor_generation == descriptor_generation)
            & jnp.array_equal(restored.descriptor_digest, descriptor_digest)
        )
        if type(raw["state_sha256"]) is not str or raw["state_sha256"] != self._state_sha256(
            restored
        ):
            raise ValueError("replacement checkpoint state digest differs")
        if not bool(jax.device_get(external_binding_valid & self.state_valid(restored))):
            raise ValueError("restored replacement state is invalid or rebound")
        return restored

    def resource_budget(
        self,
        state: AuthorizedOptionReplacementState,
        prepared: AuthorizedOptionReplacementPrepared | None = None,
    ) -> AuthorizedOptionReplacementResourceBudget:
        """Measure canonical persistence and bounded two-phase transaction work."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("resource measurement requires a valid replacement state")
        if prepared is not None and type(prepared) is not AuthorizedOptionReplacementPrepared:
            raise TypeError("prepared must be an exact preparation or None")
        persistent_nbytes = _tree_nbytes(state)
        scheduler_nbytes = _tree_nbytes(state.scheduler_state)
        installation_nbytes = _tree_nbytes(state.scheduler_state.installation_state)
        prepared_nbytes = 0 if prepared is None else _tree_nbytes(prepared)
        return AuthorizedOptionReplacementResourceBudget(
            persistent_state_nbytes=persistent_nbytes,
            scheduler_state_nbytes=scheduler_nbytes,
            installation_state_nbytes=installation_nbytes,
            retirement_binding_nbytes=persistent_nbytes - scheduler_nbytes,
            duplicated_installation_state_nbytes=0,
            prepared_state_nbytes=prepared_nbytes,
            option_slots=self._installation.discovery.config.option_budget,
            pending_proposal_slots=0,
            max_replacements=self._config.max_replacements,
            prepare_scheduler_observations=1,
            commit_preparation_recomputations=1,
            max_installations_per_commit=1,
            max_lifecycle_rebinds_per_commit=1,
            max_fresh_template_initializations_per_commit=1,
            max_rng_splits_per_commit=1,
            proposal_persisted=False,
            candidate_materialization_persisted_on_decline=False,
            host_prepare=True,
            host_commit=True,
            jit_commit=False,
            jit_atomic_adoption_kernel=True,
            assessment=AUTHORIZED_OPTION_REPLACEMENT_ASSESSMENT,
            output_writes=False,
            evidence_authority=False,
            promotion_authority=False,
            safety_authority=False,
            go_no_go_authority=False,
            retirement_authority=False,
            discovery_authority=False,
            dispatch_authority=False,
            autonomous_curation_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=AUTHORIZED_OPTION_REPLACEMENT_CHECKPOINT_SCHEMA,
        )


AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_PREPARED_SCHEMA = (
    "alberta.authorized-option-external-candidate-adoption.prepared.v2"
)
AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_RECEIPT_SCHEMA = (
    "alberta.authorized-option-external-candidate-adoption.receipt.v2"
)
AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_RESULT_SCHEMA = (
    "alberta.authorized-option-external-candidate-adoption.result.v2"
)


@chex.dataclass(frozen=True)
class AuthorizedOptionExternalCandidateAdoptionDiagnostics:
    """Replacement-level proof for one filtered cold-slot candidate."""

    all_installed_source_valid: Bool[Array, ""]
    all_installed_source_complete: Bool[Array, ""]
    retirement_result_valid: Bool[Array, ""]
    one_cold_retired_destination_valid: Bool[Array, ""]
    exact_one_cold_target: Bool[Array, ""]
    retirement_target_exact: Bool[Array, ""]
    retirement_owner_preserved: Bool[Array, ""]
    retirement_revision_exact: Bool[Array, ""]
    ordinary_preparation_valid: Bool[Array, ""]
    scheduler_preparation_exact: Bool[Array, ""]
    external_candidate_ready: Bool[Array, ""]
    replacement_capacity_available: Bool[Array, ""]
    adoption_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AuthorizedOptionExternalCandidateAdoptionPrepared:
    """Versioned source-bound replacement preparation for an external cohort."""

    all_installed_source: AuthorizedOptionReplacementState
    retirement_result: AuthorizedOptionReplacementRetirementResult
    arm_inputs: CumulantOptionSchedulerArmInputs
    replacement_arm: AuthorizedOptionReplacementArm
    observation: CumulantOptionSchedulerObservation
    live_inputs: CumulantOptionLiveInputs
    candidate_bundle: CumulantSubtaskProposalBundle
    target_mask: Bool[Array, " option_budget"]
    installation_key: Array
    successor_scheduler_key: Array
    ordinary_prepared: AuthorizedOptionReplacementPrepared
    scheduler_prepared: CumulantOptionExternalBundleAdoptionPrepared
    replacement_identity_digest: UInt[Array, " 8"]
    diagnostics: AuthorizedOptionExternalCandidateAdoptionDiagnostics
    prepared_checksum: UInt[Array, " 2"]

    SCHEMA_VERSION: ClassVar[str] = (
        AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_PREPARED_SCHEMA
    )

    @property
    def schema_version(self) -> str:
        return self.SCHEMA_VERSION


@chex.dataclass(frozen=True)
class AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt:
    """Caller declaration binding replacement and nested scheduler adoption."""

    scheduler_authority: CumulantOptionExternalBundleAdoptionAuthorityReceipt
    adoption_authorized: Bool[Array, ""]
    replacement_identity_digest: UInt[Array, " 8"]
    controller_owner_digest: UInt[Array, " 8"]
    all_installed_source_checksum: UInt[Array, " 2"]
    source_controller_revision: Int[Array, ""]
    retired_destination_checksum: UInt[Array, " 2"]
    retired_controller_revision: Int[Array, ""]
    retirement_authority_revision_words: UInt[Array, " 2"]
    replacement_authority_revision_words: UInt[Array, " 2"]
    target_mask: Bool[Array, " option_budget"]
    candidate_binding_digest: UInt[Array, " 2"]
    prepared_checksum: UInt[Array, " 2"]
    scheduler_prepared_checksum: UInt[Array, " 2"]

    SCHEMA_VERSION: ClassVar[str] = (
        AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_RECEIPT_SCHEMA
    )


@chex.dataclass(frozen=True)
class AuthorizedOptionExternalCandidateAdoptionResult:
    """Replacement successor or exact one-cold transient destination."""

    state: AuthorizedOptionReplacementState
    scheduler_result: CumulantOptionExternalBundleAdoptionResult
    destination_state_valid: Bool[Array, ""]
    destination_matches_retired: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    preparation_derivation_valid: Bool[Array, ""]
    authority_valid: Bool[Array, ""]
    candidate_ready: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    reset_slots: Bool[Array, " option_budget"]
    preserved_slots: Bool[Array, " option_budget"]
    cold_state_active: Bool[Array, ""]
    installation_key_consumed: Array
    caller_authenticated: Bool[Array, ""]

    SCHEMA_VERSION: ClassVar[str] = (
        AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_RESULT_SCHEMA
    )


def _external_candidate_replacement_identity(
    controller: AuthorizedOptionReplacementController,
) -> Array:
    from alberta_framework.core.option_lifecycle_audit import option_semantic_digest

    return option_semantic_digest(
        {
            "schema": AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_PREPARED_SCHEMA,
            "replacement": controller.to_config(),
        }
    )


def _external_candidate_prepared_payload(
    prepared: AuthorizedOptionExternalCandidateAdoptionPrepared,
) -> tuple[Array, ...]:
    return tuple(
        cast(Array, leaf)
        for leaf in jax.tree_util.tree_leaves(
            tuple(
                getattr(prepared, field.name)
                for field in dataclasses.fields(
                    AuthorizedOptionExternalCandidateAdoptionPrepared
                )
                if field.name != "prepared_checksum"
            )
        )
    )


def prepare_authorized_option_external_candidate_adoption(
    controller: AuthorizedOptionReplacementController,
    all_installed_source: AuthorizedOptionReplacementState,
    retirement_result: AuthorizedOptionReplacementRetirementResult,
    arm_inputs: CumulantOptionSchedulerArmInputs,
    observation: CumulantOptionSchedulerObservation,
    live_inputs: CumulantOptionLiveInputs,
    candidate_bundle: CumulantSubtaskProposalBundle,
    target_mask: Array,
    installation_key: Array,
    successor_scheduler_key: Array,
) -> AuthorizedOptionExternalCandidateAdoptionPrepared:
    """Stage one exact filtered candidate over an exact retirement result."""

    if type(controller) is not AuthorizedOptionReplacementController:
        raise TypeError("controller must be exact AuthorizedOptionReplacementController")
    controller._check_state_contract(all_installed_source)
    if type(retirement_result) is not AuthorizedOptionReplacementRetirementResult:
        raise TypeError("retirement_result has the wrong exact replacement type")
    retired = retirement_result.state
    controller._check_state_contract(retired)
    if type(arm_inputs) is not CumulantOptionSchedulerArmInputs:
        raise TypeError("arm_inputs must be exact CumulantOptionSchedulerArmInputs")
    controller.scheduler.discovery.check_proposal_bundle_contract(candidate_bundle)
    target = _require_array(
        target_mask,
        name="target_mask",
        shape=(controller.scheduler.discovery.config.option_budget,),
        dtype=jnp.bool_,
    )
    replacement_arm = controller.arm(retired, arm_inputs)
    ordinary_prepared = controller.prepare(
        retired,
        replacement_arm,
        observation,
        live_inputs,
    )
    scheduler_prepared = prepare_cumulant_option_external_bundle_adoption(
        controller.scheduler,
        all_installed_source.scheduler_state,
        retired.scheduler_state,
        replacement_arm.scheduler_arm,
        observation,
        live_inputs,
        candidate_bundle,
        target,
        all_installed_source.installed_slot_mask,
        retired.installed_slot_mask,
        retirement_result.retirement.reset_slots,
        retired.last_retirement_authority_revision_words,
        installation_key,
        successor_scheduler_key,
    )
    source_valid = controller.state_valid(all_installed_source)
    source_complete = jnp.all(all_installed_source.installed_slot_mask)
    retirement_valid = (
        retirement_result.source_state_valid
        & retirement_result.phase_valid
        & retirement_result.canonical_scheduler_state_valid
        & retirement_result.transaction_applied
    )
    retired_valid = controller.state_valid(retired)
    cold = ~retired.installed_slot_mask
    exact_one_cold = (jnp.sum(cold, dtype=jnp.int32) == 1) & jnp.array_equal(
        cold, target
    )
    retirement_target = (
        jnp.array_equal(retirement_result.retirement.reset_slots, target)
        & jnp.array_equal(
            retirement_result.retirement.requested_slots,
            target,
        )
    )
    owner_preserved = (
        jnp.array_equal(
            all_installed_source.controller_owner_digest,
            retired.controller_owner_digest,
        )
        & jnp.array_equal(
            all_installed_source.expected_retirement_authority_issuer_digest,
            retired.expected_retirement_authority_issuer_digest,
        )
    )
    retirement_revision = (
        retired.controller_revision
        == all_installed_source.controller_revision + jnp.asarray(1, dtype=jnp.int32)
    ) & jnp.array_equal(
        retired.last_retirement_authority_revision_words,
        retirement_result.retirement.state.last_authority_revision_words,
    )
    ordinary_valid = ordinary_prepared.diagnostics.transaction_valid
    scheduler_exact = _tree_array_equal(
        ordinary_prepared.scheduler_result,
        scheduler_prepared.ordinary_result,
    )
    replacement_capacity = _words_less(
        retired.replacement_words,
        _words(controller.config.max_replacements),
    )
    ready = (
        source_valid
        & source_complete
        & retirement_valid
        & retired_valid
        & exact_one_cold
        & retirement_target
        & owner_preserved
        & retirement_revision
        & ordinary_valid
        & scheduler_exact
        & scheduler_prepared.diagnostics.candidate_ready
        & replacement_capacity
        & (retired.controller_revision < _INT32_MAX)
    )
    prepared = AuthorizedOptionExternalCandidateAdoptionPrepared(
        all_installed_source=all_installed_source,
        retirement_result=retirement_result,
        arm_inputs=arm_inputs,
        replacement_arm=replacement_arm,
        observation=observation,
        live_inputs=live_inputs,
        candidate_bundle=candidate_bundle,
        target_mask=target,
        installation_key=installation_key,
        successor_scheduler_key=successor_scheduler_key,
        ordinary_prepared=ordinary_prepared,
        scheduler_prepared=scheduler_prepared,
        replacement_identity_digest=_external_candidate_replacement_identity(
            controller
        ),
        diagnostics=AuthorizedOptionExternalCandidateAdoptionDiagnostics(
            all_installed_source_valid=source_valid,
            all_installed_source_complete=source_complete,
            retirement_result_valid=retirement_valid,
            one_cold_retired_destination_valid=retired_valid,
            exact_one_cold_target=exact_one_cold,
            retirement_target_exact=retirement_target,
            retirement_owner_preserved=owner_preserved,
            retirement_revision_exact=retirement_revision,
            ordinary_preparation_valid=ordinary_valid,
            scheduler_preparation_exact=scheduler_exact,
            external_candidate_ready=scheduler_prepared.diagnostics.candidate_ready,
            replacement_capacity_available=replacement_capacity,
            adoption_ready=ready,
        ),
        prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
    )
    return dataclasses.replace(
        prepared,
        prepared_checksum=_checksum_arrays(_external_candidate_prepared_payload(prepared)),
    )


def authorized_option_external_candidate_adoption_authority_receipt(
    controller: AuthorizedOptionReplacementController,
    prepared: AuthorizedOptionExternalCandidateAdoptionPrepared,
    installation_authority: CumulantOptionInstallationAuthorityReceipt,
    *,
    adoption_authorized: bool | Array,
) -> AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt:
    """Bind the nested scheduler declaration and exact retirement source."""

    if type(controller) is not AuthorizedOptionReplacementController:
        raise TypeError("controller must be exact AuthorizedOptionReplacementController")
    if type(prepared) is not AuthorizedOptionExternalCandidateAdoptionPrepared:
        raise TypeError("prepared has the wrong exact external-candidate type")
    scheduler_authority = cumulant_option_external_bundle_adoption_authority_receipt(
        controller.scheduler,
        prepared.scheduler_prepared,
        installation_authority,
        adoption_authorized=adoption_authorized,
    )
    source = prepared.all_installed_source
    retired = prepared.retirement_result.state
    return AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt(
        scheduler_authority=scheduler_authority,
        adoption_authorized=jnp.asarray(adoption_authorized, dtype=jnp.bool_),
        replacement_identity_digest=prepared.replacement_identity_digest,
        controller_owner_digest=source.controller_owner_digest,
        all_installed_source_checksum=source.binding_checksum,
        source_controller_revision=source.controller_revision,
        retired_destination_checksum=retired.binding_checksum,
        retired_controller_revision=retired.controller_revision,
        retirement_authority_revision_words=(
            retired.last_retirement_authority_revision_words
        ),
        replacement_authority_revision_words=(
            installation_authority.authority_revision_words
        ),
        target_mask=prepared.target_mask,
        candidate_binding_digest=prepared.candidate_bundle.binding_digest,
        prepared_checksum=prepared.prepared_checksum,
        scheduler_prepared_checksum=prepared.scheduler_prepared.prepared_checksum,
    )


def _check_external_candidate_authority_contract(
    controller: AuthorizedOptionReplacementController,
    receipt: AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt,
) -> None:
    if type(receipt) is not AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt:
        raise TypeError("receipt has the wrong exact external-candidate type")
    n = controller.scheduler.discovery.config.option_budget
    contracts = (
        (receipt.adoption_authorized, "adoption_authorized", (), jnp.bool_),
        (receipt.replacement_identity_digest, "replacement_identity_digest", (8,), jnp.uint32),
        (receipt.controller_owner_digest, "controller_owner_digest", (8,), jnp.uint32),
        (
            receipt.all_installed_source_checksum,
            "all_installed_source_checksum",
            (2,),
            jnp.uint32,
        ),
        (receipt.source_controller_revision, "source_controller_revision", (), jnp.int32),
        (receipt.retired_destination_checksum, "retired_destination_checksum", (2,), jnp.uint32),
        (receipt.retired_controller_revision, "retired_controller_revision", (), jnp.int32),
        (
            receipt.retirement_authority_revision_words,
            "retirement_authority_revision_words",
            (2,),
            jnp.uint32,
        ),
        (
            receipt.replacement_authority_revision_words,
            "replacement_authority_revision_words",
            (2,),
            jnp.uint32,
        ),
        (receipt.target_mask, "target_mask", (n,), jnp.bool_),
        (receipt.candidate_binding_digest, "candidate_binding_digest", (2,), jnp.uint32),
        (receipt.prepared_checksum, "prepared_checksum", (2,), jnp.uint32),
        (
            receipt.scheduler_prepared_checksum,
            "scheduler_prepared_checksum",
            (2,),
            jnp.uint32,
        ),
    )
    for value, name, shape, dtype in contracts:
        _require_array(value, name=f"receipt.{name}", shape=shape, dtype=dtype)


def adopt_authorized_option_external_candidate(
    controller: AuthorizedOptionReplacementController,
    one_cold_retired_destination: AuthorizedOptionReplacementState,
    prepared: AuthorizedOptionExternalCandidateAdoptionPrepared,
    authority_receipt: AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt,
) -> AuthorizedOptionExternalCandidateAdoptionResult:
    """Adopt one exact candidate or return the exact one-cold destination."""

    if type(controller) is not AuthorizedOptionReplacementController:
        raise TypeError("controller must be exact AuthorizedOptionReplacementController")
    controller._check_state_contract(one_cold_retired_destination)
    if type(prepared) is not AuthorizedOptionExternalCandidateAdoptionPrepared:
        raise TypeError("prepared has the wrong exact external-candidate type")
    _check_external_candidate_authority_contract(controller, authority_receipt)
    integrity = jnp.array_equal(
        prepared.prepared_checksum,
        _checksum_arrays(_external_candidate_prepared_payload(prepared)),
    )
    recomputed = prepare_authorized_option_external_candidate_adoption(
        controller,
        prepared.all_installed_source,
        prepared.retirement_result,
        prepared.arm_inputs,
        prepared.observation,
        prepared.live_inputs,
        prepared.candidate_bundle,
        prepared.target_mask,
        prepared.installation_key,
        prepared.successor_scheduler_key,
    )
    derivation = _tree_array_equal(prepared, recomputed)
    retired = recomputed.retirement_result.state
    destination_valid = controller.state_valid(one_cold_retired_destination)
    destination_matches = _tree_array_equal(one_cold_retired_destination, retired)
    scheduler_result = adopt_cumulant_option_external_bundle(
        controller.scheduler,
        one_cold_retired_destination.scheduler_state,
        recomputed.scheduler_prepared,
        authority_receipt.scheduler_authority,
    )
    nested_revision = (
        authority_receipt.scheduler_authority.installation_authority.authority_revision_words
    )
    authority_valid = (
        authority_receipt.adoption_authorized
        & authority_receipt.scheduler_authority.adoption_authorized
        & jnp.array_equal(
            authority_receipt.replacement_identity_digest,
            recomputed.replacement_identity_digest,
        )
        & jnp.array_equal(
            authority_receipt.controller_owner_digest,
            recomputed.all_installed_source.controller_owner_digest,
        )
        & jnp.array_equal(
            authority_receipt.all_installed_source_checksum,
            recomputed.all_installed_source.binding_checksum,
        )
        & (
            authority_receipt.source_controller_revision
            == recomputed.all_installed_source.controller_revision
        )
        & jnp.array_equal(
            authority_receipt.retired_destination_checksum,
            retired.binding_checksum,
        )
        & (authority_receipt.retired_controller_revision == retired.controller_revision)
        & jnp.array_equal(
            authority_receipt.retirement_authority_revision_words,
            retired.last_retirement_authority_revision_words,
        )
        & jnp.array_equal(
            authority_receipt.replacement_authority_revision_words,
            nested_revision,
        )
        & _words_less(retired.last_replacement_authority_revision_words, nested_revision)
        & jnp.array_equal(authority_receipt.target_mask, recomputed.target_mask)
        & jnp.array_equal(
            authority_receipt.candidate_binding_digest,
            recomputed.candidate_bundle.binding_digest,
        )
        & jnp.array_equal(
            authority_receipt.prepared_checksum,
            recomputed.prepared_checksum,
        )
        & jnp.array_equal(
            authority_receipt.scheduler_prepared_checksum,
            recomputed.scheduler_prepared.prepared_checksum,
        )
    )
    next_replacement_words, replacement_capacity = _increment_words(
        retired.replacement_words
    )
    candidate = controller._with_checksum(
        dataclasses.replace(
            retired,
            scheduler_state=scheduler_result.state,
            canonical_scheduler_checksum=scheduler_result.state.binding_checksum,
            installed_slot_mask=retired.installed_slot_mask | recomputed.target_mask,
            descriptor_generation=recomputed.candidate_bundle.semantic_generation,
            descriptor_digest=_descriptor_digest(
                recomputed.candidate_bundle.selected_descriptors
            ),
            controller_revision=retired.controller_revision + jnp.asarray(1, dtype=jnp.int32),
            replacement_words=next_replacement_words,
            last_replacement_authority_revision_words=nested_revision,
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    candidate_valid = controller.state_valid(candidate)
    all_installed = jnp.all(candidate.installed_slot_mask)
    applied = (
        destination_valid
        & destination_matches
        & integrity
        & derivation
        & recomputed.diagnostics.adoption_ready
        & authority_valid
        & scheduler_result.transaction_applied
        & replacement_capacity
        & candidate_valid
        & all_installed
    )
    next_state = cast(
        AuthorizedOptionReplacementState,
        jax.tree_util.tree_map(
            lambda proposed, destination: jnp.where(applied, proposed, destination),
            candidate,
            one_cold_retired_destination,
        ),
    )
    return AuthorizedOptionExternalCandidateAdoptionResult(
        state=next_state,
        scheduler_result=scheduler_result,
        destination_state_valid=destination_valid,
        destination_matches_retired=destination_matches,
        prepared_integrity_valid=integrity,
        preparation_derivation_valid=derivation,
        authority_valid=authority_valid & scheduler_result.authority_valid,
        candidate_ready=recomputed.diagnostics.adoption_ready,
        transaction_applied=applied,
        reset_slots=applied & scheduler_result.reset_slots,
        preserved_slots=applied & scheduler_result.preserved_slots,
        cold_state_active=jnp.any(~next_state.installed_slot_mask),
        installation_key_consumed=scheduler_result.installation_key_consumed,
        caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
    )


__all__ = [
    "AUTHORIZED_OPTION_REPLACEMENT_ASSESSMENT",
    "AUTHORIZED_OPTION_REPLACEMENT_AUTONOMOUS_CURATION_AUTHORITY",
    "AUTHORIZED_OPTION_REPLACEMENT_CHECKPOINT_SCHEMA",
    "AUTHORIZED_OPTION_REPLACEMENT_CONFIG_SCHEMA",
    "AUTHORIZED_OPTION_REPLACEMENT_DISCOVERY_AUTHORITY",
    "AUTHORIZED_OPTION_REPLACEMENT_DISPATCH_AUTHORITY",
    "AUTHORIZED_OPTION_REPLACEMENT_EVIDENCE_AUTHORITY",
    "AUTHORIZED_OPTION_REPLACEMENT_GO_NO_GO_AUTHORITY",
    "AUTHORIZED_OPTION_REPLACEMENT_OUTPUT_WRITES",
    "AUTHORIZED_OPTION_REPLACEMENT_PROMOTION_AUTHORITY",
    "AUTHORIZED_OPTION_REPLACEMENT_RETIREMENT_AUTHORITY",
    "AUTHORIZED_OPTION_REPLACEMENT_SAFETY_AUTHORITY",
    "AUTHORIZED_OPTION_REPLACEMENT_SCIENTIFIC_PROMOTION_ALLOWED",
    "AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_PREPARED_SCHEMA",
    "AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_RECEIPT_SCHEMA",
    "AUTHORIZED_OPTION_EXTERNAL_CANDIDATE_ADOPTION_RESULT_SCHEMA",
    "AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt",
    "AuthorizedOptionExternalCandidateAdoptionDiagnostics",
    "AuthorizedOptionExternalCandidateAdoptionPrepared",
    "AuthorizedOptionExternalCandidateAdoptionResult",
    "AuthorizedOptionReplacementArm",
    "AuthorizedOptionReplacementCommitDiagnostics",
    "AuthorizedOptionReplacementConfig",
    "AuthorizedOptionReplacementController",
    "AuthorizedOptionReplacementBorrowResult",
    "AuthorizedOptionReplacementMetadataState",
    "AuthorizedOptionReplacementPrepareDiagnostics",
    "AuthorizedOptionReplacementPrepared",
    "AuthorizedOptionReplacementResourceBudget",
    "AuthorizedOptionReplacementResult",
    "AuthorizedOptionReplacementRetirementResult",
    "AuthorizedOptionReplacementStartResult",
    "AuthorizedOptionReplacementState",
    "AuthorizedOptionReplacementUpdateResult",
    "OptionReplacementAuthorityReceipt",
    "adopt_authorized_option_external_candidate",
    "authorized_option_external_candidate_adoption_authority_receipt",
    "prepare_authorized_option_external_candidate_adoption",
]
