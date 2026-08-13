# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Authorized all-installed atomic adoption of one exact fresh cold-slot cohort.

This opt-in v2 controller composes the stateless fresh-cohort filter with the
public retirement and external-candidate adoption boundaries of the unchanged
v1 replacement stack.  Its persistent state is always all-installed.  The one
cold retirement destination exists only inside a transient preparation and is
selected nowhere unless the exact filtered cohort completes that slot in the
same outer transaction.

``commit`` rederives retirement, the ordinary v1 preparation, the exact filter
source and output, and both public lower adoption preparations.  It then
bit-compares the supplied preparation before considering the separately
declared authority receipt.  A decline, veto, stale destination, replay, or
shape-valid tamper returns the exact all-installed outer destination.  Lower
receipts and local checksums are integrity declarations, not authentication.

The controller owns no safety, go/no-go, retirement, replacement, discovery,
dispatch, evidence, or promotion authority.  It does not evaluate gradients or
actors: delight is unavailable at this mechanical boundary and actor backward
calls are exactly zero.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.authorized_option_replacement import (
    AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt,
    AuthorizedOptionExternalCandidateAdoptionPrepared,
    AuthorizedOptionExternalCandidateAdoptionResult,
    AuthorizedOptionReplacementController,
    AuthorizedOptionReplacementPrepared,
    AuthorizedOptionReplacementRetirementResult,
    AuthorizedOptionReplacementState,
    adopt_authorized_option_external_candidate,
    authorized_option_external_candidate_adoption_authority_receipt,
    prepare_authorized_option_external_candidate_adoption,
)
from alberta_framework.core.authorized_option_retirement import (
    OptionRetirementAuthorityReceipt,
)
from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionLiveInputs,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionInstallationAuthorityReceipt,
    CumulantOptionRetirementHandoff,
    CumulantOptionSchedulerArmInputs,
    CumulantOptionSchedulerObservation,
)
from alberta_framework.core.fresh_cold_slot_cumulant_cohort import (
    FreshColdSlotCumulantCohortFilter,
    FreshColdSlotCumulantCohortPrepared,
    FreshColdSlotCumulantCohortSource,
)
from alberta_framework.core.option_lifecycle_audit import option_semantic_digest

AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_CONFIG_SCHEMA = (
    "alberta.authorized-fresh-cold-slot-atomic-swap.config.v2"
)
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA = (
    "alberta.authorized-fresh-cold-slot-atomic-swap.state.v2"
)
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_PREPARED_SCHEMA = (
    "alberta.authorized-fresh-cold-slot-atomic-swap.prepared.v2"
)
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RECEIPT_SCHEMA = (
    "alberta.authorized-fresh-cold-slot-atomic-swap.receipt.v2"
)
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RESULT_SCHEMA = (
    "alberta.authorized-fresh-cold-slot-atomic-swap.result.v2"
)
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RESOURCE_SCHEMA = (
    "alberta.authorized-fresh-cold-slot-atomic-swap.resource.v2"
)
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ASSESSMENT = "not_assessed"
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_OUTPUT_WRITES = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_EVIDENCE_AUTHORITY = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_PROMOTION_AUTHORITY = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_SAFETY_AUTHORITY = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_GO_NO_GO_AUTHORITY = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RETIREMENT_AUTHORITY = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_REPLACEMENT_AUTHORITY = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_DISCOVERY_AUTHORITY = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_DISPATCH_AUTHORITY = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_SCIENTIFIC_PROMOTION_ALLOWED = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_DELIGHT_AVAILABLE = False
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ACTOR_BACKWARD_CALLS = 0
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ERROR_NONE = 0
AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ERROR_CAPACITY = 1

_DIGEST_WORDS = 8
_CLOCK_WORDS = 2
_INT32_MAX = 2**31 - 1
_UINT64_MAX = 2**64 - 1


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
    expected = jnp.dtype(dtype)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    if jnp.dtype(array.dtype) != expected:
        raise TypeError(f"{name} must have dtype {expected}; got {array.dtype}")
    return array


def _require_threefry_key(value: object, *, name: str) -> Array:
    try:
        array = cast(Array, value)
        implementation = str(jr.key_impl(array))
        key_data = jr.key_data(array)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be one typed Threefry JAX key") from exc
    if (
        not jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
        or array.shape != ()
        or implementation != "threefry2x32"
        or key_data.shape != (_CLOCK_WORDS,)
        or key_data.dtype != jnp.uint32
    ):
        raise TypeError(f"{name} must be one typed Threefry JAX key")
    return array


def _words(value: int) -> Array:
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        raise ValueError("counter value must be uint64-compatible")
    return jnp.asarray((value >> 32, value & 0xFFFFFFFF), dtype=jnp.uint32)


def _words_less(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _words_less_equal(left: Array, right: Array) -> Array:
    return _words_less(left, right) | jnp.array_equal(left, right)


def _increment_words(value: Array) -> tuple[Array, Array]:
    low = value[1] + jnp.uint32(1)
    carry = (low == 0).astype(jnp.uint32)
    high = value[0] + carry
    available = ~((carry != 0) & (high == 0))
    candidate = jnp.stack((high, low), dtype=jnp.uint32)
    return jnp.where(available, candidate, value), available


def _float_bits_equal(left: Array, right: Array) -> Array:
    return jnp.array_equal(
        jax.lax.bitcast_convert_type(left, jnp.uint32),
        jax.lax.bitcast_convert_type(right, jnp.uint32),
    )


def _tree_array_equal(left: object, right: object) -> Array:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(
        right_leaves
    ):
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
        acc1 = acc1 ^ jnp.bitwise_xor.reduce(
            words ^ (indices * jnp.uint32(0x165667B1))
        )
        offset += words.shape[0]
    return jnp.stack((acc0, acc1), dtype=jnp.uint32)


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.size) * int(array.dtype.itemsize)
    return total


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedFreshColdSlotAtomicSwapConfig:
    """Static capacity and fixed nonauthority declaration for the v2 seam."""

    max_atomic_swaps: int = 1

    SCHEMA_VERSION: ClassVar[str] = AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _positive_int(self.max_atomic_swaps, name="max_atomic_swaps")
        if self.max_atomic_swaps != 1:
            raise ValueError("the v2 one-cycle contract requires max_atomic_swaps == 1")

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_atomic_swaps": self.max_atomic_swaps,
            "swap_scope": "all_installed_to_all_installed_exactly_one_cold_target",
            "freshness_source": "exact_rederived_fresh_cold_slot_cumulant_cohort",
            "proposal_persistence": "none",
            "receipt_semantics": "integrity_declaration_not_authentication",
            "assessment": AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ASSESSMENT,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "safety_authority": False,
            "go_no_go_authority": False,
            "retirement_authority": False,
            "replacement_authority": False,
            "discovery_authority": False,
            "dispatch_authority": False,
            "scientific_promotion_allowed": False,
            "delight_available": False,
            "actor_backward_calls": 0,
            "host_prepare": True,
            "host_commit": True,
            "jit_commit": False,
        }

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, object],
    ) -> AuthorizedFreshColdSlotAtomicSwapConfig:
        if type(value) is not dict:
            raise ValueError("atomic-swap config must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "max_atomic_swaps",
            "swap_scope",
            "freshness_source",
            "proposal_persistence",
            "receipt_semantics",
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
            "scientific_promotion_allowed",
            "delight_available",
            "actor_backward_calls",
            "host_prepare",
            "host_commit",
            "jit_commit",
        }
        if set(raw) != expected:
            raise ValueError("atomic-swap config keys differ from schema v2")
        fixed: dict[str, object] = {
            "schema_version": cls.SCHEMA_VERSION,
            "swap_scope": "all_installed_to_all_installed_exactly_one_cold_target",
            "freshness_source": "exact_rederived_fresh_cold_slot_cumulant_cohort",
            "proposal_persistence": "none",
            "receipt_semantics": "integrity_declaration_not_authentication",
            "assessment": AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ASSESSMENT,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "safety_authority": False,
            "go_no_go_authority": False,
            "retirement_authority": False,
            "replacement_authority": False,
            "discovery_authority": False,
            "dispatch_authority": False,
            "scientific_promotion_allowed": False,
            "delight_available": False,
            "actor_backward_calls": 0,
            "host_prepare": True,
            "host_commit": True,
            "jit_commit": False,
        }
        for name, expected_value in fixed.items():
            if raw.pop(name) != expected_value:
                raise ValueError(f"atomic-swap config {name} differs")
        return cls(max_atomic_swaps=cast(int, raw.pop("max_atomic_swaps")))


@chex.dataclass(frozen=True)
class AuthorizedFreshColdSlotAtomicSwapState:
    """All-installed v1 owner plus small source and authority bindings."""

    replacement_state: AuthorizedOptionReplacementState
    expected_authority_issuer_digest: UInt[Array, " 8"]
    controller_owner_digest: UInt[Array, " 8"]
    expected_replacement_controller_owner_digest: UInt[Array, " 8"]
    expected_installation_authority_issuer_digest: UInt[Array, " 8"]
    expected_retirement_authority_issuer_digest: UInt[Array, " 8"]
    controller_identity_digest: UInt[Array, " 8"]
    replacement_identity_digest: UInt[Array, " 8"]
    scheduler_identity_digest: UInt[Array, " 8"]
    installer_identity_digest: UInt[Array, " 8"]
    filter_identity_digest: UInt[Array, " 8"]
    swap_words: UInt[Array, " 2"]
    last_authority_revision_words: UInt[Array, " 2"]
    revision: Int[Array, ""]
    unavailable: Bool[Array, ""]
    error: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]

    SCHEMA_VERSION: ClassVar[str] = AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA


@chex.dataclass(frozen=True)
class AuthorizedFreshColdSlotAtomicSwapPrepareDiagnostics:
    """Primitive source, retirement, filter, and lower-adoption facts."""

    source_state_valid: Bool[Array, ""]
    source_all_slots_installed: Bool[Array, ""]
    swap_capacity_available: Bool[Array, ""]
    transient_retirement_applied: Bool[Array, ""]
    transient_retirement_state_valid: Bool[Array, ""]
    exact_one_transient_cold_slot: Bool[Array, ""]
    ordinary_preparation_valid: Bool[Array, ""]
    filter_source_exact: Bool[Array, ""]
    fresh_preparation_valid: Bool[Array, ""]
    fresh_preparation_exact: Bool[Array, ""]
    fresh_cohort_ready: Bool[Array, ""]
    exact_target_semantic_change: Bool[Array, ""]
    live_slots_semantically_preserved: Bool[Array, ""]
    caller_keys_valid: Bool[Array, ""]
    lower_preparation_ready: Bool[Array, ""]
    identities_exact: Bool[Array, ""]
    atomic_swap_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AuthorizedFreshColdSlotAtomicSwapPrepared:
    """Complete transient source-bound v2 derivation."""

    source_state: AuthorizedFreshColdSlotAtomicSwapState
    retirement_handoff: CumulantOptionRetirementHandoff
    retirement_authority: OptionRetirementAuthorityReceipt
    phase_one_key: Array
    phase_two_key: Array
    arm_inputs: CumulantOptionSchedulerArmInputs
    observation: CumulantOptionSchedulerObservation
    live_inputs: CumulantOptionLiveInputs
    supplied_fresh_prepared: FreshColdSlotCumulantCohortPrepared
    installation_key: Array
    successor_scheduler_key: Array
    retirement_result: AuthorizedOptionReplacementRetirementResult
    ordinary_prepared: AuthorizedOptionReplacementPrepared
    expected_filter_source: FreshColdSlotCumulantCohortSource
    rederived_fresh_prepared: FreshColdSlotCumulantCohortPrepared
    external_adoption_prepared: AuthorizedOptionExternalCandidateAdoptionPrepared
    controller_identity_digest: UInt[Array, " 8"]
    replacement_identity_digest: UInt[Array, " 8"]
    scheduler_identity_digest: UInt[Array, " 8"]
    installer_identity_digest: UInt[Array, " 8"]
    filter_identity_digest: UInt[Array, " 8"]
    diagnostics: AuthorizedFreshColdSlotAtomicSwapPrepareDiagnostics
    prepared_checksum: UInt[Array, " 2"]

    SCHEMA_VERSION: ClassVar[str] = AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_PREPARED_SCHEMA

    @property
    def schema_version(self) -> str:
        return self.SCHEMA_VERSION


@chex.dataclass(frozen=True)
class AuthorizedFreshColdSlotAtomicSwapAuthorityReceipt:
    """Outer integrity declaration binding all identities, keys, and clocks."""

    external_adoption_authority: AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt
    swap_authorized: Bool[Array, ""]
    outer_veto_passed: Bool[Array, ""]
    authority_issuer_digest: UInt[Array, " 8"]
    authority_revision_words: UInt[Array, " 2"]
    controller_owner_digest: UInt[Array, " 8"]
    controller_identity_digest: UInt[Array, " 8"]
    replacement_identity_digest: UInt[Array, " 8"]
    scheduler_identity_digest: UInt[Array, " 8"]
    installer_identity_digest: UInt[Array, " 8"]
    filter_identity_digest: UInt[Array, " 8"]
    source_binding_checksum: UInt[Array, " 2"]
    source_revision: Int[Array, ""]
    source_swap_words: UInt[Array, " 2"]
    source_last_authority_revision_words: UInt[Array, " 2"]
    source_replacement_checksum: UInt[Array, " 2"]
    source_replacement_controller_revision: Int[Array, ""]
    retired_replacement_checksum: UInt[Array, " 2"]
    retired_replacement_controller_revision: Int[Array, ""]
    retirement_authority_issuer_digest: UInt[Array, " 8"]
    retirement_authority_revision_words: UInt[Array, " 2"]
    installation_authority_issuer_digest: UInt[Array, " 8"]
    installation_authority_revision_words: UInt[Array, " 2"]
    phase_one_key_data: UInt[Array, " 2"]
    phase_two_key_data: UInt[Array, " 2"]
    installation_key_data: UInt[Array, " 2"]
    successor_scheduler_key_data: UInt[Array, " 2"]
    target_mask: Bool[Array, " option_budget"]
    candidate_binding_digest: UInt[Array, " 2"]
    fresh_prepared_checksum: UInt[Array, " 2"]
    external_adoption_prepared_checksum: UInt[Array, " 2"]
    prepared_checksum: UInt[Array, " 2"]

    SCHEMA_VERSION: ClassVar[str] = AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RECEIPT_SCHEMA


@chex.dataclass(frozen=True)
class AuthorizedFreshColdSlotAtomicSwapResult:
    """Accepted full successor or exact unchanged all-installed destination."""

    state: AuthorizedFreshColdSlotAtomicSwapState
    retirement_result: AuthorizedOptionReplacementRetirementResult
    replacement_result: AuthorizedOptionExternalCandidateAdoptionResult
    destination_state_valid: Bool[Array, ""]
    destination_matches_source: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    preparation_derivation_valid: Bool[Array, ""]
    authority_valid: Bool[Array, ""]
    atomic_swap_ready: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    retirement_applied: Bool[Array, ""]
    replacement_applied: Bool[Array, ""]
    cold_state_persisted: Bool[Array, ""]
    reset_slots: Bool[Array, " option_budget"]
    preserved_slots: Bool[Array, " option_budget"]
    installation_key_consumed: Array
    caller_authenticated: Bool[Array, ""]

    SCHEMA_VERSION: ClassVar[str] = AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RESULT_SCHEMA


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedFreshColdSlotAtomicSwapResourceBudget:
    """Exact bytes, bounded work, RNG ownership, and nonauthority facts."""

    persistent_state_nbytes: int
    replacement_state_nbytes: int
    overlay_state_nbytes: int
    prepared_state_nbytes: int
    option_slots: int
    pending_proposal_slots: int
    max_atomic_swaps: int
    prepare_retirement_derivations: int
    prepare_retirement_rebind_evaluations: int
    prepare_scheduler_observations: int
    prepare_filter_derivations: int
    prepare_candidate_installation_evaluations: int
    commit_preparation_recomputations: int
    commit_retirement_derivations: int
    commit_retirement_rebind_evaluations: int
    commit_lower_preparation_recomputations: int
    commit_scheduler_observations: int
    commit_filter_derivations: int
    commit_candidate_installation_evaluations: int
    caller_keys_per_preparation: int
    wrapper_rng_split_calls_per_commit: int
    wrapper_generated_root_keys_per_commit: int
    child_rng_uses_supplied_caller_keys_only: bool
    max_adopted_installations_per_commit: int
    max_transient_cold_destinations: int
    persistent_cold_destinations: int
    proposal_persisted: bool
    candidate_materialization_persisted_on_decline: bool
    host_prepare: bool
    host_commit: bool
    jit_commit: bool
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
    scientific_promotion_allowed: bool
    delight_available: bool
    actor_backward_calls: int
    state_schema: str
    resource_schema: str

    SCHEMA_VERSION: ClassVar[str] = AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RESOURCE_SCHEMA


class AuthorizedFreshColdSlotAtomicSwapController:
    """Host v2 seam borrowing one exact v1 controller and filter instance."""

    def __init__(
        self,
        replacement: AuthorizedOptionReplacementController,
        cohort_filter: FreshColdSlotCumulantCohortFilter,
        config: AuthorizedFreshColdSlotAtomicSwapConfig | None = None,
    ) -> None:
        if type(replacement) is not AuthorizedOptionReplacementController:
            raise TypeError("replacement must be exact AuthorizedOptionReplacementController")
        if type(cohort_filter) is not FreshColdSlotCumulantCohortFilter:
            raise TypeError("cohort_filter must be exact FreshColdSlotCumulantCohortFilter")
        if cohort_filter.installation is not replacement.scheduler.installation:
            raise ValueError("replacement and filter must borrow the same installer object")
        cfg = config or AuthorizedFreshColdSlotAtomicSwapConfig()
        if type(cfg) is not AuthorizedFreshColdSlotAtomicSwapConfig:
            raise TypeError("config must be exact AuthorizedFreshColdSlotAtomicSwapConfig")
        self._replacement = replacement
        self._filter = cohort_filter
        self._config = cfg
        self._replacement_identity_digest = option_semantic_digest(
            {
                "schema": AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA,
                "component": "replacement",
                "config": replacement.to_config(),
            }
        )
        self._scheduler_identity_digest = option_semantic_digest(
            {
                "schema": AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA,
                "component": "scheduler",
                "config": replacement.scheduler.to_config(),
            }
        )
        self._installer_identity_digest = option_semantic_digest(
            {
                "schema": AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA,
                "component": "installer",
                "config": replacement.scheduler.installation.to_config(),
            }
        )
        self._filter_identity_digest = option_semantic_digest(
            {
                "schema": AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA,
                "component": "fresh_filter",
                "config": cohort_filter.to_config(),
            }
        )
        self._controller_identity_digest = option_semantic_digest(
            {
                "schema": AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA,
                "component": "outer_controller",
                "config": self.to_config(),
            }
        )

    @property
    def replacement(self) -> AuthorizedOptionReplacementController:
        return self._replacement

    @property
    def cohort_filter(self) -> FreshColdSlotCumulantCohortFilter:
        return self._filter

    @property
    def config(self) -> AuthorizedFreshColdSlotAtomicSwapConfig:
        return self._config

    @property
    def state_schema(self) -> str:
        return AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA

    def to_config(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(
                json.dumps(
                    {
                        "schema_version": AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_CONFIG_SCHEMA,
                        "atomic_swap": self._config.to_config(),
                        "replacement": self._replacement.to_config(),
                        "fresh_filter": self._filter.to_config(),
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
        )

    def _state_payload_arrays(
        self,
        state: AuthorizedFreshColdSlotAtomicSwapState,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                tuple(
                    getattr(state, field.name)
                    for field in dataclasses.fields(AuthorizedFreshColdSlotAtomicSwapState)
                    if field.name != "binding_checksum"
                )
            )
        )

    def _with_state_checksum(
        self,
        state: AuthorizedFreshColdSlotAtomicSwapState,
    ) -> AuthorizedFreshColdSlotAtomicSwapState:
        return dataclasses.replace(
            state,
            binding_checksum=_checksum_arrays(self._state_payload_arrays(state)),
        )

    def _prepared_payload_arrays(
        self,
        prepared: AuthorizedFreshColdSlotAtomicSwapPrepared,
    ) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                tuple(
                    getattr(prepared, field.name)
                    for field in dataclasses.fields(AuthorizedFreshColdSlotAtomicSwapPrepared)
                    if field.name != "prepared_checksum"
                )
            )
        )

    def _with_prepared_checksum(
        self,
        prepared: AuthorizedFreshColdSlotAtomicSwapPrepared,
    ) -> AuthorizedFreshColdSlotAtomicSwapPrepared:
        return dataclasses.replace(
            prepared,
            prepared_checksum=_checksum_arrays(self._prepared_payload_arrays(prepared)),
        )

    def _check_state_contract(
        self,
        state: AuthorizedFreshColdSlotAtomicSwapState,
    ) -> None:
        if type(state) is not AuthorizedFreshColdSlotAtomicSwapState:
            raise TypeError("state must be exact AuthorizedFreshColdSlotAtomicSwapState")
        if type(state.replacement_state) is not AuthorizedOptionReplacementState:
            raise TypeError("replacement_state has the wrong exact v1 type")
        contracts = (
            (
                state.expected_authority_issuer_digest,
                "expected_authority_issuer_digest",
                (_DIGEST_WORDS,),
                jnp.uint32,
            ),
            (state.controller_owner_digest, "controller_owner_digest", (8,), jnp.uint32),
            (
                state.expected_replacement_controller_owner_digest,
                "expected_replacement_controller_owner_digest",
                (8,),
                jnp.uint32,
            ),
            (
                state.expected_installation_authority_issuer_digest,
                "expected_installation_authority_issuer_digest",
                (8,),
                jnp.uint32,
            ),
            (
                state.expected_retirement_authority_issuer_digest,
                "expected_retirement_authority_issuer_digest",
                (8,),
                jnp.uint32,
            ),
            (state.controller_identity_digest, "controller_identity_digest", (8,), jnp.uint32),
            (
                state.replacement_identity_digest,
                "replacement_identity_digest",
                (8,),
                jnp.uint32,
            ),
            (state.scheduler_identity_digest, "scheduler_identity_digest", (8,), jnp.uint32),
            (state.installer_identity_digest, "installer_identity_digest", (8,), jnp.uint32),
            (state.filter_identity_digest, "filter_identity_digest", (8,), jnp.uint32),
            (state.swap_words, "swap_words", (2,), jnp.uint32),
            (
                state.last_authority_revision_words,
                "last_authority_revision_words",
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

    def state_valid(
        self,
        state: AuthorizedFreshColdSlotAtomicSwapState,
    ) -> Bool[Array, ""]:
        """Validate all-installed persistence, identities, clocks, and checksum."""

        self._check_state_contract(state)
        at_capacity = jnp.array_equal(state.swap_words, _words(self._config.max_atomic_swaps))
        zero_swaps = jnp.array_equal(state.swap_words, _words(0))
        clock_valid = jnp.where(
            zero_swaps,
            jnp.all(state.last_authority_revision_words == 0),
            jnp.any(state.last_authority_revision_words != 0),
        )
        expected_error = jnp.where(
            at_capacity,
            jnp.asarray(AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ERROR_CAPACITY, jnp.int32),
            jnp.asarray(AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ERROR_NONE, jnp.int32),
        )
        identities = (
            jnp.array_equal(
                state.controller_identity_digest,
                self._controller_identity_digest,
            )
            & jnp.array_equal(
                state.replacement_identity_digest,
                self._replacement_identity_digest,
            )
            & jnp.array_equal(
                state.scheduler_identity_digest,
                self._scheduler_identity_digest,
            )
            & jnp.array_equal(
                state.installer_identity_digest,
                self._installer_identity_digest,
            )
            & jnp.array_equal(
                state.filter_identity_digest,
                self._filter_identity_digest,
            )
        )
        nested_bindings = (
            jnp.array_equal(
                state.expected_replacement_controller_owner_digest,
                state.replacement_state.controller_owner_digest,
            )
            & jnp.array_equal(
                state.expected_installation_authority_issuer_digest,
                state.replacement_state.scheduler_state.expected_authority_issuer_digest,
            )
            & jnp.array_equal(
                state.expected_retirement_authority_issuer_digest,
                state.replacement_state.expected_retirement_authority_issuer_digest,
            )
        )
        return (
            self._replacement.state_valid(state.replacement_state)
            & jnp.all(state.replacement_state.installed_slot_mask)
            & jnp.any(state.expected_authority_issuer_digest != 0)
            & jnp.any(state.controller_owner_digest != 0)
            & identities
            & nested_bindings
            & _words_less_equal(state.swap_words, _words(self._config.max_atomic_swaps))
            & (state.swap_words[0] == 0)
            & (state.revision == state.swap_words[1].astype(jnp.int32))
            & clock_valid
            & (state.unavailable == at_capacity)
            & (state.error == expected_error)
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._state_payload_arrays(state)),
            )
        )

    def init(
        self,
        replacement_state: AuthorizedOptionReplacementState,
        *,
        authority_issuer_digest: Array,
        controller_owner_digest: Array,
    ) -> AuthorizedFreshColdSlotAtomicSwapState:
        """Bind one valid all-installed, unused v1 replacement source."""

        if type(replacement_state) is not AuthorizedOptionReplacementState:
            raise TypeError("replacement_state must be exact AuthorizedOptionReplacementState")
        issuer = _require_array(
            authority_issuer_digest,
            name="authority_issuer_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        owner = _require_array(
            controller_owner_digest,
            name="controller_owner_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        source_ready = (
            self._replacement.state_valid(replacement_state)
            & jnp.all(replacement_state.installed_slot_mask)
            & jnp.array_equal(replacement_state.retirement_words, _words(0))
            & jnp.array_equal(replacement_state.replacement_words, _words(0))
        )
        if not bool(jax.device_get(source_ready)):
            raise ValueError("replacement_state must be a valid unused all-installed source")
        if not bool(jax.device_get(jnp.any(issuer != 0))):
            raise ValueError("authority_issuer_digest must be nonzero")
        if not bool(jax.device_get(jnp.any(owner != 0))):
            raise ValueError("controller_owner_digest must be nonzero")
        state = AuthorizedFreshColdSlotAtomicSwapState(
            replacement_state=replacement_state,
            expected_authority_issuer_digest=issuer,
            controller_owner_digest=owner,
            expected_replacement_controller_owner_digest=(
                replacement_state.controller_owner_digest
            ),
            expected_installation_authority_issuer_digest=(
                replacement_state.scheduler_state.expected_authority_issuer_digest
            ),
            expected_retirement_authority_issuer_digest=(
                replacement_state.expected_retirement_authority_issuer_digest
            ),
            controller_identity_digest=self._controller_identity_digest,
            replacement_identity_digest=self._replacement_identity_digest,
            scheduler_identity_digest=self._scheduler_identity_digest,
            installer_identity_digest=self._installer_identity_digest,
            filter_identity_digest=self._filter_identity_digest,
            swap_words=_words(0),
            last_authority_revision_words=_words(0),
            revision=jnp.asarray(0, dtype=jnp.int32),
            unavailable=jnp.asarray(False, dtype=jnp.bool_),
            error=jnp.asarray(AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ERROR_NONE, jnp.int32),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        state = self._with_state_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initialized v2 atomic-swap state failed its contract")
        return state

    def prepare(
        self,
        state: AuthorizedFreshColdSlotAtomicSwapState,
        retirement_handoff: CumulantOptionRetirementHandoff,
        retirement_authority: OptionRetirementAuthorityReceipt,
        phase_one_key: Array,
        phase_two_key: Array,
        arm_inputs: CumulantOptionSchedulerArmInputs,
        observation: CumulantOptionSchedulerObservation,
        live_inputs: CumulantOptionLiveInputs,
        fresh_prepared: FreshColdSlotCumulantCohortPrepared,
        installation_key: Array,
        successor_scheduler_key: Array,
    ) -> AuthorizedFreshColdSlotAtomicSwapPrepared:
        """Rederive retirement, ordinary observation, filter, and lower adoption."""

        self._check_state_contract(state)
        if type(fresh_prepared) is not FreshColdSlotCumulantCohortPrepared:
            raise TypeError("fresh_prepared has the wrong exact v2 filter type")
        key_one = _require_threefry_key(phase_one_key, name="phase_one_key")
        key_two = _require_threefry_key(phase_two_key, name="phase_two_key")
        install_key = _require_threefry_key(installation_key, name="installation_key")
        successor_key = _require_threefry_key(
            successor_scheduler_key,
            name="successor_scheduler_key",
        )
        key_rows = jnp.stack(
            (
                jr.key_data(key_one),
                jr.key_data(key_two),
                jr.key_data(install_key),
                jr.key_data(successor_key),
            ),
            axis=0,
        )
        pairwise_distinct = jnp.all(
            (~jnp.all(key_rows[:, None, :] == key_rows[None, :, :], axis=2))
            | jnp.eye(4, dtype=jnp.bool_)
        )
        source_valid = self.state_valid(state)
        source_all_installed = jnp.all(state.replacement_state.installed_slot_mask)
        capacity = _words_less(
            state.swap_words,
            _words(self._config.max_atomic_swaps),
        )
        retired = self._replacement.retire(
            state.replacement_state,
            retirement_handoff,
            retirement_authority,
            key_one,
            key_two,
        )
        retired_valid = self._replacement.state_valid(retired.state)
        cold = ~retired.state.installed_slot_mask
        exact_one_cold = jnp.sum(cold, dtype=jnp.int32) == 1
        replacement_arm = self._replacement.arm(retired.state, arm_inputs)
        ordinary = self._replacement.prepare(
            retired.state,
            replacement_arm,
            observation,
            live_inputs,
        )
        retired_installation = retired.state.scheduler_state.installation_state
        expected_filter_source = FreshColdSlotCumulantCohortSource(
            discovery_result=ordinary.scheduler_result.discovery,
            installed_bundle=retired_installation.installed_bundle,
            installed_semantic_digests=retired_installation.installed_semantic_digests,
            installed_slot_mask=retired.state.installed_slot_mask,
            previous_raw_features=arm_inputs.current_raw_features,
            previous_raw_available=arm_inputs.current_raw_available,
            live_inputs=live_inputs,
        )
        rederived_fresh = self._filter.prepare(expected_filter_source)
        filter_source_exact = _tree_array_equal(
            fresh_prepared.source,
            expected_filter_source,
        )
        fresh_valid = self._filter.validate(fresh_prepared)
        fresh_exact = fresh_valid & _tree_array_equal(fresh_prepared, rederived_fresh)
        external_prepared = prepare_authorized_option_external_candidate_adoption(
            self._replacement,
            state.replacement_state,
            retired,
            arm_inputs,
            observation,
            live_inputs,
            rederived_fresh.filtered_bundle,
            rederived_fresh.target_mask,
            install_key,
            successor_key,
        )
        exact_change = (
            jnp.array_equal(rederived_fresh.changed_slots, rederived_fresh.target_mask)
            & (jnp.sum(rederived_fresh.target_mask, dtype=jnp.int32) == 1)
            & jnp.array_equal(rederived_fresh.target_mask, cold)
        )
        live_preserved = ~jnp.any(
            rederived_fresh.changed_slots & retired.state.installed_slot_mask
        )
        identities = (
            jnp.array_equal(
                state.controller_identity_digest,
                self._controller_identity_digest,
            )
            & jnp.array_equal(
                state.replacement_identity_digest,
                self._replacement_identity_digest,
            )
            & jnp.array_equal(
                state.scheduler_identity_digest,
                self._scheduler_identity_digest,
            )
            & jnp.array_equal(
                state.installer_identity_digest,
                self._installer_identity_digest,
            )
            & jnp.array_equal(
                state.filter_identity_digest,
                self._filter_identity_digest,
            )
        )
        ready = (
            source_valid
            & source_all_installed
            & capacity
            & retired.transaction_applied
            & retired_valid
            & exact_one_cold
            & ordinary.diagnostics.transaction_valid
            & filter_source_exact
            & fresh_exact
            & rederived_fresh.diagnostics.candidate_ready
            & exact_change
            & live_preserved
            & pairwise_distinct
            & external_prepared.diagnostics.adoption_ready
            & identities
        )
        prepared = AuthorizedFreshColdSlotAtomicSwapPrepared(
            source_state=state,
            retirement_handoff=retirement_handoff,
            retirement_authority=retirement_authority,
            phase_one_key=key_one,
            phase_two_key=key_two,
            arm_inputs=arm_inputs,
            observation=observation,
            live_inputs=live_inputs,
            supplied_fresh_prepared=fresh_prepared,
            installation_key=install_key,
            successor_scheduler_key=successor_key,
            retirement_result=retired,
            ordinary_prepared=ordinary,
            expected_filter_source=expected_filter_source,
            rederived_fresh_prepared=rederived_fresh,
            external_adoption_prepared=external_prepared,
            controller_identity_digest=self._controller_identity_digest,
            replacement_identity_digest=self._replacement_identity_digest,
            scheduler_identity_digest=self._scheduler_identity_digest,
            installer_identity_digest=self._installer_identity_digest,
            filter_identity_digest=self._filter_identity_digest,
            diagnostics=AuthorizedFreshColdSlotAtomicSwapPrepareDiagnostics(
                source_state_valid=source_valid,
                source_all_slots_installed=source_all_installed,
                swap_capacity_available=capacity,
                transient_retirement_applied=retired.transaction_applied,
                transient_retirement_state_valid=retired_valid,
                exact_one_transient_cold_slot=exact_one_cold,
                ordinary_preparation_valid=ordinary.diagnostics.transaction_valid,
                filter_source_exact=filter_source_exact,
                fresh_preparation_valid=fresh_valid,
                fresh_preparation_exact=fresh_exact,
                fresh_cohort_ready=rederived_fresh.diagnostics.candidate_ready,
                exact_target_semantic_change=exact_change,
                live_slots_semantically_preserved=live_preserved,
                caller_keys_valid=pairwise_distinct,
                lower_preparation_ready=external_prepared.diagnostics.adoption_ready,
                identities_exact=identities,
                atomic_swap_ready=ready,
            ),
            prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_prepared_checksum(prepared)

    def authority_receipt(
        self,
        prepared: AuthorizedFreshColdSlotAtomicSwapPrepared,
        installation_authority: CumulantOptionInstallationAuthorityReceipt,
        *,
        authority_issuer_digest: Array,
        authority_revision_words: Array,
        swap_authorized: bool | Array,
        outer_veto_passed: bool | Array,
    ) -> AuthorizedFreshColdSlotAtomicSwapAuthorityReceipt:
        """Declare one source-bound outer authorization without authentication."""

        if type(prepared) is not AuthorizedFreshColdSlotAtomicSwapPrepared:
            raise TypeError("prepared has the wrong exact v2 atomic-swap type")
        issuer = _require_array(
            authority_issuer_digest,
            name="authority_issuer_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        authority_revision = _require_array(
            authority_revision_words,
            name="authority_revision_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        authorized = jnp.asarray(swap_authorized, dtype=jnp.bool_)
        veto = jnp.asarray(outer_veto_passed, dtype=jnp.bool_)
        nested = authorized_option_external_candidate_adoption_authority_receipt(
            self._replacement,
            prepared.external_adoption_prepared,
            installation_authority,
            adoption_authorized=authorized & veto,
        )
        source = prepared.source_state
        retired = prepared.retirement_result.state
        return AuthorizedFreshColdSlotAtomicSwapAuthorityReceipt(
            external_adoption_authority=nested,
            swap_authorized=authorized,
            outer_veto_passed=veto,
            authority_issuer_digest=issuer,
            authority_revision_words=authority_revision,
            controller_owner_digest=source.controller_owner_digest,
            controller_identity_digest=prepared.controller_identity_digest,
            replacement_identity_digest=prepared.replacement_identity_digest,
            scheduler_identity_digest=prepared.scheduler_identity_digest,
            installer_identity_digest=prepared.installer_identity_digest,
            filter_identity_digest=prepared.filter_identity_digest,
            source_binding_checksum=source.binding_checksum,
            source_revision=source.revision,
            source_swap_words=source.swap_words,
            source_last_authority_revision_words=source.last_authority_revision_words,
            source_replacement_checksum=source.replacement_state.binding_checksum,
            source_replacement_controller_revision=(
                source.replacement_state.controller_revision
            ),
            retired_replacement_checksum=retired.binding_checksum,
            retired_replacement_controller_revision=retired.controller_revision,
            retirement_authority_issuer_digest=prepared.retirement_authority.issuer_digest,
            retirement_authority_revision_words=(
                prepared.retirement_authority.authority_revision_words
            ),
            installation_authority_issuer_digest=installation_authority.issuer_digest,
            installation_authority_revision_words=(
                installation_authority.authority_revision_words
            ),
            phase_one_key_data=jr.key_data(prepared.phase_one_key),
            phase_two_key_data=jr.key_data(prepared.phase_two_key),
            installation_key_data=jr.key_data(prepared.installation_key),
            successor_scheduler_key_data=jr.key_data(prepared.successor_scheduler_key),
            target_mask=prepared.rederived_fresh_prepared.target_mask,
            candidate_binding_digest=(
                prepared.rederived_fresh_prepared.filtered_bundle.binding_digest
            ),
            fresh_prepared_checksum=prepared.rederived_fresh_prepared.prepared_checksum,
            external_adoption_prepared_checksum=(
                prepared.external_adoption_prepared.prepared_checksum
            ),
            prepared_checksum=prepared.prepared_checksum,
        )

    def _check_authority_contract(
        self,
        receipt: AuthorizedFreshColdSlotAtomicSwapAuthorityReceipt,
    ) -> None:
        if type(receipt) is not AuthorizedFreshColdSlotAtomicSwapAuthorityReceipt:
            raise TypeError("authority_receipt has the wrong exact v2 atomic-swap type")
        if type(receipt.external_adoption_authority) is not (
            AuthorizedOptionExternalCandidateAdoptionAuthorityReceipt
        ):
            raise TypeError("external_adoption_authority has the wrong exact public type")
        budget = self._replacement.scheduler.discovery.config.option_budget
        contracts = (
            (receipt.swap_authorized, "swap_authorized", (), jnp.bool_),
            (receipt.outer_veto_passed, "outer_veto_passed", (), jnp.bool_),
            (receipt.authority_issuer_digest, "authority_issuer_digest", (8,), jnp.uint32),
            (
                receipt.authority_revision_words,
                "authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (receipt.controller_owner_digest, "controller_owner_digest", (8,), jnp.uint32),
            (receipt.controller_identity_digest, "controller_identity_digest", (8,), jnp.uint32),
            (
                receipt.replacement_identity_digest,
                "replacement_identity_digest",
                (8,),
                jnp.uint32,
            ),
            (receipt.scheduler_identity_digest, "scheduler_identity_digest", (8,), jnp.uint32),
            (receipt.installer_identity_digest, "installer_identity_digest", (8,), jnp.uint32),
            (receipt.filter_identity_digest, "filter_identity_digest", (8,), jnp.uint32),
            (receipt.source_binding_checksum, "source_binding_checksum", (2,), jnp.uint32),
            (receipt.source_revision, "source_revision", (), jnp.int32),
            (receipt.source_swap_words, "source_swap_words", (2,), jnp.uint32),
            (
                receipt.source_last_authority_revision_words,
                "source_last_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (
                receipt.source_replacement_checksum,
                "source_replacement_checksum",
                (2,),
                jnp.uint32,
            ),
            (
                receipt.source_replacement_controller_revision,
                "source_replacement_controller_revision",
                (),
                jnp.int32,
            ),
            (
                receipt.retired_replacement_checksum,
                "retired_replacement_checksum",
                (2,),
                jnp.uint32,
            ),
            (
                receipt.retired_replacement_controller_revision,
                "retired_replacement_controller_revision",
                (),
                jnp.int32,
            ),
            (
                receipt.retirement_authority_issuer_digest,
                "retirement_authority_issuer_digest",
                (8,),
                jnp.uint32,
            ),
            (
                receipt.retirement_authority_revision_words,
                "retirement_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (
                receipt.installation_authority_issuer_digest,
                "installation_authority_issuer_digest",
                (8,),
                jnp.uint32,
            ),
            (
                receipt.installation_authority_revision_words,
                "installation_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (receipt.phase_one_key_data, "phase_one_key_data", (2,), jnp.uint32),
            (receipt.phase_two_key_data, "phase_two_key_data", (2,), jnp.uint32),
            (receipt.installation_key_data, "installation_key_data", (2,), jnp.uint32),
            (
                receipt.successor_scheduler_key_data,
                "successor_scheduler_key_data",
                (2,),
                jnp.uint32,
            ),
            (receipt.target_mask, "target_mask", (budget,), jnp.bool_),
            (
                receipt.candidate_binding_digest,
                "candidate_binding_digest",
                (2,),
                jnp.uint32,
            ),
            (
                receipt.fresh_prepared_checksum,
                "fresh_prepared_checksum",
                (2,),
                jnp.uint32,
            ),
            (
                receipt.external_adoption_prepared_checksum,
                "external_adoption_prepared_checksum",
                (2,),
                jnp.uint32,
            ),
            (receipt.prepared_checksum, "prepared_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"authority_receipt.{name}", shape=shape, dtype=dtype)

    def _authority_valid(
        self,
        prepared: AuthorizedFreshColdSlotAtomicSwapPrepared,
        receipt: AuthorizedFreshColdSlotAtomicSwapAuthorityReceipt,
    ) -> Array:
        source = prepared.source_state
        retired = prepared.retirement_result.state
        nested = receipt.external_adoption_authority
        expected_nested = authorized_option_external_candidate_adoption_authority_receipt(
            self._replacement,
            prepared.external_adoption_prepared,
            nested.scheduler_authority.installation_authority,
            adoption_authorized=receipt.swap_authorized & receipt.outer_veto_passed,
        )
        return (
            receipt.swap_authorized
            & receipt.outer_veto_passed
            & jnp.array_equal(
                receipt.authority_issuer_digest,
                source.expected_authority_issuer_digest,
            )
            & jnp.any(receipt.authority_issuer_digest != 0)
            & _words_less(
                source.last_authority_revision_words,
                receipt.authority_revision_words,
            )
            & jnp.any(receipt.authority_revision_words != 0)
            & jnp.array_equal(receipt.controller_owner_digest, source.controller_owner_digest)
            & jnp.array_equal(
                receipt.controller_identity_digest,
                prepared.controller_identity_digest,
            )
            & jnp.array_equal(
                receipt.replacement_identity_digest,
                prepared.replacement_identity_digest,
            )
            & jnp.array_equal(
                receipt.scheduler_identity_digest,
                prepared.scheduler_identity_digest,
            )
            & jnp.array_equal(
                receipt.installer_identity_digest,
                prepared.installer_identity_digest,
            )
            & jnp.array_equal(
                receipt.filter_identity_digest,
                prepared.filter_identity_digest,
            )
            & jnp.array_equal(receipt.source_binding_checksum, source.binding_checksum)
            & (receipt.source_revision == source.revision)
            & jnp.array_equal(receipt.source_swap_words, source.swap_words)
            & jnp.array_equal(
                receipt.source_last_authority_revision_words,
                source.last_authority_revision_words,
            )
            & jnp.array_equal(
                receipt.source_replacement_checksum,
                source.replacement_state.binding_checksum,
            )
            & (
                receipt.source_replacement_controller_revision
                == source.replacement_state.controller_revision
            )
            & jnp.array_equal(
                receipt.retired_replacement_checksum,
                retired.binding_checksum,
            )
            & (
                receipt.retired_replacement_controller_revision
                == retired.controller_revision
            )
            & jnp.array_equal(
                receipt.retirement_authority_issuer_digest,
                prepared.retirement_authority.issuer_digest,
            )
            & jnp.array_equal(
                receipt.retirement_authority_revision_words,
                prepared.retirement_authority.authority_revision_words,
            )
            & jnp.array_equal(
                receipt.installation_authority_issuer_digest,
                nested.scheduler_authority.installation_authority.issuer_digest,
            )
            & jnp.array_equal(
                receipt.installation_authority_revision_words,
                nested.scheduler_authority.installation_authority.authority_revision_words,
            )
            & jnp.array_equal(receipt.phase_one_key_data, jr.key_data(prepared.phase_one_key))
            & jnp.array_equal(receipt.phase_two_key_data, jr.key_data(prepared.phase_two_key))
            & jnp.array_equal(
                receipt.installation_key_data,
                jr.key_data(prepared.installation_key),
            )
            & jnp.array_equal(
                receipt.successor_scheduler_key_data,
                jr.key_data(prepared.successor_scheduler_key),
            )
            & jnp.array_equal(
                receipt.target_mask,
                prepared.rederived_fresh_prepared.target_mask,
            )
            & jnp.array_equal(
                receipt.candidate_binding_digest,
                prepared.rederived_fresh_prepared.filtered_bundle.binding_digest,
            )
            & jnp.array_equal(
                receipt.fresh_prepared_checksum,
                prepared.rederived_fresh_prepared.prepared_checksum,
            )
            & jnp.array_equal(
                receipt.external_adoption_prepared_checksum,
                prepared.external_adoption_prepared.prepared_checksum,
            )
            & jnp.array_equal(receipt.prepared_checksum, prepared.prepared_checksum)
            & _tree_array_equal(nested, expected_nested)
        )

    def commit(
        self,
        state: AuthorizedFreshColdSlotAtomicSwapState,
        prepared: AuthorizedFreshColdSlotAtomicSwapPrepared,
        authority_receipt: AuthorizedFreshColdSlotAtomicSwapAuthorityReceipt,
    ) -> AuthorizedFreshColdSlotAtomicSwapResult:
        """Adopt only a complete exact all-installed successor."""

        self._check_state_contract(state)
        if type(prepared) is not AuthorizedFreshColdSlotAtomicSwapPrepared:
            raise TypeError("prepared has the wrong exact v2 atomic-swap type")
        self._check_authority_contract(authority_receipt)
        integrity = jnp.array_equal(
            prepared.prepared_checksum,
            _checksum_arrays(self._prepared_payload_arrays(prepared)),
        )
        recomputed = self.prepare(
            prepared.source_state,
            prepared.retirement_handoff,
            prepared.retirement_authority,
            prepared.phase_one_key,
            prepared.phase_two_key,
            prepared.arm_inputs,
            prepared.observation,
            prepared.live_inputs,
            prepared.supplied_fresh_prepared,
            prepared.installation_key,
            prepared.successor_scheduler_key,
        )
        derivation = _tree_array_equal(prepared, recomputed)
        destination_valid = self.state_valid(state)
        destination_matches = _tree_array_equal(state, recomputed.source_state)
        outer_authority_valid = self._authority_valid(recomputed, authority_receipt)
        lower = adopt_authorized_option_external_candidate(
            self._replacement,
            recomputed.retirement_result.state,
            recomputed.external_adoption_prepared,
            authority_receipt.external_adoption_authority,
        )
        next_swap_words, counter_capacity = _increment_words(
            recomputed.source_state.swap_words
        )
        at_capacity = jnp.array_equal(next_swap_words, _words(self._config.max_atomic_swaps))
        candidate = self._with_state_checksum(
            dataclasses.replace(
                recomputed.source_state,
                replacement_state=lower.state,
                swap_words=next_swap_words,
                last_authority_revision_words=authority_receipt.authority_revision_words,
                revision=recomputed.source_state.revision
                + jnp.asarray(1, dtype=jnp.int32),
                unavailable=at_capacity,
                error=jnp.where(
                    at_capacity,
                    jnp.asarray(
                        AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ERROR_CAPACITY,
                        jnp.int32,
                    ),
                    jnp.asarray(
                        AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ERROR_NONE,
                        jnp.int32,
                    ),
                ),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        target = recomputed.rederived_fresh_prepared.target_mask
        exact_reset = jnp.array_equal(lower.reset_slots, target)
        exact_preserve = jnp.array_equal(lower.preserved_slots, ~target)
        candidate_valid = self.state_valid(candidate)
        all_installed = jnp.all(candidate.replacement_state.installed_slot_mask)
        applied = (
            destination_valid
            & destination_matches
            & integrity
            & derivation
            & recomputed.diagnostics.atomic_swap_ready
            & outer_authority_valid
            & lower.prepared_integrity_valid
            & lower.preparation_derivation_valid
            & lower.authority_valid
            & lower.transaction_applied
            & exact_reset
            & exact_preserve
            & counter_capacity
            & (recomputed.source_state.revision < _INT32_MAX)
            & candidate_valid
            & all_installed
        )
        next_state = cast(
            AuthorizedFreshColdSlotAtomicSwapState,
            jax.tree_util.tree_map(
                lambda proposed, destination: jnp.where(applied, proposed, destination),
                candidate,
                state,
            ),
        )
        cold_persisted = (
            jnp.all(recomputed.source_state.replacement_state.installed_slot_mask)
            & jnp.any(~next_state.replacement_state.installed_slot_mask)
        )
        consumed_key = jr.wrap_key_data(
            jnp.where(
                applied,
                jr.key_data(recomputed.installation_key),
                jnp.zeros((2,), dtype=jnp.uint32),
            ),
            impl="threefry2x32",
        )
        return AuthorizedFreshColdSlotAtomicSwapResult(
            state=next_state,
            retirement_result=recomputed.retirement_result,
            replacement_result=lower,
            destination_state_valid=destination_valid,
            destination_matches_source=destination_matches,
            prepared_integrity_valid=integrity,
            preparation_derivation_valid=derivation,
            authority_valid=outer_authority_valid & lower.authority_valid,
            atomic_swap_ready=recomputed.diagnostics.atomic_swap_ready,
            transaction_applied=applied,
            retirement_applied=applied,
            replacement_applied=applied,
            cold_state_persisted=cold_persisted,
            reset_slots=applied & lower.reset_slots,
            preserved_slots=applied & lower.preserved_slots,
            installation_key_consumed=consumed_key,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )

    def resource_budget(
        self,
        state: AuthorizedFreshColdSlotAtomicSwapState,
        prepared: AuthorizedFreshColdSlotAtomicSwapPrepared | None = None,
    ) -> AuthorizedFreshColdSlotAtomicSwapResourceBudget:
        """Measure exact persistence and declare bounded v2 transaction work."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("resource measurement requires a valid v2 state")
        if prepared is not None and type(prepared) is not (
            AuthorizedFreshColdSlotAtomicSwapPrepared
        ):
            raise TypeError("prepared must be an exact v2 preparation or None")
        persistent = _tree_nbytes(state)
        replacement = _tree_nbytes(state.replacement_state)
        prepared_bytes = 0 if prepared is None else _tree_nbytes(prepared)
        return AuthorizedFreshColdSlotAtomicSwapResourceBudget(
            persistent_state_nbytes=persistent,
            replacement_state_nbytes=replacement,
            overlay_state_nbytes=persistent - replacement,
            prepared_state_nbytes=prepared_bytes,
            option_slots=self._replacement.scheduler.discovery.config.option_budget,
            pending_proposal_slots=0,
            max_atomic_swaps=self._config.max_atomic_swaps,
            prepare_retirement_derivations=1,
            prepare_retirement_rebind_evaluations=2,
            prepare_scheduler_observations=3,
            prepare_filter_derivations=2,
            prepare_candidate_installation_evaluations=1,
            commit_preparation_recomputations=1,
            commit_retirement_derivations=1,
            commit_retirement_rebind_evaluations=2,
            commit_lower_preparation_recomputations=2,
            commit_scheduler_observations=6,
            commit_filter_derivations=2,
            commit_candidate_installation_evaluations=3,
            caller_keys_per_preparation=4,
            wrapper_rng_split_calls_per_commit=0,
            wrapper_generated_root_keys_per_commit=0,
            child_rng_uses_supplied_caller_keys_only=True,
            max_adopted_installations_per_commit=1,
            max_transient_cold_destinations=1,
            persistent_cold_destinations=0,
            proposal_persisted=False,
            candidate_materialization_persisted_on_decline=False,
            host_prepare=True,
            host_commit=True,
            jit_commit=False,
            assessment=AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ASSESSMENT,
            output_writes=False,
            evidence_authority=False,
            promotion_authority=False,
            safety_authority=False,
            go_no_go_authority=False,
            retirement_authority=False,
            replacement_authority=False,
            discovery_authority=False,
            dispatch_authority=False,
            scientific_promotion_allowed=False,
            delight_available=False,
            actor_backward_calls=0,
            state_schema=AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA,
            resource_schema=AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RESOURCE_SCHEMA,
        )


__all__ = [
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ACTOR_BACKWARD_CALLS",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ASSESSMENT",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_CONFIG_SCHEMA",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_DELIGHT_AVAILABLE",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_DISCOVERY_AUTHORITY",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_DISPATCH_AUTHORITY",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ERROR_CAPACITY",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_ERROR_NONE",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_EVIDENCE_AUTHORITY",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_GO_NO_GO_AUTHORITY",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_OUTPUT_WRITES",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_PREPARED_SCHEMA",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_PROMOTION_AUTHORITY",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RECEIPT_SCHEMA",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_REPLACEMENT_AUTHORITY",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RESOURCE_SCHEMA",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RESULT_SCHEMA",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_RETIREMENT_AUTHORITY",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_SAFETY_AUTHORITY",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_SCIENTIFIC_PROMOTION_ALLOWED",
    "AUTHORIZED_FRESH_COLD_SLOT_ATOMIC_SWAP_STATE_SCHEMA",
    "AuthorizedFreshColdSlotAtomicSwapAuthorityReceipt",
    "AuthorizedFreshColdSlotAtomicSwapConfig",
    "AuthorizedFreshColdSlotAtomicSwapController",
    "AuthorizedFreshColdSlotAtomicSwapPrepareDiagnostics",
    "AuthorizedFreshColdSlotAtomicSwapPrepared",
    "AuthorizedFreshColdSlotAtomicSwapResourceBudget",
    "AuthorizedFreshColdSlotAtomicSwapResult",
    "AuthorizedFreshColdSlotAtomicSwapState",
]
