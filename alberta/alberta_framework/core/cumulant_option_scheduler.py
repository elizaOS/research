# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Bounded scheduling around cumulant discovery and live STOMP installation.

The scheduler closes only the mechanical timing edge between
``CumulantSubtaskDiscovery`` and ``CumulantOptionInstallation``.  Every accepted
transition is armed and observed by the discovery mechanism.  A complete
discovered cohort may be offered to the installer at the configured cadence,
or on a bounded retry, only when all of the following hold:

* the live successor is bit/exactly the discovery successor;
* a caller-issued receipt binds the source universe and exact lifetime range;
* the real STOMP/lifecycle state is quiescent; and
* the installer still accepts the complete fresh bundle.

No proposal payload is queued.  A deferred attempt records only ``retry_due``;
the next attempt therefore requires a newly observed, transition-bound bundle.
Before the first installation, and whenever no installation commits, the
existing installer materializes the cold raw-plus-zero-tail path so its option
mask remains the sole behavior/learning authority.

The scheduler splits its typed Threefry key only for an authorized quiescent
attempt, and commits the successor key for every applied installation,
including a descriptor-identical provenance refresh.  Failed or skipped
attempts commit neither installation/rebind state nor RNG state; the ordinary
cold/live materialization may still advance on the accepted transition.

Maintenance calls the existing lifecycle scorecard at a bounded cadence and
returns a semantic-generation-bound retirement handoff.  It never executes
the proposed retirement.  The authority receipt is an integrity-bound caller
declaration, not authentication, and this module owns no safety, go/no-go,
curation, retirement, dispatch, evidence, or scientific-promotion authority.

``schedule_clock`` is the small array-only JAX boundary used for exact cadence
logic.  The composed ``observe``, ``start_control``, and ``update_control``
methods are explicitly host-orchestrated: they choose between the installer's
cold/live/optional results without tracing both full STOMP branches.  The
underlying discovery and installer array kernels retain their own JIT
contracts.  Checkpoint hashing is also host-only.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionInstallation,
    CumulantOptionInstallationBorrowResult,
    CumulantOptionInstallationMetadataState,
    CumulantOptionInstallationResult,
    CumulantOptionInstallationState,
    CumulantOptionLiveInputs,
    CumulantOptionMaterialization,
    CumulantOptionMaterializationResult,
    CumulantOptionStartResult,
    CumulantOptionUpdateResult,
)
from alberta_framework.core.cumulant_subtask_discovery import (
    CumulantSubtaskDiscovery,
    CumulantSubtaskDiscoveryArm,
    CumulantSubtaskDiscoveryResult,
    CumulantSubtaskDiscoveryState,
    CumulantSubtaskProposalBundle,
)
from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleMaintenanceReport,
)

CUMULANT_OPTION_SCHEDULER_CONFIG_SCHEMA = "alberta.cumulant-option-scheduler.config.v1"
CUMULANT_OPTION_SCHEDULER_CHECKPOINT_SCHEMA = "alberta.cumulant-option-scheduler.state.v1"
CUMULANT_OPTION_SCHEDULER_ASSESSMENT = "not_assessed"
CUMULANT_OPTION_SCHEDULER_HOST_ORCHESTRATION = True
CUMULANT_OPTION_SCHEDULER_OUTPUT_WRITES = False
CUMULANT_OPTION_SCHEDULER_EVIDENCE_AUTHORITY = False
CUMULANT_OPTION_SCHEDULER_PROMOTION_AUTHORITY = False
CUMULANT_OPTION_SCHEDULER_GO_NO_GO_AUTHORITY = False
CUMULANT_OPTION_SCHEDULER_SAFETY_AUTHORITY = False
CUMULANT_OPTION_SCHEDULER_RETIREMENT_AUTHORITY = False
CUMULANT_OPTION_SCHEDULER_DISPATCH_AUTHORITY = False
CUMULANT_OPTION_SCHEDULER_SCIENTIFIC_PROMOTION_ALLOWED = False

CUMULANT_OPTION_SCHEDULER_ERROR_NONE = 0
CUMULANT_OPTION_SCHEDULER_ERROR_CAPACITY = 1

_INT32_MAX = 2**31 - 1
_UINT64_MAX = 2**64 - 1
_DIGEST_WORDS = 8
_TRANSITION_WORDS = 2


def _positive_int(value: object, *, name: str, ceiling: int = _INT32_MAX) -> int:
    if type(value) is not int or not 1 <= value <= ceiling:
        raise ValueError(f"{name} must be a positive exact Python int <= {ceiling}")
    return value


def _nonnegative_int(value: object, *, name: str, ceiling: int = _INT32_MAX) -> int:
    if type(value) is not int or not 0 <= value <= ceiling:
        raise ValueError(f"{name} must be a non-negative exact Python int <= {ceiling}")
    return value


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with exact shape and dtype")
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _require_threefry_key(value: Any, *, name: str) -> Array:
    try:
        implementation = str(jr.key_impl(value))
        key_data = jr.key_data(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be one typed Threefry JAX key") from exc
    if (
        not jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
        or value.shape != ()
        or implementation != "threefry2x32"
        or key_data.shape != (_TRANSITION_WORDS,)
        or key_data.dtype != jnp.uint32
    ):
        raise TypeError(f"{name} must be one typed Threefry JAX key")
    return cast(Array, value)


def _int32_scalar(value: int | Array, *, name: str) -> Array:
    if type(value) is int:
        if not -(2**31) <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be signed-int32 compatible")
        return jnp.asarray(value, dtype=jnp.int32)
    return _require_array(value, name=name, shape=(), dtype=jnp.int32)


def _words(value: int) -> Array:
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        raise ValueError("exact counter value must be uint64-compatible")
    return jnp.asarray((value >> 32, value & 0xFFFFFFFF), dtype=jnp.uint32)


def _increment_words(value: Array) -> tuple[Array, Array]:
    low = value[1] + jnp.uint32(1)
    carry = (low == 0).astype(jnp.uint32)
    high = value[0] + carry
    available = ~((carry != 0) & (high == 0))
    candidate = jnp.stack((high, low), dtype=jnp.uint32)
    return jnp.where(available, candidate, value), available


def _words_less(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _words_less_equal(left: Array, right: Array) -> Array:
    return _words_less(left, right) | jnp.array_equal(left, right)


def _words_at_capacity(value: Array, capacity: int) -> Array:
    return jnp.array_equal(value, _words(capacity))


def _words_below_capacity(value: Array, capacity: int) -> Array:
    return _words_less(value, _words(capacity))


def _words_modulo(value: Array, divisor: int) -> Array:
    """Modulo for this scheduler's signed-int32-bounded exact clock."""

    return jnp.where(
        value[0] == 0,
        value[1] % jnp.uint32(divisor),
        jnp.uint32(divisor),
    )


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
            valid = valid & jnp.array_equal(jr.key_data(left_array), jr.key_data(right_array))
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


def _tree_sha256(tree: object) -> Array:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        host = np.asarray(jax.device_get(array))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.size) * int(array.dtype.itemsize)
    return total


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantOptionSchedulerConfig:
    """Static observation, retry, maintenance, and lifetime budgets."""

    proposal_period: int = 1
    maintenance_period: int = 32
    max_steps: int = 100_000
    max_install_attempts: int = 128
    max_retry_streak: int = 8
    max_maintenance_handoffs: int = 128

    SCHEMA_VERSION: ClassVar[str] = CUMULANT_OPTION_SCHEDULER_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _positive_int(self.proposal_period, name="proposal_period")
        _positive_int(self.maintenance_period, name="maintenance_period")
        _positive_int(self.max_steps, name="max_steps")
        _nonnegative_int(self.max_install_attempts, name="max_install_attempts")
        _positive_int(self.max_retry_streak, name="max_retry_streak")
        _nonnegative_int(
            self.max_maintenance_handoffs,
            name="max_maintenance_handoffs",
        )
        if self.proposal_period > self.max_steps:
            raise ValueError("proposal_period must not exceed max_steps")
        if self.maintenance_period > self.max_steps:
            raise ValueError("maintenance_period must not exceed max_steps")
        if self.max_install_attempts > self.max_steps:
            raise ValueError("max_install_attempts must not exceed max_steps")
        if self.max_maintenance_handoffs > self.max_steps:
            raise ValueError("max_maintenance_handoffs must not exceed max_steps")

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "proposal_period": self.proposal_period,
            "maintenance_period": self.maintenance_period,
            "max_steps": self.max_steps,
            "max_install_attempts": self.max_install_attempts,
            "max_retry_streak": self.max_retry_streak,
            "max_maintenance_handoffs": self.max_maintenance_handoffs,
            "assessment": CUMULANT_OPTION_SCHEDULER_ASSESSMENT,
            "host_orchestration": True,
            "pending_proposal_slots": 0,
            "fresh_bundle_required_on_retry": True,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "go_no_go_authority": False,
            "safety_authority": False,
            "retirement_authority": False,
            "dispatch_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> CumulantOptionSchedulerConfig:
        if type(value) is not dict:
            raise ValueError("scheduler config must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "proposal_period",
            "maintenance_period",
            "max_steps",
            "max_install_attempts",
            "max_retry_streak",
            "max_maintenance_handoffs",
            "assessment",
            "host_orchestration",
            "pending_proposal_slots",
            "fresh_bundle_required_on_retry",
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "go_no_go_authority",
            "safety_authority",
            "retirement_authority",
            "dispatch_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("scheduler config keys differ from schema v1")
        if raw.pop("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("scheduler config schema_version differs")
        if raw.pop("assessment") != CUMULANT_OPTION_SCHEDULER_ASSESSMENT:
            raise ValueError("scheduler assessment must remain not_assessed")
        if raw.pop("host_orchestration") is not True:
            raise ValueError("scheduler control must remain host-orchestrated")
        if raw.pop("pending_proposal_slots") != 0:
            raise ValueError("scheduler cannot persist pending proposal slots")
        if raw.pop("fresh_bundle_required_on_retry") is not True:
            raise ValueError("scheduler retries must require a fresh bundle")
        for name in (
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "go_no_go_authority",
            "safety_authority",
            "retirement_authority",
            "dispatch_authority",
            "scientific_promotion_allowed",
        ):
            if raw.pop(name) is not False:
                raise ValueError(f"scheduler cannot claim {name}")
        return cls(**cast(dict[str, Any], raw))


@chex.dataclass(frozen=True)
class CumulantOptionSchedulerArmInputs:
    """Current, predict-before-update discovery inputs for one transition."""

    current_raw_features: Float[Array, " raw_feature_dim"]
    current_raw_available: Bool[Array, " raw_feature_dim"]
    current_controllable_events: Float[Array, " controllable_event_dim"]
    current_controllable_events_available: Bool[Array, " controllable_event_dim"]
    current_transition_atoms: Float[Array, " transition_atom_dim"]
    current_transition_atoms_available: Bool[Array, " transition_atom_dim"]
    current_bottleneck_values: Float[Array, " prediction_bottleneck_dim"]
    current_bottleneck_available: Bool[Array, " prediction_bottleneck_dim"]
    probe_features: Float[Array, " probe_feature_dim"]
    current_incumbent_values: Float[Array, " incumbent_count"]
    current_incumbent_available: Bool[Array, " incumbent_count"]
    current_hand_values: Float[Array, " option_budget"]
    current_hand_available: Bool[Array, " option_budget"]
    hand_comparator_identity: UInt[Array, " 2"]
    reward_base_predictions: Float[Array, " reward_channel_count"]
    model_base_predictions: Float[Array, " model_channel_count"]
    action: Int[Array, ""]
    behavior_propensity: Float[Array, ""]
    randomized: Bool[Array, ""]
    transition_id: UInt[Array, " 2"]
    semantic_generation: Int[Array, ""]
    source_digest: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CumulantOptionSchedulerObservation:
    """Successor discovery inputs for the exact armed transition."""

    next_raw_features: Float[Array, " raw_feature_dim"]
    next_raw_available: Bool[Array, " raw_feature_dim"]
    next_controllable_events: Float[Array, " controllable_event_dim"]
    next_controllable_events_available: Bool[Array, " controllable_event_dim"]
    next_transition_atoms: Float[Array, " transition_atom_dim"]
    next_transition_atoms_available: Bool[Array, " transition_atom_dim"]
    next_bottleneck_values: Float[Array, " prediction_bottleneck_dim"]
    next_bottleneck_available: Bool[Array, " prediction_bottleneck_dim"]
    bottleneck_epistemic: Float[Array, " prediction_bottleneck_dim"]
    bottleneck_progress: Float[Array, " prediction_bottleneck_dim"]
    bottleneck_aleatoric: Float[Array, " prediction_bottleneck_dim"]
    bottleneck_evidence_available: Bool[Array, " prediction_bottleneck_dim"]
    randomized_action_evidence: Bool[Array, " n_actions"]
    next_incumbent_values: Float[Array, " incumbent_count"]
    next_incumbent_available: Bool[Array, " incumbent_count"]
    next_hand_values: Float[Array, " option_budget"]
    next_hand_available: Bool[Array, " option_budget"]
    hand_comparator_identity: UInt[Array, " 2"]
    reward_targets: Float[Array, " reward_channel_count"]
    reward_targets_available: Bool[Array, " reward_channel_count"]
    model_targets: Float[Array, " model_channel_count"]
    model_targets_available: Bool[Array, " model_channel_count"]
    transition_id: UInt[Array, " 2"]
    semantic_generation: Int[Array, ""]
    source_digest: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CumulantOptionInstallationAuthorityReceipt:
    """Caller declaration authorizing a bounded source-universe installation.

    The receipt is deliberately not cryptographic authentication.  Its issuer
    and revision bindings prevent accidental replay across scheduler owners.
    """

    go_no_go_authorized: Bool[Array, ""]
    safety_boundary_authorized: Bool[Array, ""]
    semantic_generation: Int[Array, ""]
    source_digest: UInt[Array, " 2"]
    canonical_digest: UInt[Array, " 32"]
    valid_from_step_words: UInt[Array, " 2"]
    valid_through_step_words: UInt[Array, " 2"]
    issuer_digest: UInt[Array, " 8"]
    authority_revision_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CumulantOptionSchedulerState:
    """Discovery, installer, exact schedules, and zero-payload retry state."""

    discovery_state: CumulantSubtaskDiscoveryState
    installation_state: CumulantOptionInstallationState
    installation_rng_key: Array
    expected_authority_issuer_digest: UInt[Array, " 8"]
    step_words: UInt[Array, " 2"]
    proposal_observation_words: UInt[Array, " 2"]
    install_attempt_words: UInt[Array, " 2"]
    install_applied_words: UInt[Array, " 2"]
    maintenance_handoff_words: UInt[Array, " 2"]
    control_update_words: UInt[Array, " 2"]
    last_authority_revision_words: UInt[Array, " 2"]
    retry_streak: Int[Array, ""]
    retry_due: Bool[Array, ""]
    schedule_unavailable: Bool[Array, ""]
    schedule_error: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CumulantOptionSchedulerMetadataState:
    """Scheduler metadata borrowing its installer's sole external STOMP owner."""

    discovery_state: CumulantSubtaskDiscoveryState
    installation_metadata: CumulantOptionInstallationMetadataState
    installation_rng_key: Array
    expected_authority_issuer_digest: UInt[Array, " 8"]
    step_words: UInt[Array, " 2"]
    proposal_observation_words: UInt[Array, " 2"]
    install_attempt_words: UInt[Array, " 2"]
    install_applied_words: UInt[Array, " 2"]
    maintenance_handoff_words: UInt[Array, " 2"]
    control_update_words: UInt[Array, " 2"]
    last_authority_revision_words: UInt[Array, " 2"]
    retry_streak: Int[Array, ""]
    retry_due: Bool[Array, ""]
    schedule_unavailable: Bool[Array, ""]
    schedule_error: Int[Array, ""]
    source_binding_checksum: UInt[Array, " 2"]
    metadata_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CumulantOptionSchedulerBorrowResult:
    """Fail-closed transient scheduler reconstruction around one STOMP owner."""

    state: CumulantOptionSchedulerState
    installation: CumulantOptionInstallationBorrowResult
    metadata_valid: Bool[Array, ""]
    binding_matches: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CumulantOptionSchedulerArm:
    """Discovery arm plus the exact scheduler state that issued it."""

    discovery_arm: CumulantSubtaskDiscoveryArm
    scheduler_step_words: UInt[Array, " 2"]
    scheduler_checksum: UInt[Array, " 2"]
    proposal_due: Bool[Array, ""]
    available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CumulantOptionScheduleClock:
    """Small pure-JAX cadence result used by eager, JIT, and scan callers."""

    current_step_words: UInt[Array, " 2"]
    next_step_words: UInt[Array, " 2"]
    proposal_due: Bool[Array, ""]
    maintenance_due_after_step: Bool[Array, ""]
    capacity_available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CumulantOptionRetirementHandoff:
    """Bounded maintenance proposal; it cannot mutate or retire an option."""

    report: OptionLifecycleMaintenanceReport
    available: Bool[Array, ""]
    scheduler_step_words: UInt[Array, " 2"]
    discovery_semantic_generation: Int[Array, ""]
    discovery_source_digest: UInt[Array, " 2"]
    discovery_canonical_digest: UInt[Array, " 32"]
    last_transition_id: UInt[Array, " 2"]
    consumer_source_digest: UInt[Array, " 8"]
    consumer_representation_digest: UInt[Array, " 8"]
    lifecycle_id: UInt[Array, " 2"]
    installation_revision: Int[Array, ""]
    lifecycle_revision: Int[Array, ""]
    audit_revision: Int[Array, ""]
    option_semantic_digests: UInt[Array, "option_budget 8"]
    option_semantic_generations: Int[Array, " option_budget"]
    proposed_retirement_slots: Int[Array, " maintenance_budget"]
    proposed_retirement_mask: Bool[Array, " maintenance_budget"]
    retirement_authority: Bool[Array, ""]
    go_no_go_authority: Bool[Array, ""]
    safety_authority: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CumulantOptionSchedulerResult:
    """Atomic composed transition and scheduling diagnostics."""

    state: CumulantOptionSchedulerState
    discovery: CumulantSubtaskDiscoveryResult
    materialization: CumulantOptionMaterialization
    retirement_handoff: CumulantOptionRetirementHandoff
    transaction_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    proposal_due: Bool[Array, ""]
    proposal_ready: Bool[Array, ""]
    authority_receipt_valid: Bool[Array, ""]
    authorization_requested: Bool[Array, ""]
    quiescent_boundary: Bool[Array, ""]
    installation_attempt_capacity_available: Bool[Array, ""]
    installation_attempt_capacity_exhausted: Bool[Array, ""]
    installation_attempted: Bool[Array, ""]
    installation_applied: Bool[Array, ""]
    installation_deferred: Bool[Array, ""]
    retry_scheduled: Bool[Array, ""]
    retry_exhausted_this_step: Bool[Array, ""]
    fresh_bundle_required_on_retry: Bool[Array, ""]
    cold_option_mask_active: Bool[Array, ""]
    maintenance_due: Bool[Array, ""]
    maintenance_handoff_emitted: Bool[Array, ""]
    retirement_applied: Bool[Array, ""]
    scheduler_rng_advanced: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantOptionSchedulerStartResult:
    """Host-only cold/live control start result."""

    state: CumulantOptionSchedulerState
    control: CumulantOptionStartResult | None
    cold_path: bool
    applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantOptionSchedulerUpdateResult:
    """Host-only cold/live control update result."""

    state: CumulantOptionSchedulerState
    control: CumulantOptionUpdateResult | None
    cold_path: bool
    applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantOptionSchedulerResourceBudget:
    """Exact allocation and bounded scheduler work/authority declaration."""

    persistent_state_nbytes: int
    discovery_state_nbytes: int
    installation_state_nbytes: int
    scheduler_binding_nbytes: int
    exact_counter_words: int
    pending_proposal_slots: int
    proposal_observations_per_accepted_step: int
    max_install_attempts_per_step: int
    max_materializations_per_step: int
    max_maintenance_reports_per_step: int
    rng_split_calls_at_init: int
    max_rng_split_calls_per_step: int
    rng_state_advances_on_every_applied_install: bool
    max_steps: int
    max_install_attempts: int
    max_retry_streak: int
    max_maintenance_handoffs: int
    cold_mask_delegated_to_installer: bool
    retries_require_fresh_bundle: bool
    installed_control_survives_scheduler_capacity: bool
    host_orchestration: bool
    assessment: str
    output_writes: bool
    evidence_authority: bool
    promotion_authority: bool
    go_no_go_authority: bool
    safety_authority: bool
    retirement_authority: bool
    dispatch_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str


class CumulantOptionScheduler:
    """Bounded, receipt-gated scheduler around one discovery/installer pair."""

    def __init__(
        self,
        installation: CumulantOptionInstallation,
        config: CumulantOptionSchedulerConfig | None = None,
    ) -> None:
        if type(installation) is not CumulantOptionInstallation:
            raise TypeError("installation must be an exact CumulantOptionInstallation")
        self._installation = installation
        self._discovery = installation.discovery
        self._config = config or CumulantOptionSchedulerConfig(
            max_steps=self._discovery.config.max_observations,
            max_install_attempts=min(
                installation.config.max_installations,
                self._discovery.config.max_observations,
            ),
            max_maintenance_handoffs=min(
                128,
                self._discovery.config.max_observations,
            ),
        )
        if self._config.max_steps > self._discovery.config.max_observations:
            raise ValueError("scheduler max_steps exceeds discovery max_observations")

    @property
    def config(self) -> CumulantOptionSchedulerConfig:
        return self._config

    @property
    def installation(self) -> CumulantOptionInstallation:
        return self._installation

    @property
    def discovery(self) -> CumulantSubtaskDiscovery:
        return self._discovery

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": CUMULANT_OPTION_SCHEDULER_CONFIG_SCHEMA,
            "scheduler": self._config.to_config(),
            "installation": self._installation.to_config(),
        }

    def _payload_arrays(self, state: CumulantOptionSchedulerState) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    state.discovery_state,
                    state.installation_state,
                    state.installation_rng_key,
                    state.expected_authority_issuer_digest,
                    state.step_words,
                    state.proposal_observation_words,
                    state.install_attempt_words,
                    state.install_applied_words,
                    state.maintenance_handoff_words,
                    state.control_update_words,
                    state.last_authority_revision_words,
                    state.retry_streak,
                    state.retry_due,
                    state.schedule_unavailable,
                    state.schedule_error,
                )
            )
        )

    def _with_checksum(
        self,
        state: CumulantOptionSchedulerState,
    ) -> CumulantOptionSchedulerState:
        return dataclasses.replace(
            state,
            binding_checksum=_checksum_arrays(self._payload_arrays(state)),
        )

    def _metadata_payload_arrays(
        self,
        state: CumulantOptionSchedulerMetadataState,
    ) -> tuple[Array, ...]:
        values = tuple(
            getattr(state, field.name)
            for field in dataclasses.fields(CumulantOptionSchedulerMetadataState)
            if field.name != "metadata_checksum"
        )
        return tuple(
            cast(Array, leaf) for leaf in jax.tree_util.tree_leaves(values)
        )

    def _with_metadata_checksum(
        self,
        state: CumulantOptionSchedulerMetadataState,
    ) -> CumulantOptionSchedulerMetadataState:
        return dataclasses.replace(
            state,
            metadata_checksum=_checksum_arrays(self._metadata_payload_arrays(state)),
        )

    def _check_metadata_contract(
        self,
        state: CumulantOptionSchedulerMetadataState,
    ) -> None:
        if type(state) is not CumulantOptionSchedulerMetadataState:
            raise TypeError(
                "state must be an exact CumulantOptionSchedulerMetadataState"
            )
        if type(state.discovery_state) is not CumulantSubtaskDiscoveryState:
            raise TypeError("state.discovery_state has the wrong exact type")
        if type(state.installation_metadata) is not CumulantOptionInstallationMetadataState:
            raise TypeError("state.installation_metadata has the wrong exact type")
        _require_threefry_key(
            state.installation_rng_key,
            name="state.installation_rng_key",
        )
        contracts = (
            (
                state.expected_authority_issuer_digest,
                "expected_authority_issuer_digest",
                (_DIGEST_WORDS,),
                jnp.uint32,
            ),
            (state.step_words, "step_words", (2,), jnp.uint32),
            (
                state.proposal_observation_words,
                "proposal_observation_words",
                (2,),
                jnp.uint32,
            ),
            (state.install_attempt_words, "install_attempt_words", (2,), jnp.uint32),
            (state.install_applied_words, "install_applied_words", (2,), jnp.uint32),
            (
                state.maintenance_handoff_words,
                "maintenance_handoff_words",
                (2,),
                jnp.uint32,
            ),
            (state.control_update_words, "control_update_words", (2,), jnp.uint32),
            (
                state.last_authority_revision_words,
                "last_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (state.retry_streak, "retry_streak", (), jnp.int32),
            (state.retry_due, "retry_due", (), jnp.bool_),
            (state.schedule_unavailable, "schedule_unavailable", (), jnp.bool_),
            (state.schedule_error, "schedule_error", (), jnp.int32),
            (state.source_binding_checksum, "source_binding_checksum", (2,), jnp.uint32),
            (state.metadata_checksum, "metadata_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def detach_borrowed_stomp(
        self,
        state: CumulantOptionSchedulerState,
    ) -> CumulantOptionSchedulerMetadataState:
        """Detach the scheduler stack without retaining its nested STOMP."""

        self._check_state_contract(state)
        installation_metadata = self._installation.detach_borrowed_stomp(
            state.installation_state
        )
        values = {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(CumulantOptionSchedulerState)
            if field.name not in {"installation_state", "binding_checksum"}
        }
        metadata = CumulantOptionSchedulerMetadataState(
            installation_metadata=installation_metadata,
            **values,
            source_binding_checksum=state.binding_checksum,
            metadata_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_metadata_checksum(metadata)

    def metadata_state_valid(
        self,
        state: CumulantOptionSchedulerMetadataState,
    ) -> Bool[Array, ""]:
        """Validate detached scheduling, discovery, and installer bindings."""

        self._check_metadata_contract(state)
        discovery_valid = self._discovery.validate_state(
            state.discovery_state,
            semantic_generation=state.discovery_state.semantic_generation,
            source_digest=state.discovery_state.source_digest,
        )
        installation_valid = self._installation.metadata_state_valid(
            state.installation_metadata
        )
        step_matches_discovery = (state.step_words[0] == 0) & (
            state.step_words[1]
            == state.discovery_state.observation_count.astype(jnp.uint32)
        )
        live = state.installation_metadata.has_live_observation
        child_binding = (~live) | (
            (
                state.installation_metadata.last_semantic_generation
                == state.discovery_state.semantic_generation
            )
            & jnp.array_equal(
                state.installation_metadata.last_source_digest,
                state.discovery_state.source_digest,
            )
            & jnp.array_equal(
                state.installation_metadata.last_canonical_digest,
                state.discovery_state.canonical_digest,
            )
            & (
                state.installation_metadata.last_materialization_observation_count
                == state.discovery_state.observation_count
            )
            & jnp.array_equal(
                state.installation_metadata.last_materialization_transition_id,
                state.discovery_state.last_transition_id,
            )
        )
        expected_unavailable = _words_at_capacity(
            state.step_words,
            self._config.max_steps,
        )
        expected_error = jnp.where(
            expected_unavailable,
            jnp.asarray(CUMULANT_OPTION_SCHEDULER_ERROR_CAPACITY, dtype=jnp.int32),
            jnp.asarray(CUMULANT_OPTION_SCHEDULER_ERROR_NONE, dtype=jnp.int32),
        )
        return (
            discovery_valid
            & installation_valid
            & step_matches_discovery
            & jnp.array_equal(state.proposal_observation_words, state.step_words)
            & _words_less_equal(state.step_words, _words(self._config.max_steps))
            & _words_less_equal(
                state.install_attempt_words,
                _words(self._config.max_install_attempts),
            )
            & _words_less_equal(state.install_applied_words, state.install_attempt_words)
            & _words_less_equal(
                state.maintenance_handoff_words,
                _words(self._config.max_maintenance_handoffs),
            )
            & (state.retry_streak >= 0)
            & (state.retry_streak < self._config.max_retry_streak)
            & (state.retry_due == (state.retry_streak > 0))
            & (state.schedule_unavailable == expected_unavailable)
            & (state.schedule_error == expected_error)
            & jnp.any(state.expected_authority_issuer_digest != 0)
            & child_binding
            & jnp.array_equal(
                state.metadata_checksum,
                _checksum_arrays(self._metadata_payload_arrays(state)),
            )
        )

    def attach_borrowed_stomp(
        self,
        metadata: CumulantOptionSchedulerMetadataState,
        stomp_state: Any,
    ) -> CumulantOptionSchedulerBorrowResult:
        """Build a transient scheduler around the exact borrowed STOMP owner."""

        self._check_metadata_contract(metadata)
        installation_result = self._installation.attach_borrowed_stomp(
            metadata.installation_metadata,
            stomp_state,
        )
        values = {
            field.name: getattr(metadata, field.name)
            for field in dataclasses.fields(CumulantOptionSchedulerMetadataState)
            if field.name
            not in {
                "installation_metadata",
                "source_binding_checksum",
                "metadata_checksum",
            }
        }
        candidate = CumulantOptionSchedulerState(
            installation_state=installation_result.state,
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
            & installation_result.transaction_applied
            & binding_matches
            & self.state_valid(candidate)
        )
        return CumulantOptionSchedulerBorrowResult(
            state=candidate,
            installation=installation_result,
            metadata_valid=metadata_valid,
            binding_matches=binding_matches,
            transaction_applied=transaction_applied,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _check_state_contract(self, state: CumulantOptionSchedulerState) -> None:
        if type(state) is not CumulantOptionSchedulerState:
            raise TypeError("state must be an exact CumulantOptionSchedulerState")
        if type(state.discovery_state) is not CumulantSubtaskDiscoveryState:
            raise TypeError("state.discovery_state has the wrong exact type")
        if type(state.installation_state) is not CumulantOptionInstallationState:
            raise TypeError("state.installation_state has the wrong exact type")
        _require_threefry_key(state.installation_rng_key, name="state.installation_rng_key")
        contracts = (
            (
                state.expected_authority_issuer_digest,
                "expected_authority_issuer_digest",
                (_DIGEST_WORDS,),
                jnp.uint32,
            ),
            (state.step_words, "step_words", (2,), jnp.uint32),
            (
                state.proposal_observation_words,
                "proposal_observation_words",
                (2,),
                jnp.uint32,
            ),
            (state.install_attempt_words, "install_attempt_words", (2,), jnp.uint32),
            (state.install_applied_words, "install_applied_words", (2,), jnp.uint32),
            (
                state.maintenance_handoff_words,
                "maintenance_handoff_words",
                (2,),
                jnp.uint32,
            ),
            (state.control_update_words, "control_update_words", (2,), jnp.uint32),
            (
                state.last_authority_revision_words,
                "last_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (state.retry_streak, "retry_streak", (), jnp.int32),
            (state.retry_due, "retry_due", (), jnp.bool_),
            (state.schedule_unavailable, "schedule_unavailable", (), jnp.bool_),
            (state.schedule_error, "schedule_error", (), jnp.int32),
            (state.binding_checksum, "binding_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def _check_authority_contract(
        self,
        receipt: CumulantOptionInstallationAuthorityReceipt,
    ) -> None:
        if type(receipt) is not CumulantOptionInstallationAuthorityReceipt:
            raise TypeError(
                "authority_receipt must be an exact "
                "CumulantOptionInstallationAuthorityReceipt"
            )
        contracts = (
            (receipt.go_no_go_authorized, "go_no_go_authorized", (), jnp.bool_),
            (
                receipt.safety_boundary_authorized,
                "safety_boundary_authorized",
                (),
                jnp.bool_,
            ),
            (receipt.semantic_generation, "semantic_generation", (), jnp.int32),
            (receipt.source_digest, "source_digest", (2,), jnp.uint32),
            (receipt.canonical_digest, "canonical_digest", (32,), jnp.uint8),
            (receipt.valid_from_step_words, "valid_from_step_words", (2,), jnp.uint32),
            (
                receipt.valid_through_step_words,
                "valid_through_step_words",
                (2,),
                jnp.uint32,
            ),
            (receipt.issuer_digest, "issuer_digest", (8,), jnp.uint32),
            (
                receipt.authority_revision_words,
                "authority_revision_words",
                (2,),
                jnp.uint32,
            ),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"authority_receipt.{name}", shape=shape, dtype=dtype)

    def init(
        self,
        key: Array,
        *,
        semantic_generation: int | Array,
        source_digest: Array,
        consumer_source_digest: Array,
        consumer_representation_digest: Array,
        lifecycle_id: Array,
        authority_issuer_digest: Array,
    ) -> CumulantOptionSchedulerState:
        """Initialize child mechanisms and one scheduler-owned install key."""

        checked = _require_threefry_key(key, name="key")
        issuer = _require_array(
            authority_issuer_digest,
            name="authority_issuer_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        if not bool(jax.device_get(jnp.any(issuer != 0))):
            raise ValueError("authority_issuer_digest must be nonzero")
        discovery_key, installation_key, scheduler_key = jr.split(checked, 3)
        discovery_state = self._discovery.init(
            discovery_key,
            semantic_generation=semantic_generation,
            source_digest=source_digest,
        )
        installation_state = self._installation.init(
            installation_key,
            consumer_source_digest=consumer_source_digest,
            consumer_representation_digest=consumer_representation_digest,
            lifecycle_id=lifecycle_id,
        )
        state = CumulantOptionSchedulerState(
            discovery_state=discovery_state,
            installation_state=installation_state,
            installation_rng_key=scheduler_key,
            expected_authority_issuer_digest=issuer,
            step_words=_words(0),
            proposal_observation_words=_words(0),
            install_attempt_words=_words(0),
            install_applied_words=_words(0),
            maintenance_handoff_words=_words(0),
            control_update_words=_words(0),
            last_authority_revision_words=_words(0),
            retry_streak=jnp.asarray(0, dtype=jnp.int32),
            retry_due=jnp.asarray(False, dtype=jnp.bool_),
            schedule_unavailable=jnp.asarray(False, dtype=jnp.bool_),
            schedule_error=jnp.asarray(CUMULANT_OPTION_SCHEDULER_ERROR_NONE, jnp.int32),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        state = self._with_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initialized scheduler state failed its exact contract")
        return state

    def state_valid(self, state: CumulantOptionSchedulerState) -> Bool[Array, ""]:
        """Validate child bindings, exact clocks, retry state, RNG, and checksum."""

        self._check_state_contract(state)
        discovery_valid = self._discovery.validate_state(
            state.discovery_state,
            semantic_generation=state.discovery_state.semantic_generation,
            source_digest=state.discovery_state.source_digest,
        )
        installation_valid = self._installation.state_valid(state.installation_state)
        step_matches_discovery = (
            state.step_words[0] == 0
        ) & (
            state.step_words[1]
            == state.discovery_state.observation_count.astype(jnp.uint32)
        )
        live = state.installation_state.has_live_observation
        child_binding = (~live) | (
            (state.installation_state.last_semantic_generation
             == state.discovery_state.semantic_generation)
            & jnp.array_equal(
                state.installation_state.last_source_digest,
                state.discovery_state.source_digest,
            )
            & jnp.array_equal(
                state.installation_state.last_canonical_digest,
                state.discovery_state.canonical_digest,
            )
            & (
                state.installation_state.last_materialization_observation_count
                == state.discovery_state.observation_count
            )
            & jnp.array_equal(
                state.installation_state.last_materialization_transition_id,
                state.discovery_state.last_transition_id,
            )
        )
        expected_unavailable = _words_at_capacity(state.step_words, self._config.max_steps)
        expected_error = jnp.where(
            expected_unavailable,
            jnp.asarray(CUMULANT_OPTION_SCHEDULER_ERROR_CAPACITY, dtype=jnp.int32),
            jnp.asarray(CUMULANT_OPTION_SCHEDULER_ERROR_NONE, dtype=jnp.int32),
        )
        return (
            discovery_valid
            & installation_valid
            & step_matches_discovery
            & jnp.array_equal(state.proposal_observation_words, state.step_words)
            & _words_less_equal(state.step_words, _words(self._config.max_steps))
            & _words_less_equal(
                state.install_attempt_words,
                _words(self._config.max_install_attempts),
            )
            & _words_less_equal(state.install_applied_words, state.install_attempt_words)
            & _words_less_equal(
                state.maintenance_handoff_words,
                _words(self._config.max_maintenance_handoffs),
            )
            & (state.retry_streak >= 0)
            & (state.retry_streak < self._config.max_retry_streak)
            & (state.retry_due == (state.retry_streak > 0))
            & (state.schedule_unavailable == expected_unavailable)
            & (state.schedule_error == expected_error)
            & jnp.any(state.expected_authority_issuer_digest != 0)
            & child_binding
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def proposal_due(self, state: CumulantOptionSchedulerState) -> Bool[Array, ""]:
        """Pure exact-clock schedule predicate; retry never stores a bundle."""

        self._check_state_contract(state)
        periodic = _words_modulo(state.step_words, self._config.proposal_period) == 0
        return (
            self.state_valid(state)
            & (~state.schedule_unavailable)
            & (periodic | state.retry_due)
        )

    def schedule_clock(
        self,
        step_words: Array,
        retry_due: Array,
    ) -> CumulantOptionScheduleClock:
        """Advance only exact cadence state; this boundary is JIT/scan safe."""

        words = _require_array(
            step_words,
            name="step_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        retry = _require_array(
            retry_due,
            name="retry_due",
            shape=(),
            dtype=jnp.bool_,
        )
        next_words, raw_capacity = _increment_words(words)
        within_capacity = _words_less_equal(next_words, _words(self._config.max_steps))
        available = raw_capacity & within_capacity
        proposal = (
            (_words_modulo(words, self._config.proposal_period) == 0) | retry
        ) & available
        maintenance = (
            _words_modulo(next_words, self._config.maintenance_period) == 0
        ) & available
        return CumulantOptionScheduleClock(
            current_step_words=words,
            next_step_words=jnp.where(available, next_words, words),
            proposal_due=proposal,
            maintenance_due_after_step=maintenance,
            capacity_available=available,
        )

    def arm(
        self,
        state: CumulantOptionSchedulerState,
        inputs: CumulantOptionSchedulerArmInputs,
    ) -> CumulantOptionSchedulerArm:
        """Arm discovery on every accepted scheduler opportunity."""

        self._check_state_contract(state)
        if type(inputs) is not CumulantOptionSchedulerArmInputs:
            raise TypeError("inputs must be an exact CumulantOptionSchedulerArmInputs")
        discovery_arm = self._discovery.arm(
            state.discovery_state,
            current_raw_features=inputs.current_raw_features,
            current_raw_available=inputs.current_raw_available,
            current_controllable_events=inputs.current_controllable_events,
            current_controllable_events_available=inputs.current_controllable_events_available,
            current_transition_atoms=inputs.current_transition_atoms,
            current_transition_atoms_available=inputs.current_transition_atoms_available,
            current_bottleneck_values=inputs.current_bottleneck_values,
            current_bottleneck_available=inputs.current_bottleneck_available,
            probe_features=inputs.probe_features,
            current_incumbent_values=inputs.current_incumbent_values,
            current_incumbent_available=inputs.current_incumbent_available,
            current_hand_values=inputs.current_hand_values,
            current_hand_available=inputs.current_hand_available,
            hand_comparator_identity=inputs.hand_comparator_identity,
            reward_base_predictions=inputs.reward_base_predictions,
            model_base_predictions=inputs.model_base_predictions,
            action=inputs.action,
            behavior_propensity=inputs.behavior_propensity,
            randomized=inputs.randomized,
            transition_id=inputs.transition_id,
            semantic_generation=inputs.semantic_generation,
            source_digest=inputs.source_digest,
        )
        available = (
            self.state_valid(state)
            & (~state.schedule_unavailable)
            & discovery_arm.available
        )
        return CumulantOptionSchedulerArm(
            discovery_arm=discovery_arm,
            scheduler_step_words=state.step_words,
            scheduler_checksum=state.binding_checksum,
            proposal_due=self.proposal_due(state),
            available=available,
        )

    def _live_matches_observation(
        self,
        live: CumulantOptionLiveInputs,
        observation: CumulantOptionSchedulerObservation,
        discovery: CumulantSubtaskDiscoveryResult,
    ) -> Array:
        return (
            _float_bits_equal(live.raw_features, observation.next_raw_features)
            & jnp.array_equal(live.raw_available, observation.next_raw_available)
            & _float_bits_equal(
                live.controllable_events,
                observation.next_controllable_events,
            )
            & jnp.array_equal(
                live.controllable_events_available,
                observation.next_controllable_events_available,
            )
            & _float_bits_equal(live.transition_atoms, observation.next_transition_atoms)
            & jnp.array_equal(
                live.transition_atoms_available,
                observation.next_transition_atoms_available,
            )
            & _float_bits_equal(
                live.bottleneck_values,
                observation.next_bottleneck_values,
            )
            & jnp.array_equal(
                live.bottleneck_available,
                observation.next_bottleneck_available,
            )
            & (live.semantic_generation == observation.semantic_generation)
            & jnp.array_equal(live.source_digest, observation.source_digest)
            & jnp.array_equal(live.transition_id, observation.transition_id)
            & jnp.array_equal(
                live.canonical_digest,
                discovery.state.canonical_digest,
            )
            & (
                live.state_observation_count
                == discovery.state.observation_count
            )
        )

    def _authority_valid(
        self,
        state: CumulantOptionSchedulerState,
        receipt: CumulantOptionInstallationAuthorityReceipt,
        live: CumulantOptionLiveInputs,
        next_step_words: Array,
    ) -> Array:
        return (
            receipt.go_no_go_authorized
            & receipt.safety_boundary_authorized
            & (receipt.semantic_generation == live.semantic_generation)
            & jnp.array_equal(receipt.source_digest, live.source_digest)
            & jnp.array_equal(receipt.canonical_digest, live.canonical_digest)
            & jnp.array_equal(
                receipt.issuer_digest,
                state.expected_authority_issuer_digest,
            )
            & jnp.any(receipt.issuer_digest != 0)
            & _words_less_equal(receipt.valid_from_step_words, next_step_words)
            & _words_less_equal(next_step_words, receipt.valid_through_step_words)
            & _words_less(
                state.last_authority_revision_words,
                receipt.authority_revision_words,
            )
            & jnp.any(receipt.authority_revision_words != 0)
        )

    def _fallback_materialize(
        self,
        state: CumulantOptionInstallationState,
        live: CumulantOptionLiveInputs,
    ) -> CumulantOptionMaterializationResult:
        if bool(jax.device_get(state.installed)):
            return self._installation.materialize_live(state, live)
        return self._installation.materialize_cold(state, live)

    def _retirement_handoff(
        self,
        discovery_state: CumulantSubtaskDiscoveryState,
        installation_state: CumulantOptionInstallationState,
        step_words: Array,
        *,
        available: Array,
    ) -> CumulantOptionRetirementHandoff:
        audit_state = installation_state.lifecycle_state.audit_state
        report = self._installation.lifecycle.audit.maintenance_report(audit_state)
        emitted = available & report.state_valid & installation_state.installed
        return CumulantOptionRetirementHandoff(
            report=report,
            available=emitted,
            scheduler_step_words=step_words,
            discovery_semantic_generation=discovery_state.semantic_generation,
            discovery_source_digest=discovery_state.source_digest,
            discovery_canonical_digest=discovery_state.canonical_digest,
            last_transition_id=discovery_state.last_transition_id,
            consumer_source_digest=installation_state.consumer_source_digest,
            consumer_representation_digest=(
                installation_state.consumer_representation_digest
            ),
            lifecycle_id=installation_state.lifecycle_id,
            installation_revision=installation_state.revision,
            lifecycle_revision=installation_state.lifecycle_state.revision,
            audit_revision=audit_state.revision,
            option_semantic_digests=installation_state.installed_semantic_digests,
            option_semantic_generations=audit_state.semantic_generations,
            proposed_retirement_slots=jnp.where(
                emitted,
                report.proposed_replacement_slots,
                -jnp.ones_like(report.proposed_replacement_slots),
            ),
            proposed_retirement_mask=emitted & report.proposed_replacement_mask,
            retirement_authority=jnp.asarray(False, dtype=jnp.bool_),
            go_no_go_authority=jnp.asarray(False, dtype=jnp.bool_),
            safety_authority=jnp.asarray(False, dtype=jnp.bool_),
        )

    def observe(
        self,
        state: CumulantOptionSchedulerState,
        arm: CumulantOptionSchedulerArm,
        observation: CumulantOptionSchedulerObservation,
        live_inputs: CumulantOptionLiveInputs,
        authority_receipt: CumulantOptionInstallationAuthorityReceipt,
    ) -> CumulantOptionSchedulerResult:
        """Observe, optionally install, materialize, and emit maintenance atomically."""

        self._check_state_contract(state)
        if type(arm) is not CumulantOptionSchedulerArm:
            raise TypeError("arm must be an exact CumulantOptionSchedulerArm")
        if type(observation) is not CumulantOptionSchedulerObservation:
            raise TypeError("observation must be an exact CumulantOptionSchedulerObservation")
        self._check_authority_contract(authority_receipt)
        self._installation._check_live_inputs(live_inputs)
        clock = self.schedule_clock(state.step_words, state.retry_due)
        next_step_words = clock.next_step_words
        step_capacity = clock.capacity_available
        next_observation_words, observation_capacity = _increment_words(
            state.proposal_observation_words
        )
        discovery = self._discovery.observe(
            state.discovery_state,
            arm.discovery_arm,
            next_raw_features=observation.next_raw_features,
            next_raw_available=observation.next_raw_available,
            next_controllable_events=observation.next_controllable_events,
            next_controllable_events_available=observation.next_controllable_events_available,
            next_transition_atoms=observation.next_transition_atoms,
            next_transition_atoms_available=observation.next_transition_atoms_available,
            next_bottleneck_values=observation.next_bottleneck_values,
            next_bottleneck_available=observation.next_bottleneck_available,
            bottleneck_epistemic=observation.bottleneck_epistemic,
            bottleneck_progress=observation.bottleneck_progress,
            bottleneck_aleatoric=observation.bottleneck_aleatoric,
            bottleneck_evidence_available=observation.bottleneck_evidence_available,
            randomized_action_evidence=observation.randomized_action_evidence,
            next_incumbent_values=observation.next_incumbent_values,
            next_incumbent_available=observation.next_incumbent_available,
            next_hand_values=observation.next_hand_values,
            next_hand_available=observation.next_hand_available,
            hand_comparator_identity=observation.hand_comparator_identity,
            reward_targets=observation.reward_targets,
            reward_targets_available=observation.reward_targets_available,
            model_targets=observation.model_targets,
            model_targets_available=observation.model_targets_available,
            transition_id=observation.transition_id,
            semantic_generation=observation.semantic_generation,
            source_digest=observation.source_digest,
        )
        persistent_valid = self.state_valid(state)
        arm_binding_valid = (
            arm.available
            & jnp.array_equal(arm.scheduler_step_words, state.step_words)
            & jnp.array_equal(arm.scheduler_checksum, state.binding_checksum)
            & jnp.array_equal(
                arm.discovery_arm.transition_id,
                observation.transition_id,
            )
        )
        live_binding_valid = self._live_matches_observation(
            live_inputs,
            observation,
            discovery,
        )
        fallback = self._fallback_materialize(state.installation_state, live_inputs)
        quiescent = (
            (state.installation_state.lifecycle_state.stomp_state.executing_option < 0)
            & (state.installation_state.lifecycle_state.audit_state.active_option < 0)
            & (~state.installation_state.lifecycle_state.audit_state.trial_active)
        )
        authority_valid = self._authority_valid(
            state,
            authority_receipt,
            live_inputs,
            next_step_words,
        )
        proposal_ready = discovery.discovered.ready
        install_requested = arm.proposal_due & proposal_ready
        attempt_capacity = _words_below_capacity(
            state.install_attempt_words,
            self._config.max_install_attempts,
        )
        installation_attempted = (
            install_requested
            & authority_valid
            & quiescent
            & state.installation_state.has_live_observation
            & attempt_capacity
        )
        attempted_on_host = bool(jax.device_get(installation_attempted))
        next_rng_key = state.installation_rng_key
        installation_state = state.installation_state
        installation_materialization = fallback.materialization
        installation_transaction_valid = jnp.asarray(True, dtype=jnp.bool_)
        installation_result_applied = jnp.asarray(False, dtype=jnp.bool_)
        installer_deferred = jnp.asarray(False, dtype=jnp.bool_)
        if attempted_on_host:
            next_rng_key, fresh_install_key = jr.split(state.installation_rng_key)
            installation = self._installation.install(
                state.installation_state,
                discovery.discovered,
                fresh_install_key,
                inputs=live_inputs,
            )
            installation_state = installation.state
            installation_materialization = installation.materialization
            installation_transaction_valid = installation.transaction_valid
            installation_result_applied = installation.applied
            installer_deferred = installation.deferred
        installation_applied = installation_attempted & installation_result_applied
        applied_on_host = bool(jax.device_get(installation_applied))
        selected_installation_state = (
            installation_state if applied_on_host else fallback.state
        )
        selected_materialization = (
            installation_materialization if applied_on_host else fallback.materialization
        )
        selected_materialization_applied = installation_applied | fallback.applied
        attempted_install_valid = (~installation_attempted) | installation_transaction_valid

        deferred = install_requested & (~installation_applied) & (
            (~authority_valid)
            | (~quiescent)
            | (~state.installation_state.has_live_observation)
            | installer_deferred
        )
        attempt_capacity_exhausted = install_requested & (~attempt_capacity)
        retry_increment = state.retry_streak + jnp.asarray(1, dtype=jnp.int32)
        retry_scheduled = deferred & (retry_increment < self._config.max_retry_streak)
        retry_exhausted = deferred & (~retry_scheduled)
        next_retry_streak = jnp.where(
            retry_scheduled,
            retry_increment,
            jnp.asarray(0, dtype=jnp.int32),
        )

        next_attempt_words, attempt_counter_capacity = _increment_words(
            state.install_attempt_words
        )
        next_applied_words, applied_counter_capacity = _increment_words(
            state.install_applied_words
        )
        committed_attempt_words = jnp.where(
            installation_attempted,
            next_attempt_words,
            state.install_attempt_words,
        )
        committed_applied_words = jnp.where(
            installation_applied,
            next_applied_words,
            state.install_applied_words,
        )
        # Even a descriptor-identical refresh commits new provenance and live
        # state.  Consume the successor key on every applied installation so a
        # derived fresh key is never reused after a committed transaction.
        committed_rng = (
            next_rng_key
            if bool(jax.device_get(installation_applied))
            else state.installation_rng_key
        )
        committed_authority_revision = jnp.where(
            installation_applied,
            authority_receipt.authority_revision_words,
            state.last_authority_revision_words,
        )

        maintenance_due = clock.maintenance_due_after_step
        maintenance_capacity = _words_below_capacity(
            state.maintenance_handoff_words,
            self._config.max_maintenance_handoffs,
        )
        maintenance_requested = maintenance_due & maintenance_capacity
        handoff = self._retirement_handoff(
            discovery.state,
            selected_installation_state,
            next_step_words,
            available=maintenance_requested,
        )
        next_handoff_words, handoff_counter_capacity = _increment_words(
            state.maintenance_handoff_words
        )
        committed_handoff_words = jnp.where(
            handoff.available,
            next_handoff_words,
            state.maintenance_handoff_words,
        )

        reaches_capacity = _words_at_capacity(next_step_words, self._config.max_steps)
        proposed = CumulantOptionSchedulerState(
            discovery_state=discovery.state,
            installation_state=selected_installation_state,
            installation_rng_key=committed_rng,
            expected_authority_issuer_digest=state.expected_authority_issuer_digest,
            step_words=next_step_words,
            proposal_observation_words=next_observation_words,
            install_attempt_words=committed_attempt_words,
            install_applied_words=committed_applied_words,
            maintenance_handoff_words=committed_handoff_words,
            control_update_words=state.control_update_words,
            last_authority_revision_words=committed_authority_revision,
            retry_streak=next_retry_streak,
            retry_due=retry_scheduled,
            schedule_unavailable=reaches_capacity,
            schedule_error=jnp.where(
                reaches_capacity,
                jnp.asarray(CUMULANT_OPTION_SCHEDULER_ERROR_CAPACITY, dtype=jnp.int32),
                jnp.asarray(CUMULANT_OPTION_SCHEDULER_ERROR_NONE, dtype=jnp.int32),
            ),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        transaction_valid = (
            persistent_valid
            & arm_binding_valid
            & step_capacity
            & observation_capacity
            & discovery.diagnostics.transaction_applied
            & live_binding_valid
            & selected_materialization_applied
            & attempted_install_valid
            & ((~installation_attempted) | attempt_counter_capacity)
            & ((~installation_applied) | applied_counter_capacity)
            & ((~handoff.available) | handoff_counter_capacity)
            & _words_less_equal(next_step_words, _words(self._config.max_steps))
            & self.state_valid(proposed)
        )
        next_state = proposed if bool(jax.device_get(transaction_valid)) else state
        returned_handoff = dataclasses.replace(
            handoff,
            available=transaction_valid & handoff.available,
            proposed_retirement_slots=jnp.where(
                transaction_valid & handoff.available,
                handoff.proposed_retirement_slots,
                -jnp.ones_like(handoff.proposed_retirement_slots),
            ),
            proposed_retirement_mask=(
                transaction_valid & handoff.proposed_retirement_mask
            ),
        )
        return CumulantOptionSchedulerResult(
            state=next_state,
            discovery=discovery,
            materialization=selected_materialization,
            retirement_handoff=returned_handoff,
            transaction_valid=transaction_valid,
            transaction_applied=transaction_valid,
            proposal_due=arm.proposal_due,
            proposal_ready=proposal_ready,
            authority_receipt_valid=authority_valid,
            authorization_requested=install_requested & (~authority_valid),
            quiescent_boundary=quiescent,
            installation_attempt_capacity_available=attempt_capacity,
            installation_attempt_capacity_exhausted=(
                transaction_valid & attempt_capacity_exhausted
            ),
            installation_attempted=transaction_valid & installation_attempted,
            installation_applied=transaction_valid & installation_applied,
            installation_deferred=transaction_valid & deferred,
            retry_scheduled=transaction_valid & retry_scheduled,
            retry_exhausted_this_step=transaction_valid & retry_exhausted,
            fresh_bundle_required_on_retry=jnp.asarray(True, dtype=jnp.bool_),
            cold_option_mask_active=transaction_valid & (~selected_installation_state.installed),
            maintenance_due=transaction_valid & maintenance_due,
            maintenance_handoff_emitted=transaction_valid & handoff.available,
            retirement_applied=jnp.asarray(False, dtype=jnp.bool_),
            scheduler_rng_advanced=transaction_valid & installation_applied,
        )

    def _commit_control_state(
        self,
        state: CumulantOptionSchedulerState,
        installation_state: CumulantOptionInstallationState,
    ) -> tuple[CumulantOptionSchedulerState, bool]:
        next_words, capacity = _increment_words(state.control_update_words)
        if not bool(jax.device_get(capacity)):
            return state, False
        proposed = self._with_checksum(
            dataclasses.replace(
                state,
                installation_state=installation_state,
                control_update_words=next_words,
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        if not bool(jax.device_get(self.state_valid(proposed))):
            return state, False
        return proposed, True

    def start_control(
        self,
        state: CumulantOptionSchedulerState,
        materialization: CumulantOptionMaterialization,
    ) -> CumulantOptionSchedulerStartResult:
        """Host-only primitive/cold or installed lifecycle start."""

        self._check_state_contract(state)
        if not bool(jax.device_get(self.state_valid(state))):
            return CumulantOptionSchedulerStartResult(state, None, False, False)
        cold = not bool(jax.device_get(state.installation_state.installed))
        child = (
            self._installation.start_cold(state.installation_state, materialization)
            if cold
            else self._installation.start(state.installation_state, materialization)
        )
        if not child.applied:
            return CumulantOptionSchedulerStartResult(state, None, cold, False)
        proposed, applied = self._commit_control_state(state, child.state)
        return CumulantOptionSchedulerStartResult(
            proposed if applied else state,
            child if applied else None,
            cold,
            applied,
        )

    def update_control(
        self,
        state: CumulantOptionSchedulerState,
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
    ) -> CumulantOptionSchedulerUpdateResult:
        """Host-only real control update; scheduler capacity is not a veto."""

        self._check_state_contract(state)
        if not bool(jax.device_get(self.state_valid(state))):
            return CumulantOptionSchedulerUpdateResult(state, None, False, False)
        cold = not bool(jax.device_get(state.installation_state.installed))
        if cold:
            child = self._installation.update_cold(
                state.installation_state,
                materialization,
                env_reward,
                discount,
                execution_boundary=execution_boundary,
            )
        else:
            child = self._installation.update(
                state.installation_state,
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
            return CumulantOptionSchedulerUpdateResult(state, None, cold, False)
        proposed, applied = self._commit_control_state(state, child.state)
        return CumulantOptionSchedulerUpdateResult(
            proposed if applied else state,
            child if applied else None,
            cold,
            applied,
        )

    def checkpoint_payload(self, state: CumulantOptionSchedulerState) -> dict[str, object]:
        """Return a host-only exact state payload with an integrity digest."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid cumulant option scheduler")
        return {
            "schema_version": CUMULANT_OPTION_SCHEDULER_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "state": state,
            "state_digest": _tree_sha256(state),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        expected_semantic_generation: int | Array,
        expected_source_digest: Array,
        expected_consumer_source_digest: Array,
        expected_consumer_representation_digest: Array,
        expected_lifecycle_id: Array,
        expected_authority_issuer_digest: Array,
        expected_installed_bundle: CumulantSubtaskProposalBundle | None,
    ) -> CumulantOptionSchedulerState:
        """Restore only the exact config, digest, and every external binding."""

        if type(payload) is not dict:
            raise ValueError("scheduler checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {"schema_version", "config", "state", "state_digest"}:
            raise ValueError("scheduler checkpoint keys differ from schema v1")
        if raw["schema_version"] != CUMULANT_OPTION_SCHEDULER_CHECKPOINT_SCHEMA:
            raise ValueError("scheduler checkpoint schema differs")
        if raw["config"] != self.to_config():
            raise ValueError("scheduler checkpoint config differs")
        if type(raw["state"]) is not CumulantOptionSchedulerState:
            raise ValueError("scheduler checkpoint state type differs")
        state = raw["state"]
        digest = _require_array(
            raw["state_digest"],
            name="checkpoint.state_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        if not bool(jax.device_get(jnp.array_equal(digest, _tree_sha256(state)))):
            raise ValueError("scheduler checkpoint state digest differs")
        generation = _int32_scalar(
            expected_semantic_generation,
            name="expected_semantic_generation",
        )
        source = _require_array(
            expected_source_digest,
            name="expected_source_digest",
            shape=(2,),
            dtype=jnp.uint32,
        )
        consumer_source = _require_array(
            expected_consumer_source_digest,
            name="expected_consumer_source_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        representation = _require_array(
            expected_consumer_representation_digest,
            name="expected_consumer_representation_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        lifecycle = _require_array(
            expected_lifecycle_id,
            name="expected_lifecycle_id",
            shape=(2,),
            dtype=jnp.uint32,
        )
        issuer = _require_array(
            expected_authority_issuer_digest,
            name="expected_authority_issuer_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        if expected_installed_bundle is None:
            bundle_valid = ~state.installation_state.installed
        else:
            self._discovery.check_proposal_bundle_contract(expected_installed_bundle)
            bundle_valid = state.installation_state.installed & _tree_array_equal(
                state.installation_state.installed_bundle,
                expected_installed_bundle,
            )
        binding_valid = (
            (state.discovery_state.semantic_generation == generation)
            & jnp.array_equal(state.discovery_state.source_digest, source)
            & jnp.array_equal(
                state.installation_state.consumer_source_digest,
                consumer_source,
            )
            & jnp.array_equal(
                state.installation_state.consumer_representation_digest,
                representation,
            )
            & jnp.array_equal(state.installation_state.lifecycle_id, lifecycle)
            & jnp.array_equal(state.expected_authority_issuer_digest, issuer)
            & bundle_valid
            & self.state_valid(state)
        )
        if not bool(jax.device_get(binding_valid)):
            raise ValueError("scheduler checkpoint is invalid, stale, or rebound")
        return state

    def resource_budget(
        self,
        state: CumulantOptionSchedulerState,
    ) -> CumulantOptionSchedulerResourceBudget:
        """Return exact persistent bytes and declared maximum scheduler work."""

        self._check_state_contract(state)
        total = _tree_nbytes(state)
        discovery_bytes = _tree_nbytes(state.discovery_state)
        installation_bytes = _tree_nbytes(state.installation_state)
        return CumulantOptionSchedulerResourceBudget(
            persistent_state_nbytes=total,
            discovery_state_nbytes=discovery_bytes,
            installation_state_nbytes=installation_bytes,
            scheduler_binding_nbytes=total - discovery_bytes - installation_bytes,
            exact_counter_words=14,
            pending_proposal_slots=0,
            proposal_observations_per_accepted_step=1,
            max_install_attempts_per_step=1,
            max_materializations_per_step=2,
            max_maintenance_reports_per_step=1,
            rng_split_calls_at_init=1,
            max_rng_split_calls_per_step=1,
            rng_state_advances_on_every_applied_install=True,
            max_steps=self._config.max_steps,
            max_install_attempts=self._config.max_install_attempts,
            max_retry_streak=self._config.max_retry_streak,
            max_maintenance_handoffs=self._config.max_maintenance_handoffs,
            cold_mask_delegated_to_installer=True,
            retries_require_fresh_bundle=True,
            installed_control_survives_scheduler_capacity=True,
            host_orchestration=True,
            assessment=CUMULANT_OPTION_SCHEDULER_ASSESSMENT,
            output_writes=False,
            evidence_authority=False,
            promotion_authority=False,
            go_no_go_authority=False,
            safety_authority=False,
            retirement_authority=False,
            dispatch_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=CUMULANT_OPTION_SCHEDULER_CHECKPOINT_SCHEMA,
        )


CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_PREPARED_SCHEMA = (
    "alberta.cumulant-option-external-bundle-adoption.prepared.v2"
)
CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_RECEIPT_SCHEMA = (
    "alberta.cumulant-option-external-bundle-adoption.receipt.v2"
)
CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_RESULT_SCHEMA = (
    "alberta.cumulant-option-external-bundle-adoption.result.v2"
)


@chex.dataclass(frozen=True)
class CumulantOptionExternalBundleAdoptionDiagnostics:
    """Source-bound feasibility facts for one external filtered cohort."""

    all_installed_source_valid: Bool[Array, ""]
    one_cold_retired_destination_valid: Bool[Array, ""]
    retirement_scheduler_relation_valid: Bool[Array, ""]
    all_installed_source_mask_exact: Bool[Array, ""]
    one_cold_retired_destination_mask_exact: Bool[Array, ""]
    retirement_target_exact: Bool[Array, ""]
    retirement_authority_revision_bound: Bool[Array, ""]
    arm_binding_valid: Bool[Array, ""]
    ordinary_transition_valid: Bool[Array, ""]
    bundle_binding_valid: Bool[Array, ""]
    exact_target_semantic_change: Bool[Array, ""]
    live_slots_semantically_preserved: Bool[Array, ""]
    fresh_transition: Bool[Array, ""]
    quiescent: Bool[Array, ""]
    scheduler_capacity_available: Bool[Array, ""]
    installer_capacity_available: Bool[Array, ""]
    caller_keys_valid: Bool[Array, ""]
    installation_transaction_valid: Bool[Array, ""]
    installation_applied: Bool[Array, ""]
    exact_reset_mask: Bool[Array, ""]
    exact_preserve_mask: Bool[Array, ""]
    live_policy_rng_preserved: Bool[Array, ""]
    candidate_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CumulantOptionExternalBundleAdoptionPrepared:
    """Versioned transient scheduler adoption; never a persistent proposal."""

    all_installed_source: CumulantOptionSchedulerState
    one_cold_retired_destination: CumulantOptionSchedulerState
    arm: CumulantOptionSchedulerArm
    observation: CumulantOptionSchedulerObservation
    live_inputs: CumulantOptionLiveInputs
    candidate_bundle: CumulantSubtaskProposalBundle
    target_mask: Bool[Array, " option_budget"]
    all_installed_source_slot_mask: Bool[Array, " option_budget"]
    one_cold_retired_destination_slot_mask: Bool[Array, " option_budget"]
    retirement_target_mask: Bool[Array, " option_budget"]
    retirement_authority_revision_words: UInt[Array, " 2"]
    installation_key: Array
    successor_scheduler_key: Array
    ordinary_result: CumulantOptionSchedulerResult
    installation_result: CumulantOptionInstallationResult
    scheduler_identity_digest: UInt[Array, " 8"]
    installer_identity_digest: UInt[Array, " 8"]
    diagnostics: CumulantOptionExternalBundleAdoptionDiagnostics
    prepared_checksum: UInt[Array, " 2"]

    SCHEMA_VERSION: ClassVar[str] = (
        CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_PREPARED_SCHEMA
    )

    @property
    def schema_version(self) -> str:
        return self.SCHEMA_VERSION


@chex.dataclass(frozen=True)
class CumulantOptionExternalBundleAdoptionAuthorityReceipt:
    """Integrity declaration binding one exact scheduler adoption."""

    installation_authority: CumulantOptionInstallationAuthorityReceipt
    adoption_authorized: Bool[Array, ""]
    scheduler_identity_digest: UInt[Array, " 8"]
    installer_identity_digest: UInt[Array, " 8"]
    all_installed_source_checksum: UInt[Array, " 2"]
    retired_destination_checksum: UInt[Array, " 2"]
    retired_installation_revision: Int[Array, ""]
    retired_lifecycle_revision: Int[Array, ""]
    retired_audit_revision: Int[Array, ""]
    target_mask: Bool[Array, " option_budget"]
    all_installed_source_slot_mask: Bool[Array, " option_budget"]
    one_cold_retired_destination_slot_mask: Bool[Array, " option_budget"]
    retirement_target_mask: Bool[Array, " option_budget"]
    retirement_authority_revision_words: UInt[Array, " 2"]
    candidate_binding_digest: UInt[Array, " 2"]
    candidate_semantic_digests: UInt[Array, "option_budget 8"]
    installation_key_data: UInt[Array, " 2"]
    successor_scheduler_key_data: UInt[Array, " 2"]
    prepared_checksum: UInt[Array, " 2"]

    SCHEMA_VERSION: ClassVar[str] = (
        CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_RECEIPT_SCHEMA
    )


@chex.dataclass(frozen=True)
class CumulantOptionExternalBundleAdoptionResult:
    """Installed scheduler successor or exact one-cold transient destination."""

    state: CumulantOptionSchedulerState
    ordinary_result: CumulantOptionSchedulerResult
    materialization: CumulantOptionMaterialization
    destination_state_valid: Bool[Array, ""]
    destination_matches_retired: Bool[Array, ""]
    prepared_integrity_valid: Bool[Array, ""]
    preparation_derivation_valid: Bool[Array, ""]
    authority_valid: Bool[Array, ""]
    candidate_ready: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    reset_slots: Bool[Array, " option_budget"]
    preserved_slots: Bool[Array, " option_budget"]
    installation_key_consumed: Array
    caller_authenticated: Bool[Array, ""]

    SCHEMA_VERSION: ClassVar[str] = CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_RESULT_SCHEMA


def _external_bundle_adoption_identity(scheduler: CumulantOptionScheduler) -> tuple[Array, Array]:
    from alberta_framework.core.option_lifecycle_audit import option_semantic_digest

    return (
        option_semantic_digest(
            {
                "schema": CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_PREPARED_SCHEMA,
                "scheduler": scheduler.to_config(),
            }
        ),
        option_semantic_digest(
            {
                "schema": CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_PREPARED_SCHEMA,
                "installation": scheduler.installation.to_config(),
            }
        ),
    )


def _external_bundle_prepared_payload(
    prepared: CumulantOptionExternalBundleAdoptionPrepared,
) -> tuple[Array, ...]:
    return tuple(
        cast(Array, leaf)
        for leaf in jax.tree_util.tree_leaves(
            tuple(
                getattr(prepared, field.name)
                for field in dataclasses.fields(CumulantOptionExternalBundleAdoptionPrepared)
                if field.name != "prepared_checksum"
            )
        )
    )


def _scheduler_retirement_relation_valid(
    all_installed_source: CumulantOptionSchedulerState,
    one_cold_retired_destination: CumulantOptionSchedulerState,
    all_installed_source_slot_mask: Array,
    one_cold_retired_destination_slot_mask: Array,
    retirement_target_mask: Array,
    retirement_authority_revision_words: Array,
) -> Array:
    """Pin the narrow scheduler delta produced by an authorized retirement."""

    source_installation = all_installed_source.installation_state
    retired_installation = one_cold_retired_destination.installation_state
    scheduler_fields_equal = _tree_array_equal(
        tuple(
            getattr(all_installed_source, field.name)
            for field in dataclasses.fields(CumulantOptionSchedulerState)
            if field.name not in {"installation_state", "binding_checksum"}
        ),
        tuple(
            getattr(one_cold_retired_destination, field.name)
            for field in dataclasses.fields(CumulantOptionSchedulerState)
            if field.name not in {"installation_state", "binding_checksum"}
        ),
    )
    installation_fields_equal = _tree_array_equal(
        tuple(
            getattr(source_installation, field.name)
            for field in dataclasses.fields(CumulantOptionInstallationState)
            if field.name not in {"lifecycle_state", "revision", "binding_checksum"}
        ),
        tuple(
            getattr(retired_installation, field.name)
            for field in dataclasses.fields(CumulantOptionInstallationState)
            if field.name not in {"lifecycle_state", "revision", "binding_checksum"}
        ),
    )
    semantic_binding_preserved = (
        jnp.array_equal(
            source_installation.installed_semantic_digests,
            retired_installation.installed_semantic_digests,
        )
        & jnp.array_equal(
            source_installation.lifecycle_state.audit_state.semantic_digests,
            retired_installation.lifecycle_state.audit_state.semantic_digests,
        )
        & (
            retired_installation.revision
            == source_installation.revision + jnp.asarray(1, dtype=jnp.int32)
        )
        & (
            retired_installation.lifecycle_state.revision
            == source_installation.lifecycle_state.revision + jnp.asarray(2, dtype=jnp.int32)
        )
        & (
            retired_installation.lifecycle_state.audit_state.revision
            == source_installation.lifecycle_state.audit_state.revision
            + jnp.asarray(2, dtype=jnp.int32)
        )
        & jnp.array_equal(
            retired_installation.lifecycle_state.audit_state.semantic_generations,
            source_installation.lifecycle_state.audit_state.semantic_generations
            + jnp.asarray(retirement_target_mask, dtype=jnp.int32) * jnp.int32(2),
        )
    )
    target_valid = (
        jnp.all(all_installed_source_slot_mask)
        & (jnp.sum(~one_cold_retired_destination_slot_mask, dtype=jnp.int32) == 1)
        & jnp.array_equal(
            one_cold_retired_destination_slot_mask,
            all_installed_source_slot_mask & (~retirement_target_mask),
        )
        & jnp.array_equal(
            retirement_target_mask,
            ~one_cold_retired_destination_slot_mask,
        )
        & (jnp.sum(retirement_target_mask, dtype=jnp.int32) == 1)
        & jnp.any(retirement_authority_revision_words != 0)
    )
    return (
        scheduler_fields_equal
        & installation_fields_equal
        & semantic_binding_preserved
        & target_valid
    )


def _denied_external_bundle_receipt(
    source: CumulantOptionSchedulerState,
    live_inputs: CumulantOptionLiveInputs,
) -> CumulantOptionInstallationAuthorityReceipt:
    next_revision, _ = _increment_words(source.last_authority_revision_words)
    return CumulantOptionInstallationAuthorityReceipt(
        go_no_go_authorized=jnp.asarray(False, dtype=jnp.bool_),
        safety_boundary_authorized=jnp.asarray(False, dtype=jnp.bool_),
        semantic_generation=live_inputs.semantic_generation,
        source_digest=live_inputs.source_digest,
        canonical_digest=live_inputs.canonical_digest,
        valid_from_step_words=source.step_words,
        valid_through_step_words=jnp.full((2,), 0xFFFFFFFF, dtype=jnp.uint32),
        issuer_digest=source.expected_authority_issuer_digest,
        authority_revision_words=next_revision,
    )


def prepare_cumulant_option_external_bundle_adoption(
    scheduler: CumulantOptionScheduler,
    all_installed_source: CumulantOptionSchedulerState,
    one_cold_retired_destination: CumulantOptionSchedulerState,
    arm: CumulantOptionSchedulerArm,
    observation: CumulantOptionSchedulerObservation,
    live_inputs: CumulantOptionLiveInputs,
    candidate_bundle: CumulantSubtaskProposalBundle,
    target_mask: Array,
    all_installed_source_slot_mask: Array,
    one_cold_retired_destination_slot_mask: Array,
    retirement_target_mask: Array,
    retirement_authority_revision_words: Array,
    installation_key: Array,
    successor_scheduler_key: Array,
) -> CumulantOptionExternalBundleAdoptionPrepared:
    """Rederive one ordinary transition and stage one exact external cohort."""

    if type(scheduler) is not CumulantOptionScheduler:
        raise TypeError("scheduler must be an exact CumulantOptionScheduler")
    scheduler._check_state_contract(all_installed_source)
    scheduler._check_state_contract(one_cold_retired_destination)
    if type(arm) is not CumulantOptionSchedulerArm:
        raise TypeError("arm must be an exact CumulantOptionSchedulerArm")
    if type(observation) is not CumulantOptionSchedulerObservation:
        raise TypeError("observation must be an exact CumulantOptionSchedulerObservation")
    scheduler.discovery.check_proposal_bundle_contract(candidate_bundle)
    target = _require_array(
        target_mask,
        name="target_mask",
        shape=(scheduler.discovery.config.option_budget,),
        dtype=jnp.bool_,
    )
    all_installed_mask = _require_array(
        all_installed_source_slot_mask,
        name="all_installed_source_slot_mask",
        shape=(scheduler.discovery.config.option_budget,),
        dtype=jnp.bool_,
    )
    retired_slot_mask = _require_array(
        one_cold_retired_destination_slot_mask,
        name="one_cold_retired_destination_slot_mask",
        shape=(scheduler.discovery.config.option_budget,),
        dtype=jnp.bool_,
    )
    retirement_target = _require_array(
        retirement_target_mask,
        name="retirement_target_mask",
        shape=(scheduler.discovery.config.option_budget,),
        dtype=jnp.bool_,
    )
    retirement_authority_revision = _require_array(
        retirement_authority_revision_words,
        name="retirement_authority_revision_words",
        shape=(2,),
        dtype=jnp.uint32,
    )
    install_key = _require_threefry_key(installation_key, name="installation_key")
    successor_key = _require_threefry_key(
        successor_scheduler_key,
        name="successor_scheduler_key",
    )
    ordinary = scheduler.observe(
        one_cold_retired_destination,
        arm,
        observation,
        live_inputs,
        _denied_external_bundle_receipt(one_cold_retired_destination, live_inputs),
    )
    source_valid = scheduler.state_valid(all_installed_source)
    retired_valid = scheduler.state_valid(one_cold_retired_destination)
    retirement_relation = _scheduler_retirement_relation_valid(
        all_installed_source,
        one_cold_retired_destination,
        all_installed_mask,
        retired_slot_mask,
        retirement_target,
        retirement_authority_revision,
    )
    all_installed_mask_exact = jnp.all(all_installed_mask)
    one_cold_mask_exact = (
        (jnp.sum(~retired_slot_mask, dtype=jnp.int32) == 1)
        & jnp.array_equal(retired_slot_mask, all_installed_mask & (~retirement_target))
    )
    retirement_target_exact = jnp.array_equal(target, retirement_target)
    retirement_authority_bound = jnp.any(retirement_authority_revision != 0)
    arm_binding = (
        arm.available
        & jnp.array_equal(
            arm.scheduler_checksum,
            one_cold_retired_destination.binding_checksum,
        )
        & jnp.array_equal(
            arm.scheduler_step_words,
            one_cold_retired_destination.step_words,
        )
    )
    bundle_binding = scheduler.discovery.validate_proposal_bundle(
        candidate_bundle,
        semantic_generation=live_inputs.semantic_generation,
        source_digest=live_inputs.source_digest,
        canonical_digest=live_inputs.canonical_digest,
        transition_id=live_inputs.transition_id,
        state_observation_count=live_inputs.state_observation_count,
    ) & (candidate_bundle.cohort_id == -1)
    candidate_semantics = scheduler.installation.semantic_digests_for_bundle(
        candidate_bundle
    )
    source_semantics = (
        one_cold_retired_destination.installation_state.installed_semantic_digests
    )
    changed = jnp.any(candidate_semantics != source_semantics, axis=1)
    exact_target = (
        jnp.sum(target, dtype=jnp.int32) == 1
    ) & jnp.array_equal(changed, target)
    live_preserved = ~jnp.any(changed & (~target))
    installation_source = one_cold_retired_destination.installation_state
    fresh = _words_less(
        installation_source.last_materialization_transition_id,
        candidate_bundle.transition_id,
    ) & (
        candidate_bundle.state_observation_count
        > installation_source.last_materialization_observation_count
    )
    quiescent = (
        ordinary.quiescent_boundary
        & (installation_source.lifecycle_state.stomp_state.executing_option < 0)
        & (installation_source.lifecycle_state.audit_state.active_option < 0)
        & (~installation_source.lifecycle_state.audit_state.trial_active)
    )
    scheduler_capacity = ordinary.installation_attempt_capacity_available
    installer_capacity = ~installation_source.installer_unavailable & (
        installation_source.installation_count
        < scheduler.installation.config.max_installations
    )
    keys_valid = (
        ~jnp.array_equal(jr.key_data(install_key), jr.key_data(successor_key))
        & ~jnp.array_equal(
            jr.key_data(install_key),
            jr.key_data(one_cold_retired_destination.installation_rng_key),
        )
        & ~jnp.array_equal(
            jr.key_data(successor_key),
            jr.key_data(one_cold_retired_destination.installation_rng_key),
        )
    )
    installation = scheduler.installation.install(
        installation_source,
        candidate_bundle,
        install_key,
        inputs=live_inputs,
    )
    exact_reset = jnp.array_equal(installation.reset_slots, target)
    exact_preserve = jnp.array_equal(installation.preserved_slots, ~target)
    ready = (
        source_valid
        & retired_valid
        & retirement_relation
        & all_installed_mask_exact
        & one_cold_mask_exact
        & retirement_target_exact
        & retirement_authority_bound
        & arm_binding
        & ordinary.transaction_valid
        & ordinary.transaction_applied
        & bundle_binding
        & exact_target
        & live_preserved
        & fresh
        & quiescent
        & scheduler_capacity
        & installer_capacity
        & keys_valid
        & installation.transaction_valid
        & installation.applied
        & exact_reset
        & exact_preserve
        & installation.live_policy_rng_preserved
    )
    scheduler_identity, installer_identity = _external_bundle_adoption_identity(scheduler)
    prepared = CumulantOptionExternalBundleAdoptionPrepared(
        all_installed_source=all_installed_source,
        one_cold_retired_destination=one_cold_retired_destination,
        arm=arm,
        observation=observation,
        live_inputs=live_inputs,
        candidate_bundle=candidate_bundle,
        target_mask=target,
        all_installed_source_slot_mask=all_installed_mask,
        one_cold_retired_destination_slot_mask=retired_slot_mask,
        retirement_target_mask=retirement_target,
        retirement_authority_revision_words=retirement_authority_revision,
        installation_key=install_key,
        successor_scheduler_key=successor_key,
        ordinary_result=ordinary,
        installation_result=installation,
        scheduler_identity_digest=scheduler_identity,
        installer_identity_digest=installer_identity,
        diagnostics=CumulantOptionExternalBundleAdoptionDiagnostics(
            all_installed_source_valid=source_valid,
            one_cold_retired_destination_valid=retired_valid,
            retirement_scheduler_relation_valid=retirement_relation,
            all_installed_source_mask_exact=all_installed_mask_exact,
            one_cold_retired_destination_mask_exact=one_cold_mask_exact,
            retirement_target_exact=retirement_target_exact,
            retirement_authority_revision_bound=retirement_authority_bound,
            arm_binding_valid=arm_binding,
            ordinary_transition_valid=ordinary.transaction_valid,
            bundle_binding_valid=bundle_binding,
            exact_target_semantic_change=exact_target,
            live_slots_semantically_preserved=live_preserved,
            fresh_transition=fresh,
            quiescent=quiescent,
            scheduler_capacity_available=scheduler_capacity,
            installer_capacity_available=installer_capacity,
            caller_keys_valid=keys_valid,
            installation_transaction_valid=installation.transaction_valid,
            installation_applied=installation.applied,
            exact_reset_mask=exact_reset,
            exact_preserve_mask=exact_preserve,
            live_policy_rng_preserved=installation.live_policy_rng_preserved,
            candidate_ready=ready,
        ),
        prepared_checksum=jnp.zeros((2,), dtype=jnp.uint32),
    )
    return dataclasses.replace(
        prepared,
        prepared_checksum=_checksum_arrays(_external_bundle_prepared_payload(prepared)),
    )


def cumulant_option_external_bundle_adoption_authority_receipt(
    scheduler: CumulantOptionScheduler,
    prepared: CumulantOptionExternalBundleAdoptionPrepared,
    installation_authority: CumulantOptionInstallationAuthorityReceipt,
    *,
    adoption_authorized: bool | Array,
) -> CumulantOptionExternalBundleAdoptionAuthorityReceipt:
    """Bind caller-declared authority without authenticating the caller."""

    if type(scheduler) is not CumulantOptionScheduler:
        raise TypeError("scheduler must be an exact CumulantOptionScheduler")
    if type(prepared) is not CumulantOptionExternalBundleAdoptionPrepared:
        raise TypeError("prepared has the wrong exact external-adoption type")
    scheduler._check_authority_contract(installation_authority)
    retired_installation = prepared.one_cold_retired_destination.installation_state
    candidate_semantics = scheduler.installation.semantic_digests_for_bundle(
        prepared.candidate_bundle
    )
    return CumulantOptionExternalBundleAdoptionAuthorityReceipt(
        installation_authority=installation_authority,
        adoption_authorized=jnp.asarray(adoption_authorized, dtype=jnp.bool_),
        scheduler_identity_digest=prepared.scheduler_identity_digest,
        installer_identity_digest=prepared.installer_identity_digest,
        all_installed_source_checksum=prepared.all_installed_source.binding_checksum,
        retired_destination_checksum=(
            prepared.one_cold_retired_destination.binding_checksum
        ),
        retired_installation_revision=retired_installation.revision,
        retired_lifecycle_revision=retired_installation.lifecycle_state.revision,
        retired_audit_revision=(retired_installation.lifecycle_state.audit_state.revision),
        target_mask=prepared.target_mask,
        all_installed_source_slot_mask=prepared.all_installed_source_slot_mask,
        one_cold_retired_destination_slot_mask=(
            prepared.one_cold_retired_destination_slot_mask
        ),
        retirement_target_mask=prepared.retirement_target_mask,
        retirement_authority_revision_words=(
            prepared.retirement_authority_revision_words
        ),
        candidate_binding_digest=prepared.candidate_bundle.binding_digest,
        candidate_semantic_digests=candidate_semantics,
        installation_key_data=jr.key_data(prepared.installation_key),
        successor_scheduler_key_data=jr.key_data(prepared.successor_scheduler_key),
        prepared_checksum=prepared.prepared_checksum,
    )


def _check_external_bundle_authority_contract(
    scheduler: CumulantOptionScheduler,
    receipt: CumulantOptionExternalBundleAdoptionAuthorityReceipt,
) -> None:
    if type(receipt) is not CumulantOptionExternalBundleAdoptionAuthorityReceipt:
        raise TypeError("receipt has the wrong exact external-adoption type")
    scheduler._check_authority_contract(receipt.installation_authority)
    budget = scheduler.discovery.config.option_budget
    contracts = (
        (receipt.adoption_authorized, "adoption_authorized", (), jnp.bool_),
        (receipt.scheduler_identity_digest, "scheduler_identity_digest", (8,), jnp.uint32),
        (receipt.installer_identity_digest, "installer_identity_digest", (8,), jnp.uint32),
        (
            receipt.all_installed_source_checksum,
            "all_installed_source_checksum",
            (2,),
            jnp.uint32,
        ),
        (receipt.retired_destination_checksum, "retired_destination_checksum", (2,), jnp.uint32),
        (receipt.retired_installation_revision, "retired_installation_revision", (), jnp.int32),
        (receipt.retired_lifecycle_revision, "retired_lifecycle_revision", (), jnp.int32),
        (receipt.retired_audit_revision, "retired_audit_revision", (), jnp.int32),
        (receipt.target_mask, "target_mask", (budget,), jnp.bool_),
        (
            receipt.all_installed_source_slot_mask,
            "all_installed_source_slot_mask",
            (budget,),
            jnp.bool_,
        ),
        (
            receipt.one_cold_retired_destination_slot_mask,
            "one_cold_retired_destination_slot_mask",
            (budget,),
            jnp.bool_,
        ),
        (
            receipt.retirement_target_mask,
            "retirement_target_mask",
            (budget,),
            jnp.bool_,
        ),
        (
            receipt.retirement_authority_revision_words,
            "retirement_authority_revision_words",
            (2,),
            jnp.uint32,
        ),
        (receipt.candidate_binding_digest, "candidate_binding_digest", (2,), jnp.uint32),
        (
            receipt.candidate_semantic_digests,
            "candidate_semantic_digests",
            (budget, 8),
            jnp.uint32,
        ),
        (receipt.installation_key_data, "installation_key_data", (2,), jnp.uint32),
        (
            receipt.successor_scheduler_key_data,
            "successor_scheduler_key_data",
            (2,),
            jnp.uint32,
        ),
        (receipt.prepared_checksum, "prepared_checksum", (2,), jnp.uint32),
    )
    for value, name, shape, dtype in contracts:
        _require_array(value, name=f"receipt.{name}", shape=shape, dtype=dtype)


def adopt_cumulant_option_external_bundle(
    scheduler: CumulantOptionScheduler,
    one_cold_retired_destination: CumulantOptionSchedulerState,
    prepared: CumulantOptionExternalBundleAdoptionPrepared,
    authority_receipt: CumulantOptionExternalBundleAdoptionAuthorityReceipt,
) -> CumulantOptionExternalBundleAdoptionResult:
    """Adopt one exact cohort or return the exact one-cold destination."""

    if type(scheduler) is not CumulantOptionScheduler:
        raise TypeError("scheduler must be an exact CumulantOptionScheduler")
    scheduler._check_state_contract(one_cold_retired_destination)
    if type(prepared) is not CumulantOptionExternalBundleAdoptionPrepared:
        raise TypeError("prepared has the wrong exact external-adoption type")
    _check_external_bundle_authority_contract(scheduler, authority_receipt)
    integrity = jnp.array_equal(
        prepared.prepared_checksum,
        _checksum_arrays(_external_bundle_prepared_payload(prepared)),
    )
    recomputed = prepare_cumulant_option_external_bundle_adoption(
        scheduler,
        prepared.all_installed_source,
        prepared.one_cold_retired_destination,
        prepared.arm,
        prepared.observation,
        prepared.live_inputs,
        prepared.candidate_bundle,
        prepared.target_mask,
        prepared.all_installed_source_slot_mask,
        prepared.one_cold_retired_destination_slot_mask,
        prepared.retirement_target_mask,
        prepared.retirement_authority_revision_words,
        prepared.installation_key,
        prepared.successor_scheduler_key,
    )
    derivation = _tree_array_equal(prepared, recomputed)
    destination_valid = scheduler.state_valid(one_cold_retired_destination)
    destination_matches = _tree_array_equal(
        one_cold_retired_destination,
        recomputed.one_cold_retired_destination,
    )
    installation = recomputed.one_cold_retired_destination.installation_state
    lifecycle = installation.lifecycle_state
    audit = lifecycle.audit_state
    nested = authority_receipt.installation_authority
    authority_valid = (
        authority_receipt.adoption_authorized
        & nested.go_no_go_authorized
        & nested.safety_boundary_authorized
        & scheduler._authority_valid(
            recomputed.one_cold_retired_destination,
            nested,
            recomputed.live_inputs,
            recomputed.ordinary_result.state.step_words,
        )
        & jnp.array_equal(
            authority_receipt.scheduler_identity_digest,
            recomputed.scheduler_identity_digest,
        )
        & jnp.array_equal(
            authority_receipt.installer_identity_digest,
            recomputed.installer_identity_digest,
        )
        & jnp.array_equal(
            authority_receipt.all_installed_source_checksum,
            recomputed.all_installed_source.binding_checksum,
        )
        & jnp.array_equal(
            authority_receipt.retired_destination_checksum,
            recomputed.one_cold_retired_destination.binding_checksum,
        )
        & (authority_receipt.retired_installation_revision == installation.revision)
        & (authority_receipt.retired_lifecycle_revision == lifecycle.revision)
        & (authority_receipt.retired_audit_revision == audit.revision)
        & jnp.array_equal(authority_receipt.target_mask, recomputed.target_mask)
        & jnp.array_equal(
            authority_receipt.all_installed_source_slot_mask,
            recomputed.all_installed_source_slot_mask,
        )
        & jnp.array_equal(
            authority_receipt.one_cold_retired_destination_slot_mask,
            recomputed.one_cold_retired_destination_slot_mask,
        )
        & jnp.array_equal(
            authority_receipt.retirement_target_mask,
            recomputed.retirement_target_mask,
        )
        & jnp.array_equal(
            authority_receipt.retirement_authority_revision_words,
            recomputed.retirement_authority_revision_words,
        )
        & jnp.array_equal(
            authority_receipt.candidate_binding_digest,
            recomputed.candidate_bundle.binding_digest,
        )
        & jnp.array_equal(
            authority_receipt.candidate_semantic_digests,
            scheduler.installation.semantic_digests_for_bundle(
                recomputed.candidate_bundle
            ),
        )
        & jnp.array_equal(
            authority_receipt.installation_key_data,
            jr.key_data(recomputed.installation_key),
        )
        & jnp.array_equal(
            authority_receipt.successor_scheduler_key_data,
            jr.key_data(recomputed.successor_scheduler_key),
        )
        & jnp.array_equal(
            authority_receipt.prepared_checksum,
            recomputed.prepared_checksum,
        )
    )
    next_attempt_words, attempt_capacity = _increment_words(
        recomputed.one_cold_retired_destination.install_attempt_words
    )
    next_applied_words, applied_capacity = _increment_words(
        recomputed.one_cold_retired_destination.install_applied_words
    )
    candidate = scheduler._with_checksum(
        dataclasses.replace(
            recomputed.ordinary_result.state,
            installation_state=recomputed.installation_result.state,
            installation_rng_key=recomputed.successor_scheduler_key,
            install_attempt_words=next_attempt_words,
            install_applied_words=next_applied_words,
            last_authority_revision_words=nested.authority_revision_words,
            retry_streak=jnp.asarray(0, dtype=jnp.int32),
            retry_due=jnp.asarray(False, dtype=jnp.bool_),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    candidate_valid = scheduler.state_valid(candidate)
    applied = (
        destination_valid
        & destination_matches
        & integrity
        & derivation
        & recomputed.diagnostics.candidate_ready
        & authority_valid
        & attempt_capacity
        & applied_capacity
        & candidate_valid
    )
    next_state = cast(
        CumulantOptionSchedulerState,
        jax.tree_util.tree_map(
            lambda proposed, destination: jnp.where(applied, proposed, destination),
            candidate,
            one_cold_retired_destination,
        ),
    )
    materialization = cast(
        CumulantOptionMaterialization,
        jax.tree_util.tree_map(
            lambda proposed, fallback: jnp.where(applied, proposed, fallback),
            recomputed.installation_result.materialization,
            recomputed.ordinary_result.materialization,
        ),
    )
    return CumulantOptionExternalBundleAdoptionResult(
        state=next_state,
        ordinary_result=recomputed.ordinary_result,
        materialization=materialization,
        destination_state_valid=destination_valid,
        destination_matches_retired=destination_matches,
        prepared_integrity_valid=integrity,
        preparation_derivation_valid=derivation,
        authority_valid=authority_valid,
        candidate_ready=recomputed.diagnostics.candidate_ready,
        transaction_applied=applied,
        reset_slots=applied & recomputed.installation_result.reset_slots,
        preserved_slots=applied & recomputed.installation_result.preserved_slots,
        installation_key_consumed=jr.wrap_key_data(
            jnp.where(
                applied,
                jr.key_data(recomputed.installation_key),
                jnp.zeros((2,), dtype=jnp.uint32),
            ),
            impl="threefry2x32",
        ),
        caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
    )


__all__ = [
    "CUMULANT_OPTION_SCHEDULER_ASSESSMENT",
    "CUMULANT_OPTION_SCHEDULER_CHECKPOINT_SCHEMA",
    "CUMULANT_OPTION_SCHEDULER_CONFIG_SCHEMA",
    "CUMULANT_OPTION_SCHEDULER_DISPATCH_AUTHORITY",
    "CUMULANT_OPTION_SCHEDULER_ERROR_CAPACITY",
    "CUMULANT_OPTION_SCHEDULER_ERROR_NONE",
    "CUMULANT_OPTION_SCHEDULER_EVIDENCE_AUTHORITY",
    "CUMULANT_OPTION_SCHEDULER_GO_NO_GO_AUTHORITY",
    "CUMULANT_OPTION_SCHEDULER_HOST_ORCHESTRATION",
    "CUMULANT_OPTION_SCHEDULER_OUTPUT_WRITES",
    "CUMULANT_OPTION_SCHEDULER_PROMOTION_AUTHORITY",
    "CUMULANT_OPTION_SCHEDULER_RETIREMENT_AUTHORITY",
    "CUMULANT_OPTION_SCHEDULER_SAFETY_AUTHORITY",
    "CUMULANT_OPTION_SCHEDULER_SCIENTIFIC_PROMOTION_ALLOWED",
    "CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_PREPARED_SCHEMA",
    "CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_RECEIPT_SCHEMA",
    "CUMULANT_OPTION_EXTERNAL_BUNDLE_ADOPTION_RESULT_SCHEMA",
    "CumulantOptionExternalBundleAdoptionAuthorityReceipt",
    "CumulantOptionExternalBundleAdoptionDiagnostics",
    "CumulantOptionExternalBundleAdoptionPrepared",
    "CumulantOptionExternalBundleAdoptionResult",
    "CumulantOptionInstallationAuthorityReceipt",
    "CumulantOptionRetirementHandoff",
    "CumulantOptionScheduleClock",
    "CumulantOptionScheduler",
    "CumulantOptionSchedulerBorrowResult",
    "CumulantOptionSchedulerArm",
    "CumulantOptionSchedulerArmInputs",
    "CumulantOptionSchedulerConfig",
    "CumulantOptionSchedulerMetadataState",
    "CumulantOptionSchedulerResourceBudget",
    "CumulantOptionSchedulerResult",
    "CumulantOptionSchedulerStartResult",
    "CumulantOptionSchedulerState",
    "CumulantOptionSchedulerUpdateResult",
    "CumulantOptionSchedulerObservation",
    "adopt_cumulant_option_external_bundle",
    "cumulant_option_external_bundle_adoption_authority_receipt",
    "prepare_cumulant_option_external_bundle_adoption",
]
