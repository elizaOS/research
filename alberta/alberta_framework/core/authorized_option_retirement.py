# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Authority-gated, noncompensating retirement for installed option slots.

This L0 mechanism consumes a generation-bound
:class:`~alberta_framework.core.cumulant_option_scheduler.CumulantOptionRetirementHandoff`
and a distinct caller-issued authority receipt.  It recomputes a fixed policy
from the live lifecycle facts; the scheduler proposal is never treated as an
authorization or as the policy verdict.

An accepted retirement uses two genuine, quiescent public
``STOMPOptionLifecycle.rebind`` transactions.  The first rebinds exactly the
named slots to collision-checked temporary identities and the second rebinds
them to their bit-exact installer identities using an independent caller key.
Both transactions must report the exact reset/preserve masks and the complete
post-state must satisfy the installer contract before anything commits.  Thus
policy, model, traces, optimizer state, base option heads, and audit statistics
are scrubbed without persisting the temporary identity.  The final semantic
identity is retained only because it is load-bearing for installer validity;
``installed_slot_mask`` is the authoritative retirement status.

Callers composing this controller must use :meth:`start` and :meth:`update`.
Those boundaries pass the persistent cold mask into real STOMP selection,
bootstrap, planning, and audit attribution.  Calling the underlying installer
directly would leave this composition and is not a valid controller execution.

The policy is deliberately noncompensating.  Every named slot needs minimum
support in every configured context and no positive randomized primitive
margin.  It must additionally have at least one fixed concern: poor completion
reliability, high normalized model error, low planning use, or deterministic
redundancy.  Thresholds are static configuration, not calibrated here.

There is no queued proposal and no automatic replacement.  A successful
retirement leaves an explicit cold vacancy for a later, separately authorized
installation transaction.  Authority receipts are integrity bindings, not
authentication.  This module writes no outputs, carries no safety, evidence,
promotion, scientific, or autonomous curation authority, and is
``not_assessed``.
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

from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionInstallation,
    CumulantOptionInstallationState,
    CumulantOptionLiveInputs,
    CumulantOptionMaterialization,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionRetirementHandoff,
)
from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleMaintenanceReport,
)
from alberta_framework.core.stomp_option_lifecycle import (
    STOMPOptionLifecycleStartResult,
    STOMPOptionLifecycleUpdateResult,
)

AUTHORIZED_OPTION_RETIREMENT_CONFIG_SCHEMA = "alberta.authorized-option-retirement.config.v1"
AUTHORIZED_OPTION_RETIREMENT_CHECKPOINT_SCHEMA = (
    "alberta.authorized-option-retirement.state.v1"
)
AUTHORIZED_OPTION_RETIREMENT_ASSESSMENT = "not_assessed"
AUTHORIZED_OPTION_RETIREMENT_OUTPUT_WRITES = False
AUTHORIZED_OPTION_RETIREMENT_EVIDENCE_AUTHORITY = False
AUTHORIZED_OPTION_RETIREMENT_PROMOTION_AUTHORITY = False
AUTHORIZED_OPTION_RETIREMENT_SAFETY_AUTHORITY = False
AUTHORIZED_OPTION_RETIREMENT_GO_NO_GO_AUTHORITY = False
AUTHORIZED_OPTION_RETIREMENT_AUTONOMOUS_CURATION_AUTHORITY = False
AUTHORIZED_OPTION_RETIREMENT_SCIENTIFIC_PROMOTION_ALLOWED = False

AUTHORIZED_OPTION_RETIREMENT_ERROR_NONE = 0
AUTHORIZED_OPTION_RETIREMENT_ERROR_CAPACITY = 1

_DIGEST_WORDS = 8
_COUNTER_WORDS = 2
_INT32_MAX = 2**31 - 1
_UINT64_MAX = 2**64 - 1


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
        data = jr.key_data(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be one typed Threefry JAX key") from exc
    if (
        not jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
        or value.shape != ()
        or implementation != "threefry2x32"
        or data.shape != (_COUNTER_WORDS,)
        or data.dtype != jnp.uint32
    ):
        raise TypeError(f"{name} must be one typed Threefry JAX key")
    return cast(Array, value)


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or not 1 <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be a positive exact Python int32")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be a non-negative exact Python int32")
    return value


def _finite_float(
    value: object,
    *,
    name: str,
    lower: float,
    upper: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite Python float")
    rounded = float(np.float32(value))
    if not math.isfinite(rounded) or value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside its fixed finite range")
    return value


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
            valid = valid & jnp.array_equal(jr.key_data(left_array), jr.key_data(right_array))
        elif left_array.dtype == jnp.float32:
            valid = valid & _float_bits_equal(left_array, right_array)
        else:
            valid = valid & jnp.array_equal(left_array, right_array)
    return valid


def _all_float_leaves_finite(value: object) -> Array:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.floating):
            valid = valid & jnp.all(jnp.isfinite(array))
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


def _installation_payload_arrays(
    state: CumulantOptionInstallationState,
) -> tuple[Array, ...]:
    return tuple(
        cast(Array, leaf)
        for leaf in jax.tree_util.tree_leaves(
            (
                state.lifecycle_state,
                state.installed_bundle,
                state.installed_semantic_digests,
                state.consumer_source_digest,
                state.consumer_representation_digest,
                state.lifecycle_id,
                state.installed,
                state.has_live_observation,
                state.last_semantic_generation,
                state.last_source_digest,
                state.last_canonical_digest,
                state.last_raw_features,
                state.last_raw_available,
                state.last_tail_values,
                state.last_tail_available,
                state.last_materialization_transition_id,
                state.last_materialization_observation_count,
                state.installation_count,
                state.installer_unavailable,
                state.installer_error,
                state.revision,
            )
        )
    )


def _with_installation_checksum(
    state: CumulantOptionInstallationState,
) -> CumulantOptionInstallationState:
    return dataclasses.replace(
        state,
        binding_checksum=_checksum_arrays(_installation_payload_arrays(state)),
    )


def _descriptor_digest(descriptors: Array) -> Array:
    words = jax.lax.bitcast_convert_type(descriptors, jnp.uint32).reshape((-1,))
    rows: list[Array] = []
    for digest_index in range(_DIGEST_WORDS):
        acc = jnp.uint32(0x811C9DC5 ^ ((digest_index + 1) * 0x9E3779B9 & 0xFFFFFFFF))
        for payload_index in range(words.shape[0]):
            acc = (acc ^ words[payload_index]) * jnp.uint32(0x01000193)
            acc = acc + jnp.uint32(
                ((payload_index + 1) * (digest_index + 3) * 0x85EB) & 0xFFFFFFFF
            )
        rows.append(acc)
    return jnp.stack(tuple(rows), dtype=jnp.uint32)


def _temporary_semantics(original: Array, reset_mask: Array, nonce_words: Array) -> Array:
    n_options = original.shape[0]
    slots = jnp.arange(n_options, dtype=jnp.uint32)[:, None]
    columns = jnp.arange(_DIGEST_WORDS, dtype=jnp.uint32)[None, :]
    salt = (
        jnp.uint32(0xA5A5A5A5)
        ^ ((slots + jnp.uint32(1)) * jnp.uint32(0x9E3779B9))
        ^ ((columns + jnp.uint32(3)) * jnp.uint32(0x85EBCA6B))
        ^ nonce_words[0]
        ^ (nonce_words[1] * jnp.uint32(0x27D4EB2D))
    )
    changed = original ^ salt
    return jnp.where(reset_mask[:, None], changed, original).astype(jnp.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedOptionRetirementConfig:
    """Fixed support, policy, and exact retirement-capacity thresholds."""

    minimum_context_support: int = 2
    maximum_completion_reliability: float = 0.5
    minimum_normalized_model_rmse: float = 1.0
    maximum_planning_uses: int = 0
    maximum_positive_randomized_margin: float = 0.0
    redundancy_distance_threshold: float = 0.05
    max_retirements: int = 128

    SCHEMA_VERSION: ClassVar[str] = AUTHORIZED_OPTION_RETIREMENT_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _positive_int(self.minimum_context_support, name="minimum_context_support")
        _finite_float(
            self.maximum_completion_reliability,
            name="maximum_completion_reliability",
            lower=0.0,
            upper=1.0,
        )
        _finite_float(
            self.minimum_normalized_model_rmse,
            name="minimum_normalized_model_rmse",
            lower=0.0,
        )
        _nonnegative_int(self.maximum_planning_uses, name="maximum_planning_uses")
        _finite_float(
            self.maximum_positive_randomized_margin,
            name="maximum_positive_randomized_margin",
            lower=0.0,
        )
        _finite_float(
            self.redundancy_distance_threshold,
            name="redundancy_distance_threshold",
            lower=0.0,
        )
        _nonnegative_int(self.max_retirements, name="max_retirements")

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "minimum_context_support": self.minimum_context_support,
            "maximum_completion_reliability": self.maximum_completion_reliability,
            "minimum_normalized_model_rmse": self.minimum_normalized_model_rmse,
            "maximum_planning_uses": self.maximum_planning_uses,
            "maximum_positive_randomized_margin": self.maximum_positive_randomized_margin,
            "redundancy_distance_threshold": self.redundancy_distance_threshold,
            "max_retirements": self.max_retirements,
            "policy_semantics": "hard_support_and_no_positive_margin_then_any_fixed_concern",
            "assessment": AUTHORIZED_OPTION_RETIREMENT_ASSESSMENT,
            "pending_proposal_slots": 0,
            "automatic_replacement": False,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "safety_authority": False,
            "go_no_go_authority": False,
            "autonomous_curation_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> AuthorizedOptionRetirementConfig:
        if type(value) is not dict:
            raise ValueError("retirement config must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "minimum_context_support",
            "maximum_completion_reliability",
            "minimum_normalized_model_rmse",
            "maximum_planning_uses",
            "maximum_positive_randomized_margin",
            "redundancy_distance_threshold",
            "max_retirements",
            "policy_semantics",
            "assessment",
            "pending_proposal_slots",
            "automatic_replacement",
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "safety_authority",
            "go_no_go_authority",
            "autonomous_curation_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("retirement config keys differ from schema v1")
        if raw.pop("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("retirement config schema_version differs")
        if raw.pop("policy_semantics") != (
            "hard_support_and_no_positive_margin_then_any_fixed_concern"
        ):
            raise ValueError("retirement policy semantics differ")
        if raw.pop("assessment") != AUTHORIZED_OPTION_RETIREMENT_ASSESSMENT:
            raise ValueError("retirement assessment must remain not_assessed")
        if raw.pop("pending_proposal_slots") != 0:
            raise ValueError("retirement cannot persist proposals")
        if raw.pop("automatic_replacement") is not False:
            raise ValueError("retirement cannot enable automatic replacement")
        for name in (
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "safety_authority",
            "go_no_go_authority",
            "autonomous_curation_authority",
            "scientific_promotion_allowed",
        ):
            if raw.pop(name) is not False:
                raise ValueError(f"retirement cannot claim {name}")
        return cls(**cast(dict[str, Any], raw))


@chex.dataclass(frozen=True)
class OptionRetirementAuthorityReceipt:
    """Caller declaration binding authority, slots, owners, facts, and reset keys."""

    retirement_authorized: Bool[Array, ""]
    go_no_go_authorized: Bool[Array, ""]
    safety_boundary_authorized: Bool[Array, ""]
    issuer_digest: UInt[Array, " 8"]
    controller_owner_digest: UInt[Array, " 8"]
    authority_revision_words: UInt[Array, " 2"]
    valid_from_scheduler_step_words: UInt[Array, " 2"]
    valid_through_scheduler_step_words: UInt[Array, " 2"]
    scheduler_step_words: UInt[Array, " 2"]
    descriptor_generation: Int[Array, ""]
    descriptor_digest: UInt[Array, " 8"]
    discovery_source_digest: UInt[Array, " 2"]
    discovery_canonical_digest: UInt[Array, " 32"]
    consumer_source_digest: UInt[Array, " 8"]
    consumer_representation_digest: UInt[Array, " 8"]
    lifecycle_id: UInt[Array, " 2"]
    installation_revision: Int[Array, ""]
    lifecycle_revision: Int[Array, ""]
    audit_revision: Int[Array, ""]
    controller_revision: Int[Array, ""]
    option_semantic_digests: UInt[Array, "n_options 8"]
    option_semantic_generations: Int[Array, " n_options"]
    retirement_slots: Int[Array, " maintenance_budget"]
    retirement_mask: Bool[Array, " maintenance_budget"]
    phase_one_key_data: UInt[Array, " 2"]
    phase_two_key_data: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class AuthorizedOptionRetirementState:
    """Valid installer state plus authoritative live/cold slot identities."""

    installation_state: CumulantOptionInstallationState
    installed_slot_mask: Bool[Array, " n_options"]
    descriptor_generation: Int[Array, ""]
    descriptor_digest: UInt[Array, " 8"]
    expected_authority_issuer_digest: UInt[Array, " 8"]
    controller_owner_digest: UInt[Array, " 8"]
    controller_revision: Int[Array, ""]
    retirement_words: UInt[Array, " 2"]
    last_authority_revision_words: UInt[Array, " 2"]
    last_scheduler_step_words: UInt[Array, " 2"]
    unavailable: Bool[Array, ""]
    error: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class AuthorizedOptionRetirementResult:
    """Atomic retirement verdict and persistent cold-mask result."""

    state: AuthorizedOptionRetirementState
    transaction_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    authority_valid: Bool[Array, ""]
    handoff_valid: Bool[Array, ""]
    policy_eligible: Bool[Array, " n_options"]
    requested_slots: Bool[Array, " n_options"]
    minimum_context_support: Bool[Array, " n_options"]
    no_positive_randomized_margin: Bool[Array, " n_options"]
    poor_completion_reliability: Bool[Array, " n_options"]
    high_model_error: Bool[Array, " n_options"]
    low_planning_use: Bool[Array, " n_options"]
    redundancy_loser: Bool[Array, " n_options"]
    quiescent: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    capacity_exhausted: Bool[Array, ""]
    phase_one_applied: Bool[Array, ""]
    phase_two_applied: Bool[Array, ""]
    reset_slots: Bool[Array, " n_options"]
    extended_action_mask: Bool[Array, " n_total_actions"]
    cold_mask_active: Bool[Array, ""]
    replacement_installed: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedOptionRetirementStartResult:
    """Host control start through the authoritative cold mask."""

    state: AuthorizedOptionRetirementState
    lifecycle_result: STOMPOptionLifecycleStartResult | None
    applied: bool


@chex.dataclass(frozen=True)
class AuthorizedOptionRetirementMaterializationResult:
    """Atomic live materialization that preserves the authoritative cold mask."""

    state: AuthorizedOptionRetirementState
    materialization: CumulantOptionMaterialization
    applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedOptionRetirementUpdateResult:
    """Host control update through the authoritative cold mask."""

    state: AuthorizedOptionRetirementState
    lifecycle_result: STOMPOptionLifecycleUpdateResult | None
    applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class AuthorizedOptionRetirementResourceBudget:
    """Exact allocation/work and explicit lack-of-authority declaration."""

    persistent_state_nbytes: int
    installation_state_nbytes: int
    controller_binding_nbytes: int
    option_slots: int
    maintenance_proposal_slots: int
    pending_proposal_slots: int
    public_rebind_calls_per_applied_retirement: int
    reset_keys_per_applied_retirement: int
    max_retirements: int
    assessment: str
    output_writes: bool
    evidence_authority: bool
    promotion_authority: bool
    safety_authority: bool
    go_no_go_authority: bool
    autonomous_curation_authority: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str


class AuthorizedOptionRetirementController:
    """Strict retirement and cold-mask composition over one installer."""

    def __init__(
        self,
        installation: CumulantOptionInstallation,
        config: AuthorizedOptionRetirementConfig | None = None,
    ) -> None:
        if type(installation) is not CumulantOptionInstallation:
            raise TypeError("installation must be an exact CumulantOptionInstallation")
        self._installation = installation
        self._audit = installation.lifecycle.audit
        self._config = config or AuthorizedOptionRetirementConfig()
        if self._config.minimum_context_support > self._audit.config.max_observations:
            raise ValueError("minimum_context_support exceeds audit observation capacity")

    @property
    def config(self) -> AuthorizedOptionRetirementConfig:
        return self._config

    @property
    def installation(self) -> CumulantOptionInstallation:
        return self._installation

    def to_config(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(
                json.dumps(
                    {
                        "schema_version": AUTHORIZED_OPTION_RETIREMENT_CONFIG_SCHEMA,
                        "retirement": self._config.to_config(),
                        "installation": self._installation.to_config(),
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
        )

    def _payload_arrays(self, state: AuthorizedOptionRetirementState) -> tuple[Array, ...]:
        return tuple(
            cast(Array, leaf)
            for leaf in jax.tree_util.tree_leaves(
                (
                    state.installation_state,
                    state.installed_slot_mask,
                    state.descriptor_generation,
                    state.descriptor_digest,
                    state.expected_authority_issuer_digest,
                    state.controller_owner_digest,
                    state.controller_revision,
                    state.retirement_words,
                    state.last_authority_revision_words,
                    state.last_scheduler_step_words,
                    state.unavailable,
                    state.error,
                )
            )
        )

    def _with_checksum(
        self,
        state: AuthorizedOptionRetirementState,
    ) -> AuthorizedOptionRetirementState:
        return dataclasses.replace(
            state,
            binding_checksum=_checksum_arrays(self._payload_arrays(state)),
        )

    def _check_state_contract(self, state: AuthorizedOptionRetirementState) -> None:
        if type(state) is not AuthorizedOptionRetirementState:
            raise TypeError("state must be an exact AuthorizedOptionRetirementState")
        n = self._audit.config.n_options
        contracts = (
            (state.installed_slot_mask, "installed_slot_mask", (n,), jnp.bool_),
            (state.descriptor_generation, "descriptor_generation", (), jnp.int32),
            (state.descriptor_digest, "descriptor_digest", (8,), jnp.uint32),
            (
                state.expected_authority_issuer_digest,
                "expected_authority_issuer_digest",
                (8,),
                jnp.uint32,
            ),
            (state.controller_owner_digest, "controller_owner_digest", (8,), jnp.uint32),
            (state.controller_revision, "controller_revision", (), jnp.int32),
            (state.retirement_words, "retirement_words", (2,), jnp.uint32),
            (
                state.last_authority_revision_words,
                "last_authority_revision_words",
                (2,),
                jnp.uint32,
            ),
            (state.last_scheduler_step_words, "last_scheduler_step_words", (2,), jnp.uint32),
            (state.unavailable, "unavailable", (), jnp.bool_),
            (state.error, "error", (), jnp.int32),
            (state.binding_checksum, "binding_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def _check_receipt_contract(self, receipt: OptionRetirementAuthorityReceipt) -> None:
        if type(receipt) is not OptionRetirementAuthorityReceipt:
            raise TypeError("authority_receipt must be an exact OptionRetirementAuthorityReceipt")
        n = self._audit.config.n_options
        m = self._audit.config.maintenance_budget
        contracts = (
            (receipt.retirement_authorized, "retirement_authorized", (), jnp.bool_),
            (receipt.go_no_go_authorized, "go_no_go_authorized", (), jnp.bool_),
            (
                receipt.safety_boundary_authorized,
                "safety_boundary_authorized",
                (),
                jnp.bool_,
            ),
            (receipt.issuer_digest, "issuer_digest", (8,), jnp.uint32),
            (receipt.controller_owner_digest, "controller_owner_digest", (8,), jnp.uint32),
            (receipt.authority_revision_words, "authority_revision_words", (2,), jnp.uint32),
            (
                receipt.valid_from_scheduler_step_words,
                "valid_from_scheduler_step_words",
                (2,),
                jnp.uint32,
            ),
            (
                receipt.valid_through_scheduler_step_words,
                "valid_through_scheduler_step_words",
                (2,),
                jnp.uint32,
            ),
            (receipt.scheduler_step_words, "scheduler_step_words", (2,), jnp.uint32),
            (receipt.descriptor_generation, "descriptor_generation", (), jnp.int32),
            (receipt.descriptor_digest, "descriptor_digest", (8,), jnp.uint32),
            (receipt.discovery_source_digest, "discovery_source_digest", (2,), jnp.uint32),
            (
                receipt.discovery_canonical_digest,
                "discovery_canonical_digest",
                (32,),
                jnp.uint8,
            ),
            (receipt.consumer_source_digest, "consumer_source_digest", (8,), jnp.uint32),
            (
                receipt.consumer_representation_digest,
                "consumer_representation_digest",
                (8,),
                jnp.uint32,
            ),
            (receipt.lifecycle_id, "lifecycle_id", (2,), jnp.uint32),
            (receipt.installation_revision, "installation_revision", (), jnp.int32),
            (receipt.lifecycle_revision, "lifecycle_revision", (), jnp.int32),
            (receipt.audit_revision, "audit_revision", (), jnp.int32),
            (receipt.controller_revision, "controller_revision", (), jnp.int32),
            (receipt.option_semantic_digests, "option_semantic_digests", (n, 8), jnp.uint32),
            (
                receipt.option_semantic_generations,
                "option_semantic_generations",
                (n,),
                jnp.int32,
            ),
            (receipt.retirement_slots, "retirement_slots", (m,), jnp.int32),
            (receipt.retirement_mask, "retirement_mask", (m,), jnp.bool_),
            (receipt.phase_one_key_data, "phase_one_key_data", (2,), jnp.uint32),
            (receipt.phase_two_key_data, "phase_two_key_data", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"authority_receipt.{name}", shape=shape, dtype=dtype)

    def init(
        self,
        installation_state: CumulantOptionInstallationState,
        *,
        authority_issuer_digest: Array,
        controller_owner_digest: Array,
    ) -> AuthorizedOptionRetirementState:
        """Bind one already installed, quiescent option cohort."""

        self._installation._check_state_contract(installation_state)
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
        if not bool(jax.device_get(jnp.any(issuer != 0) & jnp.any(owner != 0))):
            raise ValueError("authority issuer and controller owner digests must be nonzero")
        if not bool(jax.device_get(self._installation.state_valid(installation_state))):
            raise ValueError("installation_state must satisfy the complete installer contract")
        if not bool(jax.device_get(installation_state.installed)):
            raise ValueError("retirement requires an installed option cohort")
        descriptor_digest = _descriptor_digest(
            installation_state.installed_bundle.selected_descriptors
        )
        unavailable = self._config.max_retirements == 0
        state = AuthorizedOptionRetirementState(
            installation_state=installation_state,
            installed_slot_mask=jnp.ones((self._audit.config.n_options,), dtype=jnp.bool_),
            descriptor_generation=installation_state.installed_bundle.semantic_generation,
            descriptor_digest=descriptor_digest,
            expected_authority_issuer_digest=issuer,
            controller_owner_digest=owner,
            controller_revision=jnp.asarray(0, dtype=jnp.int32),
            retirement_words=_words(0),
            last_authority_revision_words=_words(0),
            last_scheduler_step_words=_words(0),
            unavailable=jnp.asarray(unavailable, dtype=jnp.bool_),
            error=jnp.asarray(
                AUTHORIZED_OPTION_RETIREMENT_ERROR_CAPACITY
                if unavailable
                else AUTHORIZED_OPTION_RETIREMENT_ERROR_NONE,
                dtype=jnp.int32,
            ),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        state = self._with_checksum(state)
        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("initialized retirement state failed its exact contract")
        return state

    def state_valid(self, state: AuthorizedOptionRetirementState) -> Bool[Array, ""]:
        """Validate installer, descriptor, mask, exact clocks, and checksum."""

        self._check_state_contract(state)
        installation = state.installation_state
        lifecycle = installation.lifecycle_state
        audit = lifecycle.audit_state
        executing = lifecycle.stomp_state.executing_option
        executing_safe = (executing < 0) | state.installed_slot_mask[
            jnp.clip(executing, 0, self._audit.config.n_options - 1)
        ]
        active = audit.active_option
        active_safe = (active < 0) | state.installed_slot_mask[
            jnp.clip(active, 0, self._audit.config.n_options - 1)
        ]
        at_capacity = jnp.array_equal(
            state.retirement_words,
            _words(self._config.max_retirements),
        )
        no_retirement = jnp.array_equal(state.retirement_words, _words(0))
        clock_binding = jnp.where(
            no_retirement,
            jnp.all(state.last_authority_revision_words == 0)
            & jnp.all(state.last_scheduler_step_words == 0),
            jnp.any(state.last_authority_revision_words != 0)
            & jnp.any(state.last_scheduler_step_words != 0),
        )
        expected_error = jnp.where(
            at_capacity,
            jnp.asarray(AUTHORIZED_OPTION_RETIREMENT_ERROR_CAPACITY, dtype=jnp.int32),
            jnp.asarray(AUTHORIZED_OPTION_RETIREMENT_ERROR_NONE, dtype=jnp.int32),
        )
        return (
            self._installation.state_valid(installation)
            & installation.installed
            & (state.descriptor_generation == installation.installed_bundle.semantic_generation)
            & jnp.array_equal(
                state.descriptor_digest,
                _descriptor_digest(installation.installed_bundle.selected_descriptors),
            )
            & jnp.any(state.expected_authority_issuer_digest != 0)
            & jnp.any(state.controller_owner_digest != 0)
            & (state.controller_revision >= 0)
            & _words_less_equal(
                state.retirement_words,
                _words(self._config.max_retirements),
            )
            & (state.retirement_words[0] == 0)
            & (
                state.controller_revision
                >= state.retirement_words[1].astype(jnp.int32)
            )
            & clock_binding
            & (state.unavailable == at_capacity)
            & (state.error == expected_error)
            & executing_safe
            & active_safe
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def extended_action_mask(
        self,
        state: AuthorizedOptionRetirementState,
    ) -> Bool[Array, " n_total_actions"]:
        """Return the authoritative primitive-plus-live-option mask."""

        self._check_state_contract(state)
        primitive = jnp.ones(
            (self._installation.stomp_agent.config.n_primitive_actions,),
            dtype=jnp.bool_,
        )
        return jnp.concatenate((primitive, state.installed_slot_mask), axis=0)

    def _requested_slots(self, receipt: OptionRetirementAuthorityReceipt) -> tuple[Array, Array]:
        n = self._audit.config.n_options
        clipped = jnp.clip(receipt.retirement_slots, 0, n - 1)
        in_range = (receipt.retirement_slots >= 0) & (receipt.retirement_slots < n)
        active_values = jnp.where(receipt.retirement_mask, receipt.retirement_slots, -1)
        equality = active_values[:, None] == active_values[None, :]
        duplicate = jnp.any(
            equality
            & receipt.retirement_mask[:, None]
            & receipt.retirement_mask[None, :]
            & (~jnp.eye(receipt.retirement_mask.shape[0], dtype=jnp.bool_))
        )
        requested = jnp.zeros((n,), dtype=jnp.bool_).at[clipped].max(
            receipt.retirement_mask & in_range
        )
        valid = (
            jnp.any(receipt.retirement_mask)
            & jnp.all((~receipt.retirement_mask) | in_range)
            & jnp.all(receipt.retirement_mask | (receipt.retirement_slots == -1))
            & (~duplicate)
        )
        return requested, valid

    def _policy(
        self,
        state: AuthorizedOptionRetirementState,
        report: OptionLifecycleMaintenanceReport,
    ) -> tuple[Array, ...]:
        audit_state = state.installation_state.lifecycle_state.audit_state
        support = self._config.minimum_context_support
        minimum_context_support = (
            jnp.all(audit_state.initiation_opportunities >= support, axis=1)
            & jnp.all(audit_state.comparison_treatment_counts >= support, axis=1)
            & jnp.all(audit_state.comparison_primitive_counts >= support, axis=1)
            & jnp.all(audit_state.context_signature_counts >= support, axis=1)
            & (audit_state.completion_moment_counts >= support)
            & (audit_state.model_error_counts >= support)
        )
        no_positive_margin = jnp.all(
            report.inverse_propensity_marginal_improvement_by_context
            <= jnp.asarray(
                self._config.maximum_positive_randomized_margin,
                dtype=jnp.float32,
            ),
            axis=1,
        )
        poor_completion = report.completion_reliability <= jnp.asarray(
            self._config.maximum_completion_reliability,
            dtype=jnp.float32,
        )
        aggregate_model_error = jnp.sqrt(
            jnp.mean(report.normalized_model_rmse**2, axis=1)
        )
        high_model_error = aggregate_model_error >= jnp.asarray(
            self._config.minimum_normalized_model_rmse,
            dtype=jnp.float32,
        )
        low_planning = report.planning_use_counts <= self._config.maximum_planning_uses
        installed = state.installed_slot_mask
        pair_ready = (
            report.shared_context_counts >= self._config.minimum_context_support
        )
        pair_close = report.redundancy_distances <= jnp.asarray(
            self._config.redundancy_distance_threshold,
            dtype=jnp.float32,
        )
        pair_live = installed[:, None] & installed[None, :]
        pairs = (
            pair_ready
            & pair_close
            & pair_live
            & (~jnp.eye(self._audit.config.n_options, dtype=jnp.bool_))
        )
        redundancy_loser = jnp.any(jnp.triu(pairs, k=1), axis=0)
        concern = poor_completion | high_model_error | low_planning | redundancy_loser
        eligible = (
            report.state_valid
            & minimum_context_support
            & no_positive_margin
            & concern
            & installed
        )
        return (
            eligible,
            minimum_context_support,
            no_positive_margin,
            poor_completion,
            high_model_error,
            low_planning,
            redundancy_loser,
        )

    def _handoff_valid(
        self,
        state: AuthorizedOptionRetirementState,
        handoff: CumulantOptionRetirementHandoff,
        recomputed: OptionLifecycleMaintenanceReport,
    ) -> Array:
        installation = state.installation_state
        lifecycle = installation.lifecycle_state
        audit = lifecycle.audit_state
        return (
            handoff.available
            & handoff.report.state_valid
            & (~handoff.retirement_authority)
            & (~handoff.go_no_go_authority)
            & (~handoff.safety_authority)
            & _tree_array_equal(handoff.report, recomputed)
            & (handoff.report.state_revision == audit.revision)
            & (handoff.discovery_semantic_generation == state.descriptor_generation)
            & jnp.array_equal(
                handoff.discovery_source_digest,
                installation.installed_bundle.source_digest,
            )
            & jnp.array_equal(
                handoff.discovery_canonical_digest,
                installation.installed_bundle.canonical_digest,
            )
            & jnp.array_equal(
                handoff.last_transition_id,
                installation.last_materialization_transition_id,
            )
            & jnp.array_equal(
                handoff.consumer_source_digest,
                installation.consumer_source_digest,
            )
            & jnp.array_equal(
                handoff.consumer_representation_digest,
                installation.consumer_representation_digest,
            )
            & jnp.array_equal(handoff.lifecycle_id, installation.lifecycle_id)
            & (handoff.installation_revision == installation.revision)
            & (handoff.lifecycle_revision == lifecycle.revision)
            & (handoff.audit_revision == audit.revision)
            & jnp.array_equal(
                handoff.option_semantic_digests,
                installation.installed_semantic_digests,
            )
            & jnp.array_equal(
                handoff.option_semantic_generations,
                audit.semantic_generations,
            )
            & jnp.array_equal(
                handoff.proposed_retirement_slots,
                recomputed.proposed_replacement_slots,
            )
            & jnp.array_equal(
                handoff.proposed_retirement_mask,
                recomputed.proposed_replacement_mask,
            )
            & _all_float_leaves_finite(handoff.report)
        )

    def _authority_valid(
        self,
        state: AuthorizedOptionRetirementState,
        handoff: CumulantOptionRetirementHandoff,
        receipt: OptionRetirementAuthorityReceipt,
        phase_one_key: Array,
        phase_two_key: Array,
    ) -> Array:
        installation = state.installation_state
        lifecycle = installation.lifecycle_state
        audit = lifecycle.audit_state
        requested, slots_valid = self._requested_slots(receipt)
        proposed_slots = jnp.clip(
            handoff.proposed_retirement_slots,
            0,
            self._audit.config.n_options - 1,
        )
        proposed = jnp.zeros_like(requested).at[proposed_slots].max(
            handoff.proposed_retirement_mask
            & (handoff.proposed_retirement_slots >= 0)
            & (handoff.proposed_retirement_slots < self._audit.config.n_options)
        )
        return (
            receipt.retirement_authorized
            & receipt.go_no_go_authorized
            & receipt.safety_boundary_authorized
            & slots_valid
            & jnp.all((~requested) | proposed)
            & jnp.array_equal(receipt.issuer_digest, state.expected_authority_issuer_digest)
            & jnp.any(receipt.issuer_digest != 0)
            & jnp.array_equal(receipt.controller_owner_digest, state.controller_owner_digest)
            & _words_less(
                state.last_authority_revision_words,
                receipt.authority_revision_words,
            )
            & jnp.any(receipt.authority_revision_words != 0)
            & _words_less_equal(
                receipt.valid_from_scheduler_step_words,
                handoff.scheduler_step_words,
            )
            & _words_less_equal(
                handoff.scheduler_step_words,
                receipt.valid_through_scheduler_step_words,
            )
            & jnp.array_equal(receipt.scheduler_step_words, handoff.scheduler_step_words)
            & _words_less(state.last_scheduler_step_words, handoff.scheduler_step_words)
            & (receipt.descriptor_generation == state.descriptor_generation)
            & jnp.array_equal(receipt.descriptor_digest, state.descriptor_digest)
            & jnp.array_equal(
                receipt.discovery_source_digest,
                installation.installed_bundle.source_digest,
            )
            & jnp.array_equal(
                receipt.discovery_canonical_digest,
                installation.installed_bundle.canonical_digest,
            )
            & jnp.array_equal(receipt.consumer_source_digest, installation.consumer_source_digest)
            & jnp.array_equal(
                receipt.consumer_representation_digest,
                installation.consumer_representation_digest,
            )
            & jnp.array_equal(receipt.lifecycle_id, installation.lifecycle_id)
            & (receipt.installation_revision == installation.revision)
            & (receipt.lifecycle_revision == lifecycle.revision)
            & (receipt.audit_revision == audit.revision)
            & (receipt.controller_revision == state.controller_revision)
            & jnp.array_equal(
                receipt.option_semantic_digests,
                installation.installed_semantic_digests,
            )
            & jnp.array_equal(receipt.option_semantic_generations, audit.semantic_generations)
            & jnp.array_equal(receipt.phase_one_key_data, jr.key_data(phase_one_key))
            & jnp.array_equal(receipt.phase_two_key_data, jr.key_data(phase_two_key))
            & (~jnp.array_equal(receipt.phase_one_key_data, receipt.phase_two_key_data))
        )

    def retire(
        self,
        state: AuthorizedOptionRetirementState,
        handoff: CumulantOptionRetirementHandoff,
        authority_receipt: OptionRetirementAuthorityReceipt,
        phase_one_key: Array,
        phase_two_key: Array,
    ) -> AuthorizedOptionRetirementResult:
        """Apply one exact two-rebind retirement or return a whole-state no-op."""

        self._check_state_contract(state)
        if type(handoff) is not CumulantOptionRetirementHandoff:
            raise TypeError("handoff must be an exact CumulantOptionRetirementHandoff")
        self._check_receipt_contract(authority_receipt)
        key_one = _require_threefry_key(phase_one_key, name="phase_one_key")
        key_two = _require_threefry_key(phase_two_key, name="phase_two_key")
        persistent_valid = self.state_valid(state)
        recomputed_report = self._audit.maintenance_report(
            state.installation_state.lifecycle_state.audit_state
        )
        handoff_valid = self._handoff_valid(state, handoff, recomputed_report)
        authority_valid = self._authority_valid(
            state,
            handoff,
            authority_receipt,
            key_one,
            key_two,
        )
        (
            policy_eligible,
            minimum_support,
            no_positive_margin,
            poor_completion,
            high_model_error,
            low_planning,
            redundancy_loser,
        ) = self._policy(state, recomputed_report)
        requested, slots_valid = self._requested_slots(authority_receipt)
        requested_policy_valid = jnp.any(requested) & jnp.all((~requested) | policy_eligible)
        installation = state.installation_state
        lifecycle_state = installation.lifecycle_state
        quiescent = (
            (lifecycle_state.stomp_state.executing_option < 0)
            & (lifecycle_state.audit_state.active_option < 0)
            & (~lifecycle_state.audit_state.trial_active)
        )
        next_words, counter_available = _increment_words(state.retirement_words)
        capacity_available = (
            (~state.unavailable)
            & _words_less(state.retirement_words, _words(self._config.max_retirements))
            & counter_available
            & (state.controller_revision < _INT32_MAX)
            & (installation.revision < _INT32_MAX)
        )
        preconditions = (
            persistent_valid
            & handoff_valid
            & authority_valid
            & slots_valid
            & requested_policy_valid
            & quiescent
            & capacity_available
        )
        # Eager malformed/stale/deferred requests return before either reset
        # boundary.  Traced JIT/scan calls continue through the array-only
        # transaction so the same compiled program covers both verdicts.
        if not isinstance(preconditions, jax.core.Tracer) and not bool(
            jax.device_get(preconditions)
        ):
            return AuthorizedOptionRetirementResult(
                state=state,
                transaction_valid=(
                    persistent_valid
                    & handoff_valid
                    & authority_valid
                    & slots_valid
                ),
                transaction_applied=jnp.asarray(False, dtype=jnp.bool_),
                authority_valid=authority_valid,
                handoff_valid=handoff_valid,
                policy_eligible=policy_eligible,
                requested_slots=requested,
                minimum_context_support=minimum_support,
                no_positive_randomized_margin=no_positive_margin,
                poor_completion_reliability=poor_completion,
                high_model_error=high_model_error,
                low_planning_use=low_planning,
                redundancy_loser=redundancy_loser,
                quiescent=quiescent,
                capacity_available=capacity_available,
                capacity_exhausted=~capacity_available,
                phase_one_applied=jnp.asarray(False, dtype=jnp.bool_),
                phase_two_applied=jnp.asarray(False, dtype=jnp.bool_),
                reset_slots=jnp.zeros_like(requested),
                extended_action_mask=self.extended_action_mask(state),
                cold_mask_active=jnp.any(~state.installed_slot_mask),
                replacement_installed=jnp.asarray(False, dtype=jnp.bool_),
            )

        original_semantics = installation.installed_semantic_digests
        temporary_semantics = _temporary_semantics(
            original_semantics,
            requested,
            authority_receipt.authority_revision_words,
        )
        equality = jnp.all(
            temporary_semantics[:, None, :] == temporary_semantics[None, :, :],
            axis=2,
        )
        temporary_valid = (
            jnp.all(jnp.any(temporary_semantics != 0, axis=1))
            & jnp.all(equality == jnp.eye(self._audit.config.n_options, dtype=jnp.bool_))
            & jnp.all(
                (~requested)
                | jnp.any(temporary_semantics != original_semantics, axis=1)
            )
            & jnp.all(
                (~requested[:, None])
                | jnp.any(
                    temporary_semantics[:, None, :] != original_semantics[None, :, :],
                    axis=2,
                )
            )
        )
        phase_one_wrapper = self._installation.lifecycle.with_external_semantic_digests(
            temporary_semantics
        )
        # Compile both public reset boundaries even for an eager controller
        # call.  STOMP's fresh templates contain ``jr.normal`` draws; using
        # the same compiled boundary prevents backend lowering from producing
        # one-ulp eager/JIT differences in reset parameters and their exact
        # integrity checksums.
        phase_one = jax.jit(phase_one_wrapper.rebind)(
            lifecycle_state,
            key_one,
            source_digest=installation.consumer_source_digest,
            representation_digest=installation.consumer_representation_digest,
        )
        phase_one_exact = (
            phase_one.transaction_valid
            & phase_one.applied
            & (~phase_one.deferred)
            & jnp.array_equal(phase_one.reset_slots, requested)
            & jnp.array_equal(phase_one.preserved_slots, ~requested)
            & jnp.array_equal(
                phase_one.state.audit_state.semantic_digests,
                temporary_semantics,
            )
        )
        phase_two_wrapper = self._installation.lifecycle.with_external_semantic_digests(
            original_semantics
        )
        phase_two = jax.jit(phase_two_wrapper.rebind)(
            phase_one.state,
            key_two,
            source_digest=installation.consumer_source_digest,
            representation_digest=installation.consumer_representation_digest,
        )
        phase_two_exact = (
            phase_two.transaction_valid
            & phase_two.applied
            & (~phase_two.deferred)
            & jnp.array_equal(phase_two.reset_slots, requested)
            & jnp.array_equal(phase_two.preserved_slots, ~requested)
            & jnp.array_equal(
                phase_two.state.audit_state.semantic_digests,
                original_semantics,
            )
            & (~jnp.any(
                phase_two.state.audit_state.semantic_digests
                != installation.installed_semantic_digests
            ))
        )
        rebound_installation = _with_installation_checksum(
            dataclasses.replace(
                installation,
                lifecycle_state=phase_two.state,
                revision=installation.revision + jnp.int32(1),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        post_mask = state.installed_slot_mask & (~requested)
        reaches_capacity = jnp.array_equal(next_words, _words(self._config.max_retirements))
        proposed = AuthorizedOptionRetirementState(
            installation_state=rebound_installation,
            installed_slot_mask=post_mask,
            descriptor_generation=state.descriptor_generation,
            descriptor_digest=state.descriptor_digest,
            expected_authority_issuer_digest=state.expected_authority_issuer_digest,
            controller_owner_digest=state.controller_owner_digest,
            controller_revision=state.controller_revision + jnp.int32(1),
            retirement_words=next_words,
            last_authority_revision_words=authority_receipt.authority_revision_words,
            last_scheduler_step_words=handoff.scheduler_step_words,
            unavailable=reaches_capacity,
            error=jnp.where(
                reaches_capacity,
                jnp.asarray(AUTHORIZED_OPTION_RETIREMENT_ERROR_CAPACITY, dtype=jnp.int32),
                jnp.asarray(AUTHORIZED_OPTION_RETIREMENT_ERROR_NONE, dtype=jnp.int32),
            ),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        proposed_valid = self.state_valid(proposed)
        post_valid = (
            jnp.array_equal(
                rebound_installation.installed_bundle.selected_descriptors,
                installation.installed_bundle.selected_descriptors,
            )
            & (
                rebound_installation.installed_bundle.semantic_generation
                == state.descriptor_generation
            )
            & jnp.array_equal(
                _descriptor_digest(
                    rebound_installation.installed_bundle.selected_descriptors
                ),
                state.descriptor_digest,
            )
            & jnp.array_equal(
                rebound_installation.lifecycle_state.audit_state.semantic_digests,
                installation.installed_semantic_digests,
            )
            & proposed_valid
        )
        applied = (
            preconditions
            & temporary_valid
            & phase_one_exact
            & phase_two_exact
            & post_valid
        )
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        return AuthorizedOptionRetirementResult(
            state=next_state,
            transaction_valid=(
                persistent_valid
                & handoff_valid
                & authority_valid
                & slots_valid
                & temporary_valid
                & phase_one.transaction_valid
                & phase_two.transaction_valid
            ),
            transaction_applied=applied,
            authority_valid=authority_valid,
            handoff_valid=handoff_valid,
            policy_eligible=policy_eligible,
            requested_slots=requested,
            minimum_context_support=minimum_support,
            no_positive_randomized_margin=no_positive_margin,
            poor_completion_reliability=poor_completion,
            high_model_error=high_model_error,
            low_planning_use=low_planning,
            redundancy_loser=redundancy_loser,
            quiescent=quiescent,
            capacity_available=capacity_available,
            capacity_exhausted=~capacity_available,
            phase_one_applied=applied & phase_one_exact,
            phase_two_applied=applied & phase_two_exact,
            reset_slots=applied & requested,
            extended_action_mask=self.extended_action_mask(next_state),
            cold_mask_active=jnp.any(~next_state.installed_slot_mask),
            replacement_installed=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _replace_lifecycle(
        self,
        state: AuthorizedOptionRetirementState,
        lifecycle_state: Any,
    ) -> AuthorizedOptionRetirementState:
        installation = _with_installation_checksum(
            dataclasses.replace(
                state.installation_state,
                lifecycle_state=lifecycle_state,
                revision=state.installation_state.revision + jnp.int32(1),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        return self._with_checksum(
            dataclasses.replace(
                state,
                installation_state=installation,
                controller_revision=state.controller_revision + jnp.int32(1),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )

    def materialize_live(
        self,
        state: AuthorizedOptionRetirementState,
        inputs: CumulantOptionLiveInputs,
    ) -> AuthorizedOptionRetirementMaterializationResult:
        """Advance installed descriptors without changing any retired-slot mask."""

        self._check_state_contract(state)
        raw = self._installation.materialize_live(state.installation_state, inputs)
        revision_available = (
            state.controller_revision < _INT32_MAX
        )
        proposed = self._with_checksum(
            dataclasses.replace(
                state,
                installation_state=raw.state,
                controller_revision=state.controller_revision + jnp.int32(1),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        applied = (
            self.state_valid(state)
            & raw.applied
            & revision_available
            & self.state_valid(proposed)
        )
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        return AuthorizedOptionRetirementMaterializationResult(
            state=next_state,
            materialization=raw.materialization,
            applied=applied,
        )

    def start(
        self,
        state: AuthorizedOptionRetirementState,
        materialization: CumulantOptionMaterialization,
    ) -> AuthorizedOptionRetirementStartResult:
        """Host-only real start; retired options are behavior-ineligible."""

        self._check_state_contract(state)
        valid = bool(
            jax.device_get(
                self.state_valid(state)
                & self._installation._materialization_valid_for_state(
                    state.installation_state,
                    materialization,
                )
                & (state.controller_revision < _INT32_MAX)
                & (state.installation_state.revision < _INT32_MAX)
            )
        )
        if not valid:
            return AuthorizedOptionRetirementStartResult(state, None, False)
        lifecycle = self._installation.lifecycle.with_external_semantic_digests(
            state.installation_state.installed_semantic_digests
        )
        result = lifecycle.start_with_extended_action_mask(
            state.installation_state.lifecycle_state,
            materialization.observation,
            self.extended_action_mask(state),
        )
        proposed = self._replace_lifecycle(state, result.state)
        applied = bool(jax.device_get(result.applied & self.state_valid(proposed)))
        return AuthorizedOptionRetirementStartResult(
            proposed if applied else state,
            result if applied else None,
            applied,
        )

    def update(
        self,
        state: AuthorizedOptionRetirementState,
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
    ) -> AuthorizedOptionRetirementUpdateResult:
        """Host-only real update; selection, bootstrap, planning, and audit share the mask."""

        self._check_state_contract(state)
        candidate = jnp.asarray(idle_candidate_option, dtype=jnp.int32)
        candidate_in_range = (candidate >= 0) & (candidate < self._audit.config.n_options)
        candidate_live = candidate_in_range & state.installed_slot_mask[
            jnp.clip(candidate, 0, self._audit.config.n_options - 1)
        ]
        valid = bool(
            jax.device_get(
                self.state_valid(state)
                & self._installation._materialization_valid_for_state(
                    state.installation_state,
                    materialization,
                )
                & (state.controller_revision < _INT32_MAX)
                & (state.installation_state.revision < _INT32_MAX)
            )
        )
        if not valid:
            return AuthorizedOptionRetirementUpdateResult(state, None, False)
        lifecycle = self._installation.lifecycle.with_external_semantic_digests(
            state.installation_state.installed_semantic_digests
        )
        result = lifecycle.update(
            state.installation_state.lifecycle_state,
            env_reward,
            materialization.observation,
            discount,
            execution_boundary=execution_boundary,
            context=context,
            idle_candidate_option=candidate,
            idle_initiation_eligible=(
                jnp.asarray(idle_initiation_eligible, dtype=jnp.bool_) & candidate_live
            ),
            comparator_randomized=comparator_randomized,
            treatment_propensity=treatment_propensity,
            extended_action_mask=self.extended_action_mask(state),
            enable_planning=enable_planning,
        )
        proposed = self._replace_lifecycle(state, result.state)
        applied = bool(jax.device_get(result.transaction_applied & self.state_valid(proposed)))
        return AuthorizedOptionRetirementUpdateResult(
            proposed if applied else state,
            result if applied else None,
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
            raise ValueError("encoded retirement array differs from schema v1")
        dtype = np.dtype(value["dtype"])
        shape = value["shape"]
        payload = value["bytes_hex"]
        if type(shape) is not list or any(type(cell) is not int or cell < 0 for cell in shape):
            raise ValueError("encoded retirement array shape is invalid")
        if type(payload) is not str:
            raise ValueError("encoded retirement array bytes must be hex")
        try:
            raw = bytes.fromhex(payload)
        except ValueError as exc:
            raise ValueError("encoded retirement array bytes are not hex") from exc
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected:
            raise ValueError("encoded retirement array byte length differs")
        return jnp.asarray(np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy())

    @staticmethod
    def _state_sha256(state: AuthorizedOptionRetirementState) -> str:
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
        state: AuthorizedOptionRetirementState,
    ) -> dict[str, object]:
        """Return a strict bit-preserving checkpoint without external writes."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid retirement state")
        controller_fields = {
            field.name: self._encode_array(cast(Array, getattr(state, field.name)))
            for field in dataclasses.fields(AuthorizedOptionRetirementState)
            if field.name != "installation_state"
        }
        return {
            "schema_version": AUTHORIZED_OPTION_RETIREMENT_CHECKPOINT_SCHEMA,
            "state_type": "AuthorizedOptionRetirementState",
            "config": self.to_config(),
            "installation": self._installation.checkpoint_payload(
                state.installation_state
            ),
            "controller_fields": controller_fields,
            "state_sha256": self._state_sha256(state),
            "assessment": AUTHORIZED_OPTION_RETIREMENT_ASSESSMENT,
            "evidence_authority": False,
            "scientific_promotion_allowed": False,
        }

    def restore_checkpoint(
        self,
        value: Mapping[str, object],
        *,
        expected_authority_issuer_digest: Array,
        expected_controller_owner_digest: Array,
        expected_descriptor_generation: Array,
        expected_descriptor_digest: Array,
    ) -> AuthorizedOptionRetirementState:
        """Restore only an exact config-bound and digest-valid checkpoint."""

        if type(value) is not dict:
            raise ValueError("retirement checkpoint must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "state_type",
            "config",
            "installation",
            "controller_fields",
            "state_sha256",
            "assessment",
            "evidence_authority",
            "scientific_promotion_allowed",
        }
        if set(raw) != expected:
            raise ValueError("retirement checkpoint keys differ from schema v1")
        if raw["schema_version"] != AUTHORIZED_OPTION_RETIREMENT_CHECKPOINT_SCHEMA:
            raise ValueError("retirement checkpoint schema differs")
        if raw["state_type"] != "AuthorizedOptionRetirementState":
            raise ValueError("retirement checkpoint state_type differs")
        if raw["config"] != self.to_config():
            raise ValueError("retirement checkpoint config differs")
        if raw["assessment"] != AUTHORIZED_OPTION_RETIREMENT_ASSESSMENT:
            raise ValueError("retirement checkpoint assessment differs")
        if raw["evidence_authority"] is not False:
            raise ValueError("retirement checkpoint cannot claim evidence authority")
        if raw["scientific_promotion_allowed"] is not False:
            raise ValueError("retirement checkpoint cannot permit promotion")
        fields = raw["controller_fields"]
        if type(fields) is not dict:
            raise ValueError("retirement checkpoint controller_fields must be a dict")
        expected_fields = {
            field.name
            for field in dataclasses.fields(AuthorizedOptionRetirementState)
            if field.name != "installation_state"
        }
        if set(fields) != expected_fields:
            raise ValueError("retirement checkpoint controller fields differ")
        installation_payload = raw["installation"]
        if type(installation_payload) is not dict:
            raise ValueError("retirement installation checkpoint must be an exact dict")
        persisted_installation = installation_payload.get("state")
        if type(persisted_installation) is not CumulantOptionInstallationState:
            raise ValueError("retirement installation checkpoint state type differs")
        installation = self._installation.restore_checkpoint(
            installation_payload,
            expected_consumer_source_digest=(
                persisted_installation.consumer_source_digest
            ),
            expected_consumer_representation_digest=(
                persisted_installation.consumer_representation_digest
            ),
            expected_lifecycle_id=persisted_installation.lifecycle_id,
            expected_installed_bundle=(
                persisted_installation.installed_bundle
                if bool(jax.device_get(persisted_installation.installed))
                else None
            ),
        )
        decoded = {name: self._decode_array(fields[name]) for name in expected_fields}
        restored = AuthorizedOptionRetirementState(
            installation_state=installation,
            **cast(dict[str, Any], decoded),
        )
        issuer = _require_array(
            expected_authority_issuer_digest,
            name="expected_authority_issuer_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        owner = _require_array(
            expected_controller_owner_digest,
            name="expected_controller_owner_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        generation = _require_array(
            expected_descriptor_generation,
            name="expected_descriptor_generation",
            shape=(),
            dtype=jnp.int32,
        )
        descriptor = _require_array(
            expected_descriptor_digest,
            name="expected_descriptor_digest",
            shape=(8,),
            dtype=jnp.uint32,
        )
        external_binding_valid = (
            jnp.array_equal(restored.expected_authority_issuer_digest, issuer)
            & jnp.array_equal(restored.controller_owner_digest, owner)
            & (restored.descriptor_generation == generation)
            & jnp.array_equal(restored.descriptor_digest, descriptor)
        )
        if type(raw["state_sha256"]) is not str or raw["state_sha256"] != self._state_sha256(
            restored
        ):
            raise ValueError("retirement checkpoint state digest differs")
        if not bool(jax.device_get(external_binding_valid & self.state_valid(restored))):
            raise ValueError("restored retirement state is invalid or rebound")
        return restored

    def resource_budget(
        self,
        state: AuthorizedOptionRetirementState,
    ) -> AuthorizedOptionRetirementResourceBudget:
        """Measure exact persistent bytes and declare bounded work/authority."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("resource measurement requires a valid retirement state")

        def nbytes(value: object) -> int:
            total = 0
            for leaf in jax.tree_util.tree_leaves(value):
                array = jnp.asarray(leaf)
                if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
                    array = jr.key_data(array)
                total += int(array.size) * int(array.dtype.itemsize)
            return total

        installation_nbytes = nbytes(state.installation_state)
        persistent_nbytes = nbytes(state)
        return AuthorizedOptionRetirementResourceBudget(
            persistent_state_nbytes=persistent_nbytes,
            installation_state_nbytes=installation_nbytes,
            controller_binding_nbytes=persistent_nbytes - installation_nbytes,
            option_slots=self._audit.config.n_options,
            maintenance_proposal_slots=self._audit.config.maintenance_budget,
            pending_proposal_slots=0,
            public_rebind_calls_per_applied_retirement=2,
            reset_keys_per_applied_retirement=2,
            max_retirements=self._config.max_retirements,
            assessment=AUTHORIZED_OPTION_RETIREMENT_ASSESSMENT,
            output_writes=False,
            evidence_authority=False,
            promotion_authority=False,
            safety_authority=False,
            go_no_go_authority=False,
            autonomous_curation_authority=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=AUTHORIZED_OPTION_RETIREMENT_CHECKPOINT_SCHEMA,
        )


__all__ = [
    "AUTHORIZED_OPTION_RETIREMENT_ASSESSMENT",
    "AUTHORIZED_OPTION_RETIREMENT_AUTONOMOUS_CURATION_AUTHORITY",
    "AUTHORIZED_OPTION_RETIREMENT_CHECKPOINT_SCHEMA",
    "AUTHORIZED_OPTION_RETIREMENT_CONFIG_SCHEMA",
    "AUTHORIZED_OPTION_RETIREMENT_EVIDENCE_AUTHORITY",
    "AUTHORIZED_OPTION_RETIREMENT_GO_NO_GO_AUTHORITY",
    "AUTHORIZED_OPTION_RETIREMENT_OUTPUT_WRITES",
    "AUTHORIZED_OPTION_RETIREMENT_PROMOTION_AUTHORITY",
    "AUTHORIZED_OPTION_RETIREMENT_SAFETY_AUTHORITY",
    "AUTHORIZED_OPTION_RETIREMENT_SCIENTIFIC_PROMOTION_ALLOWED",
    "AuthorizedOptionRetirementConfig",
    "AuthorizedOptionRetirementController",
    "AuthorizedOptionRetirementMaterializationResult",
    "AuthorizedOptionRetirementResourceBudget",
    "AuthorizedOptionRetirementResult",
    "AuthorizedOptionRetirementStartResult",
    "AuthorizedOptionRetirementState",
    "AuthorizedOptionRetirementUpdateResult",
    "OptionRetirementAuthorityReceipt",
]
