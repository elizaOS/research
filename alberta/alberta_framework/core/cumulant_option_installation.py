# mypy: disable-error-code="attr-defined,call-arg,arg-type,type-var"
"""Strict opt-in discovery-to-STOMP option installation.

This module closes one mechanism edge between
:class:`CumulantSubtaskDiscovery` and :class:`STOMPOptionLifecycle`.  A
complete discovered proposal may bind a fixed bank of preallocated STOMP
option slots only at the lifecycle wrapper's quiescent rebind boundary.  The
proposal's complete payload and its generation, source, canonical-universe,
transition, and discovery-revision provenance remain in persistent state.

Tail indices are storage locations, not semantic identities. Each lifecycle
semantic digest binds the external four-cell cumulant descriptor and fixed
live-materialization and option-termination procedure. Proposal-generation
and source-snapshot provenance remain persisted but are deliberately not part
of option identity. Consequently an
unchanged semantic slot is preserved by ``STOMPOptionLifecycle.rebind`` while
a changed slot is reset from its caller-keyed fresh template, including its
policy, model, traces, optimizer state, and base option head.

Installed descriptors are reevaluated on every live observation.  A selected
cumulant is already descriptor-polarized by the discovery contract.  This
adapter applies descriptor polarity exactly once while materializing the live
source value; the installed :class:`SubtaskSpec` therefore always uses
``pseudo_reward_scale=+1`` and terminates when the already-polarized tail value
reaches the configured positive threshold.  Feature-change descriptors use
the exact previous accepted raw observation retained in bounded state.

An empty STOMP template may reserve a fixed observation suffix by declaring a
width greater than the discovery raw-feature width.  Installation keeps its
option-cumulant cells immediately after the raw prefix and fills every reserved
cell with zero in standalone materializations.  A later external owner may use
those cells under its own exact binding, but this installer neither supplies
nor claims their semantics; a nonzero reserved cell in an installation-produced
token is rejected.

Before installation, a zero-tail observation can drive primitives only; an
explicit eligibility mask keeps every cold option head behavior-ineligible.
Invalid, stale, partial, misattributed, unavailable, or nonfinite
materializations are exact state no-ops and cannot reach STOMP. Semantic
cutover is deferred while
an option, lifecycle audit, or comparator trial is active.  Installer capacity
exhaustion freezes semantic replacement only; an already valid installed STOMP
controller can continue receiving live materializations and updates.
Persistent composed-state corruption blocks both installation and control.
Installation and materialization return array-only pytrees and are JIT-safe.
The optional-result ``start``/``update`` control boundary is intentionally
host-only: it performs eager fail-closed gating before returning either a real
lifecycle result or ``None``.

The composition is caller-invoked and owns no autonomous discovery, evidence,
promotion, benefit, output, or scientific authority.  Its assessment is
always ``not_assessed``.  Checkpoint hashes detect accidental or transported
payload corruption; they do not authenticate an external caller.
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

from alberta_framework.core.cumulant_subtask_discovery import (
    CUMULANT_SOURCE_CONTROLLABLE_EVENT,
    CUMULANT_SOURCE_FEATURE_CHANGE,
    CUMULANT_SOURCE_PREDICTION_BOTTLENECK,
    CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
    CumulantSubtaskDiscovery,
    CumulantSubtaskProposalBundle,
)
from alberta_framework.core.option_lifecycle_audit import OptionLifecycleAudit
from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec
from alberta_framework.core.stomp_option_lifecycle import (
    STOMPOptionLifecycle,
    STOMPOptionLifecycleBorrowResult,
    STOMPOptionLifecycleConfig,
    STOMPOptionLifecycleMetadataState,
    STOMPOptionLifecycleRebindResult,
    STOMPOptionLifecycleStartResult,
    STOMPOptionLifecycleState,
    STOMPOptionLifecycleUpdateResult,
)

CUMULANT_OPTION_INSTALLATION_CONFIG_SCHEMA = "alberta.cumulant-option-installation.config.v1"
CUMULANT_OPTION_INSTALLATION_CHECKPOINT_SCHEMA = "alberta.cumulant-option-installation.state.v1"
CUMULANT_OPTION_INSTALLATION_ASSESSMENT = "not_assessed"
CUMULANT_OPTION_INSTALLATION_THRESHOLD_SEMANTICS = (
    "selected cumulants are descriptor-polarized exactly once during live "
    "materialization; SubtaskSpec uses scale +1 and terminates at tail >= threshold"
)
CUMULANT_OPTION_INSTALLATION_OUTPUT_WRITES = False
CUMULANT_OPTION_INSTALLATION_EVIDENCE_AUTHORITY = False
CUMULANT_OPTION_INSTALLATION_PROMOTION_AUTHORITY = False
CUMULANT_OPTION_INSTALLATION_BENEFIT_CLAIM = False
CUMULANT_OPTION_INSTALLATION_AUTONOMOUS_DISCOVERY_CLAIM = False
CUMULANT_OPTION_INSTALLATION_SCIENTIFIC_PROMOTION_ALLOWED = False
CUMULANT_OPTION_INSTALLATION_CONTROL_HOST_ONLY = True

CUMULANT_OPTION_INSTALLER_ERROR_NONE = 0
CUMULANT_OPTION_INSTALLER_ERROR_CAPACITY = 1

_DIGEST_WORDS = 8
_LIFECYCLE_WORDS = 2
_DESCRIPTOR_WIDTH = 4
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_LIVE_MATERIALIZATION_PROCEDURE_WORD = 0xC011A17E
_POSITIVE_THRESHOLD_PROCEDURE_WORD = 0x7E120001


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


def _strict_int32(value: int | Array, *, name: str) -> Array:
    if type(value) is int:
        if not -(2**31) <= value <= _INT32_MAX:
            raise ValueError(f"{name} must be signed-int32 compatible")
        return jnp.asarray(value, dtype=jnp.int32)
    return _require_array(value, name=name, shape=(), dtype=jnp.int32)


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


def _saturating_increment(value: Array) -> Array:
    return jnp.where(
        value < jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        value + jnp.asarray(1, dtype=jnp.int32),
        value,
    )


def _uint64_words_less(left: Array, right: Array) -> Array:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] < right[1]))


def _float_bits_equal(left: Array, right: Array) -> Array:
    return jnp.array_equal(
        jax.lax.bitcast_convert_type(left, jnp.int32),
        jax.lax.bitcast_convert_type(right, jnp.int32),
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
        if left_array.dtype == jnp.float32:
            valid = valid & _float_bits_equal(left_array, right_array)
        else:
            valid = valid & jnp.array_equal(left_array, right_array)
    return valid


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.size * array.dtype.itemsize)
    return total


def _cryptographic_state_digest(state: object) -> Array:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(state):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        host = np.asarray(jax.device_get(array))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _canonical_json(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantOptionInstallationConfig:
    """Opt-in threshold, duration, and finite installation capacity."""

    polarized_cumulant_threshold: float = 0.5
    max_option_steps: int = 8
    max_installations: int = 128

    SCHEMA_VERSION: ClassVar[str] = CUMULANT_OPTION_INSTALLATION_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.polarized_cumulant_threshold) is not float
            or not math.isfinite(self.polarized_cumulant_threshold)
            or self.polarized_cumulant_threshold <= 0.0
            or not math.isfinite(float(np.float32(self.polarized_cumulant_threshold)))
        ):
            raise ValueError("polarized_cumulant_threshold must be a positive finite float32 value")
        if type(self.max_option_steps) is not int or not 1 <= self.max_option_steps <= _INT32_MAX:
            raise ValueError("max_option_steps must be a positive signed-int32 integer")
        if type(self.max_installations) is not int or not 0 <= self.max_installations <= _INT32_MAX:
            raise ValueError("max_installations must be a non-negative signed-int32 integer")

    def to_config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "polarized_cumulant_threshold": self.polarized_cumulant_threshold,
            "max_option_steps": self.max_option_steps,
            "max_installations": self.max_installations,
            "threshold_semantics": CUMULANT_OPTION_INSTALLATION_THRESHOLD_SEMANTICS,
            "assessment": CUMULANT_OPTION_INSTALLATION_ASSESSMENT,
            "output_writes": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "benefit_claim": False,
            "autonomous_discovery_claim": False,
            "scientific_promotion_allowed": False,
            "control_host_only": True,
        }

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> CumulantOptionInstallationConfig:
        if type(value) is not dict:
            raise ValueError("cumulant option installation config must be an exact dict")
        raw = dict(value)
        expected = {
            "schema_version",
            "polarized_cumulant_threshold",
            "max_option_steps",
            "max_installations",
            "threshold_semantics",
            "assessment",
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "benefit_claim",
            "autonomous_discovery_claim",
            "scientific_promotion_allowed",
            "control_host_only",
        }
        if set(raw) != expected:
            raise ValueError("cumulant option installation config keys differ from schema v1")
        if raw.pop("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("cumulant option installation config schema differs")
        if raw.pop("threshold_semantics") != CUMULANT_OPTION_INSTALLATION_THRESHOLD_SEMANTICS:
            raise ValueError("cumulant option installation threshold semantics differ")
        if raw.pop("assessment") != CUMULANT_OPTION_INSTALLATION_ASSESSMENT:
            raise ValueError("cumulant option installation assessment must remain not_assessed")
        for name in (
            "output_writes",
            "evidence_authority",
            "promotion_authority",
            "benefit_claim",
            "autonomous_discovery_claim",
            "scientific_promotion_allowed",
        ):
            if raw.pop(name) is not False:
                raise ValueError(f"cumulant option installation cannot claim {name}")
        if raw.pop("control_host_only") is not True:
            raise ValueError("cumulant option installation control must remain host-only")
        return cls(**cast(dict[str, Any], raw))


@chex.dataclass(frozen=True)
class CumulantOptionLiveInputs:
    """One exact live source snapshot for installed descriptor evaluation."""

    raw_features: Float[Array, " raw_feature_dim"]
    raw_available: Bool[Array, " raw_feature_dim"]
    controllable_events: Float[Array, " controllable_event_dim"]
    controllable_events_available: Bool[Array, " controllable_event_dim"]
    transition_atoms: Float[Array, " transition_atom_dim"]
    transition_atoms_available: Bool[Array, " transition_atom_dim"]
    bottleneck_values: Float[Array, " prediction_bottleneck_dim"]
    bottleneck_available: Bool[Array, " prediction_bottleneck_dim"]
    semantic_generation: Int[Array, ""]
    source_digest: UInt[Array, " 2"]
    canonical_digest: UInt[Array, " 32"]
    transition_id: UInt[Array, " 2"]
    state_observation_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class CumulantOptionMaterialization:
    """State-bound live STOMP observation; unavailable tokens cannot execute."""

    observation: Float[Array, " observation_dim"]
    tail_values: Float[Array, " option_budget"]
    tail_available: Bool[Array, " option_budget"]
    transition_id: UInt[Array, " 2"]
    state_observation_count: Int[Array, ""]
    composition_revision: Int[Array, ""]
    composition_checksum: UInt[Array, " 2"]
    available: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CumulantOptionInstallationState:
    """Installed bundle, causal live-feature state, and real lifecycle state."""

    lifecycle_state: STOMPOptionLifecycleState
    installed_bundle: CumulantSubtaskProposalBundle
    installed_semantic_digests: UInt[Array, "option_budget 8"]
    consumer_source_digest: UInt[Array, " 8"]
    consumer_representation_digest: UInt[Array, " 8"]
    lifecycle_id: UInt[Array, " 2"]
    installed: Bool[Array, ""]
    has_live_observation: Bool[Array, ""]
    last_semantic_generation: Int[Array, ""]
    last_source_digest: UInt[Array, " 2"]
    last_canonical_digest: UInt[Array, " 32"]
    last_raw_features: Float[Array, " raw_feature_dim"]
    last_raw_available: Bool[Array, " raw_feature_dim"]
    last_tail_values: Float[Array, " option_budget"]
    last_tail_available: Bool[Array, " option_budget"]
    last_materialization_transition_id: UInt[Array, " 2"]
    last_materialization_observation_count: Int[Array, ""]
    installation_count: Int[Array, ""]
    installer_unavailable: Bool[Array, ""]
    installer_error: Int[Array, ""]
    revision: Int[Array, ""]
    binding_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CumulantOptionInstallationMetadataState:
    """Installer metadata borrowing the sole externally owned STOMP state."""

    lifecycle_metadata: STOMPOptionLifecycleMetadataState
    installed_bundle: CumulantSubtaskProposalBundle
    installed_semantic_digests: UInt[Array, "option_budget 8"]
    consumer_source_digest: UInt[Array, " 8"]
    consumer_representation_digest: UInt[Array, " 8"]
    lifecycle_id: UInt[Array, " 2"]
    installed: Bool[Array, ""]
    has_live_observation: Bool[Array, ""]
    last_semantic_generation: Int[Array, ""]
    last_source_digest: UInt[Array, " 2"]
    last_canonical_digest: UInt[Array, " 32"]
    last_raw_features: Float[Array, " raw_feature_dim"]
    last_raw_available: Bool[Array, " raw_feature_dim"]
    last_tail_values: Float[Array, " option_budget"]
    last_tail_available: Bool[Array, " option_budget"]
    last_materialization_transition_id: UInt[Array, " 2"]
    last_materialization_observation_count: Int[Array, ""]
    installation_count: Int[Array, ""]
    installer_unavailable: Bool[Array, ""]
    installer_error: Int[Array, ""]
    revision: Int[Array, ""]
    source_binding_checksum: UInt[Array, " 2"]
    metadata_checksum: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CumulantOptionInstallationBorrowResult:
    """Fail-closed transient reconstruction around a borrowed STOMP owner."""

    state: CumulantOptionInstallationState
    lifecycle: STOMPOptionLifecycleBorrowResult
    metadata_valid: Bool[Array, ""]
    binding_matches: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CumulantOptionInstallationResult:
    """Atomic install/rebind result and first bound materialization."""

    state: CumulantOptionInstallationState
    materialization: CumulantOptionMaterialization
    transaction_valid: Bool[Array, ""]
    bundle_live_binding_valid: Bool[Array, ""]
    materialization_valid: Bool[Array, ""]
    semantics_changed: Bool[Array, ""]
    quiescent: Bool[Array, ""]
    capacity_available: Bool[Array, ""]
    applied: Bool[Array, ""]
    provenance_refreshed: Bool[Array, ""]
    deferred: Bool[Array, ""]
    preserved_slots: Bool[Array, " option_budget"]
    reset_slots: Bool[Array, " option_budget"]
    installer_unavailable: Bool[Array, ""]
    live_policy_rng_preserved: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CumulantOptionMaterializationResult:
    """Atomic causal live-feature advance and state-bound observation."""

    state: CumulantOptionInstallationState
    materialization: CumulantOptionMaterialization
    state_valid: Bool[Array, ""]
    binding_valid: Bool[Array, ""]
    transition_is_newer: Bool[Array, ""]
    inputs_valid: Bool[Array, ""]
    applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantOptionStartResult:
    """Optional real lifecycle start; invalid materialization is a no-op."""

    state: CumulantOptionInstallationState
    lifecycle_result: STOMPOptionLifecycleStartResult | None
    applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantOptionUpdateResult:
    """Optional real lifecycle update; invalid materialization is a no-op."""

    state: CumulantOptionInstallationState
    lifecycle_result: STOMPOptionLifecycleUpdateResult | None
    applied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class CumulantOptionInstallationResourceBudget:
    """Exact persistent bytes and bounded installation/materialization work."""

    persistent_state_nbytes: int
    lifecycle_state_nbytes: int
    installation_binding_nbytes: int
    raw_feature_dim: int
    option_budget: int
    observation_dim: int
    live_source_value_cells_per_materialization: int
    descriptor_cells_per_materialization: int
    availability_cells_per_materialization: int
    lifecycle_rebind_calls_per_changed_installation: int
    fresh_template_initializations_per_changed_installation: int
    live_policy_rng_draws_per_installation: int
    live_policy_rng_replaced_by_installer: bool
    fresh_template_key_supplied_by_caller: bool
    installer_capacity_can_block_valid_stomp_control: bool
    max_installations: int
    assessment: str
    output_writes: bool
    evidence_authority: bool
    promotion_authority: bool
    benefit_claim: bool
    autonomous_discovery_claim: bool
    scientific_promotion_allowed: bool
    checkpoint_schema: str
    control_host_only: bool


class CumulantOptionInstallation:
    """Explicit persistent proposal installation around one STOMP lifecycle."""

    def __init__(
        self,
        discovery: CumulantSubtaskDiscovery,
        stomp_template_config: STOMPConfig,
        audit: OptionLifecycleAudit,
        lifecycle_config: STOMPOptionLifecycleConfig | None = None,
        config: CumulantOptionInstallationConfig | None = None,
    ) -> None:
        if type(discovery) is not CumulantSubtaskDiscovery:
            raise TypeError("discovery must be an exact CumulantSubtaskDiscovery")
        if type(stomp_template_config) is not STOMPConfig:
            raise TypeError("stomp_template_config must be an exact STOMPConfig")
        if type(audit) is not OptionLifecycleAudit:
            raise TypeError("audit must be an exact OptionLifecycleAudit")
        if stomp_template_config.subtask_specs:
            raise ValueError("stomp_template_config must not preinstall subtask specs")
        dcfg = discovery.config
        if stomp_template_config.observation_dim < dcfg.raw_feature_dim:
            raise ValueError(
                "template observation_dim must be at least discovery raw_feature_dim"
            )
        if stomp_template_config.n_primitive_actions != dcfg.n_actions:
            raise ValueError("template primitive actions must equal discovery action count")
        self._discovery = discovery
        self._config = config or CumulantOptionInstallationConfig()
        self._lifecycle_config = lifecycle_config or STOMPOptionLifecycleConfig()
        # The template width is the sole opt-in declaration of reserved suffix
        # capacity.  It changes no config schema and Q=0 retains the historical
        # raw-plus-option-tail layout exactly.
        self._reserved_observation_suffix = (
            stomp_template_config.observation_dim - dcfg.raw_feature_dim
        )
        specs = tuple(
            SubtaskSpec(
                feature_index=dcfg.raw_feature_dim + slot,
                threshold=self._config.polarized_cumulant_threshold,
                pseudo_reward_scale=1.0,
                max_option_steps=self._config.max_option_steps,
            )
            for slot in range(dcfg.option_budget)
        )
        self._stomp_config = dataclasses.replace(
            stomp_template_config,
            subtask_specs=specs,
            observation_dim=(
                stomp_template_config.observation_dim + dcfg.option_budget
            ),
        )
        self._agent = STOMPAgent(self._stomp_config)
        self._audit = audit
        self._base_lifecycle = STOMPOptionLifecycle(
            self._agent,
            self._audit,
            self._lifecycle_config,
        )
        if audit.config.n_options != dcfg.option_budget:
            raise ValueError("audit option count must equal discovery option budget")
        if audit.config.outcome_dim != self._stomp_config.observation_dim:
            raise ValueError("audit outcome_dim must equal materialized observation_dim")

    @property
    def config(self) -> CumulantOptionInstallationConfig:
        return self._config

    @property
    def discovery(self) -> CumulantSubtaskDiscovery:
        return self._discovery

    @property
    def stomp_agent(self) -> STOMPAgent:
        return self._agent

    @property
    def lifecycle(self) -> STOMPOptionLifecycle:
        return self._base_lifecycle

    @property
    def subtask_specs(self) -> tuple[SubtaskSpec, ...]:
        return self._stomp_config.subtask_specs

    def to_config(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            _canonical_json(
                {
                    "schema_version": CUMULANT_OPTION_INSTALLATION_CONFIG_SCHEMA,
                    "installation": self._config.to_config(),
                    "discovery": self._discovery.to_config(),
                    "stomp": self._agent.to_config(),
                    "audit": self._audit.to_config(),
                    "lifecycle": self._lifecycle_config.to_config(),
                }
            ),
        )

    def _lifecycle_with_semantics(self, semantic_digests: Array) -> STOMPOptionLifecycle:
        return self._base_lifecycle.with_external_semantic_digests(
            semantic_digests,
        )

    def _semantic_digests(self, bundle: CumulantSubtaskProposalBundle) -> Array:
        """Bind descriptor and procedure semantics, excluding proposal provenance."""

        threshold_word = jax.lax.bitcast_convert_type(
            jnp.asarray(self._config.polarized_cumulant_threshold, dtype=jnp.float32),
            jnp.uint32,
        ).reshape((1,))
        duration_word = jnp.asarray((self._config.max_option_steps,), dtype=jnp.uint32)
        rows: list[Array] = []
        for slot in range(self._discovery.config.option_budget):
            descriptor_words = jax.lax.bitcast_convert_type(
                bundle.selected_descriptors[slot], jnp.uint32
            )
            payload = jnp.concatenate(
                (
                    descriptor_words,
                    threshold_word,
                    duration_word,
                    jnp.asarray(
                        (
                            1,
                            _LIVE_MATERIALIZATION_PROCEDURE_WORD,
                            _POSITIVE_THRESHOLD_PROCEDURE_WORD,
                        ),
                        dtype=jnp.uint32,
                    ),
                )
            )
            words: list[Array] = []
            for word_index in range(_DIGEST_WORDS):
                acc = jnp.uint32(0x811C9DC5) ^ jnp.uint32(
                    (word_index + 1) * 0x1B873593 & _UINT32_MAX
                )
                for payload_index in range(payload.shape[0]):
                    acc = (acc ^ payload[payload_index]) * jnp.uint32(0x01000193)
                    acc = acc + jnp.uint32(
                        ((payload_index + 1) * (word_index + 3) * 0x9E37) & _UINT32_MAX
                    )
                words.append(acc)
            rows.append(jnp.stack(tuple(words), axis=0))
        return jnp.stack(tuple(rows), axis=0).astype(jnp.uint32)

    def semantic_digests_for_bundle(
        self,
        bundle: CumulantSubtaskProposalBundle,
    ) -> UInt[Array, "option_budget 8"]:
        """Preview exact slot semantics without installing or acquiring authority.

        The returned identities are the same descriptor-and-procedure digests
        consumed by :meth:`install`.  Proposal provenance is deliberately not
        part of an option's semantic identity, and this pure preview performs
        no lifecycle mutation, RNG draw, curation, or go/no-go decision.
        """

        self._discovery.check_proposal_bundle_contract(bundle)
        return self._semantic_digests(bundle)

    def _payload_arrays(self, state: CumulantOptionInstallationState) -> tuple[Array, ...]:
        leaves = tuple(
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
        return leaves

    def _with_checksum(
        self,
        state: CumulantOptionInstallationState,
    ) -> CumulantOptionInstallationState:
        return dataclasses.replace(
            state,
            binding_checksum=_checksum_arrays(self._payload_arrays(state)),
        )

    def _metadata_payload_arrays(
        self,
        state: CumulantOptionInstallationMetadataState,
    ) -> tuple[Array, ...]:
        values = tuple(
            getattr(state, field.name)
            for field in dataclasses.fields(CumulantOptionInstallationMetadataState)
            if field.name != "metadata_checksum"
        )
        return tuple(
            cast(Array, leaf) for leaf in jax.tree_util.tree_leaves(values)
        )

    def _with_metadata_checksum(
        self,
        state: CumulantOptionInstallationMetadataState,
    ) -> CumulantOptionInstallationMetadataState:
        return dataclasses.replace(
            state,
            metadata_checksum=_checksum_arrays(self._metadata_payload_arrays(state)),
        )

    def _check_metadata_contract(
        self,
        state: CumulantOptionInstallationMetadataState,
    ) -> None:
        if type(state) is not CumulantOptionInstallationMetadataState:
            raise TypeError(
                "state must be an exact CumulantOptionInstallationMetadataState"
            )
        cfg = self._discovery.config
        self._discovery.check_proposal_bundle_contract(state.installed_bundle)
        contracts = (
            (
                state.installed_semantic_digests,
                "installed_semantic_digests",
                (cfg.option_budget, _DIGEST_WORDS),
                jnp.uint32,
            ),
            (state.consumer_source_digest, "consumer_source_digest", (8,), jnp.uint32),
            (
                state.consumer_representation_digest,
                "consumer_representation_digest",
                (8,),
                jnp.uint32,
            ),
            (state.lifecycle_id, "lifecycle_id", (2,), jnp.uint32),
            (state.installed, "installed", (), jnp.bool_),
            (state.has_live_observation, "has_live_observation", (), jnp.bool_),
            (
                state.last_semantic_generation,
                "last_semantic_generation",
                (),
                jnp.int32,
            ),
            (state.last_source_digest, "last_source_digest", (2,), jnp.uint32),
            (
                state.last_canonical_digest,
                "last_canonical_digest",
                (32,),
                jnp.uint8,
            ),
            (
                state.last_raw_features,
                "last_raw_features",
                (cfg.raw_feature_dim,),
                jnp.float32,
            ),
            (
                state.last_raw_available,
                "last_raw_available",
                (cfg.raw_feature_dim,),
                jnp.bool_,
            ),
            (
                state.last_tail_values,
                "last_tail_values",
                (cfg.option_budget,),
                jnp.float32,
            ),
            (
                state.last_tail_available,
                "last_tail_available",
                (cfg.option_budget,),
                jnp.bool_,
            ),
            (
                state.last_materialization_transition_id,
                "last_materialization_transition_id",
                (2,),
                jnp.uint32,
            ),
            (
                state.last_materialization_observation_count,
                "last_materialization_observation_count",
                (),
                jnp.int32,
            ),
            (state.installation_count, "installation_count", (), jnp.int32),
            (state.installer_unavailable, "installer_unavailable", (), jnp.bool_),
            (state.installer_error, "installer_error", (), jnp.int32),
            (state.revision, "revision", (), jnp.int32),
            (state.source_binding_checksum, "source_binding_checksum", (2,), jnp.uint32),
            (state.metadata_checksum, "metadata_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def detach_borrowed_stomp(
        self,
        state: CumulantOptionInstallationState,
    ) -> CumulantOptionInstallationMetadataState:
        """Detach installer metadata without retaining a STOMP state."""

        self._check_state_contract(state)
        lifecycle = self._lifecycle_with_semantics(state.installed_semantic_digests)
        lifecycle_metadata = lifecycle.detach_borrowed_stomp(state.lifecycle_state)
        values = {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(CumulantOptionInstallationState)
            if field.name not in {"lifecycle_state", "binding_checksum"}
        }
        metadata = CumulantOptionInstallationMetadataState(
            lifecycle_metadata=lifecycle_metadata,
            **values,
            source_binding_checksum=state.binding_checksum,
            metadata_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_metadata_checksum(metadata)

    def metadata_state_valid(
        self,
        state: CumulantOptionInstallationMetadataState,
    ) -> Bool[Array, ""]:
        """Validate detached installer state against its lifecycle metadata."""

        self._check_metadata_contract(state)
        empty = self._empty_bundle()
        base_semantics = self._base_lifecycle.semantic_digests
        bundle_self_valid = self._discovery.validate_proposal_bundle(
            state.installed_bundle,
            semantic_generation=state.installed_bundle.semantic_generation,
            source_digest=state.installed_bundle.source_digest,
            canonical_digest=state.installed_bundle.canonical_digest,
            transition_id=state.installed_bundle.transition_id,
            state_observation_count=state.installed_bundle.state_observation_count,
        )
        derived_semantics = self._semantic_digests(state.installed_bundle)
        expected_semantics = jnp.where(state.installed, derived_semantics, base_semantics)
        bound_lifecycle = self._lifecycle_with_semantics(expected_semantics)
        installed_contract = (
            bundle_self_valid
            & (state.installed_bundle.cohort_id == -1)
            & state.has_live_observation
            & (
                state.last_semantic_generation
                == state.installed_bundle.semantic_generation
            )
            & jnp.array_equal(
                state.last_source_digest,
                state.installed_bundle.source_digest,
            )
            & jnp.array_equal(
                state.last_canonical_digest,
                state.installed_bundle.canonical_digest,
            )
            & jnp.all(state.last_raw_available)
            & jnp.all(state.last_tail_available)
            & (
                state.last_materialization_observation_count
                >= state.installed_bundle.state_observation_count
            )
            & (
                jnp.array_equal(
                    state.last_materialization_transition_id,
                    state.installed_bundle.transition_id,
                )
                | _uint64_words_less(
                    state.installed_bundle.transition_id,
                    state.last_materialization_transition_id,
                )
            )
            & (state.installation_count > 0)
        )
        pristine_contract = (
            ~state.has_live_observation
            & (state.last_semantic_generation == -1)
            & jnp.all(state.last_source_digest == 0)
            & jnp.all(state.last_canonical_digest == 0)
            & jnp.all(state.last_raw_features == 0.0)
            & ~jnp.any(state.last_raw_available)
            & jnp.all(state.last_tail_values == 0.0)
            & ~jnp.any(state.last_tail_available)
            & jnp.all(state.last_materialization_transition_id == 0)
            & (state.last_materialization_observation_count == -1)
        )
        cold_live_contract = (
            state.has_live_observation
            & (state.last_semantic_generation >= 0)
            & jnp.any(state.last_source_digest != 0)
            & jnp.any(state.last_canonical_digest != 0)
            & jnp.all(state.last_raw_available)
            & jnp.all(state.last_tail_values == 0.0)
            & ~jnp.any(state.last_tail_available)
            & (state.last_materialization_observation_count >= 0)
            & (state.lifecycle_metadata.stomp_executing_option < 0)
            & (state.lifecycle_metadata.audit_state.active_option < 0)
            & (~state.lifecycle_metadata.audit_state.trial_active)
        )
        dormant_contract = (
            _tree_array_equal(state.installed_bundle, empty)
            & (state.installation_count == 0)
            & (pristine_contract | cold_live_contract)
        )
        expected_unavailable = state.installation_count >= self._config.max_installations
        expected_error = jnp.where(
            expected_unavailable,
            jnp.asarray(CUMULANT_OPTION_INSTALLER_ERROR_CAPACITY, dtype=jnp.int32),
            jnp.asarray(CUMULANT_OPTION_INSTALLER_ERROR_NONE, dtype=jnp.int32),
        )
        return (
            jnp.any(state.consumer_source_digest != 0)
            & jnp.any(state.consumer_representation_digest != 0)
            & jnp.any(state.lifecycle_id != 0)
            & jnp.array_equal(
                state.lifecycle_id,
                state.lifecycle_metadata.lifecycle_id,
            )
            & jnp.array_equal(
                state.lifecycle_metadata.audit_state.source_digest,
                state.consumer_source_digest,
            )
            & jnp.array_equal(
                state.lifecycle_metadata.audit_state.representation_digest,
                state.consumer_representation_digest,
            )
            & jnp.array_equal(state.installed_semantic_digests, expected_semantics)
            & bound_lifecycle.metadata_state_valid(state.lifecycle_metadata)
            & jnp.where(state.installed, installed_contract, dormant_contract)
            & (state.installation_count >= 0)
            & (state.installation_count <= self._config.max_installations)
            & (state.installer_unavailable == expected_unavailable)
            & (state.installer_error == expected_error)
            & (state.revision >= 0)
            & jnp.all(jnp.isfinite(state.last_raw_features))
            & jnp.all(jnp.isfinite(state.last_tail_values))
            & jnp.array_equal(
                state.metadata_checksum,
                _checksum_arrays(self._metadata_payload_arrays(state)),
            )
        )

    def attach_borrowed_stomp(
        self,
        metadata: CumulantOptionInstallationMetadataState,
        stomp_state: Any,
    ) -> CumulantOptionInstallationBorrowResult:
        """Build a transient full installer around the exact borrowed owner."""

        self._check_metadata_contract(metadata)
        lifecycle = self._lifecycle_with_semantics(
            metadata.installed_semantic_digests
        )
        lifecycle_result = lifecycle.attach_borrowed_stomp(
            metadata.lifecycle_metadata,
            stomp_state,
        )
        values = {
            field.name: getattr(metadata, field.name)
            for field in dataclasses.fields(CumulantOptionInstallationMetadataState)
            if field.name
            not in {
                "lifecycle_metadata",
                "source_binding_checksum",
                "metadata_checksum",
            }
        }
        candidate = CumulantOptionInstallationState(
            lifecycle_state=lifecycle_result.state,
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
            & lifecycle_result.transaction_applied
            & binding_matches
            & self.state_valid(candidate)
        )
        return CumulantOptionInstallationBorrowResult(
            state=candidate,
            lifecycle=lifecycle_result,
            metadata_valid=metadata_valid,
            binding_matches=binding_matches,
            transaction_applied=transaction_applied,
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
        )

    def _check_live_inputs(self, inputs: CumulantOptionLiveInputs) -> None:
        if type(inputs) is not CumulantOptionLiveInputs:
            raise TypeError("inputs must be an exact CumulantOptionLiveInputs")
        cfg = self._discovery.config
        contracts = (
            (inputs.raw_features, "raw_features", (cfg.raw_feature_dim,), jnp.float32),
            (inputs.raw_available, "raw_available", (cfg.raw_feature_dim,), jnp.bool_),
            (
                inputs.controllable_events,
                "controllable_events",
                (cfg.controllable_event_dim,),
                jnp.float32,
            ),
            (
                inputs.controllable_events_available,
                "controllable_events_available",
                (cfg.controllable_event_dim,),
                jnp.bool_,
            ),
            (
                inputs.transition_atoms,
                "transition_atoms",
                (cfg.transition_atom_dim,),
                jnp.float32,
            ),
            (
                inputs.transition_atoms_available,
                "transition_atoms_available",
                (cfg.transition_atom_dim,),
                jnp.bool_,
            ),
            (
                inputs.bottleneck_values,
                "bottleneck_values",
                (cfg.prediction_bottleneck_dim,),
                jnp.float32,
            ),
            (
                inputs.bottleneck_available,
                "bottleneck_available",
                (cfg.prediction_bottleneck_dim,),
                jnp.bool_,
            ),
            (inputs.semantic_generation, "semantic_generation", (), jnp.int32),
            (inputs.source_digest, "source_digest", (2,), jnp.uint32),
            (inputs.canonical_digest, "canonical_digest", (32,), jnp.uint8),
            (inputs.transition_id, "transition_id", (2,), jnp.uint32),
            (
                inputs.state_observation_count,
                "state_observation_count",
                (),
                jnp.int32,
            ),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"inputs.{name}", shape=shape, dtype=dtype)

    def _check_state_contract(self, state: CumulantOptionInstallationState) -> None:
        if type(state) is not CumulantOptionInstallationState:
            raise TypeError("state must be an exact CumulantOptionInstallationState")
        cfg = self._discovery.config
        self._discovery.check_proposal_bundle_contract(state.installed_bundle)
        contracts = (
            (
                state.installed_semantic_digests,
                "installed_semantic_digests",
                (cfg.option_budget, _DIGEST_WORDS),
                jnp.uint32,
            ),
            (state.consumer_source_digest, "consumer_source_digest", (8,), jnp.uint32),
            (
                state.consumer_representation_digest,
                "consumer_representation_digest",
                (8,),
                jnp.uint32,
            ),
            (state.lifecycle_id, "lifecycle_id", (2,), jnp.uint32),
            (state.installed, "installed", (), jnp.bool_),
            (state.has_live_observation, "has_live_observation", (), jnp.bool_),
            (
                state.last_semantic_generation,
                "last_semantic_generation",
                (),
                jnp.int32,
            ),
            (state.last_source_digest, "last_source_digest", (2,), jnp.uint32),
            (
                state.last_canonical_digest,
                "last_canonical_digest",
                (32,),
                jnp.uint8,
            ),
            (
                state.last_raw_features,
                "last_raw_features",
                (cfg.raw_feature_dim,),
                jnp.float32,
            ),
            (
                state.last_raw_available,
                "last_raw_available",
                (cfg.raw_feature_dim,),
                jnp.bool_,
            ),
            (
                state.last_tail_values,
                "last_tail_values",
                (cfg.option_budget,),
                jnp.float32,
            ),
            (
                state.last_tail_available,
                "last_tail_available",
                (cfg.option_budget,),
                jnp.bool_,
            ),
            (
                state.last_materialization_transition_id,
                "last_materialization_transition_id",
                (2,),
                jnp.uint32,
            ),
            (
                state.last_materialization_observation_count,
                "last_materialization_observation_count",
                (),
                jnp.int32,
            ),
            (state.installation_count, "installation_count", (), jnp.int32),
            (state.installer_unavailable, "installer_unavailable", (), jnp.bool_),
            (state.installer_error, "installer_error", (), jnp.int32),
            (state.revision, "revision", (), jnp.int32),
            (state.binding_checksum, "binding_checksum", (2,), jnp.uint32),
        )
        for value, name, shape, dtype in contracts:
            _require_array(value, name=f"state.{name}", shape=shape, dtype=dtype)

    def _empty_bundle(self) -> CumulantSubtaskProposalBundle:
        return self._discovery.empty_proposal_bundle(-1)

    def init(
        self,
        key: Array,
        *,
        consumer_source_digest: Array,
        consumer_representation_digest: Array,
        lifecycle_id: Array,
    ) -> CumulantOptionInstallationState:
        """Initialize dormant opt-in slots with no behavior-eligible options."""

        source = _require_array(
            consumer_source_digest,
            name="consumer_source_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        representation = _require_array(
            consumer_representation_digest,
            name="consumer_representation_digest",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        lifecycle = _require_array(
            lifecycle_id,
            name="lifecycle_id",
            shape=(_LIFECYCLE_WORDS,),
            dtype=jnp.uint32,
        )
        base_semantics = self._base_lifecycle.semantic_digests
        lifecycle_state = self._base_lifecycle.init(
            key,
            source_digest=source,
            representation_digest=representation,
            lifecycle_id=lifecycle,
        )
        unavailable = self._config.max_installations == 0
        state = CumulantOptionInstallationState(
            lifecycle_state=lifecycle_state,
            installed_bundle=self._empty_bundle(),
            installed_semantic_digests=base_semantics,
            consumer_source_digest=source,
            consumer_representation_digest=representation,
            lifecycle_id=lifecycle,
            installed=jnp.asarray(False, dtype=jnp.bool_),
            has_live_observation=jnp.asarray(False, dtype=jnp.bool_),
            last_semantic_generation=jnp.asarray(-1, dtype=jnp.int32),
            last_source_digest=jnp.zeros((2,), dtype=jnp.uint32),
            last_canonical_digest=jnp.zeros((32,), dtype=jnp.uint8),
            last_raw_features=jnp.zeros(
                (self._discovery.config.raw_feature_dim,), dtype=jnp.float32
            ),
            last_raw_available=jnp.zeros(
                (self._discovery.config.raw_feature_dim,), dtype=jnp.bool_
            ),
            last_tail_values=jnp.zeros((self._discovery.config.option_budget,), dtype=jnp.float32),
            last_tail_available=jnp.zeros((self._discovery.config.option_budget,), dtype=jnp.bool_),
            last_materialization_transition_id=jnp.zeros((2,), dtype=jnp.uint32),
            last_materialization_observation_count=jnp.asarray(-1, dtype=jnp.int32),
            installation_count=jnp.asarray(0, dtype=jnp.int32),
            installer_unavailable=jnp.asarray(unavailable, dtype=jnp.bool_),
            installer_error=jnp.asarray(
                CUMULANT_OPTION_INSTALLER_ERROR_CAPACITY
                if unavailable
                else CUMULANT_OPTION_INSTALLER_ERROR_NONE,
                dtype=jnp.int32,
            ),
            revision=jnp.asarray(0, dtype=jnp.int32),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._with_checksum(state)

    def state_valid(self, state: CumulantOptionInstallationState) -> Bool[Array, ""]:
        """Validate every persistent binding before installation or control."""

        self._check_state_contract(state)
        empty = self._empty_bundle()
        base_semantics = self._base_lifecycle.semantic_digests
        bundle_self_valid = self._discovery.validate_proposal_bundle(
            state.installed_bundle,
            semantic_generation=state.installed_bundle.semantic_generation,
            source_digest=state.installed_bundle.source_digest,
            canonical_digest=state.installed_bundle.canonical_digest,
            transition_id=state.installed_bundle.transition_id,
            state_observation_count=state.installed_bundle.state_observation_count,
        )
        derived_semantics = self._semantic_digests(state.installed_bundle)
        expected_semantics = jnp.where(
            state.installed,
            derived_semantics,
            base_semantics,
        )
        bound_lifecycle = self._lifecycle_with_semantics(expected_semantics)
        installed_contract = (
            bundle_self_valid
            & (state.installed_bundle.cohort_id == -1)
            & state.has_live_observation
            & (state.last_semantic_generation == state.installed_bundle.semantic_generation)
            & jnp.array_equal(
                state.last_source_digest,
                state.installed_bundle.source_digest,
            )
            & jnp.array_equal(
                state.last_canonical_digest,
                state.installed_bundle.canonical_digest,
            )
            & jnp.all(state.last_raw_available)
            & jnp.all(state.last_tail_available)
            & (
                state.last_materialization_observation_count
                >= state.installed_bundle.state_observation_count
            )
            & (
                jnp.array_equal(
                    state.last_materialization_transition_id,
                    state.installed_bundle.transition_id,
                )
                | _uint64_words_less(
                    state.installed_bundle.transition_id,
                    state.last_materialization_transition_id,
                )
            )
            & (state.installation_count > 0)
        )
        pristine_contract = (
            ~state.has_live_observation
            & (state.last_semantic_generation == -1)
            & jnp.all(state.last_source_digest == 0)
            & jnp.all(state.last_canonical_digest == 0)
            & jnp.all(state.last_raw_features == 0.0)
            & ~jnp.any(state.last_raw_available)
            & jnp.all(state.last_tail_values == 0.0)
            & ~jnp.any(state.last_tail_available)
            & jnp.all(state.last_materialization_transition_id == 0)
            & (state.last_materialization_observation_count == -1)
        )
        cold_live_contract = (
            state.has_live_observation
            & (state.last_semantic_generation >= 0)
            & jnp.any(state.last_source_digest != 0)
            & jnp.any(state.last_canonical_digest != 0)
            & jnp.all(state.last_raw_available)
            & jnp.all(state.last_tail_values == 0.0)
            & ~jnp.any(state.last_tail_available)
            & (state.last_materialization_observation_count >= 0)
            & (state.lifecycle_state.stomp_state.executing_option < 0)
            & (state.lifecycle_state.audit_state.active_option < 0)
            & (~state.lifecycle_state.audit_state.trial_active)
        )
        dormant_contract = (
            _tree_array_equal(state.installed_bundle, empty)
            & (state.installation_count == 0)
            & (pristine_contract | cold_live_contract)
        )
        expected_unavailable = state.installation_count >= self._config.max_installations
        expected_error = jnp.where(
            expected_unavailable,
            jnp.asarray(CUMULANT_OPTION_INSTALLER_ERROR_CAPACITY, dtype=jnp.int32),
            jnp.asarray(CUMULANT_OPTION_INSTALLER_ERROR_NONE, dtype=jnp.int32),
        )
        return (
            jnp.any(state.consumer_source_digest != 0)
            & jnp.any(state.consumer_representation_digest != 0)
            & jnp.any(state.lifecycle_id != 0)
            & jnp.array_equal(state.lifecycle_id, state.lifecycle_state.lifecycle_id)
            & jnp.array_equal(
                state.lifecycle_state.audit_state.source_digest,
                state.consumer_source_digest,
            )
            & jnp.array_equal(
                state.lifecycle_state.audit_state.representation_digest,
                state.consumer_representation_digest,
            )
            & jnp.array_equal(state.installed_semantic_digests, expected_semantics)
            & bound_lifecycle.state_valid(state.lifecycle_state)
            & jnp.where(state.installed, installed_contract, dormant_contract)
            & (state.installation_count >= 0)
            & (state.installation_count <= self._config.max_installations)
            & (state.installer_unavailable == expected_unavailable)
            & (state.installer_error == expected_error)
            & (state.revision >= 0)
            & jnp.all(jnp.isfinite(state.last_raw_features))
            & jnp.all(jnp.isfinite(state.last_tail_values))
            & jnp.array_equal(
                state.binding_checksum,
                _checksum_arrays(self._payload_arrays(state)),
            )
        )

    def _inputs_finite(self, inputs: CumulantOptionLiveInputs) -> Array:
        return (
            jnp.all(jnp.isfinite(inputs.raw_features))
            & jnp.all(jnp.isfinite(inputs.controllable_events))
            & jnp.all(jnp.isfinite(inputs.transition_atoms))
            & jnp.all(jnp.isfinite(inputs.bottleneck_values))
        )

    def _compute_live_tail(
        self,
        descriptors: Array,
        previous_raw_features: Array,
        previous_raw_available: Array,
        inputs: CumulantOptionLiveInputs,
    ) -> tuple[Array, Array]:
        cfg = self._discovery.config
        family = descriptors[:, 0]
        indices = descriptors[:, 1]
        polarity = descriptors[:, 2].astype(jnp.float32)
        event_indices = jnp.clip(indices, 0, cfg.controllable_event_dim - 1)
        raw_indices = jnp.clip(indices, 0, cfg.raw_feature_dim - 1)
        atom_indices = jnp.clip(indices, 0, cfg.transition_atom_dim - 1)
        bottleneck_indices = jnp.clip(indices, 0, cfg.prediction_bottleneck_dim - 1)
        event_values = inputs.controllable_events[event_indices]
        feature_changes = inputs.raw_features[raw_indices] - previous_raw_features[raw_indices]
        atom_values = inputs.transition_atoms[atom_indices]
        bottleneck_values = inputs.bottleneck_values[bottleneck_indices]
        values = jnp.where(
            family == CUMULANT_SOURCE_CONTROLLABLE_EVENT,
            event_values,
            jnp.where(
                family == CUMULANT_SOURCE_FEATURE_CHANGE,
                feature_changes,
                jnp.where(
                    family == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
                    atom_values,
                    bottleneck_values,
                ),
            ),
        )
        event_available = inputs.controllable_events_available[event_indices]
        feature_available = inputs.raw_available[raw_indices] & previous_raw_available[raw_indices]
        atom_available = inputs.transition_atoms_available[atom_indices]
        bottleneck_available = inputs.bottleneck_available[bottleneck_indices]
        available = jnp.where(
            family == CUMULANT_SOURCE_CONTROLLABLE_EVENT,
            event_available,
            jnp.where(
                family == CUMULANT_SOURCE_FEATURE_CHANGE,
                feature_available,
                jnp.where(
                    family == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
                    atom_available,
                    bottleneck_available,
                ),
            ),
        )
        supported = (
            (family == CUMULANT_SOURCE_CONTROLLABLE_EVENT)
            | (family == CUMULANT_SOURCE_FEATURE_CHANGE)
            | (family == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM)
            | (family == CUMULANT_SOURCE_PREDICTION_BOTTLENECK)
        )
        # Polarity is applied here exactly once. STOMP's installed scale is +1.
        return (values * polarity).astype(jnp.float32), available & supported

    def _bundle_matches_inputs(
        self,
        bundle: CumulantSubtaskProposalBundle,
        inputs: CumulantOptionLiveInputs,
    ) -> Array:
        return self._discovery.validate_proposal_bundle(
            bundle,
            semantic_generation=inputs.semantic_generation,
            source_digest=inputs.source_digest,
            canonical_digest=inputs.canonical_digest,
            transition_id=inputs.transition_id,
            state_observation_count=inputs.state_observation_count,
        ) & (bundle.cohort_id == -1)

    def _materialization_token(
        self,
        state: CumulantOptionInstallationState,
        inputs: CumulantOptionLiveInputs,
        tail: Array,
        tail_available: Array,
        available: Array,
    ) -> CumulantOptionMaterialization:
        if self._reserved_observation_suffix == 0:
            observation = jnp.concatenate((inputs.raw_features, tail), axis=0)
        else:
            observation = jnp.concatenate(
                (
                    inputs.raw_features,
                    tail,
                    jnp.zeros(
                        (self._reserved_observation_suffix,),
                        dtype=jnp.float32,
                    ),
                ),
                axis=0,
            )
        zeros = jnp.zeros_like(observation)
        return CumulantOptionMaterialization(
            observation=jnp.where(available, observation, zeros),
            tail_values=jnp.where(available, tail, jnp.zeros_like(tail)),
            tail_available=jnp.where(
                available,
                tail_available,
                jnp.zeros_like(tail, dtype=jnp.bool_),
            ),
            transition_id=jnp.where(
                available,
                inputs.transition_id,
                jnp.zeros_like(inputs.transition_id),
            ),
            state_observation_count=jnp.where(
                available,
                inputs.state_observation_count,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            composition_revision=state.revision,
            composition_checksum=state.binding_checksum,
            available=jnp.asarray(available, dtype=jnp.bool_),
        )

    def materialize_cold(
        self,
        state: CumulantOptionInstallationState,
        inputs: CumulantOptionLiveInputs,
    ) -> CumulantOptionMaterializationResult:
        """Advance a raw prior while every uninstalled option remains masked."""

        self._check_state_contract(state)
        self._check_live_inputs(inputs)
        persistent_valid = self.state_valid(state)
        first = ~state.has_live_observation
        binding_valid = (~state.installed) & (
            first
            | (
                (inputs.semantic_generation == state.last_semantic_generation)
                & jnp.array_equal(inputs.source_digest, state.last_source_digest)
                & jnp.array_equal(inputs.canonical_digest, state.last_canonical_digest)
            )
        )
        transition_newer = first | (
            (inputs.state_observation_count > state.last_materialization_observation_count)
            & _uint64_words_less(
                state.last_materialization_transition_id,
                inputs.transition_id,
            )
        )
        inputs_valid = (
            self._inputs_finite(inputs)
            & (inputs.semantic_generation >= 0)
            & jnp.any(inputs.source_digest != 0)
            & jnp.any(inputs.canonical_digest != 0)
            & jnp.all(inputs.raw_available)
        )
        transaction_valid = (
            persistent_valid & binding_valid & transition_newer & inputs_valid
        )
        zero_tail = jnp.zeros(
            (self._discovery.config.option_budget,),
            dtype=jnp.float32,
        )
        no_tail = jnp.zeros_like(zero_tail, dtype=jnp.bool_)
        proposed = dataclasses.replace(
            state,
            has_live_observation=jnp.asarray(True, dtype=jnp.bool_),
            last_semantic_generation=inputs.semantic_generation,
            last_source_digest=inputs.source_digest,
            last_canonical_digest=inputs.canonical_digest,
            last_raw_features=inputs.raw_features,
            last_raw_available=inputs.raw_available,
            last_tail_values=zero_tail,
            last_tail_available=no_tail,
            last_materialization_transition_id=inputs.transition_id,
            last_materialization_observation_count=inputs.state_observation_count,
            revision=_saturating_increment(state.revision),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        applied = transaction_valid & self.state_valid(proposed)
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        token = self._materialization_token(
            next_state,
            inputs,
            zero_tail,
            no_tail,
            applied,
        )
        return CumulantOptionMaterializationResult(
            state=next_state,
            materialization=token,
            state_valid=persistent_valid,
            binding_valid=binding_valid,
            transition_is_newer=transition_newer,
            inputs_valid=inputs_valid,
            applied=applied,
        )

    def install(
        self,
        state: CumulantOptionInstallationState,
        bundle: CumulantSubtaskProposalBundle,
        fresh_key: Array,
        *,
        inputs: CumulantOptionLiveInputs,
    ) -> CumulantOptionInstallationResult:
        """Install or refresh one exact live bundle at a quiescent boundary."""

        self._check_state_contract(state)
        self._discovery.check_proposal_bundle_contract(bundle)
        self._check_live_inputs(inputs)
        _require_array(
            jr.key_data(fresh_key),
            name="fresh key data",
            shape=(2,),
            dtype=jnp.uint32,
        )
        persistent_valid = self.state_valid(state)
        bundle_valid = self._bundle_matches_inputs(bundle, inputs)
        prior_binding_valid = state.has_live_observation & (
            state.installed
            | (
                (inputs.semantic_generation == state.last_semantic_generation)
                & jnp.array_equal(inputs.source_digest, state.last_source_digest)
                & jnp.array_equal(inputs.canonical_digest, state.last_canonical_digest)
            )
        )
        transition_newer = (
            inputs.state_observation_count > state.last_materialization_observation_count
        ) & _uint64_words_less(
            state.last_materialization_transition_id,
            inputs.transition_id,
        )
        inputs_finite = self._inputs_finite(inputs)
        tail, tail_available = self._compute_live_tail(
            bundle.selected_descriptors,
            state.last_raw_features,
            state.last_raw_available,
            inputs,
        )
        materialization_valid = (
            inputs_finite
            & jnp.all(inputs.raw_available)
            & jnp.all(tail_available)
            & jnp.all(jnp.isfinite(tail))
            & _float_bits_equal(tail, bundle.selected_cumulants)
        )
        candidate_semantics = self._semantic_digests(bundle)
        same_slots = jnp.all(
            candidate_semantics == state.installed_semantic_digests,
            axis=1,
        )
        semantics_changed = (~state.installed) | jnp.any(~same_slots)
        quiescent = (
            (state.lifecycle_state.stomp_state.executing_option < 0)
            & (state.lifecycle_state.audit_state.active_option < 0)
            & (~state.lifecycle_state.audit_state.trial_active)
        )
        capacity_available = (~semantics_changed) | (
            (~state.installer_unavailable)
            & (state.installation_count < self._config.max_installations)
        )
        transaction_valid = (
            persistent_valid
            & bundle_valid
            & prior_binding_valid
            & transition_newer
            & materialization_valid
        )
        candidate_lifecycle = self._lifecycle_with_semantics(candidate_semantics)
        no_rebind = STOMPOptionLifecycleRebindResult(
            state=state.lifecycle_state,
            transaction_valid=jnp.asarray(True, dtype=jnp.bool_),
            applied=jnp.asarray(False, dtype=jnp.bool_),
            deferred=jnp.asarray(False, dtype=jnp.bool_),
            preserved_slots=jnp.ones(
                (self._discovery.config.option_budget,), dtype=jnp.bool_
            ),
            reset_slots=jnp.zeros(
                (self._discovery.config.option_budget,), dtype=jnp.bool_
            ),
        )
        rebound = jax.lax.cond(
            semantics_changed,
            lambda _: candidate_lifecycle.rebind(
                state.lifecycle_state,
                fresh_key,
                source_digest=state.consumer_source_digest,
                representation_digest=state.consumer_representation_digest,
            ),
            lambda _: no_rebind,
            None,
        )
        changed_applied = semantics_changed & rebound.applied
        refresh_applied = ~semantics_changed
        commit_requested = (
            transaction_valid
            & capacity_available
            & quiescent
            & (changed_applied | refresh_applied)
        )
        next_lifecycle = jax.lax.cond(
            changed_applied,
            lambda _: rebound.state,
            lambda _: state.lifecycle_state,
            None,
        )
        next_count = state.installation_count + semantics_changed.astype(jnp.int32)
        next_unavailable = next_count >= self._config.max_installations
        proposed = CumulantOptionInstallationState(
            lifecycle_state=next_lifecycle,
            installed_bundle=bundle,
            installed_semantic_digests=candidate_semantics,
            consumer_source_digest=state.consumer_source_digest,
            consumer_representation_digest=state.consumer_representation_digest,
            lifecycle_id=state.lifecycle_id,
            installed=jnp.asarray(True, dtype=jnp.bool_),
            has_live_observation=jnp.asarray(True, dtype=jnp.bool_),
            last_semantic_generation=inputs.semantic_generation,
            last_source_digest=inputs.source_digest,
            last_canonical_digest=inputs.canonical_digest,
            last_raw_features=inputs.raw_features,
            last_raw_available=inputs.raw_available,
            last_tail_values=tail,
            last_tail_available=tail_available,
            last_materialization_transition_id=inputs.transition_id,
            last_materialization_observation_count=inputs.state_observation_count,
            installation_count=next_count,
            installer_unavailable=next_unavailable,
            installer_error=jnp.where(
                next_unavailable,
                jnp.asarray(CUMULANT_OPTION_INSTALLER_ERROR_CAPACITY, dtype=jnp.int32),
                jnp.asarray(CUMULANT_OPTION_INSTALLER_ERROR_NONE, dtype=jnp.int32),
            ),
            revision=_saturating_increment(state.revision),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        proposed_valid = self.state_valid(proposed)
        applied = commit_requested & proposed_valid
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        materialization = self._materialization_token(
            next_state,
            inputs,
            tail,
            tail_available,
            applied,
        )
        return CumulantOptionInstallationResult(
            state=next_state,
            materialization=materialization,
            transaction_valid=transaction_valid,
            bundle_live_binding_valid=bundle_valid,
            materialization_valid=materialization_valid,
            semantics_changed=semantics_changed,
            quiescent=quiescent,
            capacity_available=capacity_available,
            applied=applied,
            provenance_refreshed=applied & (~semantics_changed),
            deferred=transaction_valid & capacity_available & (~quiescent),
            preserved_slots=applied
            & jnp.where(semantics_changed, rebound.preserved_slots, True),
            reset_slots=applied & semantics_changed & rebound.reset_slots,
            installer_unavailable=next_state.installer_unavailable,
            live_policy_rng_preserved=jnp.array_equal(
                next_state.lifecycle_state.stomp_state.rng_key,
                state.lifecycle_state.stomp_state.rng_key,
            ),
        )

    def materialize_live(
        self,
        state: CumulantOptionInstallationState,
        inputs: CumulantOptionLiveInputs,
    ) -> CumulantOptionMaterializationResult:
        """Reevaluate installed descriptors and advance the bounded raw prior."""

        self._check_state_contract(state)
        self._check_live_inputs(inputs)
        persistent_valid = self.state_valid(state)
        bundle = state.installed_bundle
        binding_valid = (
            state.installed
            & (inputs.semantic_generation == bundle.semantic_generation)
            & jnp.array_equal(inputs.source_digest, bundle.source_digest)
            & jnp.array_equal(inputs.canonical_digest, bundle.canonical_digest)
        )
        transition_newer = (
            inputs.state_observation_count > state.last_materialization_observation_count
        ) & _uint64_words_less(
            state.last_materialization_transition_id,
            inputs.transition_id,
        )
        inputs_valid = self._inputs_finite(inputs) & jnp.all(inputs.raw_available)
        tail, tail_available = self._compute_live_tail(
            bundle.selected_descriptors,
            state.last_raw_features,
            state.last_raw_available,
            inputs,
        )
        transaction_valid = (
            persistent_valid
            & binding_valid
            & transition_newer
            & inputs_valid
            & jnp.all(tail_available)
            & jnp.all(jnp.isfinite(tail))
        )
        proposed = dataclasses.replace(
            state,
            has_live_observation=jnp.asarray(True, dtype=jnp.bool_),
            last_semantic_generation=inputs.semantic_generation,
            last_source_digest=inputs.source_digest,
            last_canonical_digest=inputs.canonical_digest,
            last_raw_features=inputs.raw_features,
            last_raw_available=inputs.raw_available,
            last_tail_values=tail,
            last_tail_available=tail_available,
            last_materialization_transition_id=inputs.transition_id,
            last_materialization_observation_count=inputs.state_observation_count,
            revision=_saturating_increment(state.revision),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
        proposed = self._with_checksum(proposed)
        applied = transaction_valid & self.state_valid(proposed)
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        token = self._materialization_token(
            next_state,
            inputs,
            tail,
            tail_available,
            applied,
        )
        return CumulantOptionMaterializationResult(
            state=next_state,
            materialization=token,
            state_valid=persistent_valid,
            binding_valid=binding_valid,
            transition_is_newer=transition_newer,
            inputs_valid=inputs_valid & jnp.all(tail_available),
            applied=applied,
        )

    def _materialization_valid_for_state(
        self,
        state: CumulantOptionInstallationState,
        materialization: CumulantOptionMaterialization,
    ) -> Array:
        """Validate a complete installed-tail materialization token."""

        common = self._materialization_common_valid(state, materialization)
        raw_dim = self._discovery.config.raw_feature_dim
        option_stop = raw_dim + self._discovery.config.option_budget
        return (
            common
            & state.installed
            & jnp.all(materialization.tail_available)
            & jnp.all(state.last_tail_available)
            & _float_bits_equal(materialization.tail_values, state.last_tail_values)
            & _float_bits_equal(
                materialization.observation[raw_dim:option_stop],
                state.last_tail_values,
            )
            & jnp.all(materialization.observation[option_stop:] == 0.0)
        )

    def _materialization_common_valid(
        self,
        state: CumulantOptionInstallationState,
        materialization: CumulantOptionMaterialization,
    ) -> Array:
        if type(materialization) is not CumulantOptionMaterialization:
            raise TypeError("materialization must be an exact CumulantOptionMaterialization")
        cfg = self._stomp_config
        _require_array(
            materialization.observation,
            name="materialization.observation",
            shape=(cfg.observation_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            materialization.tail_values,
            name="materialization.tail_values",
            shape=(self._discovery.config.option_budget,),
            dtype=jnp.float32,
        )
        _require_array(
            materialization.tail_available,
            name="materialization.tail_available",
            shape=(self._discovery.config.option_budget,),
            dtype=jnp.bool_,
        )
        _require_array(
            materialization.transition_id,
            name="materialization.transition_id",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            materialization.state_observation_count,
            name="materialization.state_observation_count",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            materialization.composition_revision,
            name="materialization.composition_revision",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            materialization.composition_checksum,
            name="materialization.composition_checksum",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            materialization.available,
            name="materialization.available",
            shape=(),
            dtype=jnp.bool_,
        )
        raw_dim = self._discovery.config.raw_feature_dim
        return (
            materialization.available
            & state.has_live_observation
            & self.state_valid(state)
            & jnp.all(jnp.isfinite(materialization.observation))
            & _float_bits_equal(
                materialization.observation[:raw_dim],
                state.last_raw_features,
            )
            & jnp.array_equal(
                materialization.transition_id,
                state.last_materialization_transition_id,
            )
            & (
                materialization.state_observation_count
                == state.last_materialization_observation_count
            )
            & (materialization.composition_revision == state.revision)
            & jnp.array_equal(
                materialization.composition_checksum,
                state.binding_checksum,
            )
        )

    def _cold_materialization_valid_for_state(
        self,
        state: CumulantOptionInstallationState,
        materialization: CumulantOptionMaterialization,
    ) -> Array:
        """Validate a raw-plus-zero-tail token for primitive-only control."""

        common = self._materialization_common_valid(state, materialization)
        raw_dim = self._discovery.config.raw_feature_dim
        return (
            common
            & (~state.installed)
            & (~jnp.any(materialization.tail_available))
            & (~jnp.any(state.last_tail_available))
            & jnp.all(materialization.tail_values == 0.0)
            & jnp.all(state.last_tail_values == 0.0)
            & jnp.all(materialization.observation[raw_dim:] == 0.0)
        )

    def _cold_action_mask(self) -> Array:
        cfg = self._stomp_config
        return jnp.concatenate(
            (
                jnp.ones((cfg.n_primitive_actions,), dtype=jnp.bool_),
                jnp.zeros((cfg.n_options,), dtype=jnp.bool_),
            )
        )

    def start(
        self,
        state: CumulantOptionInstallationState,
        materialization: CumulantOptionMaterialization,
    ) -> CumulantOptionStartResult:
        """Host-only start from an exact state-bound live observation."""

        self._check_state_contract(state)
        valid = bool(
            jax.device_get(
                self._materialization_valid_for_state(state, materialization)
            )
        )
        if not valid:
            return CumulantOptionStartResult(state=state, lifecycle_result=None, applied=False)
        lifecycle = self._lifecycle_with_semantics(state.installed_semantic_digests)
        result = lifecycle.start(state.lifecycle_state, materialization.observation)
        proposed = self._with_checksum(
            dataclasses.replace(
                state,
                lifecycle_state=result.state,
                revision=_saturating_increment(state.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        applied = bool(jax.device_get(result.applied & self.state_valid(proposed)))
        if not applied:
            return CumulantOptionStartResult(state=state, lifecycle_result=None, applied=False)
        return CumulantOptionStartResult(
            state=proposed,
            lifecycle_result=result,
            applied=True,
        )

    def update(
        self,
        state: CumulantOptionInstallationState,
        materialization: CumulantOptionMaterialization,
        env_reward: float | Array,
        discount: float | Array | None = None,
        *,
        decision_observation: Array | None = None,
        execution_boundary: bool | Array = False,
        context: int | Array = 0,
        idle_candidate_option: int | Array = 0,
        idle_initiation_eligible: bool | Array = False,
        comparator_randomized: bool | Array = False,
        treatment_propensity: float | Array = 0.0,
        enable_planning: bool = True,
    ) -> CumulantOptionUpdateResult:
        """Host-only update; installer exhaustion has no control veto."""

        if decision_observation is not None:
            raise ValueError(
                "separate decision_observation is not bound to the materialization token"
            )
        self._check_state_contract(state)
        valid = bool(
            jax.device_get(
                self._materialization_valid_for_state(state, materialization)
            )
        )
        if not valid:
            return CumulantOptionUpdateResult(state=state, lifecycle_result=None, applied=False)
        lifecycle = self._lifecycle_with_semantics(state.installed_semantic_digests)
        result = lifecycle.update(
            state.lifecycle_state,
            env_reward,
            materialization.observation,
            discount,
            execution_boundary=execution_boundary,
            context=context,
            idle_candidate_option=idle_candidate_option,
            idle_initiation_eligible=idle_initiation_eligible,
            comparator_randomized=comparator_randomized,
            treatment_propensity=treatment_propensity,
            enable_planning=enable_planning,
        )
        proposed = self._with_checksum(
            dataclasses.replace(
                state,
                lifecycle_state=result.state,
                revision=_saturating_increment(state.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        applied = bool(jax.device_get(result.transaction_applied & self.state_valid(proposed)))
        if not applied:
            return CumulantOptionUpdateResult(state=state, lifecycle_result=None, applied=False)
        return CumulantOptionUpdateResult(
            state=proposed,
            lifecycle_result=result,
            applied=True,
        )

    def start_cold(
        self,
        state: CumulantOptionInstallationState,
        materialization: CumulantOptionMaterialization,
    ) -> CumulantOptionStartResult:
        """Host-only primitive start from a bound raw-plus-zero-tail token."""

        self._check_state_contract(state)
        if not bool(
            jax.device_get(
                self._cold_materialization_valid_for_state(state, materialization)
            )
        ):
            return CumulantOptionStartResult(state=state, lifecycle_result=None, applied=False)
        result = self._base_lifecycle.start_with_extended_action_mask(
            state.lifecycle_state,
            materialization.observation,
            self._cold_action_mask(),
        )
        proposed = self._with_checksum(
            dataclasses.replace(
                state,
                lifecycle_state=result.state,
                revision=_saturating_increment(state.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        applied = bool(jax.device_get(result.applied & self.state_valid(proposed)))
        return CumulantOptionStartResult(
            state=proposed if applied else state,
            lifecycle_result=result if applied else None,
            applied=applied,
        )

    def update_cold(
        self,
        state: CumulantOptionInstallationState,
        materialization: CumulantOptionMaterialization,
        env_reward: float | Array,
        discount: float | Array | None = None,
        *,
        execution_boundary: bool | Array = False,
    ) -> CumulantOptionUpdateResult:
        """Host-only primitive update with option dispatch and planning masked."""

        self._check_state_contract(state)
        if not bool(
            jax.device_get(
                self._cold_materialization_valid_for_state(state, materialization)
            )
        ):
            return CumulantOptionUpdateResult(state=state, lifecycle_result=None, applied=False)
        result = self._base_lifecycle.update(
            state.lifecycle_state,
            env_reward,
            materialization.observation,
            discount,
            execution_boundary=execution_boundary,
            idle_initiation_eligible=False,
            comparator_randomized=False,
            extended_action_mask=self._cold_action_mask(),
            enable_planning=False,
        )
        proposed = self._with_checksum(
            dataclasses.replace(
                state,
                lifecycle_state=result.state,
                revision=_saturating_increment(state.revision),
                binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
            )
        )
        applied = bool(jax.device_get(result.transaction_applied & self.state_valid(proposed)))
        return CumulantOptionUpdateResult(
            state=proposed if applied else state,
            lifecycle_result=result if applied else None,
            applied=applied,
        )

    def checkpoint_payload(
        self,
        state: CumulantOptionInstallationState,
    ) -> dict[str, object]:
        """Return an exact state payload; the digest is integrity, not authentication."""

        if not bool(jax.device_get(self.state_valid(state))):
            raise ValueError("cannot checkpoint an invalid cumulant option installation")
        return {
            "schema_version": CUMULANT_OPTION_INSTALLATION_CHECKPOINT_SCHEMA,
            "config": self.to_config(),
            "state": state,
            "state_digest": _cryptographic_state_digest(state),
        }

    def restore_checkpoint(
        self,
        payload: object,
        *,
        expected_consumer_source_digest: Array,
        expected_consumer_representation_digest: Array,
        expected_lifecycle_id: Array,
        expected_installed_bundle: CumulantSubtaskProposalBundle | None,
    ) -> CumulantOptionInstallationState:
        """Restore only an exact config, state digest, and external live binding."""

        if type(payload) is not dict:
            raise ValueError("cumulant option checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        if set(raw) != {"schema_version", "config", "state", "state_digest"}:
            raise ValueError("cumulant option checkpoint keys differ from schema v1")
        if raw["schema_version"] != CUMULANT_OPTION_INSTALLATION_CHECKPOINT_SCHEMA:
            raise ValueError("cumulant option checkpoint schema differs")
        if raw["config"] != self.to_config():
            raise ValueError("cumulant option checkpoint config differs")
        restored = raw["state"]
        if type(restored) is not CumulantOptionInstallationState:
            raise ValueError("cumulant option checkpoint state type differs")
        state = restored
        persisted = _require_array(
            raw["state_digest"],
            name="checkpoint.state_digest",
            shape=(32,),
            dtype=jnp.uint8,
        )
        if not bool(jax.device_get(jnp.array_equal(persisted, _cryptographic_state_digest(state)))):
            raise ValueError("cumulant option checkpoint state digest differs")
        source = _require_array(
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
        binding_valid = (
            jnp.array_equal(state.consumer_source_digest, source)
            & jnp.array_equal(state.consumer_representation_digest, representation)
            & jnp.array_equal(state.lifecycle_id, lifecycle)
        )
        if expected_installed_bundle is None:
            bundle_valid = ~state.installed
        else:
            self._discovery.check_proposal_bundle_contract(expected_installed_bundle)
            bundle_valid = state.installed & _tree_array_equal(
                state.installed_bundle,
                expected_installed_bundle,
            )
        if not bool(jax.device_get(binding_valid & bundle_valid & self.state_valid(state))):
            raise ValueError("cumulant option checkpoint is invalid, stale, or rebound")
        return state

    def resource_budget(
        self,
        state: CumulantOptionInstallationState,
    ) -> CumulantOptionInstallationResourceBudget:
        """Return exact allocation plus bounded work and RNG ownership."""

        self._check_state_contract(state)
        lifecycle = self._lifecycle_with_semantics(state.installed_semantic_digests)
        lifecycle_bytes = lifecycle.resource_budget(
            state.lifecycle_state
        ).wrapped_persistent_state_nbytes
        persistent_bytes = _tree_nbytes(state)
        cfg = self._discovery.config
        live_values = (
            cfg.raw_feature_dim
            + cfg.controllable_event_dim
            + cfg.transition_atom_dim
            + cfg.prediction_bottleneck_dim
        )
        live_availability = live_values + cfg.raw_feature_dim
        return CumulantOptionInstallationResourceBudget(
            persistent_state_nbytes=persistent_bytes,
            lifecycle_state_nbytes=lifecycle_bytes,
            installation_binding_nbytes=persistent_bytes - lifecycle_bytes,
            raw_feature_dim=cfg.raw_feature_dim,
            option_budget=cfg.option_budget,
            observation_dim=self._stomp_config.observation_dim,
            live_source_value_cells_per_materialization=live_values,
            descriptor_cells_per_materialization=cfg.option_budget * _DESCRIPTOR_WIDTH,
            availability_cells_per_materialization=live_availability,
            lifecycle_rebind_calls_per_changed_installation=1,
            fresh_template_initializations_per_changed_installation=1,
            live_policy_rng_draws_per_installation=0,
            live_policy_rng_replaced_by_installer=False,
            fresh_template_key_supplied_by_caller=True,
            installer_capacity_can_block_valid_stomp_control=False,
            max_installations=self._config.max_installations,
            assessment=CUMULANT_OPTION_INSTALLATION_ASSESSMENT,
            output_writes=False,
            evidence_authority=False,
            promotion_authority=False,
            benefit_claim=False,
            autonomous_discovery_claim=False,
            scientific_promotion_allowed=False,
            checkpoint_schema=CUMULANT_OPTION_INSTALLATION_CHECKPOINT_SCHEMA,
            control_host_only=True,
        )


__all__ = [
    "CUMULANT_OPTION_INSTALLATION_ASSESSMENT",
    "CUMULANT_OPTION_INSTALLATION_AUTONOMOUS_DISCOVERY_CLAIM",
    "CUMULANT_OPTION_INSTALLATION_BENEFIT_CLAIM",
    "CUMULANT_OPTION_INSTALLATION_CHECKPOINT_SCHEMA",
    "CUMULANT_OPTION_INSTALLATION_CONFIG_SCHEMA",
    "CUMULANT_OPTION_INSTALLATION_CONTROL_HOST_ONLY",
    "CUMULANT_OPTION_INSTALLATION_EVIDENCE_AUTHORITY",
    "CUMULANT_OPTION_INSTALLATION_OUTPUT_WRITES",
    "CUMULANT_OPTION_INSTALLATION_PROMOTION_AUTHORITY",
    "CUMULANT_OPTION_INSTALLATION_SCIENTIFIC_PROMOTION_ALLOWED",
    "CUMULANT_OPTION_INSTALLATION_THRESHOLD_SEMANTICS",
    "CUMULANT_OPTION_INSTALLER_ERROR_CAPACITY",
    "CUMULANT_OPTION_INSTALLER_ERROR_NONE",
    "CumulantOptionInstallation",
    "CumulantOptionInstallationBorrowResult",
    "CumulantOptionInstallationConfig",
    "CumulantOptionInstallationMetadataState",
    "CumulantOptionInstallationResourceBudget",
    "CumulantOptionInstallationResult",
    "CumulantOptionInstallationState",
    "CumulantOptionLiveInputs",
    "CumulantOptionMaterialization",
    "CumulantOptionMaterializationResult",
    "CumulantOptionStartResult",
    "CumulantOptionUpdateResult",
]
