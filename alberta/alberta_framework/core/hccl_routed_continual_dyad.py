# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return"
"""Atomic HCCL dyad with routed R35 memory and planner-v2 ownership.

This additive, host/eager L0 composition owns one HCCL world/attribution
transaction, two learning coordinators, two context-lineage owners, two full
feature-birth ledgers, two feature-bound learned memories, one paired
factorized planner-v2, and two authenticated B/M/P cache projections.  It is
the first owner in this tree for which the memory layer ``M`` and planner layer
``P`` consume the same routed R35 representation as the Prototype agents.

The transaction is deliberately functional.  Every child may form a
candidate, but a single failed invariant returns the complete outer source
bit-for-bit.  Integrity tokens are unkeyed SHA-256 bindings; they detect
accidental mutation and cross-transaction mixing but do not authenticate a
caller, establish scientific evidence, or authorize dispatch or promotion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from numbers import Real
from typing import Any, ClassVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.context_lineage_retention_seam import (
    ContextLineageRetentionPreparation,
    ContextLineageRetentionSeam,
    ContextLineageRetentionSeamConfig,
    ContextLineageRetentionSeamState,
    ContextLineageRetentionStepResult,
)
from alberta_framework.core.experiential_memory import ExperientialMemoryEntry
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateRouterAuditCoordinator,
    ExternalLearnedStateRouterAuditCoordinatorConfig,
    ExternalLearnedStateRouterAuditCoordinatorResult,
    ExternalLearnedStateRouterAuditCoordinatorState,
    ExternalLearnedStateTransition,
)
from alberta_framework.core.hccl_authenticated_bmp_projection import (
    HCCLAuthenticatedBMPActionBinding,
    HCCLAuthenticatedBMPMemoryProjection,
    HCCLAuthenticatedBMPPreparedProjection,
    HCCLAuthenticatedBMPProjection,
    HCCLAuthenticatedBMPProjectionConfig,
    HCCLAuthenticatedBMPProjectionIntegrityReceipt,
    HCCLAuthenticatedBMPProjectionResult,
)
from alberta_framework.core.hccl_causal_attribution import (
    HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER,
    HCCLActionLayer,
    HCCLActionReceipt,
)
from alberta_framework.core.hccl_feature_bound_memory import (
    HCCLFeatureBoundMemory,
    HCCLFeatureBoundMemoryConfig,
    HCCLFeatureBoundMemoryRebindResult,
    HCCLFeatureBoundMemorySettleResult,
    HCCLFeatureBoundMemoryState,
    HCCLFeatureBoundMemoryStepResult,
)
from alberta_framework.core.hccl_feature_consumer_route import (
    HCCL_FEATURE_CONTEXT_START,
    HCCL_FEATURE_FAST_START,
    HCCL_FEATURE_PAIR_START,
    HCCL_FEATURE_PHYSICAL_DIM,
    HCCL_FEATURE_TOTAL_DIM,
    HCCLFeatureBirthLedger,
    HCCLFeatureConsumerRoute,
    HCCLFeatureConsumerRouteResult,
)
from alberta_framework.core.hccl_memory_credit_estimands import (
    HCCLMemoryCreditEstimandPanel,
    derive_hccl_memory_credit_estimands,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapter,
    HCCLWorldAttributionAdapterConfig,
    HCCLWorldAttributionAdapterResult,
    HCCLWorldAttributionAdapterState,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryFeedback,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgentState,
    PrototypeFeatureRepresentationState,
    PrototypeUpdateResult,
)
from alberta_framework.core.prototype_factorized_partner_planner_v2 import (
    PrototypeFactorizedPartnerPlannerV2,
    PrototypeFactorizedPartnerPlannerV2Config,
    PrototypeFactorizedPartnerPlannerV2Result,
    PrototypeFactorizedPartnerPlannerV2State,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleState,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCLCausalCoreEventReceipt,
    HCCLCausalCoreProposal,
    HCCLCausalCoreTypedSignals,
    hccl_causal_core_lifetime_for_profile,
)

HCCL_ROUTED_CONTINUAL_DYAD_CONFIG_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.config.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_STATE_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.state.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_ACTION_RECORD_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.action-record.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_ACTION_BUNDLE_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.action-bundle.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_PREPARED_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.prepared.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_RECEIPT_SCHEMA = (
    "alberta.hccl-routed-continual-dyad.receipt.v1"
)
HCCL_ROUTED_CONTINUAL_DYAD_STATUS = (
    "l0-development-r35-memory-planner-v2-atomic-owner"
)
HCCL_ROUTED_CONTINUAL_DYAD_SCIENTIFIC_PROMOTION_ALLOWED = False

_N_AGENTS = 2
_N_ACTIONS = 2
_PHYSICAL_DIM = 16
_CONTEXT_DIM = 3
_FAST_DIM = 4
_BASE_DIM = 23
_PAIR_SLOTS = 12
_PAIR_CANDIDATE_SLOTS = 120
_CANONICAL_AGENT_LIFETIME = 8_998
_HORDE_NAMES = (
    "task_discount_0p5",
    "task_discount_0p9",
    "task_discount_0p99",
    "partner_action",
    "safety_cost",
    "tv_occupancy",
    "target_zone_occupancy",
    "option_success_unavailable",
)
_HORDE_GAMMAS = (0.5, 0.9, 0.99, 0.9, 0.9, 0.9, 0.9, 0.9)
_TOKEN_NBYTES = 32
_DIGEST_WORDS = 8
_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1
_PP_SLOT = HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER.index("PP-planner")
_MM_SLOT = 0
_B0M1_SLOT = 1
_M0B1_SLOT = 2
_BB_SLOT = 3


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _host(value: object) -> np.ndarray[Any, Any]:
    array = jnp.asarray(value)
    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
        array = jr.key_data(array)
    return np.asarray(jax.device_get(array))


def _host_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _array_exact_equal(left: object, right: object) -> bool:
    left_host = np.ascontiguousarray(_host(left))
    right_host = np.ascontiguousarray(_host(right))
    return bool(
        left_host.dtype == right_host.dtype
        and left_host.shape == right_host.shape
        and left_host.tobytes(order="C") == right_host.tobytes(order="C")
    )


def _tree_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return False
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        _array_exact_equal(a, b)
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _digest_bytes(schema: str, *values: object) -> UInt[Array, " 32"]:
    if _contains_tracer(values):
        raise TypeError("routed continual-dyad integrity is host/eager-only")
    digest = hashlib.sha256(schema.encode("ascii"))
    for value in values:
        digest.update(type(value).__module__.encode("utf-8"))
        digest.update(type(value).__qualname__.encode("utf-8"))
        leaves, structure = jax.tree.flatten(value)
        digest.update(repr(structure).encode("utf-8"))
        digest.update(len(leaves).to_bytes(8, "big"))
        for leaf in leaves:
            host = np.ascontiguousarray(_host(leaf))
            digest.update(str(host.dtype).encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _digest_words(schema: str, *values: object) -> UInt[Array, " 8"]:
    raw = np.asarray(_digest_bytes(schema, *values), dtype=np.uint8).tobytes()
    return jnp.asarray(
        tuple(int.from_bytes(raw[offset : offset + 4], "little") for offset in range(0, 32, 4)),
        dtype=jnp.uint32,
    )


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose exact array metadata")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    if jnp.dtype(array.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}; got {array.dtype}")
    return array


def _checked_successor(before: Array, after: Array) -> bool:
    source = _host(before).astype(np.uint64)
    destination = _host(after).astype(np.uint64)
    source_value = (int(source[0]) << 32) | int(source[1])
    destination_value = (int(destination[0]) << 32) | int(destination[1])
    return source_value < 2**64 - 1 and destination_value == source_value + 1


def _float32_positive_zero(value: Array) -> bool:
    bits = _host(jax.lax.bitcast_convert_type(value, jnp.uint32))
    return bool(np.all(bits == np.uint32(0)))


def _signals_at(proposals: HCCLCausalCoreProposal, index: int) -> HCCLCausalCoreTypedSignals:
    return cast(
        HCCLCausalCoreTypedSignals,
        jax.tree.map(lambda leaf: leaf[index], proposals.signals),
    )


def _proposal_at(proposals: HCCLCausalCoreProposal, index: int) -> HCCLCausalCoreProposal:
    return cast(HCCLCausalCoreProposal, jax.tree.map(lambda leaf: leaf[index], proposals))


def _prototype(state: ExternalLearnedStateRouterAuditCoordinatorState) -> PrototypeAgentState:
    return state.inner_state.prototype_state


def _feature_state(prototype: PrototypeAgentState) -> PrototypeFeatureLifecycleState:
    slot = prototype.state_builder_state
    if type(slot) is not PrototypeFeatureRepresentationState:
        raise TypeError("canonical routed dyad requires PrototypeFeatureRepresentationState")
    return slot.feature_lifecycle_state


def _pair_descriptors(state: PrototypeFeatureLifecycleState) -> Int[Array, "12 2"]:
    return jnp.asarray(state.router_state.descriptors, dtype=jnp.int32)


def _candidate_descriptors(state: PrototypeFeatureLifecycleState) -> Int[Array, "120 2"]:
    return jnp.stack(
        (state.learner_state.candidate_left, state.learner_state.candidate_right),
        axis=1,
    ).astype(jnp.int32)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLRoutedContinualDyadConfig:
    """Exact fixed-geometry configuration for the routed outer owner."""

    hccl: HCCLWorldAttributionAdapterConfig
    coordinator: ExternalLearnedStateRouterAuditCoordinatorConfig
    memory_agent_0: HCCLFeatureBoundMemoryConfig
    memory_agent_1: HCCLFeatureBoundMemoryConfig
    planner: PrototypeFactorizedPartnerPlannerV2Config
    context: ContextLineageRetentionSeamConfig
    bmp_agent_0: HCCLAuthenticatedBMPProjectionConfig
    bmp_agent_1: HCCLAuthenticatedBMPProjectionConfig
    binding_owner_digest: tuple[int, ...]
    discount: float = 0.99

    SCHEMA_VERSION: ClassVar[str] = HCCL_ROUTED_CONTINUAL_DYAD_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        exact = (
            (self.hccl, HCCLWorldAttributionAdapterConfig, "hccl"),
            (
                self.coordinator,
                ExternalLearnedStateRouterAuditCoordinatorConfig,
                "coordinator",
            ),
            (self.memory_agent_0, HCCLFeatureBoundMemoryConfig, "memory_agent_0"),
            (self.memory_agent_1, HCCLFeatureBoundMemoryConfig, "memory_agent_1"),
            (self.planner, PrototypeFactorizedPartnerPlannerV2Config, "planner"),
            (self.context, ContextLineageRetentionSeamConfig, "context"),
            (self.bmp_agent_0, HCCLAuthenticatedBMPProjectionConfig, "bmp_agent_0"),
            (self.bmp_agent_1, HCCLAuthenticatedBMPProjectionConfig, "bmp_agent_1"),
        )
        for value, expected, name in exact:
            if type(value) is not expected:
                raise TypeError(f"{name} must be an exact {expected.__name__}")
        if self.memory_agent_0.agent_index != 0 or self.memory_agent_1.agent_index != 1:
            raise ValueError("feature-bound memories must own exact agent indices 0 and 1")
        prototype = self.coordinator.inner.prototype
        feature = prototype.prototype_feature_lifecycle
        if feature is None:
            raise ValueError("coordinator Prototype feature lifecycle is required")
        if (
            self.coordinator.builder.observation_dim != _PHYSICAL_DIM + _CONTEXT_DIM
            or self.coordinator.builder.hidden_dim != _FAST_DIM
            or feature.base_feature_dim != _BASE_DIM
            or feature.active_pair_slots != _PAIR_SLOTS
            or feature.candidate_pair_slots != _PAIR_CANDIDATE_SLOTS
            or feature.pair_source_feature_dim != _PHYSICAL_DIM
            or prototype.oak.observation_dim != HCCL_FEATURE_TOTAL_DIM
        ):
            raise ValueError(
                "coordinator must implement canonical physical16/context3/fast4/pair12"
            )
        if (
            prototype.oak.n_primitive_actions != _N_ACTIONS
            or prototype.oak.stomp.n_total_actions != _N_ACTIONS
            or prototype.oak.n_options != 0
            or feature.n_primitive_actions != _N_ACTIONS
            or feature.n_options != 0
        ):
            raise ValueError("coordinator must be exact primitive-only A2 with no options")
        horde = prototype.horde_spec
        if horde is None or feature.managed_horde_demons != len(_HORDE_NAMES):
            raise ValueError("coordinator must own the exact canonical eight-head Horde")
        horde_questions = tuple(
            (
                demon.name,
                demon.demon_type.value,
                demon.gamma,
                demon.lamda,
                demon.cumulant_index,
                demon.terminal_reward,
            )
            for demon in horde.demons
        )
        expected_horde_questions = tuple(
            (name, "prediction", gamma, 0.0, index, 0.0)
            for index, (name, gamma) in enumerate(
                zip(_HORDE_NAMES, _HORDE_GAMMAS, strict=True)
            )
        )
        if (
            horde_questions != expected_horde_questions
            or tuple(horde.gammas.shape) != (len(_HORDE_NAMES),)
        ):
            raise ValueError("coordinator Horde question order is noncanonical")
        planner = PrototypeFactorizedPartnerPlannerV2(self.planner)
        planner_behavior = planner.behavior_model.config
        planner_grounded = planner.grounded_world_model.config
        if (
            planner_behavior.n_actions != _N_ACTIONS
            or planner_grounded.representation_dim != HCCL_FEATURE_TOTAL_DIM
            or planner_grounded.target_observation_dim != _PHYSICAL_DIM + 3
            or planner_grounded.n_focal_actions != _N_ACTIONS
            or planner_grounded.n_partner_actions != _N_ACTIONS
        ):
            raise ValueError("planner must implement canonical D23/R35/A2 geometry")
        context = self.context.context
        if (
            context.max_contexts != _CONTEXT_DIM
            or context.observation_dim != _N_ACTIONS
            or context.n_actions != _N_ACTIONS
        ):
            raise ValueError("context must implement canonical K3/D2/A2 geometry")
        world_lifetime = hccl_causal_core_lifetime_for_profile(
            self.hccl.world_config.schedule_profile
        )
        if self.hccl.world_config.maximum_committed_transitions != world_lifetime:
            raise ValueError("world lifetime must equal its selected schedule profile")
        required_lifetime = max(_CANONICAL_AGENT_LIFETIME, world_lifetime)
        if self.coordinator.max_events != required_lifetime:
            raise ValueError("coordinator max_events must close the complete world life")
        if self.coordinator.learning_value_router.max_steps != required_lifetime:
            raise ValueError("LearningValueRouter max_steps must close the complete world life")
        if self.coordinator.inner.ensemble.max_events != required_lifetime:
            raise ValueError("routed ensemble max_events must close the complete world life")
        if feature.max_observations != required_lifetime:
            raise ValueError(
                "feature lifecycle max_observations must close the complete world life"
            )
        for name, memory_config in (
            ("memory_agent_0", self.memory_agent_0),
            ("memory_agent_1", self.memory_agent_1),
        ):
            memory = memory_config.controller.memory
            if memory.max_age != required_lifetime:
                raise ValueError(f"{name} max_age must close the complete world life")
            if memory.staleness_scale != float(required_lifetime):
                raise ValueError(
                    f"{name} staleness_scale must equal the complete world life"
                )
        if type(self.binding_owner_digest) is not tuple or len(
            self.binding_owner_digest
        ) != _DIGEST_WORDS:
            raise ValueError("binding_owner_digest must be an exact eight-word tuple")
        for index, word in enumerate(self.binding_owner_digest):
            if type(word) is not int or not 0 <= word <= _UINT32_MAX:
                raise ValueError(f"binding_owner_digest[{index}] must be uint32")
        if not any(self.binding_owner_digest):
            raise ValueError("binding_owner_digest must be nonzero")
        owner_sets = (
            self.binding_owner_digest,
            self.bmp_agent_0.owner_digest,
            self.bmp_agent_1.owner_digest,
        )
        if len(set(owner_sets)) != len(owner_sets):
            raise ValueError("outer and per-agent BMP owners must be distinct")
        if not isinstance(self.discount, Real) or isinstance(self.discount, bool):
            raise ValueError("discount must be a real scalar")
        represented = float(np.float32(float(self.discount)))
        if not np.isfinite(represented) or not 0.0 < represented <= 1.0:
            raise ValueError("continuing discount must remain finite in (0, 1] in float32")
        object.__setattr__(self, "discount", represented)

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": self.SCHEMA_VERSION,
            "mechanism_status": HCCL_ROUTED_CONTINUAL_DYAD_STATUS,
            "scientific_promotion_allowed": False,
            "hccl": HCCLWorldAttributionAdapter(self.hccl).to_config(),
            "coordinator": self.coordinator.to_config(),
            "memory_agent_0": self.memory_agent_0.to_config(),
            "memory_agent_1": self.memory_agent_1.to_config(),
            "planner": self.planner.to_config(),
            "context": self.context.to_config(),
            "bmp_agent_0": self.bmp_agent_0.to_config(),
            "bmp_agent_1": self.bmp_agent_1.to_config(),
            "binding_owner_digest": list(self.binding_owner_digest),
            "discount": self.discount,
            "persistent_state_owners": {
                "hccl": 1,
                "coordinator": 2,
                "context_lineage": 2,
                "feature_birth_ledger": 2,
                "r35_memory": 2,
                "paired_planner_v2": 1,
                "bmp_action_record": 2,
            },
            "memory_order": "optional-settle-then-source-bank-step-then-rebind",
            "planner_order": "source-update-then-route-then-four-cell-plan",
            "world_lifetime_events": self.hccl.world_config.maximum_committed_transitions,
            "agent_lifetime_events": self.coordinator.max_events,
            "planner_geometry": {
                "stable_base_dim": _BASE_DIM,
                "routed_representation_dim": HCCL_FEATURE_TOTAL_DIM,
                "n_actions": _N_ACTIONS,
            },
            "context_geometry": {
                "max_contexts": _CONTEXT_DIM,
                "observation_dim": _N_ACTIONS,
                "n_actions": _N_ACTIONS,
            },
            "dynamic_primitive_safety_masks_supported": False,
            "primitive_masks_required_all_true": True,
            "caller_supplied_pair_admission": False,
            "dispatch_authority": False,
            "artifact_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLRoutedContinualDyadConfig:
        if type(payload) is not dict:
            raise TypeError("routed continual-dyad config must be an exact dict")
        for name in (
            "hccl",
            "coordinator",
            "memory_agent_0",
            "memory_agent_1",
            "planner",
            "context",
            "bmp_agent_0",
            "bmp_agent_1",
        ):
            if type(payload.get(name)) is not dict:
                raise ValueError(f"{name} must serialize as an exact dict")
        owner = payload.get("binding_owner_digest")
        if type(owner) is not list or not all(type(word) is int for word in owner):
            raise ValueError("binding_owner_digest must serialize as an integer list")
        discount = payload.get("discount")
        if type(discount) is not float:
            raise ValueError("discount must serialize as an exact float")
        candidate = cls(
            hccl=HCCLWorldAttributionAdapter.from_config(
                cast(dict[str, object], payload["hccl"])
            ).config,
            coordinator=ExternalLearnedStateRouterAuditCoordinatorConfig.from_config(
                cast(dict[str, object], payload["coordinator"])
            ),
            memory_agent_0=HCCLFeatureBoundMemoryConfig.from_config(
                cast(dict[str, object], payload["memory_agent_0"])
            ),
            memory_agent_1=HCCLFeatureBoundMemoryConfig.from_config(
                cast(dict[str, object], payload["memory_agent_1"])
            ),
            planner=PrototypeFactorizedPartnerPlannerV2Config.from_config(
                cast(dict[str, object], payload["planner"])
            ),
            context=ContextLineageRetentionSeamConfig.from_config(
                cast(dict[str, object], payload["context"])
            ),
            bmp_agent_0=HCCLAuthenticatedBMPProjectionConfig.from_config(
                cast(dict[str, object], payload["bmp_agent_0"])
            ),
            bmp_agent_1=HCCLAuthenticatedBMPProjectionConfig.from_config(
                cast(dict[str, object], payload["bmp_agent_1"])
            ),
            binding_owner_digest=tuple(cast(list[int], owner)),
            discount=discount,
        )
        if _canonical_json_bytes(candidate.to_config()) != _canonical_json_bytes(payload):
            raise ValueError("routed continual-dyad config is noncanonical")
        return candidate


@chex.dataclass(frozen=True)
class HCCLRoutedContinualDyadActionRecord:
    """Persistent prior-decision identity needed for B/M/P and settlement."""

    agent_index: Int[Array, ""]
    source_clock_words: UInt[Array, " 2"]
    coordinator_decision_id: UInt[Array, " 4"]
    ledger_content_token: UInt[Array, " 32"]
    memory_content_token: UInt[Array, " 32"]
    memory_transaction_words: UInt[Array, " 2"]
    memory_pending_available: Bool[Array, ""]
    retrieval_used_expected: Bool[Array, ""]
    planner_content_token: UInt[Array, " 32"]
    bmp_binding: HCCLAuthenticatedBMPActionBinding
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLRoutedContinualDyadState:
    """Single persistent owner for the complete routed dyad."""

    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]
    hccl_state: HCCLWorldAttributionAdapterState
    coordinator_0_state: ExternalLearnedStateRouterAuditCoordinatorState
    coordinator_1_state: ExternalLearnedStateRouterAuditCoordinatorState
    context_0_state: ContextLineageRetentionSeamState
    context_1_state: ContextLineageRetentionSeamState
    ledger_0: HCCLFeatureBirthLedger
    ledger_1: HCCLFeatureBirthLedger
    memory_0_state: HCCLFeatureBoundMemoryState
    memory_1_state: HCCLFeatureBoundMemoryState
    planner_state: PrototypeFactorizedPartnerPlannerV2State
    action_record_0: HCCLRoutedContinualDyadActionRecord
    action_record_1: HCCLRoutedContinualDyadActionRecord


@chex.dataclass(frozen=True)
class HCCLRoutedContinualDyadActionBundle:
    """Event-bound HCCL B/M/P receipts issued only from persistent records."""

    source_state_token: UInt[Array, " 32"]
    event_token: UInt[Array, " 32"]
    base_actions: Int[Array, " 2"]
    memory_actions_before_mask: Int[Array, " 2"]
    memory_actions: Int[Array, " 2"]
    planner_actions_before_mask: Int[Array, " 2"]
    final_actions: Int[Array, " 2"]
    hard_action_masks: Bool[Array, "2 2"]
    base: HCCLActionReceipt
    memory: HCCLActionReceipt
    planner: HCCLActionReceipt
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLRoutedLifecycleRouteProof:
    """Caller-free derivation of one pair admission mask."""

    source_pair_descriptors: Int[Array, "12 2"]
    destination_pair_descriptors: Int[Array, "12 2"]
    pair_admission_mask: Bool[Array, " 12"]
    selected_active_slot: Int[Array, ""]
    selected_candidate_slot: Int[Array, ""]
    lifecycle_committed: Bool[Array, ""]
    diagnostics_bound: Bool[Array, ""]
    descriptor_transition_valid: Bool[Array, ""]
    clock_transition_valid: Bool[Array, ""]
    proof_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLRoutedContinualDyadAgentPreparation:
    """Complete attempted child transaction for one agent."""

    agent_index: Int[Array, ""]
    context_preparation: ContextLineageRetentionPreparation
    context_result: ContextLineageRetentionStepResult
    transition: ExternalLearnedStateTransition
    coordinator_result: ExternalLearnedStateRouterAuditCoordinatorResult
    lifecycle_proof: HCCLRoutedLifecycleRouteProof
    route_result: HCCLFeatureConsumerRouteResult
    memory_feedback: LearnedExperientialMemoryFeedback | None
    memory_settle_result: HCCLFeatureBoundMemorySettleResult | None
    memory_query: Float[Array, " 35"]
    memory_entry: ExperientialMemoryEntry
    memory_step_result: HCCLFeatureBoundMemoryStepResult
    memory_rebind_result: HCCLFeatureBoundMemoryRebindResult
    memory_retrieval_categorical: Bool[Array, ""]
    memory_consumed: Bool[Array, ""]
    memory_proposed_action: Int[Array, ""]
    bmp_memory_projection: HCCLAuthenticatedBMPMemoryProjection
    bmp_prepared_projection: HCCLAuthenticatedBMPPreparedProjection
    bmp_integrity_receipt: HCCLAuthenticatedBMPProjectionIntegrityReceipt
    bmp_result: HCCLAuthenticatedBMPProjectionResult
    action_record: HCCLRoutedContinualDyadActionRecord
    child_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLRoutedContinualDyadWork:
    """Exact named logical call counts for one attempted recurring event."""

    hccl_stage_calls: Int[Array, ""]
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    context_steps: Int[Array, " 2"]
    coordinator_steps: Int[Array, " 2"]
    lifecycle_route_derivations: Int[Array, " 2"]
    memory_settlements: Int[Array, " 2"]
    memory_steps: Int[Array, " 2"]
    memory_rebinds: Int[Array, " 2"]
    planner_behavior_updates: Int[Array, ""]
    planner_grounded_updates: Int[Array, ""]
    planner_joint_cells: Int[Array, ""]
    bmp_memory_replacements: Int[Array, " 2"]
    bmp_planner_replacements: Int[Array, " 2"]
    outer_commit_decisions: Int[Array, ""]
    output_writes: Int[Array, ""]
    rng_draws_after_event: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLRoutedContinualDyadPreparedTransaction:
    """Complete transient candidate awaiting one outer integrity adoption."""

    source_state: HCCLRoutedContinualDyadState
    event: HCCLCausalCoreEventReceipt
    action_bundle: HCCLRoutedContinualDyadActionBundle
    hccl_result: HCCLWorldAttributionAdapterResult
    memory_credit_panel: HCCLMemoryCreditEstimandPanel
    planner_result: PrototypeFactorizedPartnerPlannerV2Result
    agent_0: HCCLRoutedContinualDyadAgentPreparation
    agent_1: HCCLRoutedContinualDyadAgentPreparation
    candidate_state: HCCLRoutedContinualDyadState
    next_hard_action_masks: Bool[Array, "2 2"]
    work: HCCLRoutedContinualDyadWork
    source_state_valid: Bool[Array, ""]
    event_valid: Bool[Array, ""]
    action_bundle_valid: Bool[Array, ""]
    hccl_valid: Bool[Array, ""]
    credit_valid: Bool[Array, ""]
    context_valid: Bool[Array, " 2"]
    coordinator_valid: Bool[Array, " 2"]
    lifecycle_route_valid: Bool[Array, " 2"]
    memory_valid: Bool[Array, " 2"]
    planner_valid: Bool[Array, ""]
    bmp_valid: Bool[Array, " 2"]
    candidate_state_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLRoutedContinualDyadIntegrityReceipt:
    """Unkeyed exact-content binding for one prepared transaction."""

    config_token: UInt[Array, " 32"]
    source_state_token: UInt[Array, " 32"]
    prepared_content_token: UInt[Array, " 32"]
    integrity_bound: Bool[Array, ""]
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLRoutedContinualDyadResult:
    """Selected outer state or complete source, plus the attempted audit."""

    state: HCCLRoutedContinualDyadState
    prepared: HCCLRoutedContinualDyadPreparedTransaction
    receipt: HCCLRoutedContinualDyadIntegrityReceipt
    source_state_matches: Bool[Array, ""]
    prepared_content_valid: Bool[Array, ""]
    receipt_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]


class HCCLRoutedContinualDyad:
    """Genesis and recurring all-or-none owner for the routed dyad."""

    def __init__(self, config: HCCLRoutedContinualDyadConfig) -> None:
        if type(config) is not HCCLRoutedContinualDyadConfig:
            raise TypeError("config must be an exact HCCLRoutedContinualDyadConfig")
        self._config = config
        self._hccl = HCCLWorldAttributionAdapter(config.hccl)
        self._coordinators = (
            ExternalLearnedStateRouterAuditCoordinator(config.coordinator),
            ExternalLearnedStateRouterAuditCoordinator(config.coordinator),
        )
        self._context = ContextLineageRetentionSeam(config.context)
        self._memories = (
            HCCLFeatureBoundMemory(config.memory_agent_0),
            HCCLFeatureBoundMemory(config.memory_agent_1),
        )
        self._routes = (
            HCCLFeatureConsumerRoute(agent_index=0),
            HCCLFeatureConsumerRoute(agent_index=1),
        )
        self._planner = PrototypeFactorizedPartnerPlannerV2(config.planner)
        self._bmps = (
            HCCLAuthenticatedBMPProjection(self._coordinators[0], config.bmp_agent_0),
            HCCLAuthenticatedBMPProjection(self._coordinators[1], config.bmp_agent_1),
        )
        self._owner_words = jnp.asarray(config.binding_owner_digest, dtype=jnp.uint32)
        self._config_token = _digest_bytes(
            HCCL_ROUTED_CONTINUAL_DYAD_CONFIG_SCHEMA,
            self._owner_words,
            jnp.asarray(tuple(_canonical_json_bytes(config.to_config())), dtype=jnp.uint8),
        )

    @property
    def config(self) -> HCCLRoutedContinualDyadConfig:
        return self._config

    @property
    def hccl(self) -> HCCLWorldAttributionAdapter:
        return self._hccl

    @property
    def coordinators(
        self,
    ) -> tuple[
        ExternalLearnedStateRouterAuditCoordinator,
        ExternalLearnedStateRouterAuditCoordinator,
    ]:
        return self._coordinators

    @property
    def context(self) -> ContextLineageRetentionSeam:
        return self._context

    @property
    def memories(self) -> tuple[HCCLFeatureBoundMemory, HCCLFeatureBoundMemory]:
        return self._memories

    @property
    def planner(self) -> PrototypeFactorizedPartnerPlannerV2:
        return self._planner

    @property
    def bmp_projections(
        self,
    ) -> tuple[HCCLAuthenticatedBMPProjection, HCCLAuthenticatedBMPProjection]:
        return self._bmps

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLRoutedContinualDyad:
        return cls(HCCLRoutedContinualDyadConfig.from_config(payload))

    @staticmethod
    def _coordinators_from_state(
        state: HCCLRoutedContinualDyadState,
    ) -> tuple[
        ExternalLearnedStateRouterAuditCoordinatorState,
        ExternalLearnedStateRouterAuditCoordinatorState,
    ]:
        return state.coordinator_0_state, state.coordinator_1_state

    @staticmethod
    def _contexts_from_state(
        state: HCCLRoutedContinualDyadState,
    ) -> tuple[ContextLineageRetentionSeamState, ContextLineageRetentionSeamState]:
        return state.context_0_state, state.context_1_state

    @staticmethod
    def _ledgers_from_state(
        state: HCCLRoutedContinualDyadState,
    ) -> tuple[HCCLFeatureBirthLedger, HCCLFeatureBirthLedger]:
        return state.ledger_0, state.ledger_1

    @staticmethod
    def _memories_from_state(
        state: HCCLRoutedContinualDyadState,
    ) -> tuple[HCCLFeatureBoundMemoryState, HCCLFeatureBoundMemoryState]:
        return state.memory_0_state, state.memory_1_state

    @staticmethod
    def _records_from_state(
        state: HCCLRoutedContinualDyadState,
    ) -> tuple[HCCLRoutedContinualDyadActionRecord, HCCLRoutedContinualDyadActionRecord]:
        return state.action_record_0, state.action_record_1

    @staticmethod
    def _planner_agents(
        state: PrototypeFactorizedPartnerPlannerV2State,
    ) -> tuple[Any, Any]:
        return state.agent_0, state.agent_1

    def _record_token(
        self,
        record: HCCLRoutedContinualDyadActionRecord,
    ) -> UInt[Array, " 32"]:
        bare = cast(
            HCCLRoutedContinualDyadActionRecord,
            record.replace(content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)),
        )
        return _digest_bytes(HCCL_ROUTED_CONTINUAL_DYAD_ACTION_RECORD_SCHEMA, bare)

    def _seal_record(
        self,
        record: HCCLRoutedContinualDyadActionRecord,
    ) -> HCCLRoutedContinualDyadActionRecord:
        return cast(
            HCCLRoutedContinualDyadActionRecord,
            record.replace(content_token=self._record_token(record)),
        )

    def _state_token(self, state: HCCLRoutedContinualDyadState) -> UInt[Array, " 32"]:
        bare = cast(
            HCCLRoutedContinualDyadState,
            state.replace(content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)),
        )
        return _digest_bytes(HCCL_ROUTED_CONTINUAL_DYAD_STATE_SCHEMA, bare)

    def _seal_state(self, state: HCCLRoutedContinualDyadState) -> HCCLRoutedContinualDyadState:
        return cast(
            HCCLRoutedContinualDyadState,
            state.replace(content_token=self._state_token(state)),
        )

    def _bundle_token(
        self,
        bundle: HCCLRoutedContinualDyadActionBundle,
    ) -> UInt[Array, " 32"]:
        bare = cast(
            HCCLRoutedContinualDyadActionBundle,
            bundle.replace(content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)),
        )
        return _digest_bytes(HCCL_ROUTED_CONTINUAL_DYAD_ACTION_BUNDLE_SCHEMA, bare)

    def _seal_bundle(
        self,
        bundle: HCCLRoutedContinualDyadActionBundle,
    ) -> HCCLRoutedContinualDyadActionBundle:
        return cast(
            HCCLRoutedContinualDyadActionBundle,
            bundle.replace(content_token=self._bundle_token(bundle)),
        )

    def _prepared_token(
        self,
        prepared: HCCLRoutedContinualDyadPreparedTransaction,
    ) -> UInt[Array, " 32"]:
        bare = cast(
            HCCLRoutedContinualDyadPreparedTransaction,
            prepared.replace(content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)),
        )
        return _digest_bytes(HCCL_ROUTED_CONTINUAL_DYAD_PREPARED_SCHEMA, bare)

    def _seal_prepared(
        self,
        prepared: HCCLRoutedContinualDyadPreparedTransaction,
    ) -> HCCLRoutedContinualDyadPreparedTransaction:
        return cast(
            HCCLRoutedContinualDyadPreparedTransaction,
            prepared.replace(content_token=self._prepared_token(prepared)),
        )

    @staticmethod
    def _receipt_token(
        receipt: HCCLRoutedContinualDyadIntegrityReceipt,
    ) -> UInt[Array, " 32"]:
        bare = cast(
            HCCLRoutedContinualDyadIntegrityReceipt,
            receipt.replace(content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)),
        )
        return _digest_bytes(HCCL_ROUTED_CONTINUAL_DYAD_RECEIPT_SCHEMA, bare)

    @staticmethod
    def _representation_valid(
        representation: Array,
        ledger: HCCLFeatureBirthLedger,
    ) -> bool:
        if tuple(representation.shape) != (HCCL_FEATURE_TOTAL_DIM,) or jnp.dtype(
            representation.dtype
        ) != jnp.dtype(jnp.float32):
            return False
        parents = ledger.parents[HCCL_FEATURE_PAIR_START:]
        safe_left = jnp.clip(parents[:, 0], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        safe_right = jnp.clip(parents[:, 1], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        physical = representation[:HCCL_FEATURE_PHYSICAL_DIM]
        expected = physical[safe_left] * physical[safe_right]
        active_pairs = ledger.active[HCCL_FEATURE_PAIR_START:]
        expected = jnp.where(active_pairs, expected, jnp.float32(0.0))
        bits = jax.lax.bitcast_convert_type(representation, jnp.uint32)
        expected_bits = jax.lax.bitcast_convert_type(expected, jnp.uint32)
        return _host_bool(
            jnp.all(jnp.isfinite(representation))
            & jnp.all(bits[HCCL_FEATURE_PAIR_START:] == expected_bits)
            & jnp.all(ledger.active | (bits == jnp.uint32(0)))
        )

    def _composed_observation(
        self,
        physical: Array,
        context_state: ContextLineageRetentionSeamState,
    ) -> Float[Array, " 19"]:
        value = _require_array(
            physical,
            name="physical_observation",
            shape=(_PHYSICAL_DIM,),
            dtype=jnp.float32,
        )
        return jnp.concatenate((value, self._context.context_coordinates(context_state))).astype(
            jnp.float32
        )

    def _make_action_record(
        self,
        *,
        agent_index: int,
        coordinator: ExternalLearnedStateRouterAuditCoordinatorState,
        ledger: HCCLFeatureBirthLedger,
        memory: HCCLFeatureBoundMemoryState,
        planner: PrototypeFactorizedPartnerPlannerV2State,
        binding: HCCLAuthenticatedBMPActionBinding,
    ) -> HCCLRoutedContinualDyadActionRecord:
        pending = memory.controller_state.pending.available
        retrieval_used = (
            pending
            & binding.memory_consumed
            & (binding.memory_action_before_mask == binding.memory_action)
        )
        bare = HCCLRoutedContinualDyadActionRecord(
            agent_index=jnp.asarray(agent_index, dtype=jnp.int32),
            source_clock_words=coordinator.event_words,
            coordinator_decision_id=coordinator.current_decision_id,
            ledger_content_token=ledger.content_token,
            memory_content_token=memory.content_token,
            memory_transaction_words=memory.controller_state.transaction_words,
            memory_pending_available=pending,
            retrieval_used_expected=retrieval_used,
            planner_content_token=planner.content_token,
            bmp_binding=binding,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return self._seal_record(bare)

    def _memory_owner_words(
        self,
        *,
        agent_index: int,
        memory: HCCLFeatureBoundMemoryState,
        ledger: HCCLFeatureBirthLedger,
        decision_id: Array,
        base_action: Array,
        proposed_action: Array,
        consumed: Array,
    ) -> UInt[Array, " 8"]:
        """Re-derivable typed memory-owner identity for one persisted M."""

        return _digest_words(
            "hccl-routed-persisted-memory-owner-v1",
            self._owner_words,
            jnp.asarray(agent_index, dtype=jnp.int32),
            memory,
            ledger,
            decision_id,
            base_action,
            proposed_action,
            consumed,
        )

    def _planner_owner_words(
        self,
        *,
        agent_index: int,
        planner: PrototypeFactorizedPartnerPlannerV2State,
        ledger: HCCLFeatureBirthLedger,
        decision_id: Array,
        proposed_action: Array,
        consumed: Array,
    ) -> UInt[Array, " 8"]:
        """Re-derivable agent-salted identity for one persisted paired P."""

        return _digest_words(
            "hccl-routed-persisted-planner-owner-v1",
            self._owner_words,
            jnp.asarray(agent_index, dtype=jnp.int32),
            planner,
            ledger,
            decision_id,
            proposed_action,
            consumed,
        )

    def _action_record_valid(
        self,
        record: HCCLRoutedContinualDyadActionRecord,
        *,
        agent_index: int,
        coordinator: ExternalLearnedStateRouterAuditCoordinatorState,
        ledger: HCCLFeatureBirthLedger,
        memory: HCCLFeatureBoundMemoryState,
        planner: PrototypeFactorizedPartnerPlannerV2State,
    ) -> bool:
        if type(record) is not HCCLRoutedContinualDyadActionRecord:
            return False
        try:
            binding = record.bmp_binding
            planner_agent = self._planner_agents(planner)[agent_index]
            pending = _host_bool(memory.controller_state.pending.available)
            consumed = _host_bool(binding.memory_consumed)
            expected_used = bool(
                pending
                and consumed
                and int(_host(binding.memory_action_before_mask))
                == int(_host(binding.memory_action))
            )
            actions = tuple(
                int(_host(getattr(binding, name)))
                for name in (
                    "base_action",
                    "memory_action_before_mask",
                    "memory_action",
                    "planner_action_before_mask",
                    "final_action",
                )
            )
            mask = _host(binding.hard_action_mask).astype(np.bool_)
            action_relations = bool(
                all(0 <= action < _N_ACTIONS for action in actions)
                and mask[actions[0]]
                and mask[actions[2]]
                and mask[actions[4]]
                and actions[2] in (actions[0], actions[1])
                and actions[4] in (actions[2], actions[3])
                and (consumed or actions[0] == actions[2])
                and _host_bool(binding.planner_consumed)
            )
            expected_memory_owner = self._memory_owner_words(
                agent_index=agent_index,
                memory=memory,
                ledger=ledger,
                decision_id=coordinator.current_decision_id,
                base_action=binding.base_action,
                proposed_action=binding.memory_action_before_mask,
                consumed=binding.memory_consumed,
            )
            expected_planner_owner = self._planner_owner_words(
                agent_index=agent_index,
                planner=planner,
                ledger=ledger,
                decision_id=coordinator.current_decision_id,
                proposed_action=binding.planner_action_before_mask,
                consumed=binding.planner_consumed,
            )
            return bool(
                int(_host(record.agent_index)) == agent_index
                and _array_exact_equal(record.source_clock_words, coordinator.event_words)
                and _array_exact_equal(
                    record.coordinator_decision_id,
                    coordinator.current_decision_id,
                )
                and _array_exact_equal(record.ledger_content_token, ledger.content_token)
                and _array_exact_equal(record.memory_content_token, memory.content_token)
                and _array_exact_equal(
                    record.memory_transaction_words,
                    memory.controller_state.transaction_words,
                )
                and _host_bool(record.memory_pending_available) == pending
                and (not consumed or pending)
                and _host_bool(record.retrieval_used_expected) == expected_used
                and _array_exact_equal(record.planner_content_token, planner.content_token)
                and _host_bool(self._bmps[agent_index].binding_valid(coordinator, binding))
                and _array_exact_equal(binding.decision_id, coordinator.current_decision_id)
                and int(_host(binding.final_action)) == int(_host(coordinator.current_action))
                and int(_host(binding.final_action))
                == int(_host(_prototype(coordinator).current_action))
                and int(_host(binding.planner_action_before_mask))
                == int(_host(planner_agent.cache.prepared_action))
                and _array_exact_equal(
                    binding.memory_external_owner_words,
                    expected_memory_owner,
                )
                and _array_exact_equal(
                    binding.planner_external_owner_words,
                    expected_planner_owner,
                )
                and action_relations
                and _array_exact_equal(record.content_token, self._record_token(record))
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def state_valid(self, state: HCCLRoutedContinualDyadState) -> Bool[Array, ""]:
        """Validate every owner, routed identity, action cache, and exact clock."""

        if type(state) is not HCCLRoutedContinualDyadState:
            raise TypeError("state must be an exact HCCLRoutedContinualDyadState")
        if _contains_tracer(state):
            raise TypeError("routed continual-dyad validity is host/eager-only")
        try:
            coordinators = self._coordinators_from_state(state)
            contexts = self._contexts_from_state(state)
            ledgers = self._ledgers_from_state(state)
            memories = self._memories_from_state(state)
            records = self._records_from_state(state)
            planner_agents = self._planner_agents(state.planner_state)
            prototypes = tuple(_prototype(item) for item in coordinators)
            features = tuple(_feature_state(item) for item in prototypes)
            physical = self._hccl.world.observe(state.hccl_state.world_state)
            clock = state.hccl_state.world_state.step_words
            common = bool(
                _array_exact_equal(state.config_token, self._config_token)
                and _array_exact_equal(state.content_token, self._state_token(state))
                and _host_bool(self._hccl.state_valid(state.hccl_state))
                and _host_bool(self._planner.state_valid(state.planner_state))
            )
            for index in range(_N_AGENTS):
                coordinator = coordinators[index]
                context = contexts[index]
                ledger = ledgers[index]
                memory = memories[index]
                planner_agent = planner_agents[index]
                prototype = prototypes[index]
                feature = features[index]
                expected_raw = self._composed_observation(physical[index], context)
                common = common and all(
                    (
                        _host_bool(self._coordinators[index].state_valid(coordinator)),
                        _host_bool(self._context.state_is_valid(context)),
                        _host_bool(self._routes[index].ledger_valid(ledger)),
                        _host_bool(self._memories[index].state_valid(memory, ledger)),
                        _tree_exact_equal(memory.feature_ledger, ledger),
                        _tree_exact_equal(planner_agent.ledger, ledger),
                        _array_exact_equal(
                            coordinator.current_raw_observation,
                            expected_raw,
                        ),
                        _array_exact_equal(
                            coordinator.current_representation,
                            prototype.current_raw_observation,
                        ),
                        self._representation_valid(
                            prototype.current_representation,
                            ledger,
                        ),
                        _array_exact_equal(
                            planner_agent.cache.representation,
                            prototype.current_representation,
                        ),
                        _array_exact_equal(
                            ledger.active[
                                HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
                            ],
                            context.context.in_use,
                        ),
                        _array_exact_equal(
                            ledger.birth_words[
                                HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
                            ],
                            context.slot_birth_words,
                        ),
                        _array_exact_equal(
                            ledger.descriptor[HCCL_FEATURE_PAIR_START:],
                            feature.router_state.descriptors,
                        ),
                        _array_exact_equal(coordinator.event_words, clock),
                        _array_exact_equal(context.context.step_words, clock),
                        _array_exact_equal(prototype.step_words, clock),
                        _array_exact_equal(feature.observe_words, clock),
                        _array_exact_equal(ledger.source_clock_words, clock),
                        _array_exact_equal(
                            memory.controller_state.transaction_words,
                            clock,
                        ),
                        _array_exact_equal(
                            memory.controller_state.memory.step_words,
                            clock,
                        ),
                        _array_exact_equal(planner_agent.behavior.step_words, clock),
                        _array_exact_equal(planner_agent.grounded.update_words, clock),
                        self._action_record_valid(
                            records[index],
                            agent_index=index,
                            coordinator=coordinator,
                            ledger=ledger,
                            memory=memory,
                            planner=state.planner_state,
                        ),
                    )
                )
            return jnp.asarray(common, dtype=jnp.bool_)
        except (AttributeError, IndexError, TypeError, ValueError):
            return jnp.asarray(False, dtype=jnp.bool_)

    @staticmethod
    def _hard_action_masks(value: object, *, name: str) -> Bool[Array, "2 2"]:
        masks = _require_array(
            value,
            name=name,
            shape=(_N_AGENTS, _N_ACTIONS),
            dtype=jnp.bool_,
        )
        if not _host_bool(jnp.all(jnp.any(masks, axis=1))):
            raise ValueError(f"{name} must admit at least one action per agent")
        if not _host_bool(jnp.all(masks)):
            raise ValueError(
                f"{name} must be all true until an authenticated primitive B-safety "
                "projection is owned"
            )
        return masks

    def init(
        self,
        key: Array,
        *,
        initial_hard_action_masks: Array | None = None,
    ) -> HCCLRoutedContinualDyadState:
        """Initialize every owner and persist the first planner-v2 P cache."""

        try:
            key_valid = bool(
                getattr(key, "shape", None) == ()
                and jax.dtypes.issubdtype(
                    getattr(key, "dtype", None),
                    jax.dtypes.prng_key,
                )
                and str(jr.key_impl(key)) == "threefry2x32"
            )
        except (TypeError, ValueError):
            key_valid = False
        if not key_valid:
            raise TypeError("key must be a scalar typed Threefry PRNG key")
        masks = self._hard_action_masks(
            jnp.ones((_N_AGENTS, _N_ACTIONS), dtype=jnp.bool_)
            if initial_hard_action_masks is None
            else initial_hard_action_masks,
            name="initial_hard_action_masks",
        )
        if _contains_tracer((key, masks)):
            raise TypeError("routed continual-dyad initialization is host/eager-only")

        hccl_key, coordinator_0_key, coordinator_1_key, planner_key = jr.split(key, 4)
        hccl_state = self._hccl.init(hccl_key)
        contexts = (self._context.init(), self._context.init())
        physical = self._hccl.world.observe(hccl_state.world_state)
        coordinator_sources = tuple(
            self._coordinators[index].start(
                self._coordinators[index].init(
                    (coordinator_0_key, coordinator_1_key)[index]
                ),
                self._composed_observation(physical[index], contexts[index]),
            )
            for index in range(_N_AGENTS)
        )
        source_prototypes = tuple(_prototype(item) for item in coordinator_sources)
        feature_states = tuple(_feature_state(item) for item in source_prototypes)
        ledgers = tuple(
            self._routes[index].init(
                context_active=contexts[index].context.in_use,
                pair_descriptors=_pair_descriptors(feature_states[index]),
            )
            for index in range(_N_AGENTS)
        )
        memories = tuple(
            self._memories[index].init(ledgers[index])
            for index in range(_N_AGENTS)
        )
        representations = jnp.stack(
            tuple(item.current_representation for item in source_prototypes)
        ).astype(jnp.float32)
        planner_state = self._planner.init(
            planner_key,
            ledger_agent_0=ledgers[0],
            ledger_agent_1=ledgers[1],
            representations=representations,
        )
        planner_agents = self._planner_agents(planner_state)
        final_coordinators: list[ExternalLearnedStateRouterAuditCoordinatorState] = []
        bindings: list[HCCLAuthenticatedBMPActionBinding] = []
        for index in range(_N_AGENTS):
            base_action = source_prototypes[index].current_action
            if not _host_bool(masks[index, base_action]):
                raise RuntimeError("coordinator genesis ignored its primitive action mask")
            memory_owner = self._memory_owner_words(
                agent_index=index,
                memory=memories[index],
                ledger=ledgers[index],
                decision_id=source_prototypes[index].current_decision_id,
                base_action=base_action,
                proposed_action=base_action,
                consumed=jnp.asarray(False, dtype=jnp.bool_),
            )
            memory_projection = self._bmps[index].prepare_memory(
                coordinator_sources[index],
                proposed_action=base_action,
                hard_action_mask=masks[index],
                consumed=jnp.asarray(False, dtype=jnp.bool_),
                external_owner_words=memory_owner,
            )
            planner_owner = self._planner_owner_words(
                agent_index=index,
                planner=planner_state,
                ledger=ledgers[index],
                decision_id=source_prototypes[index].current_decision_id,
                proposed_action=planner_agents[index].cache.prepared_action,
                consumed=jnp.asarray(True, dtype=jnp.bool_),
            )
            prepared = self._bmps[index].prepare_planner(
                memory_projection,
                proposed_action=planner_agents[index].cache.prepared_action,
                consumed=jnp.asarray(True, dtype=jnp.bool_),
                external_owner_words=planner_owner,
            )
            receipt = self._bmps[index].integrity_receipt(prepared)
            adopted = self._bmps[index].adopt(
                coordinator_sources[index],
                prepared,
                receipt,
            )
            if not _host_bool(adopted.update_applied):
                raise RuntimeError(f"agent {index} genesis B/M/P projection was rejected")
            final_coordinators.append(adopted.state)
            bindings.append(prepared.binding)
        records = tuple(
            self._make_action_record(
                agent_index=index,
                coordinator=final_coordinators[index],
                ledger=ledgers[index],
                memory=memories[index],
                planner=planner_state,
                binding=bindings[index],
            )
            for index in range(_N_AGENTS)
        )
        unsigned = HCCLRoutedContinualDyadState(
            config_token=self._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            hccl_state=hccl_state,
            coordinator_0_state=final_coordinators[0],
            coordinator_1_state=final_coordinators[1],
            context_0_state=contexts[0],
            context_1_state=contexts[1],
            ledger_0=ledgers[0],
            ledger_1=ledgers[1],
            memory_0_state=memories[0],
            memory_1_state=memories[1],
            planner_state=planner_state,
            action_record_0=records[0],
            action_record_1=records[1],
        )
        state = self._seal_state(unsigned)
        if not _host_bool(self.state_valid(state)):
            raise RuntimeError("routed continual-dyad genesis violates its state contract")
        return state

    def prepare_event(
        self,
        state: HCCLRoutedContinualDyadState,
    ) -> HCCLCausalCoreEventReceipt:
        """Prepare the one action-independent event for the exact source."""

        if not _host_bool(self.state_valid(state)):
            raise ValueError("event preparation requires a valid routed dyad state")
        return self._hccl.world.prepare_event(state.hccl_state.world_state)

    def _action_identity_rows(
        self,
        state: HCCLRoutedContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
        layer: HCCLActionLayer,
    ) -> UInt[Array, "2 4"]:
        records = self._records_from_state(state)
        return jnp.stack(
            tuple(
                _digest_words(
                    "hccl-routed-action-identity-v1",
                    self._owner_words,
                    state.content_token,
                    event.content_tag_words,
                    jnp.asarray(int(layer), dtype=jnp.int32),
                    jnp.asarray(index, dtype=jnp.int32),
                    records[index],
                )[:4]
                for index in range(_N_AGENTS)
            )
        ).astype(jnp.uint32)

    def _make_action_bundle(
        self,
        state: HCCLRoutedContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
    ) -> HCCLRoutedContinualDyadActionBundle:
        records = self._records_from_state(state)
        bindings = tuple(item.bmp_binding for item in records)
        base = jnp.stack(tuple(item.base_action for item in bindings)).astype(jnp.int32)
        memory_before = jnp.stack(
            tuple(item.memory_action_before_mask for item in bindings)
        ).astype(jnp.int32)
        memory = jnp.stack(tuple(item.memory_action for item in bindings)).astype(jnp.int32)
        planner_before = jnp.stack(
            tuple(item.planner_action_before_mask for item in bindings)
        ).astype(jnp.int32)
        final = jnp.stack(tuple(item.final_action for item in bindings)).astype(jnp.int32)
        masks = jnp.stack(tuple(item.hard_action_mask for item in bindings)).astype(jnp.bool_)
        receipts = tuple(
            self._hccl.bind_action_receipt(
                state.hccl_state,
                event,
                layer=layer,
                actions_before_mask=before,
                actions_after_mask=after,
                hard_action_masks=masks,
                action_receipt_identity_words=self._action_identity_rows(
                    state,
                    event,
                    layer,
                ),
            )
            for layer, before, after in (
                (HCCLActionLayer.BASE, base, base),
                (HCCLActionLayer.MEMORY, memory_before, memory),
                (HCCLActionLayer.PLANNER, planner_before, final),
            )
        )
        identities = np.reshape(
            np.stack(
                tuple(np.asarray(item.action_receipt_identity_words) for item in receipts)
            ),
            (_N_AGENTS * 3, 4),
        )
        if len({tuple(int(word) for word in row) for row in identities}) != 6:
            raise RuntimeError("B/M/P HCCL action identities must be pairwise distinct")
        bare = HCCLRoutedContinualDyadActionBundle(
            source_state_token=state.content_token,
            event_token=_digest_bytes("hccl-routed-event-v1", event),
            base_actions=base,
            memory_actions_before_mask=memory_before,
            memory_actions=memory,
            planner_actions_before_mask=planner_before,
            final_actions=final,
            hard_action_masks=masks,
            base=receipts[0],
            memory=receipts[1],
            planner=receipts[2],
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return self._seal_bundle(bare)

    def bind_actions(
        self,
        state: HCCLRoutedContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
    ) -> HCCLRoutedContinualDyadActionBundle:
        """Issue exact event-bound action receipts from persisted B/M/P records."""

        if not _host_bool(self.state_valid(state)):
            raise ValueError("action binding requires a valid routed dyad state")
        if not _host_bool(
            self._hccl.world.event_receipt_valid(state.hccl_state.world_state, event)
        ):
            raise ValueError("action binding requires the exact prepared event")
        return self._make_action_bundle(state, event)

    def action_bundle_valid(
        self,
        state: HCCLRoutedContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
        bundle: HCCLRoutedContinualDyadActionBundle,
    ) -> Bool[Array, ""]:
        """Reject a stale, foreign, replay-mixed, or mutated B/M/P bundle."""

        if type(bundle) is not HCCLRoutedContinualDyadActionBundle:
            raise TypeError("bundle must be an exact routed action bundle")
        if _contains_tracer((state, event, bundle)):
            raise TypeError("routed action-bundle validity is host/eager-only")
        try:
            expected = self._make_action_bundle(state, event)
            return jnp.asarray(
                _tree_exact_equal(bundle, expected)
                and _array_exact_equal(bundle.content_token, self._bundle_token(bundle)),
                dtype=jnp.bool_,
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return jnp.asarray(False, dtype=jnp.bool_)

    def _horde_targets(
        self,
        proposal: HCCLCausalCoreProposal,
        signals: HCCLCausalCoreTypedSignals,
        executed_actions: Array,
    ) -> tuple[Float[Array, "2 8"], Float[Array, "2 8"]]:
        next_positions = proposal.next_observation[:, 0]
        tv_occupancy = (next_positions < jnp.float32(-0.8)).astype(jnp.float32)
        target = jnp.float32(0.6) * proposal.current_hidden_sign
        target_occupancy = (
            jnp.abs(next_positions - target) <= jnp.float32(0.1)
        ).astype(jnp.float32)
        rows = tuple(
            jnp.asarray(
                (
                    signals.task_score,
                    signals.task_score,
                    signals.task_score,
                    executed_actions[1 - index].astype(jnp.float32),
                    signals.safety_cost[index],
                    tv_occupancy[index],
                    target_occupancy[index],
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
                dtype=jnp.float32,
            )
            for index in range(_N_AGENTS)
        )
        horde = self._config.coordinator.inner.prototype.horde_spec
        if horde is None or tuple(horde.gammas.shape) != (8,):
            raise RuntimeError("routed HCCL dyad requires the canonical eight-head Horde")
        discounts = jnp.broadcast_to(horde.gammas, (_N_AGENTS, 8)).astype(jnp.float32)
        return jnp.stack(rows).astype(jnp.float32), discounts

    @staticmethod
    def _transition(
        state: ExternalLearnedStateRouterAuditCoordinatorState,
        *,
        executed_action: Array,
        reward: Array,
        discount: Array,
        next_observation: Array,
        horde_cumulants: Array,
        horde_discounts: Array,
    ) -> ExternalLearnedStateTransition:
        return ExternalLearnedStateTransition(
            source_event_words=state.event_words,
            source_builder_step_words=state.cached_builder_step_words,
            source_prototype_step_words=state.cached_prototype_step_words,
            source_feature_generation_words=state.cached_feature_generation_words,
            observation=state.current_raw_observation,
            representation=state.current_representation,
            action=jnp.asarray(executed_action, dtype=jnp.int32),
            decision_id=state.current_decision_id,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            next_observation=jnp.asarray(next_observation, dtype=jnp.float32),
            next_decision_observation=jnp.asarray(next_observation, dtype=jnp.float32),
            horde_cumulants=horde_cumulants,
            horde_discounts=horde_discounts,
        )

    @staticmethod
    def _coordinator_prototype_result(
        result: ExternalLearnedStateRouterAuditCoordinatorResult,
    ) -> PrototypeUpdateResult:
        return result.evaluated.prepared.inner_result.prototype_result

    def _lifecycle_route_proof(
        self,
        *,
        source_coordinator: ExternalLearnedStateRouterAuditCoordinatorState,
        coordinator_result: ExternalLearnedStateRouterAuditCoordinatorResult,
        source_ledger: HCCLFeatureBirthLedger,
    ) -> HCCLRoutedLifecycleRouteProof:
        source_prototype = _prototype(source_coordinator)
        destination_prototype = _prototype(coordinator_result.state)
        source_feature = _feature_state(source_prototype)
        destination_feature = _feature_state(destination_prototype)
        prototype_result = self._coordinator_prototype_result(coordinator_result)
        integration = prototype_result.prototype_feature_lifecycle_diagnostics
        source_descriptors = _pair_descriptors(source_feature)
        destination_descriptors = _pair_descriptors(destination_feature)
        zero_mask = jnp.zeros((_PAIR_SLOTS,), dtype=jnp.bool_)
        if integration is None:
            return HCCLRoutedLifecycleRouteProof(
                source_pair_descriptors=source_descriptors,
                destination_pair_descriptors=destination_descriptors,
                pair_admission_mask=zero_mask,
                selected_active_slot=jnp.asarray(-1, dtype=jnp.int32),
                selected_candidate_slot=jnp.asarray(-1, dtype=jnp.int32),
                lifecycle_committed=jnp.asarray(False, dtype=jnp.bool_),
                diagnostics_bound=jnp.asarray(False, dtype=jnp.bool_),
                descriptor_transition_valid=jnp.asarray(False, dtype=jnp.bool_),
                clock_transition_valid=jnp.asarray(False, dtype=jnp.bool_),
                proof_valid=jnp.asarray(False, dtype=jnp.bool_),
            )
        lifecycle = integration.lifecycle
        selected_active = int(_host(lifecycle.curation_selected_active_worst_slot))
        selected_candidate = int(
            _host(lifecycle.curation_selected_promotion_candidate)
        )
        committed = _host_bool(lifecycle.curation_committed)
        admission = zero_mask
        if committed and 0 <= selected_active < _PAIR_SLOTS:
            admission = admission.at[selected_active].set(True)
        diagnostics_bound = bool(
            _host_bool(coordinator_result.diagnostics.transaction_applied)
            and _host_bool(
                coordinator_result.evaluated.prepared.inner_result.diagnostics.transaction_applied
            )
            and _host_bool(prototype_result.transition_diagnostics.valid)
            and _host_bool(integration.available)
            and _host_bool(integration.outer_transaction_committed)
            and _host_bool(lifecycle.transaction_applied)
            and _tree_exact_equal(prototype_result.state, destination_prototype)
            and _array_exact_equal(
                source_ledger.descriptor[HCCL_FEATURE_PAIR_START:],
                source_descriptors,
            )
            and _array_exact_equal(
                lifecycle.semantic_generation_words_before,
                source_feature.router_state.generation_words,
            )
            and _array_exact_equal(
                lifecycle.semantic_generation_words_after,
                destination_feature.router_state.generation_words,
            )
            and _array_exact_equal(
                lifecycle.observe_words_before,
                source_feature.observe_words,
            )
            and _array_exact_equal(
                lifecycle.observe_words_after,
                destination_feature.observe_words,
            )
        )
        clock_valid = bool(
            _checked_successor(source_feature.observe_words, destination_feature.observe_words)
            and _array_exact_equal(
                destination_feature.observe_words,
                coordinator_result.state.event_words,
            )
        )
        if committed:
            indexes_valid = bool(
                0 <= selected_active < _PAIR_SLOTS
                and 0
                <= selected_candidate
                < int(source_feature.learner_state.candidate_left.shape[0])
            )
            if indexes_valid:
                candidate_descriptor = np.asarray(
                    _host(_candidate_descriptors(source_feature)[selected_candidate]),
                    dtype=np.int32,
                )
                expected = np.asarray(_host(source_descriptors), dtype=np.int32).copy()
                expected[selected_active] = candidate_descriptor
                changed = np.any(
                    np.asarray(_host(source_descriptors))
                    != np.asarray(_host(destination_descriptors)),
                    axis=1,
                )
                exact_selected_change = bool(
                    np.array_equal(
                        changed,
                        np.arange(_PAIR_SLOTS, dtype=np.int32) == selected_active,
                    )
                )
                expected_matches = np.array_equal(
                    np.asarray(_host(destination_descriptors)),
                    expected,
                )
                candidate_was_not_live = not bool(
                    np.any(
                        np.all(
                            np.asarray(_host(source_descriptors), dtype=np.int32)
                            == candidate_descriptor,
                            axis=1,
                        )
                    )
                )
            else:
                exact_selected_change = False
                expected_matches = False
                candidate_was_not_live = False
            descriptor_valid = bool(
                indexes_valid
                and exact_selected_change
                and expected_matches
                and candidate_was_not_live
                and _host_bool(lifecycle.routing_attempted)
                and _host_bool(lifecycle.input_route_valid)
                and _host_bool(lifecycle.output_route_valid)
                and _host_bool(lifecycle.route_states_match)
                and _host_bool(lifecycle.routed_values_finite)
                and _host_bool(lifecycle.postcondition_checked)
                and _host_bool(lifecycle.postcondition_valid)
                and not _host_bool(lifecycle.curation_rolled_back)
                and not _host_bool(lifecycle.postcondition_rolled_back)
                and _checked_successor(
                    source_feature.router_state.generation_words,
                    destination_feature.router_state.generation_words,
                )
            )
        else:
            descriptor_valid = bool(
                _array_exact_equal(source_descriptors, destination_descriptors)
                and _array_exact_equal(
                    source_feature.router_state.generation_words,
                    destination_feature.router_state.generation_words,
                )
                and not _host_bool(jnp.any(admission))
            )
        valid = diagnostics_bound and descriptor_valid and clock_valid
        return HCCLRoutedLifecycleRouteProof(
            source_pair_descriptors=source_descriptors,
            destination_pair_descriptors=destination_descriptors,
            pair_admission_mask=admission,
            selected_active_slot=lifecycle.curation_selected_active_worst_slot,
            selected_candidate_slot=lifecycle.curation_selected_promotion_candidate,
            lifecycle_committed=lifecycle.curation_committed,
            diagnostics_bound=jnp.asarray(diagnostics_bound, dtype=jnp.bool_),
            descriptor_transition_valid=jnp.asarray(
                descriptor_valid,
                dtype=jnp.bool_,
            ),
            clock_transition_valid=jnp.asarray(clock_valid, dtype=jnp.bool_),
            proof_valid=jnp.asarray(valid, dtype=jnp.bool_),
        )

    @staticmethod
    def _encode_source_pairs(
        base: Array,
        source_ledger: HCCLFeatureBirthLedger,
    ) -> Float[Array, " 35"]:
        parents = source_ledger.parents[HCCL_FEATURE_PAIR_START:]
        safe_left = jnp.clip(parents[:, 0], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        safe_right = jnp.clip(parents[:, 1], 0, HCCL_FEATURE_PHYSICAL_DIM - 1)
        physical = base[:HCCL_FEATURE_PHYSICAL_DIM]
        products = physical[safe_left] * physical[safe_right]
        products = jnp.where(
            source_ledger.active[HCCL_FEATURE_PAIR_START:],
            products,
            jnp.float32(0.0),
        )
        return jnp.concatenate((base, products)).astype(jnp.float32)

    def _source_bank_bootstrap(
        self,
        prototype_result: PrototypeUpdateResult,
        source_ledger: HCCLFeatureBirthLedger,
        route_result: HCCLFeatureConsumerRouteResult,
    ) -> Float[Array, " 35"]:
        raw = prototype_result.oak_bootstrap_observation
        base = raw[:HCCL_FEATURE_PAIR_START]
        context_survivor = route_result.witness.route_map.survivor_mask[
            HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        context = jnp.where(
            context_survivor,
            base[HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START],
            jnp.float32(0.0),
        )
        projected_base = jnp.concatenate(
            (
                base[:HCCL_FEATURE_CONTEXT_START],
                context,
                base[HCCL_FEATURE_FAST_START:HCCL_FEATURE_PAIR_START],
            )
        ).astype(jnp.float32)
        return self._encode_source_pairs(projected_base, source_ledger)

    def _canonical_source_representation(
        self,
        representation: Array,
        source_ledger: HCCLFeatureBirthLedger,
    ) -> Float[Array, " 35"]:
        base = representation[:HCCL_FEATURE_PAIR_START]
        active_context = source_ledger.active[
            HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START
        ]
        context = jnp.where(
            active_context,
            base[HCCL_FEATURE_CONTEXT_START:HCCL_FEATURE_FAST_START],
            jnp.float32(0.0),
        )
        canonical_base = jnp.concatenate(
            (
                base[:HCCL_FEATURE_CONTEXT_START],
                context,
                base[HCCL_FEATURE_FAST_START:HCCL_FEATURE_PAIR_START],
            )
        ).astype(jnp.float32)
        return self._encode_source_pairs(canonical_base, source_ledger)

    @staticmethod
    def _event_provenance(event: HCCLCausalCoreEventReceipt, agent_index: int) -> Array:
        words = _host(event.source_step_words).astype(np.uint64)
        step = (int(words[0]) << 32) | int(words[1])
        if step > (_INT32_MAX - (_N_AGENTS - 1)) // _N_AGENTS:
            raise ValueError("routed memory provenance exceeds int32 capacity")
        return jnp.asarray(_N_AGENTS * step + agent_index, dtype=jnp.int32)

    def _memory_entry(
        self,
        *,
        agent_index: int,
        source_representation: Array,
        outcome_representation: Array,
        executed_action: Array,
        signals: HCCLCausalCoreTypedSignals,
        event: HCCLCausalCoreEventReceipt,
    ) -> ExperientialMemoryEntry:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        false = jnp.asarray(False, dtype=jnp.bool_)
        return ExperientialMemoryEntry(
            observation=jnp.asarray(source_representation, dtype=jnp.float32),
            key=jnp.asarray(source_representation, dtype=jnp.float32),
            action=jax.nn.one_hot(executed_action, _N_ACTIONS, dtype=jnp.float32),
            outcome=jnp.asarray(outcome_representation, dtype=jnp.float32),
            reward=jnp.asarray(signals.net_reward[agent_index], dtype=jnp.float32),
            uncertainty=zero,
            uncertainty_available=false,
            safety_cost=jnp.asarray(signals.safety_cost[agent_index], dtype=jnp.float32),
            safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
            reliability=jnp.asarray(1.0, dtype=jnp.float32),
            utility=zero,
            utility_available=false,
            representation_version=jnp.asarray(0, dtype=jnp.int32),
            valid=jnp.asarray(True, dtype=jnp.bool_),
            age=jnp.asarray(0, dtype=jnp.int32),
            provenance_id=self._event_provenance(event, agent_index),
            source_id=jnp.asarray(agent_index, dtype=jnp.int32),
        )

    @staticmethod
    def _categorical_retrieval_action(result: HCCLFeatureBoundMemoryStepResult) -> tuple[bool, int]:
        retrieval = result.controller_result.retrieval
        action = np.asarray(_host(retrieval.action), dtype=np.float32)
        categorical = bool(
            np.array_equal(action.view(np.uint32), np.asarray((0x3F800000, 0), np.uint32))
            or np.array_equal(action.view(np.uint32), np.asarray((0, 0x3F800000), np.uint32))
        )
        return categorical, int(np.argmax(action)) if categorical else 0

    def prepare_transaction(
        self,
        state: HCCLRoutedContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
        action_bundle: HCCLRoutedContinualDyadActionBundle,
        *,
        next_hard_action_masks: Array,
    ) -> HCCLRoutedContinualDyadPreparedTransaction:
        """Evaluate one complete event, retaining every child attempt for audit."""

        if type(state) is not HCCLRoutedContinualDyadState:
            raise TypeError("state must be an exact routed continual-dyad state")
        if type(event) is not HCCLCausalCoreEventReceipt:
            raise TypeError("event must be an exact HCCL causal-core event receipt")
        if type(action_bundle) is not HCCLRoutedContinualDyadActionBundle:
            raise TypeError("action_bundle must be an exact routed action bundle")
        masks = self._hard_action_masks(
            next_hard_action_masks,
            name="next_hard_action_masks",
        )
        if _contains_tracer((state, event, action_bundle, masks)):
            raise TypeError("routed continual-dyad preparation is host/eager-only")

        source_valid = _host_bool(self.state_valid(state))
        event_valid = _host_bool(
            self._hccl.world.event_receipt_valid(state.hccl_state.world_state, event)
        )
        bundle_valid = _host_bool(self.action_bundle_valid(state, event, action_bundle))
        source_coordinators = self._coordinators_from_state(state)
        source_contexts = self._contexts_from_state(state)
        source_ledgers = self._ledgers_from_state(state)
        source_memories = self._memories_from_state(state)
        source_records = self._records_from_state(state)

        context_preparations = tuple(
            self._context.prepare(
                source_contexts[index],
                jax.nn.one_hot(
                    action_bundle.final_actions[1 - index],
                    _N_ACTIONS,
                    dtype=jnp.float32,
                ),
                action_bundle.final_actions[index],
            )
            for index in range(_N_AGENTS)
        )
        hccl_result = self._hccl.stage(
            state.hccl_state,
            event,
            action_bundle.base,
            action_bundle.memory,
            action_bundle.planner,
            downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
        )
        pp_signals = _signals_at(hccl_result.world_proposals, _PP_SLOT)
        pp_proposal = _proposal_at(hccl_result.world_proposals, _PP_SLOT)
        credit_panel = derive_hccl_memory_credit_estimands(
            mm=_signals_at(hccl_result.world_proposals, _MM_SLOT),
            b0m1=_signals_at(hccl_result.world_proposals, _B0M1_SLOT),
            m0b1=_signals_at(hccl_result.world_proposals, _M0B1_SLOT),
            bb=_signals_at(hccl_result.world_proposals, _BB_SLOT),
        )
        memory_credits = (
            credit_panel.baseline_context_direct_effect.net_reward[0, 0],
            credit_panel.baseline_context_direct_effect.net_reward[1, 1],
        )
        context_results = tuple(
            self._context.step(
                source_contexts[index],
                context_preparations[index],
                pp_signals.task_score,
            )
            for index in range(_N_AGENTS)
        )
        next_raw = jnp.stack(
            tuple(
                self._composed_observation(
                    pp_proposal.next_observation[index],
                    context_results[index].state,
                )
                for index in range(_N_AGENTS)
            )
        ).astype(jnp.float32)
        horde_cumulants, horde_discounts = self._horde_targets(
            pp_proposal,
            pp_signals,
            action_bundle.final_actions,
        )
        discount = jnp.asarray(self._config.discount, dtype=jnp.float32)
        transitions = tuple(
            self._transition(
                source_coordinators[index],
                executed_action=action_bundle.final_actions[index],
                reward=pp_signals.net_reward[index],
                discount=discount,
                next_observation=next_raw[index],
                horde_cumulants=horde_cumulants[index],
                horde_discounts=horde_discounts[index],
            )
            for index in range(_N_AGENTS)
        )
        coordinator_results = tuple(
            self._coordinators[index].step(
                source_coordinators[index],
                transitions[index],
            )
            for index in range(_N_AGENTS)
        )
        lifecycle_proofs = tuple(
            self._lifecycle_route_proof(
                source_coordinator=source_coordinators[index],
                coordinator_result=coordinator_results[index],
                source_ledger=source_ledgers[index],
            )
            for index in range(_N_AGENTS)
        )
        route_results = tuple(
            self._routes[index].prepare_successor(
                source_ledgers[index],
                destination_source_clock_words=coordinator_results[index].state.event_words,
                context_active=context_results[index].state.context.in_use,
                context_birth_words=context_results[index].state.slot_birth_words,
                pair_descriptors=lifecycle_proofs[index].destination_pair_descriptors,
                pair_admission_mask=lifecycle_proofs[index].pair_admission_mask,
            )
            for index in range(_N_AGENTS)
        )

        feedbacks: list[LearnedExperientialMemoryFeedback | None] = []
        settle_results: list[HCCLFeatureBoundMemorySettleResult | None] = []
        memory_queries: list[Array] = []
        memory_entries: list[ExperientialMemoryEntry] = []
        memory_steps: list[HCCLFeatureBoundMemoryStepResult] = []
        memory_rebinds: list[HCCLFeatureBoundMemoryRebindResult] = []
        memory_categorical: list[bool] = []
        memory_consumed: list[bool] = []
        memory_actions: list[Array] = []
        for index in range(_N_AGENTS):
            source_memory = source_memories[index]
            record = source_records[index]
            if _host_bool(source_memory.controller_state.pending.available):
                used = record.retrieval_used_expected
                feedback = LearnedExperientialMemoryFeedback(
                    transaction_words=source_memory.controller_state.pending.transaction_words,
                    retrieval_used=used,
                    counterfactual_available=used,
                    counterfactual_delta=jnp.where(
                        used,
                        jnp.asarray(memory_credits[index], dtype=jnp.float32),
                        jnp.asarray(0.0, dtype=jnp.float32),
                    ),
                )
                settle = self._memories[index].settle(source_memory, feedback)
                step_source = settle.state
            else:
                feedback = None
                settle = None
                step_source = source_memory
            prototype_result = self._coordinator_prototype_result(
                coordinator_results[index]
            )
            query = self._source_bank_bootstrap(
                prototype_result,
                source_ledgers[index],
                route_results[index],
            )
            source_representation = self._canonical_source_representation(
                _prototype(source_coordinators[index]).current_representation,
                source_ledgers[index],
            )
            entry = self._memory_entry(
                agent_index=index,
                source_representation=source_representation,
                outcome_representation=query,
                executed_action=action_bundle.final_actions[index],
                signals=pp_signals,
                event=event,
            )
            step_result = self._memories[index].step(
                step_source,
                query,
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(False, dtype=jnp.bool_),
                entry,
            )
            rebind = self._memories[index].rebind(
                step_result.state,
                source_ledgers[index],
                route_results[index],
            )
            categorical, retrieved_action = self._categorical_retrieval_action(step_result)
            admitted = bool(
                _host_bool(step_result.diagnostics.transaction_applied)
                and _host_bool(
                    step_result.controller_result.diagnostics.learned_retrieval_admitted
                )
                and _host_bool(step_result.controller_result.retrieval.accepted)
            )
            consumed = admitted and categorical
            proposed = (
                jnp.asarray(retrieved_action, dtype=jnp.int32)
                if consumed
                else coordinator_results[index].state.current_action
            )
            feedbacks.append(feedback)
            settle_results.append(settle)
            memory_queries.append(query)
            memory_entries.append(entry)
            memory_steps.append(step_result)
            memory_rebinds.append(rebind)
            memory_categorical.append(categorical)
            memory_consumed.append(consumed)
            memory_actions.append(proposed)

        source_representations = jnp.stack(
            tuple(
                _prototype(source_coordinators[index]).current_representation
                for index in range(_N_AGENTS)
            )
        ).astype(jnp.float32)
        destination_representations = jnp.stack(
            tuple(
                _prototype(coordinator_results[index].state).current_representation
                for index in range(_N_AGENTS)
            )
        ).astype(jnp.float32)
        planner_result = self._planner.observe_route_and_plan(
            state.planner_state,
            route_result_agent_0=route_results[0],
            route_result_agent_1=route_results[1],
            source_representations=source_representations,
            destination_representations=destination_representations,
            executed_actions=action_bundle.final_actions,
            next_physical_observations=pp_proposal.next_observation,
            task_score=pp_signals.task_score,
            safety_costs=pp_signals.safety_cost,
            message_charges=pp_signals.message_charge,
            net_rewards=pp_signals.net_reward,
            discount=discount,
        )

        bmp_memories: list[HCCLAuthenticatedBMPMemoryProjection] = []
        bmp_prepared: list[HCCLAuthenticatedBMPPreparedProjection] = []
        bmp_receipts: list[HCCLAuthenticatedBMPProjectionIntegrityReceipt] = []
        bmp_results: list[HCCLAuthenticatedBMPProjectionResult] = []
        action_records: list[HCCLRoutedContinualDyadActionRecord] = []
        for index in range(_N_AGENTS):
            coordinator_base = coordinator_results[index].state
            memory_owner = self._memory_owner_words(
                agent_index=index,
                memory=memory_rebinds[index].state,
                ledger=route_results[index].ledger,
                decision_id=coordinator_base.current_decision_id,
                base_action=coordinator_base.current_action,
                proposed_action=memory_actions[index],
                consumed=jnp.asarray(memory_consumed[index], dtype=jnp.bool_),
            )
            memory_projection = self._bmps[index].prepare_memory(
                coordinator_base,
                proposed_action=memory_actions[index],
                hard_action_mask=masks[index],
                consumed=jnp.asarray(memory_consumed[index], dtype=jnp.bool_),
                external_owner_words=memory_owner,
            )
            planner_action = planner_result.prepared_actions[index]
            planner_owner = self._planner_owner_words(
                agent_index=index,
                planner=planner_result.state,
                ledger=route_results[index].ledger,
                decision_id=coordinator_base.current_decision_id,
                proposed_action=planner_action,
                consumed=jnp.asarray(True, dtype=jnp.bool_),
            )
            prepared_projection = self._bmps[index].prepare_planner(
                memory_projection,
                proposed_action=planner_action,
                consumed=jnp.asarray(True, dtype=jnp.bool_),
                external_owner_words=planner_owner,
            )
            projection_receipt = self._bmps[index].integrity_receipt(
                prepared_projection
            )
            projection_result = self._bmps[index].adopt(
                coordinator_base,
                prepared_projection,
                projection_receipt,
            )
            action_record = self._make_action_record(
                agent_index=index,
                coordinator=projection_result.state,
                ledger=route_results[index].ledger,
                memory=memory_rebinds[index].state,
                planner=planner_result.state,
                binding=prepared_projection.binding,
            )
            bmp_memories.append(memory_projection)
            bmp_prepared.append(prepared_projection)
            bmp_receipts.append(projection_receipt)
            bmp_results.append(projection_result)
            action_records.append(action_record)

        context_flags = tuple(
            bool(
                _host_bool(context_results[index].update_applied)
                and _host_bool(self._context.state_is_valid(context_results[index].state))
                and _array_exact_equal(
                    context_results[index].state.context.step_words,
                    hccl_result.post_transaction_words,
                )
                and _tree_exact_equal(
                    context_results[index].preparation,
                    context_preparations[index],
                )
            )
            for index in range(_N_AGENTS)
        )
        coordinator_flags = tuple(
            bool(
                _host_bool(coordinator_results[index].diagnostics.transaction_applied)
                and _array_exact_equal(
                    coordinator_results[index].state.event_words,
                    hccl_result.post_transaction_words,
                )
                and int(_host(transitions[index].action))
                == int(_host(action_bundle.final_actions[index]))
            )
            for index in range(_N_AGENTS)
        )
        lifecycle_route_flags = tuple(
            bool(
                _host_bool(lifecycle_proofs[index].proof_valid)
                and _host_bool(
                    self._routes[index].result_integrity_valid(
                        source_ledgers[index],
                        route_results[index],
                    )
                )
                and _host_bool(route_results[index].witness.transaction_applied)
                and _array_exact_equal(
                    route_results[index].ledger.source_clock_words,
                    hccl_result.post_transaction_words,
                )
                and _array_exact_equal(
                    route_results[index].witness.requested_pair_admission_mask,
                    lifecycle_proofs[index].pair_admission_mask,
                )
                and _array_exact_equal(
                    route_results[index].witness.requested_pair_descriptors,
                    lifecycle_proofs[index].destination_pair_descriptors,
                )
                and _array_exact_equal(
                    route_results[index].witness.requested_context_active,
                    context_results[index].state.context.in_use,
                )
                and _array_exact_equal(
                    route_results[index].witness.requested_context_birth_words,
                    context_results[index].state.slot_birth_words,
                )
            )
            for index in range(_N_AGENTS)
        )
        memory_flags = tuple(
            bool(
                (
                    _host_bool(
                        cast(HCCLFeatureBoundMemorySettleResult, settle_results[index])
                        .diagnostics.transaction_applied
                    )
                    if settle_results[index] is not None
                    else not _host_bool(source_memories[index].controller_state.pending.available)
                )
                and _host_bool(memory_steps[index].diagnostics.transaction_applied)
                and _host_bool(memory_rebinds[index].diagnostics.transaction_applied)
                and _host_bool(
                    self._memories[index].state_valid(
                        memory_rebinds[index].state,
                        route_results[index].ledger,
                    )
                )
                and _array_exact_equal(
                    memory_rebinds[index].state.controller_state.transaction_words,
                    hccl_result.post_transaction_words,
                )
                and _host_bool(
                    memory_rebinds[index].state.controller_state.pending.available
                )
                == _host_bool(
                    memory_steps[index].controller_result.diagnostics.learned_retrieval_admitted
                )
                and memory_consumed[index]
                == (
                    _host_bool(
                        memory_steps[index]
                        .controller_result.diagnostics.learned_retrieval_admitted
                    )
                    and memory_categorical[index]
                )
                and (not memory_consumed[index] or memory_categorical[index])
            )
            for index in range(_N_AGENTS)
        )
        planner_valid = _host_bool(planner_result.transaction_applied)
        bmp_flags = tuple(
            bool(
                _host_bool(bmp_results[index].update_applied)
                and _host_bool(
                    self._bmps[index].binding_valid(
                        bmp_results[index].state,
                        bmp_prepared[index].binding,
                    )
                )
                and int(_host(_prototype(bmp_results[index].state).current_action))
                == int(_host(bmp_prepared[index].binding.final_action))
            )
            for index in range(_N_AGENTS)
        )
        credit_valid = _host_bool(credit_panel.algebra.all_identities_hold)
        hccl_valid = bool(
            _host_bool(hccl_result.update_applied)
            and int(_host(hccl_result.work.world_proposal_calls)) == 8
            and int(_host(hccl_result.work.attribution_proposal_calls)) == 8
        )
        child_flags = tuple(
            bool(
                context_flags[index]
                and coordinator_flags[index]
                and lifecycle_route_flags[index]
                and memory_flags[index]
                and planner_valid
                and bmp_flags[index]
            )
            for index in range(_N_AGENTS)
        )
        agent_preparations = tuple(
            HCCLRoutedContinualDyadAgentPreparation(
                agent_index=jnp.asarray(index, dtype=jnp.int32),
                context_preparation=context_preparations[index],
                context_result=context_results[index],
                transition=transitions[index],
                coordinator_result=coordinator_results[index],
                lifecycle_proof=lifecycle_proofs[index],
                route_result=route_results[index],
                memory_feedback=feedbacks[index],
                memory_settle_result=settle_results[index],
                memory_query=jnp.asarray(memory_queries[index], dtype=jnp.float32),
                memory_entry=memory_entries[index],
                memory_step_result=memory_steps[index],
                memory_rebind_result=memory_rebinds[index],
                memory_retrieval_categorical=jnp.asarray(
                    memory_categorical[index],
                    dtype=jnp.bool_,
                ),
                memory_consumed=jnp.asarray(memory_consumed[index], dtype=jnp.bool_),
                memory_proposed_action=memory_actions[index],
                bmp_memory_projection=bmp_memories[index],
                bmp_prepared_projection=bmp_prepared[index],
                bmp_integrity_receipt=bmp_receipts[index],
                bmp_result=bmp_results[index],
                action_record=action_records[index],
                child_valid=jnp.asarray(child_flags[index], dtype=jnp.bool_),
            )
            for index in range(_N_AGENTS)
        )
        unsigned_candidate = HCCLRoutedContinualDyadState(
            config_token=state.config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            hccl_state=hccl_result.state,
            coordinator_0_state=bmp_results[0].state,
            coordinator_1_state=bmp_results[1].state,
            context_0_state=context_results[0].state,
            context_1_state=context_results[1].state,
            ledger_0=route_results[0].ledger,
            ledger_1=route_results[1].ledger,
            memory_0_state=memory_rebinds[0].state,
            memory_1_state=memory_rebinds[1].state,
            planner_state=planner_result.state,
            action_record_0=action_records[0],
            action_record_1=action_records[1],
        )
        candidate_state = self._seal_state(unsigned_candidate)
        candidate_valid = _host_bool(self.state_valid(candidate_state))
        preparation_valid = bool(
            source_valid
            and event_valid
            and bundle_valid
            and hccl_valid
            and credit_valid
            and all(child_flags)
            and candidate_valid
        )
        work = HCCLRoutedContinualDyadWork(
            hccl_stage_calls=jnp.asarray(1, dtype=jnp.int32),
            world_proposal_calls=hccl_result.work.world_proposal_calls,
            attribution_proposal_calls=hccl_result.work.attribution_proposal_calls,
            context_steps=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            coordinator_steps=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            lifecycle_route_derivations=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            memory_settlements=jnp.asarray(
                tuple(int(item is not None) for item in settle_results),
                dtype=jnp.int32,
            ),
            memory_steps=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            memory_rebinds=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            planner_behavior_updates=planner_result.work.behavior_model_updates,
            planner_grounded_updates=planner_result.work.grounded_model_updates,
            planner_joint_cells=planner_result.work.grounded_joint_cell_evaluations,
            bmp_memory_replacements=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            bmp_planner_replacements=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            outer_commit_decisions=jnp.asarray(1, dtype=jnp.int32),
            output_writes=jnp.asarray(0, dtype=jnp.int32),
            rng_draws_after_event=jnp.asarray(0, dtype=jnp.int32),
        )
        bare = HCCLRoutedContinualDyadPreparedTransaction(
            source_state=state,
            event=event,
            action_bundle=action_bundle,
            hccl_result=hccl_result,
            memory_credit_panel=credit_panel,
            planner_result=planner_result,
            agent_0=agent_preparations[0],
            agent_1=agent_preparations[1],
            candidate_state=candidate_state,
            next_hard_action_masks=masks,
            work=work,
            source_state_valid=jnp.asarray(source_valid, dtype=jnp.bool_),
            event_valid=jnp.asarray(event_valid, dtype=jnp.bool_),
            action_bundle_valid=jnp.asarray(bundle_valid, dtype=jnp.bool_),
            hccl_valid=jnp.asarray(hccl_valid, dtype=jnp.bool_),
            credit_valid=jnp.asarray(credit_valid, dtype=jnp.bool_),
            context_valid=jnp.asarray(context_flags, dtype=jnp.bool_),
            coordinator_valid=jnp.asarray(coordinator_flags, dtype=jnp.bool_),
            lifecycle_route_valid=jnp.asarray(
                lifecycle_route_flags,
                dtype=jnp.bool_,
            ),
            memory_valid=jnp.asarray(memory_flags, dtype=jnp.bool_),
            planner_valid=jnp.asarray(planner_valid, dtype=jnp.bool_),
            bmp_valid=jnp.asarray(bmp_flags, dtype=jnp.bool_),
            candidate_state_valid=jnp.asarray(candidate_valid, dtype=jnp.bool_),
            preparation_valid=jnp.asarray(preparation_valid, dtype=jnp.bool_),
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return self._seal_prepared(bare)

    def _prepared_agent_valid(
        self,
        prepared: HCCLRoutedContinualDyadPreparedTransaction,
        agent: HCCLRoutedContinualDyadAgentPreparation,
        *,
        index: int,
    ) -> bool:
        source = prepared.source_state
        source_coordinator = self._coordinators_from_state(source)[index]
        source_context = self._contexts_from_state(source)[index]
        source_ledger = self._ledgers_from_state(source)[index]
        source_memory = self._memories_from_state(source)[index]
        source_record = self._records_from_state(source)[index]
        pp_signals = _signals_at(prepared.hccl_result.world_proposals, _PP_SLOT)
        pp_proposal = _proposal_at(prepared.hccl_result.world_proposals, _PP_SLOT)
        expected_context_preparation = self._context.prepare(
            source_context,
            jax.nn.one_hot(
                prepared.action_bundle.final_actions[1 - index],
                _N_ACTIONS,
                dtype=jnp.float32,
            ),
            prepared.action_bundle.final_actions[index],
        )
        expected_next_raw = self._composed_observation(
            pp_proposal.next_observation[index],
            agent.context_result.state,
        )
        horde_cumulants, horde_discounts = self._horde_targets(
            pp_proposal,
            pp_signals,
            prepared.action_bundle.final_actions,
        )
        expected_transition = self._transition(
            source_coordinator,
            executed_action=prepared.action_bundle.final_actions[index],
            reward=pp_signals.net_reward[index],
            discount=jnp.asarray(self._config.discount, dtype=jnp.float32),
            next_observation=expected_next_raw,
            horde_cumulants=horde_cumulants[index],
            horde_discounts=horde_discounts[index],
        )
        expected_proof = self._lifecycle_route_proof(
            source_coordinator=source_coordinator,
            coordinator_result=agent.coordinator_result,
            source_ledger=source_ledger,
        )
        route = agent.route_result
        expected_query = self._source_bank_bootstrap(
            self._coordinator_prototype_result(agent.coordinator_result),
            source_ledger,
            route,
        )
        expected_source_representation = self._canonical_source_representation(
            _prototype(source_coordinator).current_representation,
            source_ledger,
        )
        expected_entry = self._memory_entry(
            agent_index=index,
            source_representation=expected_source_representation,
            outcome_representation=expected_query,
            executed_action=prepared.action_bundle.final_actions[index],
            signals=pp_signals,
            event=prepared.event,
        )
        pending = _host_bool(source_memory.controller_state.pending.available)
        if pending:
            expected_feedback = LearnedExperientialMemoryFeedback(
                transaction_words=source_memory.controller_state.pending.transaction_words,
                retrieval_used=source_record.retrieval_used_expected,
                counterfactual_available=source_record.retrieval_used_expected,
                counterfactual_delta=jnp.where(
                    source_record.retrieval_used_expected,
                    prepared.memory_credit_panel.baseline_context_direct_effect.net_reward[
                        index,
                        index,
                    ],
                    jnp.asarray(0.0, dtype=jnp.float32),
                ),
            )
            feedback_valid = bool(
                agent.memory_feedback is not None
                and _tree_exact_equal(agent.memory_feedback, expected_feedback)
                and agent.memory_settle_result is not None
                and _host_bool(
                    agent.memory_settle_result.diagnostics.transaction_applied
                )
            )
        else:
            feedback_valid = bool(
                agent.memory_feedback is None and agent.memory_settle_result is None
            )
        categorical, retrieved_action = self._categorical_retrieval_action(
            agent.memory_step_result
        )
        admitted = bool(
            _host_bool(agent.memory_step_result.diagnostics.transaction_applied)
            and _host_bool(
                agent.memory_step_result.controller_result.diagnostics.learned_retrieval_admitted
            )
            and _host_bool(agent.memory_step_result.controller_result.retrieval.accepted)
        )
        consumed = admitted and categorical
        expected_memory_action = (
            retrieved_action
            if consumed
            else int(_host(agent.coordinator_result.state.current_action))
        )
        bmp_memory_owner = self._memory_owner_words(
            agent_index=index,
            memory=agent.memory_rebind_result.state,
            ledger=route.ledger,
            decision_id=agent.coordinator_result.state.current_decision_id,
            base_action=agent.coordinator_result.state.current_action,
            proposed_action=agent.memory_proposed_action,
            consumed=agent.memory_consumed,
        )
        bmp_planner_owner = self._planner_owner_words(
            agent_index=index,
            planner=prepared.planner_result.state,
            ledger=route.ledger,
            decision_id=agent.coordinator_result.state.current_decision_id,
            proposed_action=prepared.planner_result.prepared_actions[index],
            consumed=jnp.asarray(True, dtype=jnp.bool_),
        )
        expected_record = self._make_action_record(
            agent_index=index,
            coordinator=agent.bmp_result.state,
            ledger=route.ledger,
            memory=agent.memory_rebind_result.state,
            planner=prepared.planner_result.state,
            binding=agent.bmp_prepared_projection.binding,
        )
        return bool(
            int(_host(agent.agent_index)) == index
            and _tree_exact_equal(agent.context_preparation, expected_context_preparation)
            and _tree_exact_equal(
                agent.context_result.preparation,
                expected_context_preparation,
            )
            and _host_bool(agent.context_result.update_applied)
            and _tree_exact_equal(agent.transition, expected_transition)
            and _host_bool(agent.coordinator_result.diagnostics.transaction_applied)
            and _tree_exact_equal(agent.lifecycle_proof, expected_proof)
            and _host_bool(expected_proof.proof_valid)
            and _host_bool(
                self._routes[index].result_integrity_valid(source_ledger, route)
            )
            and _host_bool(route.witness.transaction_applied)
            and _array_exact_equal(
                route.witness.requested_pair_admission_mask,
                expected_proof.pair_admission_mask,
            )
            and _array_exact_equal(
                route.witness.requested_pair_descriptors,
                expected_proof.destination_pair_descriptors,
            )
            and _array_exact_equal(
                route.witness.destination_source_clock_words,
                agent.coordinator_result.state.event_words,
            )
            and _array_exact_equal(
                route.ledger.source_clock_words,
                agent.coordinator_result.state.event_words,
            )
            and _array_exact_equal(
                route.witness.requested_context_active,
                agent.context_result.state.context.in_use,
            )
            and _array_exact_equal(
                route.witness.requested_context_birth_words,
                agent.context_result.state.slot_birth_words,
            )
            and feedback_valid
            and _tree_exact_equal(agent.memory_query, expected_query)
            and _tree_exact_equal(agent.memory_entry, expected_entry)
            and self._representation_valid(agent.memory_query, source_ledger)
            and self._representation_valid(agent.memory_entry.observation, source_ledger)
            and _host_bool(agent.memory_step_result.diagnostics.transaction_applied)
            and _host_bool(agent.memory_rebind_result.diagnostics.transaction_applied)
            and _host_bool(
                self._memories[index].state_valid(
                    agent.memory_rebind_result.state,
                    route.ledger,
                )
            )
            and _host_bool(agent.memory_retrieval_categorical) == categorical
            and _host_bool(agent.memory_consumed) == consumed
            and int(_host(agent.memory_proposed_action)) == expected_memory_action
            and _tree_exact_equal(
                agent.bmp_memory_projection.source_coordinator_state,
                agent.coordinator_result.state,
            )
            and _array_exact_equal(
                agent.bmp_memory_projection.external_owner_words,
                bmp_memory_owner,
            )
            and _array_exact_equal(
                agent.bmp_memory_projection.hard_action_mask,
                prepared.next_hard_action_masks[index],
            )
            and _host_bool(
                self._bmps[index]._memory_projection_valid(
                    agent.bmp_memory_projection
                )
            )
            and _array_exact_equal(
                agent.bmp_prepared_projection.planner_external_owner_words,
                bmp_planner_owner,
            )
            and int(_host(agent.bmp_prepared_projection.planner_proposed_action))
            == int(_host(prepared.planner_result.prepared_actions[index]))
            and _host_bool(agent.bmp_prepared_projection.planner_consumed)
            and self._bmps[index]._prepared_valid(
                agent.coordinator_result.state,
                agent.bmp_prepared_projection,
            )
            and self._bmps[index]._receipt_valid(
                agent.coordinator_result.state,
                agent.bmp_prepared_projection,
                agent.bmp_integrity_receipt,
            )
            and _host_bool(agent.bmp_result.update_applied)
            and _tree_exact_equal(
                agent.bmp_result.state,
                agent.bmp_prepared_projection.candidate_coordinator_state,
            )
            and _tree_exact_equal(agent.action_record, expected_record)
            and self._action_record_valid(
                agent.action_record,
                agent_index=index,
                coordinator=agent.bmp_result.state,
                ledger=route.ledger,
                memory=agent.memory_rebind_result.state,
                planner=prepared.planner_result.state,
            )
        )

    def _prepared_semantics_valid(
        self,
        prepared: HCCLRoutedContinualDyadPreparedTransaction,
    ) -> bool:
        if type(prepared) is not HCCLRoutedContinualDyadPreparedTransaction:
            return False
        try:
            source = prepared.source_state
            source_valid = _host_bool(self.state_valid(source))
            event_valid = _host_bool(
                self._hccl.world.event_receipt_valid(
                    source.hccl_state.world_state,
                    prepared.event,
                )
            )
            bundle_valid = _host_bool(
                self.action_bundle_valid(source, prepared.event, prepared.action_bundle)
            )
            hccl_valid = bool(
                _host_bool(prepared.hccl_result.update_applied)
                and _array_exact_equal(
                    prepared.hccl_result.pre_transaction_words,
                    source.hccl_state.world_state.step_words,
                )
                and int(_host(prepared.hccl_result.work.world_proposal_calls)) == 8
                and int(_host(prepared.hccl_result.work.attribution_proposal_calls)) == 8
            )
            expected_credit = derive_hccl_memory_credit_estimands(
                mm=_signals_at(prepared.hccl_result.world_proposals, _MM_SLOT),
                b0m1=_signals_at(prepared.hccl_result.world_proposals, _B0M1_SLOT),
                m0b1=_signals_at(prepared.hccl_result.world_proposals, _M0B1_SLOT),
                bb=_signals_at(prepared.hccl_result.world_proposals, _BB_SLOT),
            )
            credit_valid = bool(
                _tree_exact_equal(prepared.memory_credit_panel, expected_credit)
                and _host_bool(expected_credit.algebra.all_identities_hold)
            )
            planner = prepared.planner_result
            planner_source = planner.receipt.source_update
            planner_route = planner.receipt.feature_route
            planner_plan = planner.receipt.plan
            agents = (prepared.agent_0, prepared.agent_1)
            source_coordinators = self._coordinators_from_state(source)
            expected_source_representations = jnp.stack(
                tuple(
                    _prototype(source_coordinators[index]).current_representation
                    for index in range(_N_AGENTS)
                )
            ).astype(jnp.float32)
            expected_destination_representations = jnp.stack(
                tuple(
                    _prototype(agents[index].coordinator_result.state).current_representation
                    for index in range(_N_AGENTS)
                )
            ).astype(jnp.float32)
            planner_valid = bool(
                _host_bool(planner.transaction_applied)
                and _host_bool(planner.receipt.transaction_applied)
                and _host_bool(self._planner.state_valid(planner.state))
                and _array_exact_equal(
                    planner_source.source_state_content_token,
                    source.planner_state.content_token,
                )
                and _array_exact_equal(
                    planner_source.source_representations,
                    expected_source_representations,
                )
                and _array_exact_equal(
                    planner_source.executed_actions,
                    prepared.action_bundle.final_actions,
                )
                and _array_exact_equal(
                    planner_route.route_witness_content_tokens,
                    jnp.stack(
                        tuple(agent.route_result.witness.content_token for agent in agents)
                    ),
                )
                and _array_exact_equal(
                    planner_plan.destination_representations,
                    expected_destination_representations,
                )
                and _array_exact_equal(planner.prepared_actions, planner_plan.proposed_actions)
            )
            agent_valid = tuple(
                self._prepared_agent_valid(prepared, agents[index], index=index)
                for index in range(_N_AGENTS)
            )
            context_valid = tuple(
                bool(
                    _host_bool(agents[index].context_result.update_applied)
                    and _array_exact_equal(
                        agents[index].context_result.state.context.step_words,
                        prepared.hccl_result.post_transaction_words,
                    )
                )
                for index in range(_N_AGENTS)
            )
            coordinator_valid = tuple(
                bool(
                    _host_bool(
                        agents[index].coordinator_result.diagnostics.transaction_applied
                    )
                    and _array_exact_equal(
                        agents[index].coordinator_result.state.event_words,
                        prepared.hccl_result.post_transaction_words,
                    )
                )
                for index in range(_N_AGENTS)
            )
            route_valid = tuple(
                bool(
                    _host_bool(agents[index].lifecycle_proof.proof_valid)
                    and _host_bool(
                        agents[index].route_result.witness.transaction_applied
                    )
                )
                for index in range(_N_AGENTS)
            )
            memory_valid = tuple(
                bool(
                    _host_bool(
                        agents[index].memory_step_result.diagnostics.transaction_applied
                    )
                    and _host_bool(
                        agents[index].memory_rebind_result.diagnostics.transaction_applied
                    )
                )
                for index in range(_N_AGENTS)
            )
            bmp_valid = tuple(
                _host_bool(agents[index].bmp_result.update_applied)
                for index in range(_N_AGENTS)
            )
            expected_candidate = self._seal_state(
                HCCLRoutedContinualDyadState(
                    config_token=source.config_token,
                    content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
                    hccl_state=prepared.hccl_result.state,
                    coordinator_0_state=agents[0].bmp_result.state,
                    coordinator_1_state=agents[1].bmp_result.state,
                    context_0_state=agents[0].context_result.state,
                    context_1_state=agents[1].context_result.state,
                    ledger_0=agents[0].route_result.ledger,
                    ledger_1=agents[1].route_result.ledger,
                    memory_0_state=agents[0].memory_rebind_result.state,
                    memory_1_state=agents[1].memory_rebind_result.state,
                    planner_state=planner.state,
                    action_record_0=agents[0].action_record,
                    action_record_1=agents[1].action_record,
                )
            )
            candidate_valid = _host_bool(self.state_valid(expected_candidate))
            all_valid = bool(
                source_valid
                and event_valid
                and bundle_valid
                and hccl_valid
                and credit_valid
                and planner_valid
                and all(agent_valid)
                and candidate_valid
            )
            expected_work = HCCLRoutedContinualDyadWork(
                hccl_stage_calls=jnp.asarray(1, dtype=jnp.int32),
                world_proposal_calls=prepared.hccl_result.work.world_proposal_calls,
                attribution_proposal_calls=(
                    prepared.hccl_result.work.attribution_proposal_calls
                ),
                context_steps=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
                coordinator_steps=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
                lifecycle_route_derivations=jnp.ones(
                    (_N_AGENTS,),
                    dtype=jnp.int32,
                ),
                memory_settlements=jnp.asarray(
                    tuple(int(agent.memory_settle_result is not None) for agent in agents),
                    dtype=jnp.int32,
                ),
                memory_steps=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
                memory_rebinds=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
                planner_behavior_updates=planner.work.behavior_model_updates,
                planner_grounded_updates=planner.work.grounded_model_updates,
                planner_joint_cells=planner.work.grounded_joint_cell_evaluations,
                bmp_memory_replacements=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
                bmp_planner_replacements=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
                outer_commit_decisions=jnp.asarray(1, dtype=jnp.int32),
                output_writes=jnp.asarray(0, dtype=jnp.int32),
                rng_draws_after_event=jnp.asarray(0, dtype=jnp.int32),
            )
            stored = bool(
                _tree_exact_equal(prepared.candidate_state, expected_candidate)
                and _tree_exact_equal(prepared.work, expected_work)
                and _host_bool(prepared.source_state_valid) == source_valid
                and _host_bool(prepared.event_valid) == event_valid
                and _host_bool(prepared.action_bundle_valid) == bundle_valid
                and _host_bool(prepared.hccl_valid) == hccl_valid
                and _host_bool(prepared.credit_valid) == credit_valid
                and _array_exact_equal(
                    prepared.context_valid,
                    jnp.asarray(context_valid, dtype=jnp.bool_),
                )
                and _array_exact_equal(
                    prepared.coordinator_valid,
                    jnp.asarray(coordinator_valid, dtype=jnp.bool_),
                )
                and _array_exact_equal(
                    prepared.lifecycle_route_valid,
                    jnp.asarray(route_valid, dtype=jnp.bool_),
                )
                and _array_exact_equal(
                    prepared.memory_valid,
                    jnp.asarray(memory_valid, dtype=jnp.bool_),
                )
                and _host_bool(prepared.planner_valid) == planner_valid
                and _array_exact_equal(
                    prepared.bmp_valid,
                    jnp.asarray(bmp_valid, dtype=jnp.bool_),
                )
                and _host_bool(prepared.candidate_state_valid) == candidate_valid
                and _host_bool(prepared.preparation_valid) == all_valid
                and all(
                    _host_bool(agents[index].child_valid) == agent_valid[index]
                    for index in range(_N_AGENTS)
                )
            )
            return bool(
                stored
                and _array_exact_equal(
                    prepared.content_token,
                    self._prepared_token(prepared),
                )
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def integrity_receipt(
        self,
        prepared: HCCLRoutedContinualDyadPreparedTransaction,
    ) -> HCCLRoutedContinualDyadIntegrityReceipt:
        """Bind a complete prepared transaction without reevaluating learners."""

        if type(prepared) is not HCCLRoutedContinualDyadPreparedTransaction:
            raise TypeError("prepared must be an exact routed prepared transaction")
        if _contains_tracer(prepared):
            raise TypeError("routed continual-dyad receipts are host/eager-only")
        integrity = self._prepared_semantics_valid(prepared)
        bare = HCCLRoutedContinualDyadIntegrityReceipt(
            config_token=self._config_token,
            source_state_token=prepared.source_state.content_token,
            prepared_content_token=prepared.content_token,
            integrity_bound=jnp.asarray(integrity, dtype=jnp.bool_),
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return cast(
            HCCLRoutedContinualDyadIntegrityReceipt,
            bare.replace(content_token=self._receipt_token(bare)),
        )

    def _receipt_valid(
        self,
        prepared: HCCLRoutedContinualDyadPreparedTransaction,
        receipt: HCCLRoutedContinualDyadIntegrityReceipt,
    ) -> bool:
        if type(receipt) is not HCCLRoutedContinualDyadIntegrityReceipt:
            return False
        try:
            return bool(
                _array_exact_equal(receipt.config_token, self._config_token)
                and _array_exact_equal(
                    receipt.source_state_token,
                    prepared.source_state.content_token,
                )
                and _array_exact_equal(
                    receipt.prepared_content_token,
                    prepared.content_token,
                )
                and _host_bool(receipt.integrity_bound)
                and _array_exact_equal(receipt.content_token, self._receipt_token(receipt))
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def adopt(
        self,
        source: HCCLRoutedContinualDyadState,
        prepared: HCCLRoutedContinualDyadPreparedTransaction,
        receipt: HCCLRoutedContinualDyadIntegrityReceipt,
    ) -> HCCLRoutedContinualDyadResult:
        """Select all child owners together or return ``source`` bit-for-bit."""

        if type(source) is not HCCLRoutedContinualDyadState:
            raise TypeError("source must be an exact routed continual-dyad state")
        if type(prepared) is not HCCLRoutedContinualDyadPreparedTransaction:
            raise TypeError("prepared must be an exact routed prepared transaction")
        if type(receipt) is not HCCLRoutedContinualDyadIntegrityReceipt:
            raise TypeError("receipt must be an exact routed integrity receipt")
        if _contains_tracer((source, prepared, receipt)):
            raise TypeError("routed continual-dyad adoption is host/eager-only")
        source_matches = _tree_exact_equal(source, prepared.source_state)
        prepared_valid = self._prepared_semantics_valid(prepared)
        receipt_valid = self._receipt_valid(prepared, receipt)
        candidate_valid = _host_bool(self.state_valid(prepared.candidate_state))
        applied = bool(
            source_matches
            and prepared_valid
            and receipt_valid
            and candidate_valid
            and _host_bool(prepared.preparation_valid)
        )
        selected = prepared.candidate_state if applied else source
        return HCCLRoutedContinualDyadResult(
            state=selected,
            prepared=prepared,
            receipt=receipt,
            source_state_matches=jnp.asarray(source_matches, dtype=jnp.bool_),
            prepared_content_valid=jnp.asarray(prepared_valid, dtype=jnp.bool_),
            receipt_valid=jnp.asarray(receipt_valid, dtype=jnp.bool_),
            candidate_state_valid=jnp.asarray(candidate_valid, dtype=jnp.bool_),
            update_applied=jnp.asarray(applied, dtype=jnp.bool_),
            complete_source_returned=jnp.asarray(not applied, dtype=jnp.bool_),
        )

    def step(
        self,
        state: HCCLRoutedContinualDyadState,
        *,
        next_hard_action_masks: Array | None = None,
    ) -> HCCLRoutedContinualDyadResult:
        """Prepare the current event, bind B/M/P, evaluate once, and adopt."""

        event = self.prepare_event(state)
        bundle = self.bind_actions(state, event)
        masks = (
            bundle.hard_action_masks
            if next_hard_action_masks is None
            else next_hard_action_masks
        )
        prepared = self.prepare_transaction(
            state,
            event,
            bundle,
            next_hard_action_masks=masks,
        )
        return self.adopt(state, prepared, self.integrity_receipt(prepared))


__all__ = [
    "HCCL_ROUTED_CONTINUAL_DYAD_ACTION_BUNDLE_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_ACTION_RECORD_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_CONFIG_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_PREPARED_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_RECEIPT_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_SCIENTIFIC_PROMOTION_ALLOWED",
    "HCCL_ROUTED_CONTINUAL_DYAD_STATE_SCHEMA",
    "HCCL_ROUTED_CONTINUAL_DYAD_STATUS",
    "HCCLRoutedContinualDyad",
    "HCCLRoutedContinualDyadActionBundle",
    "HCCLRoutedContinualDyadActionRecord",
    "HCCLRoutedContinualDyadAgentPreparation",
    "HCCLRoutedContinualDyadConfig",
    "HCCLRoutedContinualDyadIntegrityReceipt",
    "HCCLRoutedContinualDyadPreparedTransaction",
    "HCCLRoutedContinualDyadResult",
    "HCCLRoutedContinualDyadState",
    "HCCLRoutedContinualDyadWork",
    "HCCLRoutedLifecycleRouteProof",
]
