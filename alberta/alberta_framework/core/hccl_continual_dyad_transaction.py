# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var,union-attr"
"""Atomic HCCL event transaction for two continual B/M/P learning agents.

This development-only, host/eager boundary composes existing pure donors.  It
owns one HCCL world/attribution state, two versioned live-memory B/M/P action
stacks, one paired factorized partner planner, and two context/lineage seams.
One preparation evaluates every donor at most once.  Adoption performs exact
content checks and the two action-stack integrity adoptions, then advances all
owners together or returns the complete source bit-for-bit.

The stable Prototype base is ordered ``physical16, context3, fast4``.  Only the
physical prefix participates in pair discovery, so 12 routed pair slots draw
from the exact 120 unordered pairs of the first 16 coordinates.  Slow context
preparation precedes the HCCL outcome; the completed task score updates context
before its three coordinates enter the next 19-wide recurrent input.  Memory
credit remains bound to M while the completed transition records executed P.

All SHA-256 values in this module are unkeyed integrity bindings.  They do not
authenticate a caller, authorize dispatch, execute a benchmark life, or grant
artifact, evidence, threshold, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt
from numpy.typing import NDArray

from alberta_framework.core.context_lineage_retention_seam import (
    CONTEXT_LINEAGE_RETENTION_CONFIRMATION_HORIZON,
    ContextLineageRetentionPreparation,
    ContextLineageRetentionResourceRecord,
    ContextLineageRetentionSeam,
    ContextLineageRetentionSeamConfig,
    ContextLineageRetentionSeamState,
    ContextLineageRetentionStepResult,
)
from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    ExternalLearnedStateLiveMemoryActionStackAdapter,
    ExternalLearnedStateLiveMemoryActionStackAdoptionWork,
    ExternalLearnedStateLiveMemoryActionStackConfig,
    ExternalLearnedStateLiveMemoryActionStackDiagnostics,
    ExternalLearnedStateLiveMemoryActionStackFeedback,
    ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
    ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt,
    ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
    ExternalLearnedStateLiveMemoryActionStackResult,
    ExternalLearnedStateLiveMemoryActionStackStartedFinalization,
    ExternalLearnedStateLiveMemoryActionStackStartedResult,
    ExternalLearnedStateLiveMemoryActionStackState,
)
from alberta_framework.core.external_learned_state_live_memory_adapter import (
    ExternalLearnedStateLiveMemoryEventInput,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalBuilderCandidateAuditEvidence,
    ExternalLearnedStateTransition,
)
from alberta_framework.core.hccl_causal_attribution import (
    HCCLActionLayer,
    HCCLActionReceipt,
)
from alberta_framework.core.hccl_causal_attribution import (
    _derive_contrasts as _derive_hccl_contrasts,
)
from alberta_framework.core.hccl_causal_attribution import (
    _increment_words as _increment_hccl_attribution_words,
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
    HCCLWorldAttributionResourceBudget,
)
from alberta_framework.core.prototype_agent import (
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
)
from alberta_framework.core.prototype_factorized_partner_planner import (
    PrototypeFactorizedPartnerPlanner,
    PrototypeFactorizedPartnerPlannerConfig,
    PrototypeFactorizedPartnerPlannerResourceBudget,
    PrototypeFactorizedPartnerPlannerState,
    PrototypeFactorizedPartnerTransitionResult,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCLCausalCoreEventReceipt,
    HCCLCausalCoreProposal,
    HCCLCausalCoreTypedSignals,
)
from alberta_framework.streams.hccl_causal_core import (
    _proposal_tag as _hccl_world_proposal_tag,
)

HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA = (
    "alberta.hccl-continual-dyad-transaction-config.v2"
)
HCCL_CONTINUAL_DYAD_STATE_SCHEMA = (
    "alberta.hccl-continual-dyad-transaction-state.v2"
)
HCCL_CONTINUAL_DYAD_BINDING_SCHEMA = (
    "alberta.hccl-continual-dyad-action-binding.v2"
)
HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_AGENT_SCHEMA = (
    "alberta.hccl-continual-dyad-through-memory-agent.v2"
)
HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA = (
    "alberta.hccl-continual-dyad-through-memory-transaction.v2"
)
HCCL_CONTINUAL_DYAD_PREPARED_SCHEMA = (
    "alberta.hccl-continual-dyad-prepared-transaction.v2"
)
HCCL_CONTINUAL_DYAD_RECEIPT_SCHEMA = (
    "alberta.hccl-continual-dyad-preparation-receipt.v2"
)
HCCL_CONTINUAL_DYAD_RESOURCE_SCHEMA = (
    "alberta.hccl-continual-dyad-resource.v2"
)
HCCL_CONTINUAL_DYAD_STATUS = (
    "l0-development-hccl-continual-dyad-atomic-transaction"
)
HCCL_CONTINUAL_DYAD_EVIDENCE_LEVEL = "L0"
HCCL_CONTINUAL_DYAD_LIMITATIONS = (
    "host-eager-only",
    "first-atomic-integrated-rung-not-causal-core-completion",
    "unkeyed-integrity-is-not-caller-authentication",
    "hccl-source-component-provenance-remains-synthetic",
    "planner-model-input-is-stable-base23-not-generated-tail35",
    "full-generated-feature-consumer-routing-is-not-implemented",
    "planner-sidecars-do-not-consume-generated-pair-features",
    "learned-memory-raw-key-includes-context-coordinates",
    "learned-memory-rows-are-not-feature-generation-bound",
    "memory-uncertainty-metadata-is-neutral-unavailable-not-model-derived",
    "option-success-horde-head-is-finite-neutral-and-unavailable",
    "preparation-is-transient-and-not-checkpointed",
    "no-schedule-seed-output-artifact-threshold-evidence-or-promotion-authority",
)

__all__ = (
    "HCCL_CONTINUAL_DYAD_BINDING_SCHEMA",
    "HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA",
    "HCCL_CONTINUAL_DYAD_EVIDENCE_LEVEL",
    "HCCL_CONTINUAL_DYAD_LIMITATIONS",
    "HCCL_CONTINUAL_DYAD_PREPARED_SCHEMA",
    "HCCL_CONTINUAL_DYAD_RECEIPT_SCHEMA",
    "HCCL_CONTINUAL_DYAD_RESOURCE_SCHEMA",
    "HCCL_CONTINUAL_DYAD_STATE_SCHEMA",
    "HCCL_CONTINUAL_DYAD_STATUS",
    "HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_AGENT_SCHEMA",
    "HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA",
    "HCCLContinualDyadActionBinding",
    "HCCLContinualDyadAdoptionWork",
    "HCCLContinualDyadPreparationReceipt",
    "HCCLContinualDyadPrepareWork",
    "HCCLContinualDyadPreparedAgent",
    "HCCLContinualDyadPreparedTransaction",
    "HCCLContinualDyadResourceRecord",
    "HCCLContinualDyadResult",
    "HCCLContinualDyadState",
    "HCCLContinualDyadThroughMemoryAgent",
    "HCCLContinualDyadThroughMemoryTransaction",
    "HCCLContinualDyadThroughMemoryWork",
    "HCCLContinualDyadTransaction",
    "HCCLContinualDyadTransactionConfig",
)

_N_AGENTS = 2
_N_ACTIONS = 2
_PHYSICAL_DIM = 16
_CONTEXT_DIM = 3
_FAST_DIM = 4
_EXTERNAL_RAW_DIM = _PHYSICAL_DIM + _CONTEXT_DIM
_BASE_DIM = _EXTERNAL_RAW_DIM + _FAST_DIM
_ACTIVE_PAIR_SLOTS = 12
_PAIR_CANDIDATE_SLOTS = 120
_ROUTED_DIM = _BASE_DIM + _ACTIVE_PAIR_SLOTS
_FEATURE_REPLACEMENT_INTERVAL = 64
_MEMORY_CAPACITY = 64
_CORE_L1_EVENTS = 8_998
_HCCL_HORDE_HEADS = (
    "task_discount_0p5",
    "task_discount_0p9",
    "task_discount_0p99",
    "partner_action",
    "safety_cost",
    "tv_occupancy",
    "target_zone_occupancy",
    "option_success_unavailable",
)
_DIGEST_WORDS = 8
_TOKEN_NBYTES = 32
_MM_SLOT = 0
_B0M1_SLOT = 1
_M0B1_SLOT = 2
_BB_SLOT = 3
_PP_SLOT = 4
_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1


@dataclasses.dataclass(slots=True)
class _OuterValidationWork:
    """Host-side exact counters for structural checks that execute models."""

    planner_pair_authentication_calls: int = 0
    child_finalization_structural_recomputations: list[int] = dataclasses.field(
        default_factory=lambda: [0, 0]
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest_bytes(payload: bytes) -> UInt[Array, " 8"]:
    digest = hashlib.sha256(payload).digest()
    return jnp.asarray(
        tuple(
            int.from_bytes(digest[offset : offset + 4], "big")
            for offset in range(0, len(digest), 4)
        ),
        dtype=jnp.uint32,
    )


def _tree_digest(*values: object) -> UInt[Array, " 8"]:
    """Hash exact host material at this deliberately host-only boundary."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(type(value).__module__.encode("utf-8"))
        digest.update(type(value).__qualname__.encode("utf-8"))
        leaves, tree = jax.tree.flatten(value)
        digest.update(repr(tree).encode("utf-8"))
        digest.update(len(leaves).to_bytes(8, "big"))
        for leaf in leaves:
            if hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
                array = jnp.asarray(leaf)
                material = (
                    jr.key_data(array)
                    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
                    else array
                )
                host = np.asarray(jax.device_get(material))
                digest.update(str(host.dtype).encode("ascii"))
                digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
                digest.update(np.ascontiguousarray(host).tobytes())
            else:
                digest.update(type(leaf).__module__.encode("utf-8"))
                digest.update(type(leaf).__qualname__.encode("utf-8"))
                digest.update(repr(leaf).encode("utf-8"))
    return _digest_bytes(digest.digest())


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(value):
        if not hasattr(leaf, "dtype"):
            continue
        array = jnp.asarray(leaf)
        material = (
            jr.key_data(array)
            if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
            else array
        )
        total += int(material.size) * int(material.dtype.itemsize)
    return total


def _tree_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return False
    if len(left_leaves) != len(right_leaves):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if not (hasattr(left_leaf, "dtype") and hasattr(right_leaf, "dtype")):
            if type(left_leaf) is not type(right_leaf) or left_leaf != right_leaf:
                return False
            continue
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return False
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        if not np.array_equal(
            np.asarray(jax.device_get(left_array)),
            np.asarray(jax.device_get(right_array)),
        ):
            return False
    return True


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    expected = jnp.dtype(dtype)
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if getattr(value, "dtype", None) != expected:
        raise TypeError(f"{name} must have dtype {expected}")
    return jnp.asarray(value)


def _bool(value: object) -> bool:
    return bool(jax.device_get(jnp.asarray(value, dtype=jnp.bool_)))


def _signals_at(
    proposals: HCCLCausalCoreProposal,
    index: int,
) -> HCCLCausalCoreTypedSignals:
    return cast(
        HCCLCausalCoreTypedSignals,
        jax.tree.map(lambda leaf: leaf[index], proposals.signals),
    )


def _words_token(words: Array) -> UInt[Array, " 32"]:
    value = _require_array(
        words,
        name="digest_words",
        shape=(_DIGEST_WORDS,),
        dtype=jnp.uint32,
    )
    shifts = jnp.asarray((24, 16, 8, 0), dtype=jnp.uint32)
    return jnp.reshape(
        jnp.bitwise_and(jnp.right_shift(value[:, None], shifts[None, :]), 0xFF),
        (_TOKEN_NBYTES,),
    ).astype(jnp.uint8)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadTransactionConfig:
    """Exact donor geometry for the smallest integrated continual dyad."""

    hccl: HCCLWorldAttributionAdapterConfig
    agent_0: ExternalLearnedStateLiveMemoryActionStackConfig
    agent_1: ExternalLearnedStateLiveMemoryActionStackConfig
    planner: PrototypeFactorizedPartnerPlannerConfig
    context: ContextLineageRetentionSeamConfig
    binding_owner_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.hccl) is not HCCLWorldAttributionAdapterConfig:
            raise TypeError("hccl must be an exact HCCL world-attribution config")
        for index, agent in enumerate((self.agent_0, self.agent_1)):
            if type(agent) is not ExternalLearnedStateLiveMemoryActionStackConfig:
                raise TypeError(f"agent_{index} must be an exact action-stack config")
        if type(self.planner) is not PrototypeFactorizedPartnerPlannerConfig:
            raise TypeError("planner must be an exact factorized planner config")
        if type(self.context) is not ContextLineageRetentionSeamConfig:
            raise TypeError("context must be an exact context-lineage config")
        owner = self.binding_owner_digest
        if type(owner) is not tuple or len(owner) != _DIGEST_WORDS:
            raise ValueError("binding_owner_digest must be an exact eight-word tuple")
        for index, word in enumerate(owner):
            if type(word) is not int or not 0 <= word <= _UINT32_MAX:
                raise ValueError(f"binding_owner_digest[{index}] must be uint32")
        if not any(owner):
            raise ValueError("binding_owner_digest must be nonzero")
        if self.agent_0.final_action_owner_digest == self.agent_1.final_action_owner_digest:
            raise ValueError("the two final-action owners must be distinct")

        required_agent_lifetime = max(
            _CORE_L1_EVENTS,
            self.hccl.world_config.maximum_committed_transitions,
        )
        prototype_payloads: list[dict[str, object]] = []
        gammas: list[float] = []
        for index, agent in enumerate((self.agent_0, self.agent_1)):
            coordinator = agent.coordinator
            builder = coordinator.builder
            if builder.observation_dim != _EXTERNAL_RAW_DIM:
                raise ValueError(f"agent_{index} builder observation_dim must equal 19")
            if builder.hidden_dim != _FAST_DIM:
                raise ValueError(f"agent_{index} builder hidden_dim must equal 4")
            if builder.include_raw_observation is not True:
                raise ValueError(f"agent_{index} builder must include raw observation")
            if builder.n_actions != _N_ACTIONS:
                raise ValueError(f"agent_{index} must expose exactly two actions")
            if builder.feature_dim() != _BASE_DIM:
                raise ValueError(f"agent_{index} stable base width must equal 23")
            prototype = coordinator.inner.prototype
            lifecycle = prototype.prototype_feature_lifecycle
            if lifecycle is None:
                raise ValueError(f"agent_{index} must own a Prototype feature lifecycle")
            if lifecycle.base_feature_dim != _BASE_DIM:
                raise ValueError(f"agent_{index} lifecycle base width must equal 23")
            if lifecycle.effective_pair_source_feature_dim != _PHYSICAL_DIM:
                raise ValueError(f"agent_{index} pair source width must equal 16")
            if lifecycle.active_pair_slots != _ACTIVE_PAIR_SLOTS:
                raise ValueError(f"agent_{index} active pair slots must equal 12")
            if lifecycle.candidate_pair_slots != _PAIR_CANDIDATE_SLOTS:
                raise ValueError(f"agent_{index} pair candidates must equal 120")
            if lifecycle.total_feature_dim != _ROUTED_DIM:
                raise ValueError(f"agent_{index} routed width must equal 35")
            if (
                lifecycle.n_options != 0
                or lifecycle.option_subtask_feature_indices != ()
            ):
                raise ValueError(
                    f"agent_{index} lifecycle must use primitive-only zero-option geometry"
                )
            if lifecycle.replacement_interval != _FEATURE_REPLACEMENT_INTERVAL:
                raise ValueError(
                    f"agent_{index} feature replacement interval must equal 64"
                )
            if lifecycle.max_observations < required_agent_lifetime:
                raise ValueError(
                    f"agent_{index} feature lifetime must cover all "
                    f"{required_agent_lifetime:,} profile events"
                )
            stomp = prototype.oak.stomp
            if stomp.observation_dim != _ROUTED_DIM:
                raise ValueError(f"agent_{index} OaK width must equal 35")
            if stomp.n_options != 0 or stomp.subtask_specs != ():
                raise ValueError(
                    f"agent_{index} STOMP must use primitive-only zero-option geometry"
                )
            if (
                stomp.n_primitive_actions != _N_ACTIONS
                or stomp.n_total_actions != _N_ACTIONS
            ):
                raise ValueError(
                    f"agent_{index} primitive-only STOMP must expose exactly two actions"
                )
            if stomp.option_planning_backups_per_step != 0:
                raise ValueError(
                    f"agent_{index} primitive-only STOMP must disable option backups"
                )
            if prototype.option_search_control is not None:
                raise ValueError(
                    f"agent_{index} primitive-only Prototype must disable option search"
                )
            if prototype.auto_curate_every != 0:
                raise ValueError(
                    f"agent_{index} primitive-only Prototype must disable auto-curation"
                )
            horde = prototype.horde_spec
            if horde is None or len(horde.demons) != len(_HCCL_HORDE_HEADS):
                raise ValueError(f"agent_{index} must own the exact eight-head HCCL Horde")
            if lifecycle.managed_horde_demons != len(_HCCL_HORDE_HEADS):
                raise ValueError(f"agent_{index} lifecycle must route all eight Horde heads")
            if tuple(demon.name for demon in horde.demons) != _HCCL_HORDE_HEADS:
                raise ValueError(f"agent_{index} Horde head names/order are not canonical")
            if tuple(demon.cumulant_index for demon in horde.demons) != tuple(
                range(len(_HCCL_HORDE_HEADS))
            ):
                raise ValueError(f"agent_{index} Horde cumulant indices must be 0..7")
            if not np.array_equal(
                np.asarray(horde.gammas[:3]),
                np.asarray((0.5, 0.9, 0.99), dtype=np.float32),
            ):
                raise ValueError(f"agent_{index} first Horde horizons must be .5/.9/.99")
            memory = agent.learned_memory.memory
            if memory.capacity != _MEMORY_CAPACITY:
                raise ValueError(f"agent_{index} memory capacity must equal 64")
            if memory.max_age < required_agent_lifetime:
                raise ValueError(
                    f"agent_{index} memory age horizon must cover all "
                    f"{required_agent_lifetime:,} profile events"
                )
            if memory.staleness_scale < float(required_agent_lifetime):
                raise ValueError(
                    f"agent_{index} memory staleness horizon must cover all "
                    f"{required_agent_lifetime:,} profile events"
                )
            if (
                memory.observation_dim != _EXTERNAL_RAW_DIM
                or memory.key_dim != _EXTERNAL_RAW_DIM
                or memory.outcome_dim != _EXTERNAL_RAW_DIM
                or memory.action_dim != _N_ACTIONS
            ):
                raise ValueError(
                    f"agent_{index} memory obs/key/outcome/action widths must be 19/19/19/2"
                )
            coordinator = agent.coordinator
            if coordinator.learning_value_router.max_steps < required_agent_lifetime:
                raise ValueError(
                    f"agent_{index} learning router must cover all "
                    f"{required_agent_lifetime:,} profile events"
                )
            if coordinator.max_events < required_agent_lifetime:
                raise ValueError(
                    f"agent_{index} coordinator lifetime must cover all "
                    f"{required_agent_lifetime:,} profile events"
                )
            if coordinator.inner.ensemble.max_events < required_agent_lifetime:
                raise ValueError(
                    f"agent_{index} routed ensemble must cover all "
                    f"{required_agent_lifetime:,} profile events"
                )
            if coordinator.inner.ensemble.ensemble_size != 1:
                raise ValueError(
                    f"agent_{index} uncertainty ensemble must contain exactly one member"
                )
            prototype_payloads.append(cast(dict[str, object], prototype.to_config()))
            gammas.append(float(coordinator.inner.ensemble.world_model.gamma))
        if _canonical_json_bytes(prototype_payloads[0]) != _canonical_json_bytes(
            prototype_payloads[1]
        ):
            raise ValueError("the two action stacks must use the same Prototype config")
        if gammas[0] != gammas[1]:
            raise ValueError("the two action stacks must use the same discount")
        if (
            self.planner.observation_dim != _BASE_DIM
            or self.planner.prototype_representation_dim != _ROUTED_DIM
            or self.planner.n_actions != _N_ACTIONS
        ):
            raise ValueError("planner geometry must be D=23, R=35, A=2")
        if (
            self.context.context.max_contexts != _CONTEXT_DIM
            or self.context.context.observation_dim != _N_ACTIONS
            or self.context.context.n_actions != _N_ACTIONS
            or CONTEXT_LINEAGE_RETENTION_CONFIRMATION_HORIZON != 2
        ):
            raise ValueError("context geometry must be K=3, D=2, A=2, H=2")

    @property
    def discount(self) -> float:
        return float(self.agent_0.coordinator.inner.ensemble.world_model.gamma)

    def to_config(self) -> dict[str, object]:
        required_agent_lifetime = max(
            _CORE_L1_EVENTS,
            self.hccl.world_config.maximum_committed_transitions,
        )
        return {
            "type": type(self).__name__,
            "schema": HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA,
            "state_schema": HCCL_CONTINUAL_DYAD_STATE_SCHEMA,
            "binding_schema": HCCL_CONTINUAL_DYAD_BINDING_SCHEMA,
            "through_memory_agent_schema": (
                HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_AGENT_SCHEMA
            ),
            "through_memory_schema": HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA,
            "prepared_schema": HCCL_CONTINUAL_DYAD_PREPARED_SCHEMA,
            "receipt_schema": HCCL_CONTINUAL_DYAD_RECEIPT_SCHEMA,
            "resource_schema": HCCL_CONTINUAL_DYAD_RESOURCE_SCHEMA,
            "mechanism_status": HCCL_CONTINUAL_DYAD_STATUS,
            "evidence_level": HCCL_CONTINUAL_DYAD_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": False,
            "hccl": HCCLWorldAttributionAdapter(self.hccl).to_config(),
            "agent_0": self.agent_0.to_config(),
            "agent_1": self.agent_1.to_config(),
            "planner": self.planner.to_config(),
            "context": self.context.to_config(),
            "binding_owner_digest": list(self.binding_owner_digest),
            "physical_observation_dim": _PHYSICAL_DIM,
            "context_coordinate_dim": _CONTEXT_DIM,
            "fast_state_dim": _FAST_DIM,
            "external_recurrent_input_dim": _EXTERNAL_RAW_DIM,
            "stable_base_dim": _BASE_DIM,
            "pair_source_dim": _PHYSICAL_DIM,
            "active_pair_slots": _ACTIVE_PAIR_SLOTS,
            "pair_candidate_slots": _PAIR_CANDIDATE_SLOTS,
            "routed_representation_dim": _ROUTED_DIM,
            "feature_replacement_interval": _FEATURE_REPLACEMENT_INTERVAL,
            "experiential_memory_rows_per_agent": _MEMORY_CAPACITY,
            "experiential_memory_minimum_age_horizon": required_agent_lifetime,
            "experiential_memory_minimum_staleness_scale": float(
                required_agent_lifetime
            ),
            "uncertainty_ensemble_members_per_agent": (
                self.agent_0.coordinator.inner.ensemble.ensemble_size
            ),
            "installed_option_slots_per_agent": (
                cast(
                    Any,
                    self.agent_0.coordinator.inner.prototype
                    .prototype_feature_lifecycle,
                ).n_options
            ),
            "causal_core_target_uncertainty_ensemble_members": 1,
            "causal_core_target_installed_option_slots": 0,
            "minimum_supported_core_events": required_agent_lifetime,
            "horde_head_order": list(_HCCL_HORDE_HEADS),
            "horde_cumulant_source": "PP-proposal-only",
            "horde_discount_source": "canonical-configured-head-horizons",
            "option_success_horde_head_available": False,
            "base_order": ["physical16", "context3", "fast4"],
            "context_observation": "completed-partner-P-onehot",
            "context_action": "completed-own-P",
            "context_reward": "PP-task-score",
            "memory_feedback": "baseline-context-own-direct-M-effect",
            "memory_query_uncertainty": "unavailable-positive-zero",
            "memory_entry_uncertainty": "unavailable-positive-zero",
            "memory_entry_safety": "PP-available-positive-zero",
            "memory_entry_reliability": "one",
            "memory_provenance_id": "2*source-event-index+agent-index",
            "memory_source_id": "agent-index",
            "completed_transition_action": "P",
            "planner_candidate_binding": "shared-paired-result-sha256",
            "preparation_persisted": False,
            "composite_jit_supported": False,
            "caller_authenticated": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_CONTINUAL_DYAD_LIMITATIONS),
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HCCLContinualDyadTransactionConfig:
        if type(payload) is not dict:
            raise TypeError("continual-dyad config must be an exact dict")
        values = dict(payload)
        for name in ("hccl", "agent_0", "agent_1", "planner", "context"):
            if type(values.get(name)) is not dict:
                raise ValueError(f"continual-dyad {name} config must be an exact dict")
        owner = values.get("binding_owner_digest")
        if type(owner) is not list:
            raise ValueError("binding_owner_digest must serialize as a list")
        candidate = cls(
            hccl=HCCLWorldAttributionAdapter.from_config(
                cast(dict[str, object], values["hccl"])
            ).config,
            agent_0=ExternalLearnedStateLiveMemoryActionStackConfig.from_config(
                cast(dict[str, object], values["agent_0"])
            ),
            agent_1=ExternalLearnedStateLiveMemoryActionStackConfig.from_config(
                cast(dict[str, object], values["agent_1"])
            ),
            planner=PrototypeFactorizedPartnerPlannerConfig.from_config(
                cast(dict[str, object], values["planner"])
            ),
            context=ContextLineageRetentionSeamConfig.from_config(
                cast(dict[str, object], values["context"])
            ),
            binding_owner_digest=tuple(cast(list[int], owner)),
        )
        if _canonical_json_bytes(values) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("continual-dyad config is noncanonical or unsupported")
        return candidate


@chex.dataclass(frozen=True)
class HCCLContinualDyadState:
    """One persistent owner for every component of the paired transaction."""

    config_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]
    hccl_state: HCCLWorldAttributionAdapterState
    agent_0_state: ExternalLearnedStateLiveMemoryActionStackState
    agent_1_state: ExternalLearnedStateLiveMemoryActionStackState
    planner_state: PrototypeFactorizedPartnerPlannerState
    context_0_state: ContextLineageRetentionSeamState
    context_1_state: ContextLineageRetentionSeamState


@chex.dataclass(frozen=True)
class HCCLContinualDyadActionBinding:
    """Exact current B/M/P sources and their HCCL action receipts."""

    source_state_words: UInt[Array, " 8"]
    event_words: UInt[Array, " 8"]
    action_stack_words: UInt[Array, "2 8"]
    planner_agent_words: UInt[Array, "2 8"]
    context_content_tokens: UInt[Array, "2 32"]
    base_actions: Int[Array, " 2"]
    memory_actions_before_mask: Int[Array, " 2"]
    memory_actions: Int[Array, " 2"]
    planner_actions_before_mask: Int[Array, " 2"]
    final_actions: Int[Array, " 2"]
    hard_action_masks: Bool[Array, "2 2"]
    base: HCCLActionReceipt
    memory: HCCLActionReceipt
    planner: HCCLActionReceipt
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLContinualDyadThroughMemoryAgent:
    """One exact post-memory, pre-planner agent proposal."""

    agent_index: Int[Array, ""]
    context_preparation: ContextLineageRetentionPreparation
    context_result: ContextLineageRetentionStepResult
    memory_credit: Float[Array, ""]
    memory_feedback: ExternalLearnedStateLiveMemoryActionStackFeedback | None
    transition: ExternalLearnedStateTransition
    memory_preparation: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLContinualDyadThroughMemoryWork:
    """Exact work completed before the factorized planner boundary."""

    supplied_event_receipts: Int[Array, ""]
    supplied_action_binding_bundles: Int[Array, ""]
    event_receipt_preparations: Int[Array, ""]
    event_random_draws: Int[Array, ""]
    action_receipt_validation_rebindings: Int[Array, ""]
    action_identity_validation_recomputations: Int[Array, ""]
    planner_validation_pair_authentication_calls: Int[Array, ""]
    planner_validation_agent_cache_authentication_evaluations: Int[Array, ""]
    planner_validation_behavior_probability_vector_evaluations: Int[Array, ""]
    planner_validation_grounded_joint_cell_prediction_equivalents: Int[Array, ""]
    planner_validation_expected_reward_marginalization_products: Int[Array, ""]
    context_preparations: Int[Array, " 2"]
    hccl_stage_calls: Int[Array, ""]
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    designated_counterfactual_slots: Int[Array, ""]
    inner_discarded_world_proposal_calls: Int[Array, ""]
    inner_selected_pp_world_successors: Int[Array, ""]
    outer_committed_pp_world_successors: Int[Array, ""]
    world_duplicate_mm_checks: Int[Array, ""]
    attribution_duplicate_mm_checks: Int[Array, ""]
    memory_credit_panel_derivations: Int[Array, ""]
    memory_credit_readouts: Int[Array, " 2"]
    context_steps: Int[Array, " 2"]
    lineage_proposals: Int[Array, " 2"]
    action_stack_memory_preparations: Int[Array, " 2"]
    feedback_settlement_evaluations: Int[Array, " 2"]
    coordinator_update_evaluations: Int[Array, " 2"]
    memory_action_replacement_evaluations: Int[Array, " 2"]
    fast_state_transition_attempts: Int[Array, " 2"]
    prototype_transition_attempts: Int[Array, " 2"]
    feature_lifecycle_route_attempts: Int[Array, " 2"]
    feature_lifecycle_arithmetic_count_available: Bool[Array, " 2"]
    active_pair_value_materializations: Int[Array, " 2"]
    candidate_pair_product_materializations: Int[Array, " 2"]
    lifecycle_router_candidate_evaluations: Int[Array, " 2"]
    active_pair_slot_capacity: Int[Array, " 2"]
    pair_candidate_capacity: Int[Array, " 2"]
    routed_representation_width: Int[Array, " 2"]
    coordinator_base_action_candidates: Int[Array, " 2"]
    memory_action_candidates: Int[Array, " 2"]
    learned_memory_query_evaluations: Int[Array, " 2"]
    learned_memory_write_evaluations: Int[Array, " 2"]
    learned_memory_reencode_evaluations: Int[Array, " 2"]
    learned_memory_reencode_count_available: Bool[Array, " 2"]
    agent_content_digest_evaluations: Int[Array, " 2"]
    transaction_content_digest_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLContinualDyadThroughMemoryTransaction:
    """Transient content-bound HCCL/context/M result awaiting one planner call."""

    source_state: HCCLContinualDyadState
    event: HCCLCausalCoreEventReceipt
    binding: HCCLContinualDyadActionBinding
    hccl_result: HCCLWorldAttributionAdapterResult
    memory_credit_panel: HCCLMemoryCreditEstimandPanel
    next_hard_action_masks: Bool[Array, "2 2"]
    agent_0: HCCLContinualDyadThroughMemoryAgent
    agent_1: HCCLContinualDyadThroughMemoryAgent
    work: HCCLContinualDyadThroughMemoryWork
    source_state_valid: Bool[Array, ""]
    event_valid: Bool[Array, ""]
    binding_valid: Bool[Array, ""]
    binding_matches_source: Bool[Array, ""]
    pre_outcome_context_bound: Bool[Array, " 2"]
    hccl_staged_once: Bool[Array, ""]
    credit_algebra_valid: Bool[Array, ""]
    memory_preparations_valid: Bool[Array, " 2"]
    context_candidates_valid: Bool[Array, " 2"]
    through_memory_valid: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLContinualDyadPreparedAgent:
    """One agent's pre-outcome snapshot, memory proposal, and final P binding."""

    agent_index: Int[Array, ""]
    context_preparation: ContextLineageRetentionPreparation
    context_result: ContextLineageRetentionStepResult
    memory_credit: Float[Array, ""]
    memory_feedback: ExternalLearnedStateLiveMemoryActionStackFeedback | None
    transition: ExternalLearnedStateTransition
    memory_preparation: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation
    finalization: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition
    integrity_receipt: ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt | None


@chex.dataclass(frozen=True)
class HCCLContinualDyadPrepareWork:
    supplied_event_receipts: Int[Array, ""]
    supplied_action_binding_bundles: Int[Array, ""]
    event_receipt_preparations: Int[Array, ""]
    event_random_draws: Int[Array, ""]
    action_receipt_validation_rebindings: Int[Array, ""]
    action_identity_validation_recomputations: Int[Array, ""]
    context_preparations: Int[Array, " 2"]
    hccl_stage_calls: Int[Array, ""]
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    designated_counterfactual_slots: Int[Array, ""]
    inner_discarded_world_proposal_calls: Int[Array, ""]
    inner_selected_pp_world_successors: Int[Array, ""]
    outer_committed_pp_world_successors: Int[Array, ""]
    world_duplicate_mm_checks: Int[Array, ""]
    attribution_duplicate_mm_checks: Int[Array, ""]
    memory_credit_panel_derivations: Int[Array, ""]
    memory_credit_readouts: Int[Array, " 2"]
    context_steps: Int[Array, " 2"]
    lineage_proposals: Int[Array, " 2"]
    action_stack_memory_preparations: Int[Array, " 2"]
    feedback_settlement_evaluations: Int[Array, " 2"]
    coordinator_update_evaluations: Int[Array, " 2"]
    memory_action_replacement_evaluations: Int[Array, " 2"]
    fast_state_transition_attempts: Int[Array, " 2"]
    prototype_transition_attempts: Int[Array, " 2"]
    feature_lifecycle_route_attempts: Int[Array, " 2"]
    feature_lifecycle_arithmetic_count_available: Bool[Array, " 2"]
    active_pair_value_materializations: Int[Array, " 2"]
    candidate_pair_product_materializations: Int[Array, " 2"]
    lifecycle_router_candidate_evaluations: Int[Array, " 2"]
    active_pair_slot_capacity: Int[Array, " 2"]
    pair_candidate_capacity: Int[Array, " 2"]
    routed_representation_width: Int[Array, " 2"]
    coordinator_base_action_candidates: Int[Array, " 2"]
    memory_action_candidates: Int[Array, " 2"]
    learned_memory_query_evaluations: Int[Array, " 2"]
    learned_memory_write_evaluations: Int[Array, " 2"]
    learned_memory_reencode_evaluations: Int[Array, " 2"]
    learned_memory_reencode_count_available: Bool[Array, " 2"]
    planner_completed_transition_calls: Int[Array, ""]
    behavior_update_attempts: Int[Array, ""]
    grounded_update_attempts: Int[Array, ""]
    planner_pair_authentication_calls: Int[Array, ""]
    planner_validation_pair_authentication_calls: Int[Array, ""]
    planner_transition_pair_authentication_calls: Int[Array, ""]
    planner_cache_authentication_evaluations: Int[Array, ""]
    planner_behavior_probability_vector_evaluations: Int[Array, ""]
    planner_grounded_joint_cell_prediction_equivalents: Int[Array, ""]
    planner_expected_reward_marginalization_products: Int[Array, ""]
    planner_replacement_candidates: Int[Array, ""]
    planner_atomic_pair_commit_decisions: Int[Array, ""]
    planner_decision_evaluations: Int[Array, ""]
    planner_decision_joint_cells: Int[Array, ""]
    planner_environment_transition_proposals: Int[Array, ""]
    planner_replay_updates: Int[Array, ""]
    planner_post_init_random_draws: Int[Array, ""]
    final_action_bindings: Int[Array, " 2"]
    final_binding_donor_reevaluations: Int[Array, " 2"]
    child_finalization_structural_recomputations: Int[Array, " 2"]
    child_integrity_receipts: Int[Array, " 2"]
    prepared_content_digest_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLContinualDyadPreparedTransaction:
    """Transient complete proposal; never part of persistent state."""

    source_state: HCCLContinualDyadState
    event: HCCLCausalCoreEventReceipt
    binding: HCCLContinualDyadActionBinding
    hccl_result: HCCLWorldAttributionAdapterResult
    memory_credit_panel: HCCLMemoryCreditEstimandPanel
    planner_result: PrototypeFactorizedPartnerTransitionResult
    planner_candidate_words: UInt[Array, " 8"]
    agent_0: HCCLContinualDyadPreparedAgent
    agent_1: HCCLContinualDyadPreparedAgent
    candidate_state: HCCLContinualDyadState
    work: HCCLContinualDyadPrepareWork
    source_state_valid: Bool[Array, ""]
    event_valid: Bool[Array, ""]
    binding_valid: Bool[Array, ""]
    binding_matches_source: Bool[Array, ""]
    pre_outcome_context_bound: Bool[Array, " 2"]
    hccl_staged_once: Bool[Array, ""]
    credit_algebra_valid: Bool[Array, ""]
    memory_preparations_valid: Bool[Array, " 2"]
    context_candidates_valid: Bool[Array, " 2"]
    planner_transition_valid: Bool[Array, ""]
    finalizations_valid: Bool[Array, " 2"]
    shared_planner_binding_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLContinualDyadPreparationReceipt:
    source_state_words: UInt[Array, " 8"]
    event_words: UInt[Array, " 8"]
    binding_words: UInt[Array, " 8"]
    prepared_content_tag_words: UInt[Array, " 8"]
    config_words: UInt[Array, " 8"]
    integrity_bound: Bool[Array, ""]
    content_tag_words: UInt[Array, " 8"]


@chex.dataclass(frozen=True)
class HCCLContinualDyadAdoptionWork:
    source_state_integrity_checks: Int[Array, ""]
    preparation_integrity_checks: Int[Array, ""]
    receipt_integrity_checks: Int[Array, ""]
    outer_child_finalization_structural_recomputations: Int[Array, " 2"]
    action_stack_integrity_adoptions: Int[Array, " 2"]
    child_adoption_structural_recomputations: Int[Array, " 2"]
    outer_commit_decisions: Int[Array, ""]
    outer_committed_pp_world_successors: Int[Array, ""]
    outer_discarded_world_proposals: Int[Array, ""]
    world_reevaluations: Int[Array, ""]
    context_reevaluations: Int[Array, " 2"]
    planner_reevaluations: Int[Array, ""]
    planner_validation_pair_authentication_calls: Int[Array, ""]
    planner_validation_agent_cache_authentication_evaluations: Int[Array, ""]
    planner_validation_behavior_probability_vector_evaluations: Int[Array, ""]
    planner_validation_grounded_joint_cell_prediction_equivalents: Int[Array, ""]
    planner_validation_expected_reward_marginalization_products: Int[Array, ""]
    coordinator_reevaluations: Int[Array, " 2"]
    prototype_reevaluations: Int[Array, " 2"]
    learned_memory_reevaluations: Int[Array, " 2"]


@chex.dataclass(frozen=True)
class HCCLContinualDyadResult:
    state: HCCLContinualDyadState
    prepared: HCCLContinualDyadPreparedTransaction
    receipt: HCCLContinualDyadPreparationReceipt
    agent_0_adoption: ExternalLearnedStateLiveMemoryActionStackResult | None
    agent_1_adoption: ExternalLearnedStateLiveMemoryActionStackResult | None
    adoption_work: HCCLContinualDyadAdoptionWork
    source_state_matches: Bool[Array, ""]
    source_state_valid: Bool[Array, ""]
    prepared_content_valid: Bool[Array, ""]
    receipt_valid: Bool[Array, ""]
    child_adoptions_valid: Bool[Array, " 2"]
    candidate_state_valid: Bool[Array, ""]
    hccl_owner_committed: Bool[Array, ""]
    action_stack_owners_committed: Bool[Array, " 2"]
    planner_owner_committed: Bool[Array, ""]
    context_owners_committed: Bool[Array, " 2"]
    lineage_owners_committed: Bool[Array, " 2"]
    update_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadResourceRecord:
    schema: str
    hccl_state_owners: int
    action_stack_state_owners: int
    planner_pair_state_owners: int
    context_state_owners: int
    lineage_state_owners: int
    outer_integrity_owners: int
    hccl_state_nbytes: int
    agent_0_action_stack_nbytes: int
    agent_1_action_stack_nbytes: int
    planner_pair_state_nbytes: int
    context_pair_state_nbytes: int
    outer_integrity_nbytes: int
    fast_state_pair_nbytes: int
    prototype_state_pair_nbytes: int
    feature_lifecycle_pair_nbytes: int
    action_binding_pair_nbytes: int
    learned_memory_pair_nbytes: int
    nested_breakdowns_excluded_from_total: bool
    total_persistent_state_nbytes: int
    measured_total_persistent_state_nbytes: int
    event_receipt_nbytes: int
    outer_action_binding_nbytes: int
    outer_action_binding_measurement_available: bool
    prepared_transaction_nbytes: int
    prepared_transaction_measurement_available: bool
    preparation_receipt_nbytes: int
    preparation_receipt_measurement_available: bool
    planner_validation_pair_authentication_calls: int
    planner_validation_agent_cache_authentication_evaluations: int
    planner_validation_behavior_probability_vector_evaluations: int
    planner_validation_grounded_joint_cell_prediction_equivalents: int
    planner_validation_expected_reward_marginalization_products: int
    child_finalization_structural_recomputations: tuple[int, int]
    maximum_transient_world_proposal_stack_nbytes: int
    planner: PrototypeFactorizedPartnerPlannerResourceBudget
    context: ContextLineageRetentionResourceRecord
    hccl: HCCLWorldAttributionResourceBudget
    physical_observation_dim: int
    external_recurrent_input_dim: int
    stable_base_dim: int
    pair_source_dim: int
    active_pair_slots: int
    pair_candidate_slots: int
    routed_representation_dim: int
    preparation_persisted: bool
    prepared_checkpoint_supported: bool
    full_generated_feature_consumer_routing: bool
    planner_generated_feature_tail_consumed: bool
    learned_memory_rows_feature_generation_bound: bool
    learned_memory_reencode_count_available: bool
    composite_jit_supported: bool
    output_write_calls: int
    artifact_bytes_written: int

    def to_config(self) -> dict[str, object]:
        """Return a JSON-compatible exact resource declaration."""

        return cast(dict[str, object], dataclasses.asdict(self))


class HCCLContinualDyadTransaction:
    """Host-only two-phase transaction over every current causal-core owner."""

    def __init__(self, config: HCCLContinualDyadTransactionConfig) -> None:
        if type(config) is not HCCLContinualDyadTransactionConfig:
            raise TypeError("config must be an exact continual-dyad config")
        self._config = config
        self._hccl = HCCLWorldAttributionAdapter(config.hccl)
        self._agent_0 = ExternalLearnedStateLiveMemoryActionStackAdapter(
            config.agent_0
        )
        self._agent_1 = ExternalLearnedStateLiveMemoryActionStackAdapter(
            config.agent_1
        )
        prototype = self._agent_0.coordinator.inner.prototype
        self._planner = PrototypeFactorizedPartnerPlanner(prototype, config.planner)
        self._context = ContextLineageRetentionSeam(config.context)
        self._owner_words = jnp.asarray(config.binding_owner_digest, dtype=jnp.uint32)
        config_bytes = hashlib.sha256(_canonical_json_bytes(config.to_config())).digest()
        self._config_token = jnp.asarray(tuple(config_bytes), dtype=jnp.uint8)
        self._config_words = _digest_bytes(config_bytes)

    @property
    def config(self) -> HCCLContinualDyadTransactionConfig:
        return self._config

    @property
    def hccl(self) -> HCCLWorldAttributionAdapter:
        return self._hccl

    @property
    def agent_0(self) -> ExternalLearnedStateLiveMemoryActionStackAdapter:
        return self._agent_0

    @property
    def agent_1(self) -> ExternalLearnedStateLiveMemoryActionStackAdapter:
        return self._agent_1

    @property
    def planner(self) -> PrototypeFactorizedPartnerPlanner:
        return self._planner

    @property
    def context(self) -> ContextLineageRetentionSeam:
        return self._context

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HCCLContinualDyadTransaction:
        return cls(HCCLContinualDyadTransactionConfig.from_config(payload))

    def _state_token(self, state: HCCLContinualDyadState) -> Array:
        return _words_token(
            _tree_digest(
                HCCL_CONTINUAL_DYAD_STATE_SCHEMA,
                state.config_token,
                state.hccl_state,
                state.agent_0_state,
                state.agent_1_state,
                state.planner_state,
                state.context_0_state,
                state.context_1_state,
            )
        )

    def _seal_state(self, state: HCCLContinualDyadState) -> HCCLContinualDyadState:
        return cast(
            HCCLContinualDyadState,
            state.replace(content_token=self._state_token(state)),
        )

    @staticmethod
    def _prototype(state: ExternalLearnedStateLiveMemoryActionStackState) -> Any:
        return state.coordinator_state.inner_state.prototype_state

    def _planner_binding_words(
        self,
        planner_state: PrototypeFactorizedPartnerPlannerState,
        prototype_agent_0: Any,
        prototype_agent_1: Any,
    ) -> UInt[Array, " 8"]:
        """Bind one shared digest to the exact persisted planner/Prototype pair."""

        return _tree_digest(
            "continual-dyad-persisted-planner-pair-v2",
            self._config_words,
            planner_state,
            prototype_agent_0,
            prototype_agent_1,
        )

    def _composed_observation(
        self,
        physical: Array,
        context_state: ContextLineageRetentionSeamState,
    ) -> Array:
        physical_value = _require_array(
            physical,
            name="physical_observation",
            shape=(_PHYSICAL_DIM,),
            dtype=jnp.float32,
        )
        return jnp.concatenate(
            (physical_value, self._context.context_coordinates(context_state))
        ).astype(jnp.float32)

    def _state_contract(self, state: HCCLContinualDyadState) -> None:
        if type(state) is not HCCLContinualDyadState:
            raise TypeError("state must be an exact HCCLContinualDyadState")
        _require_array(
            state.config_token,
            name="state.config_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.uint8,
        )
        _require_array(
            state.content_token,
            name="state.content_token",
            shape=(_TOKEN_NBYTES,),
            dtype=jnp.uint8,
        )

    def _authenticate_planner_pair(
        self,
        planner_state: PrototypeFactorizedPartnerPlannerState,
        prototype_agent_0: Any,
        prototype_agent_1: Any,
        work: _OuterValidationWork | None,
    ) -> Bool[Array, " 2"]:
        if work is not None:
            work.planner_pair_authentication_calls += 1
        return self._planner.authenticate_pair(
            planner_state,
            prototype_agent_0,
            prototype_agent_1,
        )

    def state_valid(
        self,
        state: HCCLContinualDyadState,
        *,
        _work: _OuterValidationWork | None = None,
    ) -> Bool[Array, ""]:
        """Validate every child and all cross-owner observation/action clocks."""

        self._state_contract(state)
        if _contains_tracer(state):
            raise TypeError("continual-dyad validity is host/eager-only")
        token_valid = np.array_equal(
            np.asarray(state.config_token), np.asarray(self._config_token)
        ) and np.array_equal(
            np.asarray(state.content_token), np.asarray(self._state_token(state))
        )
        child_valid = (
            _bool(self._hccl.state_valid(state.hccl_state))
            and _bool(self._agent_0.state_valid(state.agent_0_state))
            and _bool(self._agent_1.state_valid(state.agent_1_state))
            and _bool(self._context.state_is_valid(state.context_0_state))
            and _bool(self._context.state_is_valid(state.context_1_state))
        )
        if not (token_valid and child_valid):
            return jnp.asarray(False, dtype=jnp.bool_)
        physical = self._hccl.world.observe(state.hccl_state.world_state)
        agents = (state.agent_0_state, state.agent_1_state)
        contexts = (state.context_0_state, state.context_1_state)
        planner_agents = (state.planner_state.agent_0, state.planner_state.agent_1)
        cross_valid = True
        for index, (agent, context_state, planner_agent) in enumerate(
            zip(agents, contexts, planner_agents, strict=True)
        ):
            coordinator = agent.coordinator_state
            prototype = self._prototype(agent)
            expected_raw = self._composed_observation(physical[index], context_state)
            expected_base = jnp.concatenate(
                (expected_raw, coordinator.builder_state.hidden)
            ).astype(jnp.float32)
            binding = agent.action_binding
            cross_valid = cross_valid and all(
                (
                    np.array_equal(
                        np.asarray(coordinator.current_raw_observation),
                        np.asarray(expected_raw),
                    ),
                    np.array_equal(
                        np.asarray(coordinator.current_representation),
                        np.asarray(expected_base),
                    ),
                    np.array_equal(
                        np.asarray(prototype.current_raw_observation),
                        np.asarray(expected_base),
                    ),
                    np.array_equal(
                        np.asarray(coordinator.event_words),
                        np.asarray(state.hccl_state.world_state.step_words),
                    ),
                    np.array_equal(
                        np.asarray(context_state.context.step_words),
                        np.asarray(state.hccl_state.world_state.step_words),
                    ),
                    np.array_equal(
                        np.asarray(prototype.step_words),
                        np.asarray(state.hccl_state.world_state.step_words),
                    ),
                    np.array_equal(
                        np.asarray(planner_agent.behavior.step_words),
                        np.asarray(state.hccl_state.world_state.step_words),
                    ),
                    np.array_equal(
                        np.asarray(planner_agent.grounded.update_words),
                        np.asarray(state.hccl_state.world_state.step_words),
                    ),
                    bool(binding.available),
                    bool(binding.planner_bound),
                    int(planner_agent.cache.base_action) == int(binding.memory_action),
                    int(planner_agent.cache.effective_action) == int(binding.final_action),
                    bool(planner_agent.cache.planner_consumed)
                    == bool(binding.planner_consumed),
                )
            )
        expected_planner_words = self._planner_binding_words(
            state.planner_state,
            self._prototype(state.agent_0_state),
            self._prototype(state.agent_1_state),
        )
        shared_words = np.array_equal(
            np.asarray(state.agent_0_state.action_binding.planner_candidate_words),
            np.asarray(state.agent_1_state.action_binding.planner_candidate_words),
        ) and all(
            np.array_equal(
                np.asarray(agent.action_binding.planner_candidate_words),
                np.asarray(expected_planner_words),
            )
            for agent in agents
        )
        authenticated = self._authenticate_planner_pair(
            state.planner_state,
            self._prototype(state.agent_0_state),
            self._prototype(state.agent_1_state),
            _work,
        )
        return jnp.asarray(
            cross_valid and shared_words and bool(jnp.all(authenticated)),
            dtype=jnp.bool_,
        )

    @staticmethod
    def _hard_action_masks(value: Array, *, name: str) -> Bool[Array, "2 2"]:
        masks = _require_array(
            value,
            name=name,
            shape=(_N_AGENTS, _N_ACTIONS),
            dtype=jnp.bool_,
        )
        if not bool(jax.device_get(jnp.all(jnp.any(masks, axis=1)))):
            raise ValueError(f"{name} must admit at least one action per agent")
        return masks

    def init(
        self,
        key: Array,
        *,
        initial_hard_action_masks: Array | None = None,
        initial_extended_action_masks: tuple[Array | None, Array | None] = (None, None),
    ) -> HCCLContinualDyadState:
        """Initialize all owners and authenticate the first P decision without a step."""

        if getattr(key, "shape", None) != () or not jax.dtypes.issubdtype(
            getattr(key, "dtype", None),
            jax.dtypes.prng_key,
        ):
            raise TypeError("key must be a scalar typed PRNG key")
        if type(initial_extended_action_masks) is not tuple or len(
            initial_extended_action_masks
        ) != _N_AGENTS:
            raise TypeError("initial_extended_action_masks must be an exact pair")
        masks = self._hard_action_masks(
            jnp.ones((_N_AGENTS, _N_ACTIONS), dtype=jnp.bool_)
            if initial_hard_action_masks is None
            else initial_hard_action_masks,
            name="initial_hard_action_masks",
        )
        if _contains_tracer((key, masks, initial_extended_action_masks)):
            raise TypeError("continual-dyad initialization is host/eager-only")

        hccl_key, agent_0_key, agent_1_key, planner_key = jr.split(key, 4)
        hccl_state = self._hccl.init(hccl_key)
        contexts = (self._context.init(), self._context.init())
        physical = self._hccl.world.observe(hccl_state.world_state)
        agent_0_source = self._agent_0.start(
            self._agent_0.init(agent_0_key),
            self._composed_observation(physical[0], contexts[0]),
            hard_action_mask=masks[0],
            extended_action_mask=initial_extended_action_masks[0],
        )
        agent_1_source = self._agent_1.start(
            self._agent_1.init(agent_1_key),
            self._composed_observation(physical[1], contexts[1]),
            hard_action_mask=masks[1],
            extended_action_mask=initial_extended_action_masks[1],
        )
        planner_preparation = self._planner.prepare_pair(
            self._planner.init(planner_key),
            self._prototype(agent_0_source),
            self._prototype(agent_1_source),
            masks,
        )
        planner_words = self._planner_binding_words(
            planner_preparation.state,
            planner_preparation.prototype_agent_0,
            planner_preparation.prototype_agent_1,
        )
        source_agents = (agent_0_source, agent_1_source)
        selected_prototypes = (
            planner_preparation.prototype_agent_0,
            planner_preparation.prototype_agent_1,
        )
        planner_agents = (
            planner_preparation.state.agent_0,
            planner_preparation.state.agent_1,
        )
        adapters = (self._agent_0, self._agent_1)
        finalized: list[ExternalLearnedStateLiveMemoryActionStackStartedFinalization] = []
        adopted: list[ExternalLearnedStateLiveMemoryActionStackStartedResult] = []
        for index in range(_N_AGENTS):
            consumed = planner_agents[index].cache.planner_consumed
            planner_before = jnp.where(
                consumed,
                planner_preparation.diagnostics.proposed_actions[index],
                source_agents[index].action_binding.memory_action,
            ).astype(jnp.int32)
            item = adapters[index].prepare_started_final_action(
                source_agents[index],
                selected_prototypes[index],
                planner_action_before_mask=planner_before,
                planner_candidate_words=planner_words,
                planner_consumed=consumed,
            )
            receipt = adapters[index].started_final_action_integrity_receipt(item)
            result = adapters[index].adopt_started_final_action(
                source_agents[index],
                item,
                receipt,
            )
            finalized.append(item)
            adopted.append(result)
        genesis_valid = bool(planner_preparation.diagnostics.pair_committed) and all(
            _bool(item.finalization_valid) for item in finalized
        ) and all(_bool(item.diagnostics.transaction_applied) for item in adopted)
        if not genesis_valid:
            raise RuntimeError("continual-dyad genesis P finalization was rejected")
        unsigned = HCCLContinualDyadState(
            config_token=self._config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            hccl_state=hccl_state,
            agent_0_state=adopted[0].state,
            agent_1_state=adopted[1].state,
            planner_state=planner_preparation.state,
            context_0_state=contexts[0],
            context_1_state=contexts[1],
        )
        state = self._seal_state(unsigned)
        if not _bool(self.state_valid(state)):
            raise RuntimeError("continual-dyad genesis violates the outer state contract")
        return state

    def prepare_event(
        self,
        state: HCCLContinualDyadState,
    ) -> HCCLCausalCoreEventReceipt:
        """Prepare the one action-independent HCCL event receipt for ``state``."""

        self._state_contract(state)
        if _contains_tracer(state):
            raise TypeError("continual-dyad event preparation is host/eager-only")
        if not _bool(self.state_valid(state)):
            raise ValueError("event preparation requires a valid continual-dyad state")
        return self._hccl.world.prepare_event(state.hccl_state.world_state)

    @staticmethod
    def _causal_core_memory_event_inputs(
        event: HCCLCausalCoreEventReceipt,
    ) -> tuple[
        ExternalLearnedStateLiveMemoryEventInput,
        ExternalLearnedStateLiveMemoryEventInput,
    ]:
        """Derive neutral typed metadata and source identities from one event.

        The causal core has one model member, so epistemic uncertainty is
        unavailable rather than spuriously reported as zero-confidence
        evidence.  PP safety is exact available positive zero.  Provenance is
        the event-major ``2 * source_step + agent_index`` identity; source ID
        is the local agent index in this two-owner memory topology.
        """

        words = np.asarray(jax.device_get(event.source_step_words), dtype=np.uint32)
        step = (int(words[0]) << 32) | int(words[1])
        if step > (_INT32_MAX - (_N_AGENTS - 1)) // _N_AGENTS:
            raise ValueError("causal-core memory provenance exceeds int32 capacity")
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        unavailable = jnp.asarray(False, dtype=jnp.bool_)
        available = jnp.asarray(True, dtype=jnp.bool_)

        def make(index: int) -> ExternalLearnedStateLiveMemoryEventInput:
            return ExternalLearnedStateLiveMemoryEventInput(
                query_uncertainty=zero,
                query_uncertainty_available=unavailable,
                entry_uncertainty=zero,
                entry_uncertainty_available=unavailable,
                entry_safety_cost=zero,
                entry_safety_cost_available=available,
                entry_reliability=jnp.asarray(1.0, dtype=jnp.float32),
                provenance_id=jnp.asarray(
                    _N_AGENTS * step + index,
                    dtype=jnp.int32,
                ),
                source_id=jnp.asarray(index, dtype=jnp.int32),
            )

        return make(0), make(1)

    def causal_core_memory_event_inputs(
        self,
        state: HCCLContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
    ) -> tuple[
        ExternalLearnedStateLiveMemoryEventInput,
        ExternalLearnedStateLiveMemoryEventInput,
    ]:
        """Issue the only two memory-metadata records accepted for ``event``."""

        self._state_contract(state)
        self._hccl.world._require_event_contract(event)
        if _contains_tracer((state, event)):
            raise TypeError("causal-core memory metadata issuance is host/eager-only")
        if not _bool(
            self._hccl.world.event_receipt_valid(
                state.hccl_state.world_state,
                event,
            )
        ):
            raise ValueError("memory metadata issuance requires the exact prepared event")
        return self._causal_core_memory_event_inputs(event)

    def _action_identity_rows(
        self,
        *,
        source_words: Array,
        event_words: Array,
        action_stack_words: Array,
        planner_agent_words: Array,
        context_tokens: Array,
        layer: HCCLActionLayer,
    ) -> UInt[Array, "2 4"]:
        rows = jnp.stack(
            tuple(
                _tree_digest(
                    "continual-dyad-hccl-action-identity-v2",
                    self._owner_words,
                    source_words,
                    event_words,
                    jnp.asarray(int(layer), dtype=jnp.int32),
                    jnp.asarray(index, dtype=jnp.int32),
                    action_stack_words[index],
                    planner_agent_words[index],
                    context_tokens[index],
                )[:4]
                for index in range(_N_AGENTS)
            )
        ).astype(jnp.uint32)
        return rows

    def _binding_tag(
        self,
        binding: HCCLContinualDyadActionBinding,
    ) -> UInt[Array, " 8"]:
        bare = cast(
            HCCLContinualDyadActionBinding,
            binding.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(HCCL_CONTINUAL_DYAD_BINDING_SCHEMA, bare)

    def _make_binding(
        self,
        state: HCCLContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
    ) -> HCCLContinualDyadActionBinding:
        agents = (state.agent_0_state, state.agent_1_state)
        planner_agents = (state.planner_state.agent_0, state.planner_state.agent_1)
        contexts = (state.context_0_state, state.context_1_state)
        source_words = _tree_digest("continual-dyad-action-source-v2", state)
        event_words = _tree_digest("continual-dyad-event-v2", event)
        action_stack_words = jnp.stack(
            tuple(
                _tree_digest("continual-dyad-action-stack-v2", agent)
                for agent in agents
            )
        ).astype(jnp.uint32)
        planner_agent_words = jnp.stack(
            tuple(
                _tree_digest("continual-dyad-planner-agent-v2", planner_agent)
                for planner_agent in planner_agents
            )
        ).astype(jnp.uint32)
        context_tokens = jnp.stack(
            tuple(context.content_token for context in contexts)
        ).astype(jnp.uint8)
        base_actions = jnp.stack(
            tuple(agent.action_binding.base_action for agent in agents)
        ).astype(jnp.int32)
        memory_before = jnp.stack(
            tuple(agent.action_binding.memory_action_before_mask for agent in agents)
        ).astype(jnp.int32)
        memory_actions = jnp.stack(
            tuple(agent.action_binding.memory_action for agent in agents)
        ).astype(jnp.int32)
        planner_before = jnp.stack(
            tuple(agent.action_binding.planner_action_before_mask for agent in agents)
        ).astype(jnp.int32)
        final_actions = jnp.stack(
            tuple(agent.action_binding.final_action for agent in agents)
        ).astype(jnp.int32)
        masks = jnp.stack(
            tuple(agent.action_binding.hard_action_mask for agent in agents)
        ).astype(jnp.bool_)
        receipts: list[HCCLActionReceipt] = []
        for layer, before, after in (
            (HCCLActionLayer.BASE, base_actions, base_actions),
            (HCCLActionLayer.MEMORY, memory_before, memory_actions),
            (HCCLActionLayer.PLANNER, planner_before, final_actions),
        ):
            identities = self._action_identity_rows(
                source_words=source_words,
                event_words=event_words,
                action_stack_words=action_stack_words,
                planner_agent_words=planner_agent_words,
                context_tokens=context_tokens,
                layer=layer,
            )
            receipts.append(
                self._hccl.bind_action_receipt(
                    state.hccl_state,
                    event,
                    layer=layer,
                    actions_before_mask=before,
                    actions_after_mask=after,
                    hard_action_masks=masks,
                    action_receipt_identity_words=identities,
                )
            )
        all_identities: NDArray[np.uint32] = np.reshape(
            np.stack(
                tuple(np.asarray(item.action_receipt_identity_words) for item in receipts)
            ),
            (6, 4),
        )
        if len({tuple(int(word) for word in row) for row in all_identities}) != 6:
            raise RuntimeError("HCCL action receipt identities must be pairwise distinct")
        bare = HCCLContinualDyadActionBinding(
            source_state_words=source_words,
            event_words=event_words,
            action_stack_words=action_stack_words,
            planner_agent_words=planner_agent_words,
            context_content_tokens=context_tokens,
            base_actions=base_actions,
            memory_actions_before_mask=memory_before,
            memory_actions=memory_actions,
            planner_actions_before_mask=planner_before,
            final_actions=final_actions,
            hard_action_masks=masks,
            base=receipts[0],
            memory=receipts[1],
            planner=receipts[2],
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLContinualDyadActionBinding,
            bare.replace(content_tag_words=self._binding_tag(bare)),
        )

    def _binding_contract(self, binding: HCCLContinualDyadActionBinding) -> None:
        if type(binding) is not HCCLContinualDyadActionBinding:
            raise TypeError("binding must be an exact continual-dyad action binding")
        for name, shape, dtype in (
            ("source_state_words", (_DIGEST_WORDS,), jnp.uint32),
            ("event_words", (_DIGEST_WORDS,), jnp.uint32),
            ("action_stack_words", (_N_AGENTS, _DIGEST_WORDS), jnp.uint32),
            ("planner_agent_words", (_N_AGENTS, _DIGEST_WORDS), jnp.uint32),
            ("context_content_tokens", (_N_AGENTS, _TOKEN_NBYTES), jnp.uint8),
            ("base_actions", (_N_AGENTS,), jnp.int32),
            ("memory_actions_before_mask", (_N_AGENTS,), jnp.int32),
            ("memory_actions", (_N_AGENTS,), jnp.int32),
            ("planner_actions_before_mask", (_N_AGENTS,), jnp.int32),
            ("final_actions", (_N_AGENTS,), jnp.int32),
            ("hard_action_masks", (_N_AGENTS, _N_ACTIONS), jnp.bool_),
            ("content_tag_words", (_DIGEST_WORDS,), jnp.uint32),
        ):
            _require_array(
                getattr(binding, name),
                name=f"binding.{name}",
                shape=shape,
                dtype=dtype,
            )
        for receipt in (binding.base, binding.memory, binding.planner):
            if type(receipt) is not HCCLActionReceipt:
                raise TypeError("binding action receipts must be exact HCCLActionReceipt")
            self._hccl.attribution._require_action_contract(receipt)

    def binding_valid(
        self,
        state: HCCLContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
        binding: HCCLContinualDyadActionBinding,
        *,
        _work: _OuterValidationWork | None = None,
    ) -> Bool[Array, ""]:
        """Recompute the exact source/event/B/M/P receipt bundle."""

        self._state_contract(state)
        self._hccl.world._require_event_contract(event)
        self._binding_contract(binding)
        if _contains_tracer((state, event, binding)):
            raise TypeError("continual-dyad binding validation is host/eager-only")
        expected = self._make_binding(state, event)
        valid = (
            _bool(self.state_valid(state, _work=_work))
            and _bool(self._hccl.world.event_receipt_valid(state.hccl_state.world_state, event))
            and _tree_exact_equal(binding, expected)
            and np.array_equal(
                np.asarray(binding.content_tag_words),
                np.asarray(self._binding_tag(binding)),
            )
        )
        return jnp.asarray(valid, dtype=jnp.bool_)

    def bind_current_actions(
        self,
        state: HCCLContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
    ) -> HCCLContinualDyadActionBinding:
        """Issue the exact six-identity B/M/P receipt bundle for one event."""

        self._state_contract(state)
        self._hccl.world._require_event_contract(event)
        if _contains_tracer((state, event)):
            raise TypeError("continual-dyad action binding is host/eager-only")
        if not _bool(self.state_valid(state)):
            raise ValueError("action binding requires a valid continual-dyad state")
        if not _bool(
            self._hccl.world.event_receipt_valid(state.hccl_state.world_state, event)
        ):
            raise ValueError("action binding requires the exact prepared event")
        binding = self._make_binding(state, event)
        if not _bool(self.binding_valid(state, event, binding)):
            raise RuntimeError("new continual-dyad action binding is invalid")
        return binding

    @staticmethod
    def _pair_option(value: object, *, name: str) -> tuple[Any, Any]:
        if type(value) is not tuple or len(value) != _N_AGENTS:
            raise TypeError(f"{name} must be an exact pair")
        return cast(tuple[Any, Any], value)

    @staticmethod
    def _memory_feedback(
        state: ExternalLearnedStateLiveMemoryActionStackState,
        credit: Array,
    ) -> ExternalLearnedStateLiveMemoryActionStackFeedback | None:
        binding = state.action_binding
        if not _bool(binding.memory_feedback_required):
            return None
        retrieval_used = binding.retrieval_used_expected
        available = retrieval_used
        delta = jnp.where(
            retrieval_used,
            jnp.asarray(credit, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return ExternalLearnedStateLiveMemoryActionStackFeedback(
            action_binding_words=binding.content_tag_words,
            memory_transaction_words=binding.memory_transaction_words,
            prototype_decision_id=binding.prototype_decision_id,
            base_action=binding.base_action,
            memory_action=binding.memory_action,
            final_action=binding.final_action,
            hard_action_mask=binding.hard_action_mask,
            retrieval_used=retrieval_used,
            counterfactual_available=available,
            counterfactual_delta=delta,
        )

    def _transition(
        self,
        state: ExternalLearnedStateLiveMemoryActionStackState,
        *,
        executed_action: Array,
        reward: Array,
        next_observation: Array,
        horde_cumulants: Any,
        horde_discounts: Any,
    ) -> ExternalLearnedStateTransition:
        coordinator = state.coordinator_state
        return ExternalLearnedStateTransition(
            source_event_words=coordinator.event_words,
            source_builder_step_words=coordinator.cached_builder_step_words,
            source_prototype_step_words=coordinator.cached_prototype_step_words,
            source_feature_generation_words=coordinator.cached_feature_generation_words,
            observation=coordinator.current_raw_observation,
            representation=coordinator.current_representation,
            action=jnp.asarray(executed_action, dtype=jnp.int32),
            decision_id=coordinator.current_decision_id,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(self._config.discount, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            next_observation=jnp.asarray(next_observation, dtype=jnp.float32),
            next_decision_observation=jnp.asarray(next_observation, dtype=jnp.float32),
            horde_cumulants=horde_cumulants,
            horde_discounts=horde_discounts,
        )

    def _horde_targets(
        self,
        proposal: HCCLCausalCoreProposal,
        signals: HCCLCausalCoreTypedSignals,
        executed_actions: Array,
    ) -> tuple[Float[Array, "2 8"], Float[Array, "2 8"]]:
        """Derive the fixed causal-core Horde questions from the committed PP row."""

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
        prototype = self._config.agent_0.coordinator.inner.prototype
        horde = prototype.horde_spec
        if horde is None:
            raise RuntimeError("validated HCCL Horde config disappeared")
        discounts = jnp.broadcast_to(
            horde.gammas,
            (_N_AGENTS, len(_HCCL_HORDE_HEADS)),
        ).astype(jnp.float32)
        return jnp.stack(rows).astype(jnp.float32), discounts

    def _child_finalization_bound(
        self,
        adapter: ExternalLearnedStateLiveMemoryActionStackAdapter,
        agent: HCCLContinualDyadPreparedAgent,
        selected_prototype: Any,
        planner_before: Array,
        planner_consumed: Array,
        planner_candidate_words: Array,
    ) -> bool:
        """Reconstruct the child P projection without trusting its cached valid bit."""

        preparation = agent.memory_preparation
        finalized = agent.finalization
        final_binding = finalized.final_action_binding
        memory_state = preparation.memory_candidate_state
        memory_binding = memory_state.action_binding
        candidate = finalized.candidate_state
        candidate_binding = candidate.action_binding
        expected_coordinator = adapter._install_selected_prototype(
            memory_state,
            selected_prototype,
        )
        inherited = all(
            _tree_exact_equal(
                getattr(candidate_binding, name),
                getattr(memory_binding, name),
            )
            for name in (
                "available",
                "memory_feedback_required",
                "memory_transaction_words",
                "prototype_decision_id",
                "base_action",
                "memory_action_before_mask",
                "memory_action",
                "hard_action_mask",
                "categorical_retrieval",
                "retrieval_used_expected",
                "memory_candidate_words",
            )
        )
        work = finalized.bind_work
        zero_donor_work = all(
            int(jax.device_get(getattr(work, name))) == 0
            for name in (
                "prototype_replacement_evaluations",
                "coordinator_update_evaluations",
                "planner_model_evaluations",
                "learned_memory_evaluations",
            )
        )
        return all(
            (
                _bool(adapter._recomputed_standard_finalized_valid(finalized)),
                _tree_exact_equal(finalized.memory_preparation, preparation),
                np.array_equal(
                    np.asarray(preparation.content_tag_words),
                    np.asarray(adapter._memory_preparation_tag(preparation)),
                ),
                _bool(preparation.preparation_valid),
                np.array_equal(
                    np.asarray(final_binding.source_memory_preparation_words),
                    np.asarray(preparation.content_tag_words),
                ),
                np.array_equal(
                    np.asarray(final_binding.final_action_owner_words),
                    np.asarray(adapter._owner_words),
                ),
                np.array_equal(
                    np.asarray(final_binding.prototype_decision_id),
                    np.asarray(memory_binding.prototype_decision_id),
                ),
                int(final_binding.memory_action) == int(memory_binding.memory_action),
                int(final_binding.planner_action_before_mask) == int(planner_before),
                int(final_binding.final_action) == int(selected_prototype.current_action),
                np.array_equal(
                    np.asarray(final_binding.hard_action_mask),
                    np.asarray(preparation.hard_action_mask),
                ),
                np.array_equal(
                    np.asarray(final_binding.planner_candidate_words),
                    np.asarray(planner_candidate_words),
                ),
                bool(final_binding.planner_consumed) == bool(planner_consumed),
                _tree_exact_equal(
                    final_binding.selected_prototype_state,
                    selected_prototype,
                ),
                np.array_equal(
                    np.asarray(final_binding.content_tag_words),
                    np.asarray(adapter._final_binding_tag(final_binding)),
                ),
                _tree_exact_equal(candidate.coordinator_state, expected_coordinator),
                _tree_exact_equal(
                    candidate.learned_memory_state,
                    memory_state.learned_memory_state,
                ),
                np.array_equal(
                    np.asarray(candidate.schema_digest),
                    np.asarray(memory_state.schema_digest),
                ),
                inherited,
                _bool(candidate_binding.planner_bound),
                bool(candidate_binding.planner_consumed) == bool(planner_consumed),
                int(candidate_binding.planner_action_before_mask) == int(planner_before),
                int(candidate_binding.final_action) == int(selected_prototype.current_action),
                np.array_equal(
                    np.asarray(candidate_binding.planner_candidate_words),
                    np.asarray(planner_candidate_words),
                ),
                np.array_equal(
                    np.asarray(candidate_binding.final_action_owner_words),
                    np.asarray(adapter._owner_words),
                ),
                np.array_equal(
                    np.asarray(candidate_binding.final_prototype_words),
                    np.asarray(final_binding.final_prototype_words),
                ),
                np.array_equal(
                    np.asarray(finalized.content_tag_words),
                    np.asarray(adapter._finalized_tag(finalized)),
                ),
                _bool(adapter.state_valid(candidate)),
                _bool(finalized.finalization_valid),
                int(jax.device_get(work.final_action_binding_evaluations)) == 1,
                zero_donor_work,
            )
        )

    @staticmethod
    def _feature_lifecycle_work(
        preparation: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
    ) -> tuple[bool, int]:
        donor = preparation.donor_prepared
        if donor is None or donor.coordinator_result is None:
            return False, 0
        diagnostics = (
            donor.coordinator_result.evaluated.prepared.inner_result.prototype_result
            .prototype_feature_lifecycle_diagnostics
        )
        if diagnostics is None or not _bool(diagnostics.available):
            return False, 0
        return True, int(_bool(diagnostics.lifecycle.routing_attempted))

    def _make_through_memory_work(
        self,
        hccl_result: HCCLWorldAttributionAdapterResult,
        memory_preparations: tuple[
            ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
            ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
        ],
        validation_work: _OuterValidationWork,
    ) -> HCCLContinualDyadThroughMemoryWork:
        lifecycle_work = tuple(
            self._feature_lifecycle_work(item) for item in memory_preparations
        )
        return HCCLContinualDyadThroughMemoryWork(
            supplied_event_receipts=jnp.asarray(1, dtype=jnp.int32),
            supplied_action_binding_bundles=jnp.asarray(1, dtype=jnp.int32),
            event_receipt_preparations=jnp.asarray(0, dtype=jnp.int32),
            event_random_draws=jnp.asarray(0, dtype=jnp.int32),
            action_receipt_validation_rebindings=jnp.asarray(3, dtype=jnp.int32),
            action_identity_validation_recomputations=jnp.asarray(6, dtype=jnp.int32),
            planner_validation_pair_authentication_calls=jnp.asarray(
                validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            planner_validation_agent_cache_authentication_evaluations=jnp.asarray(
                _N_AGENTS * validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            planner_validation_behavior_probability_vector_evaluations=jnp.asarray(
                _N_AGENTS * validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            planner_validation_grounded_joint_cell_prediction_equivalents=jnp.asarray(
                _N_AGENTS
                * _N_ACTIONS**2
                * validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            planner_validation_expected_reward_marginalization_products=jnp.asarray(
                _N_AGENTS
                * _N_ACTIONS**2
                * validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            context_preparations=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            hccl_stage_calls=jnp.asarray(1, dtype=jnp.int32),
            world_proposal_calls=hccl_result.work.world_proposal_calls,
            attribution_proposal_calls=hccl_result.work.attribution_proposal_calls,
            designated_counterfactual_slots=(
                hccl_result.work.designated_counterfactual_world_slots
            ),
            inner_discarded_world_proposal_calls=(
                hccl_result.work.discarded_world_proposal_calls
            ),
            inner_selected_pp_world_successors=(
                hccl_result.work.committed_pp_world_successors
            ),
            outer_committed_pp_world_successors=jnp.asarray(0, dtype=jnp.int32),
            world_duplicate_mm_checks=(
                hccl_result.work.duplicate_mm_world_equality_checks
            ),
            attribution_duplicate_mm_checks=(
                hccl_result.attribution.work.duplicate_mm_equality_checks
            ),
            memory_credit_panel_derivations=jnp.asarray(1, dtype=jnp.int32),
            memory_credit_readouts=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            context_steps=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            lineage_proposals=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            action_stack_memory_preparations=jnp.ones(
                (_N_AGENTS,),
                dtype=jnp.int32,
            ),
            feedback_settlement_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.feedback_settlement_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            coordinator_update_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.coordinator_update_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            memory_action_replacement_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.memory_action_replacement_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            fast_state_transition_attempts=jnp.asarray(
                tuple(
                    item.prepare_work.coordinator_update_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            prototype_transition_attempts=jnp.asarray(
                tuple(
                    item.prepare_work.coordinator_update_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            feature_lifecycle_route_attempts=jnp.asarray(
                tuple(item[1] for item in lifecycle_work),
                dtype=jnp.int32,
            ),
            feature_lifecycle_arithmetic_count_available=jnp.asarray(
                tuple(item[0] for item in lifecycle_work),
                dtype=jnp.bool_,
            ),
            active_pair_value_materializations=jnp.asarray(
                tuple(5 * _ACTIVE_PAIR_SLOTS if item[0] else 0 for item in lifecycle_work),
                dtype=jnp.int32,
            ),
            candidate_pair_product_materializations=jnp.asarray(
                tuple(_PAIR_CANDIDATE_SLOTS if item[0] else 0 for item in lifecycle_work),
                dtype=jnp.int32,
            ),
            lifecycle_router_candidate_evaluations=jnp.asarray(
                tuple(2 if item[0] else 0 for item in lifecycle_work),
                dtype=jnp.int32,
            ),
            active_pair_slot_capacity=jnp.full(
                (_N_AGENTS,),
                _ACTIVE_PAIR_SLOTS,
                dtype=jnp.int32,
            ),
            pair_candidate_capacity=jnp.full(
                (_N_AGENTS,),
                _PAIR_CANDIDATE_SLOTS,
                dtype=jnp.int32,
            ),
            routed_representation_width=jnp.full(
                (_N_AGENTS,),
                _ROUTED_DIM,
                dtype=jnp.int32,
            ),
            coordinator_base_action_candidates=jnp.asarray(
                tuple(
                    item.prepare_work.coordinator_update_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            memory_action_candidates=jnp.asarray(
                tuple(
                    item.prepare_work.memory_action_replacement_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            learned_memory_query_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.learned_memory_query_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            learned_memory_write_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.learned_memory_write_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            learned_memory_reencode_evaluations=jnp.zeros(
                (_N_AGENTS,),
                dtype=jnp.int32,
            ),
            learned_memory_reencode_count_available=jnp.zeros(
                (_N_AGENTS,),
                dtype=jnp.bool_,
            ),
            agent_content_digest_evaluations=jnp.ones(
                (_N_AGENTS,),
                dtype=jnp.int32,
            ),
            transaction_content_digest_evaluations=jnp.asarray(1, dtype=jnp.int32),
        )

    def _through_memory_agent_tag(
        self,
        agent: HCCLContinualDyadThroughMemoryAgent,
    ) -> UInt[Array, " 8"]:
        bare = cast(
            HCCLContinualDyadThroughMemoryAgent,
            agent.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_AGENT_SCHEMA,
            self._config_words,
            bare,
        )

    def _seal_through_memory_agent(
        self,
        agent: HCCLContinualDyadThroughMemoryAgent,
    ) -> HCCLContinualDyadThroughMemoryAgent:
        return cast(
            HCCLContinualDyadThroughMemoryAgent,
            agent.replace(content_tag_words=self._through_memory_agent_tag(agent)),
        )

    def _through_memory_tag(
        self,
        prepared: HCCLContinualDyadThroughMemoryTransaction,
    ) -> UInt[Array, " 8"]:
        bare = cast(
            HCCLContinualDyadThroughMemoryTransaction,
            prepared.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(
            HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA,
            self._config_words,
            bare,
        )

    def _seal_through_memory(
        self,
        prepared: HCCLContinualDyadThroughMemoryTransaction,
    ) -> HCCLContinualDyadThroughMemoryTransaction:
        return cast(
            HCCLContinualDyadThroughMemoryTransaction,
            prepared.replace(content_tag_words=self._through_memory_tag(prepared)),
        )

    def _through_memory_contract(
        self,
        prepared: HCCLContinualDyadThroughMemoryTransaction,
    ) -> None:
        if type(prepared) is not HCCLContinualDyadThroughMemoryTransaction:
            raise TypeError("prepared must be an exact through-memory transaction")
        if type(prepared.source_state) is not HCCLContinualDyadState:
            raise TypeError("prepared.source_state has the wrong type")
        if type(prepared.event) is not HCCLCausalCoreEventReceipt:
            raise TypeError("prepared.event has the wrong type")
        self._binding_contract(prepared.binding)
        if type(prepared.hccl_result) is not HCCLWorldAttributionAdapterResult:
            raise TypeError("prepared.hccl_result has the wrong type")
        if type(prepared.memory_credit_panel) is not HCCLMemoryCreditEstimandPanel:
            raise TypeError("prepared.memory_credit_panel has the wrong type")
        _require_array(
            prepared.next_hard_action_masks,
            name="prepared.next_hard_action_masks",
            shape=(_N_AGENTS, _N_ACTIONS),
            dtype=jnp.bool_,
        )
        for index, agent in enumerate((prepared.agent_0, prepared.agent_1)):
            if type(agent) is not HCCLContinualDyadThroughMemoryAgent:
                raise TypeError(f"prepared.agent_{index} has the wrong type")
            _require_array(
                agent.agent_index,
                name=f"prepared.agent_{index}.agent_index",
                shape=(),
                dtype=jnp.int32,
            )
            _require_array(
                agent.memory_credit,
                name=f"prepared.agent_{index}.memory_credit",
                shape=(),
                dtype=jnp.float32,
            )
            if type(agent.context_preparation) is not ContextLineageRetentionPreparation:
                raise TypeError(f"prepared.agent_{index} context preparation is malformed")
            if type(agent.context_result) is not ContextLineageRetentionStepResult:
                raise TypeError(f"prepared.agent_{index} context result is malformed")
            if type(agent.transition) is not ExternalLearnedStateTransition:
                raise TypeError(f"prepared.agent_{index} transition is malformed")
            if type(agent.memory_preparation) is not (
                ExternalLearnedStateLiveMemoryActionStackMemoryPreparation
            ):
                raise TypeError(f"prepared.agent_{index} memory preparation is malformed")
            if agent.memory_feedback is not None and type(agent.memory_feedback) is not (
                ExternalLearnedStateLiveMemoryActionStackFeedback
            ):
                raise TypeError(f"prepared.agent_{index} feedback is malformed")
            _require_array(
                agent.content_tag_words,
                name=f"prepared.agent_{index}.content_tag_words",
                shape=(_DIGEST_WORDS,),
                dtype=jnp.uint32,
            )
        if type(prepared.work) is not HCCLContinualDyadThroughMemoryWork:
            raise TypeError("prepared.work has the wrong type")
        vector_bool = {
            "feature_lifecycle_arithmetic_count_available",
            "learned_memory_reencode_count_available",
        }
        scalar_int = {
            "supplied_event_receipts",
            "supplied_action_binding_bundles",
            "event_receipt_preparations",
            "event_random_draws",
            "action_receipt_validation_rebindings",
            "action_identity_validation_recomputations",
            "planner_validation_pair_authentication_calls",
            "planner_validation_agent_cache_authentication_evaluations",
            "planner_validation_behavior_probability_vector_evaluations",
            "planner_validation_grounded_joint_cell_prediction_equivalents",
            "planner_validation_expected_reward_marginalization_products",
            "hccl_stage_calls",
            "world_proposal_calls",
            "attribution_proposal_calls",
            "designated_counterfactual_slots",
            "inner_discarded_world_proposal_calls",
            "inner_selected_pp_world_successors",
            "outer_committed_pp_world_successors",
            "world_duplicate_mm_checks",
            "attribution_duplicate_mm_checks",
            "memory_credit_panel_derivations",
            "transaction_content_digest_evaluations",
        }
        for field in dataclasses.fields(HCCLContinualDyadThroughMemoryWork):
            name = field.name
            _require_array(
                getattr(prepared.work, name),
                name=f"prepared.work.{name}",
                shape=() if name in scalar_int else (_N_AGENTS,),
                dtype=jnp.bool_ if name in vector_bool else jnp.int32,
            )
        vector_flags = {
            "pre_outcome_context_bound",
            "memory_preparations_valid",
            "context_candidates_valid",
        }
        for name in (
            "source_state_valid",
            "event_valid",
            "binding_valid",
            "binding_matches_source",
            "pre_outcome_context_bound",
            "hccl_staged_once",
            "credit_algebra_valid",
            "memory_preparations_valid",
            "context_candidates_valid",
            "through_memory_valid",
        ):
            _require_array(
                getattr(prepared, name),
                name=f"prepared.{name}",
                shape=(_N_AGENTS,) if name in vector_flags else (),
                dtype=jnp.bool_,
            )
        _require_array(
            prepared.content_tag_words,
            name="prepared.content_tag_words",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )

    def _through_memory_work_valid(
        self,
        prepared: HCCLContinualDyadThroughMemoryTransaction,
    ) -> bool:
        work = prepared.work
        agents = (prepared.agent_0, prepared.agent_1)
        exact_scalars = {
            "supplied_event_receipts": 1,
            "supplied_action_binding_bundles": 1,
            "event_receipt_preparations": 0,
            "event_random_draws": 0,
            "action_receipt_validation_rebindings": 3,
            "action_identity_validation_recomputations": 6,
            "planner_validation_pair_authentication_calls": 2,
            "planner_validation_agent_cache_authentication_evaluations": 4,
            "planner_validation_behavior_probability_vector_evaluations": 4,
            "planner_validation_grounded_joint_cell_prediction_equivalents": 16,
            "planner_validation_expected_reward_marginalization_products": 16,
            "hccl_stage_calls": 1,
            "memory_credit_panel_derivations": 1,
            "transaction_content_digest_evaluations": 1,
        }
        if any(
            int(jax.device_get(getattr(work, name))) != expected
            for name, expected in exact_scalars.items()
        ):
            return False
        exact_vectors = (
            ("context_preparations", (1, 1)),
            ("memory_credit_readouts", (1, 1)),
            ("context_steps", (1, 1)),
            ("lineage_proposals", (1, 1)),
            ("action_stack_memory_preparations", (1, 1)),
            ("active_pair_slot_capacity", (12, 12)),
            ("pair_candidate_capacity", (120, 120)),
            ("routed_representation_width", (35, 35)),
            ("learned_memory_reencode_evaluations", (0, 0)),
            ("agent_content_digest_evaluations", (1, 1)),
        )
        if not all(
            np.array_equal(
                np.asarray(getattr(work, name)),
                np.asarray(expected, dtype=np.int32),
            )
            for name, expected in exact_vectors
        ):
            return False
        if np.any(np.asarray(work.learned_memory_reencode_count_available)):
            return False
        child_fields = (
            "feedback_settlement_evaluations",
            "coordinator_update_evaluations",
            "memory_action_replacement_evaluations",
            "learned_memory_query_evaluations",
            "learned_memory_write_evaluations",
        )
        for name in child_fields:
            child_expected: NDArray[np.int32] = np.asarray(
                tuple(
                    int(jax.device_get(getattr(agent.memory_preparation.prepare_work, name)))
                    for agent in agents
                ),
                dtype=np.int32,
            )
            if not np.array_equal(np.asarray(getattr(work, name)), child_expected):
                return False
        coordinator_counts: NDArray[np.int32] = np.asarray(
            tuple(
                int(
                    jax.device_get(
                        agent.memory_preparation.prepare_work.coordinator_update_evaluations
                    )
                )
                for agent in agents
            ),
            dtype=np.int32,
        )
        replacement_counts: NDArray[np.int32] = np.asarray(
            tuple(
                int(
                    jax.device_get(
                        agent.memory_preparation.prepare_work
                        .memory_action_replacement_evaluations
                    )
                )
                for agent in agents
            ),
            dtype=np.int32,
        )
        for name, count_expected in (
            ("fast_state_transition_attempts", coordinator_counts),
            ("prototype_transition_attempts", coordinator_counts),
            ("coordinator_base_action_candidates", coordinator_counts),
            ("memory_action_candidates", replacement_counts),
        ):
            if not np.array_equal(np.asarray(getattr(work, name)), count_expected):
                return False
        lifecycle = tuple(
            self._feature_lifecycle_work(agent.memory_preparation) for agent in agents
        )
        if not np.array_equal(
            np.asarray(work.feature_lifecycle_arithmetic_count_available),
            np.asarray(tuple(item[0] for item in lifecycle), dtype=np.bool_),
        ):
            return False
        for name, lifecycle_expected in (
            ("feature_lifecycle_route_attempts", tuple(item[1] for item in lifecycle)),
            (
                "active_pair_value_materializations",
                tuple(5 * _ACTIVE_PAIR_SLOTS if item[0] else 0 for item in lifecycle),
            ),
            (
                "candidate_pair_product_materializations",
                tuple(_PAIR_CANDIDATE_SLOTS if item[0] else 0 for item in lifecycle),
            ),
            (
                "lifecycle_router_candidate_evaluations",
                tuple(2 if item[0] else 0 for item in lifecycle),
            ),
        ):
            if not np.array_equal(
                np.asarray(getattr(work, name)),
                np.asarray(lifecycle_expected, dtype=np.int32),
            ):
                return False
        return all(
            (
                int(work.world_proposal_calls)
                == int(prepared.hccl_result.work.world_proposal_calls),
                int(work.attribution_proposal_calls)
                == int(prepared.hccl_result.work.attribution_proposal_calls),
                int(work.designated_counterfactual_slots)
                == int(
                    prepared.hccl_result.work.designated_counterfactual_world_slots
                ),
                int(work.inner_discarded_world_proposal_calls)
                == int(prepared.hccl_result.work.discarded_world_proposal_calls),
                int(work.inner_selected_pp_world_successors)
                == int(prepared.hccl_result.work.committed_pp_world_successors),
                int(work.outer_committed_pp_world_successors) == 0,
                int(work.world_duplicate_mm_checks)
                == int(prepared.hccl_result.work.duplicate_mm_world_equality_checks),
                int(work.attribution_duplicate_mm_checks)
                == int(
                    prepared.hccl_result.attribution.work.duplicate_mm_equality_checks
                ),
            )
        )

    def _through_memory_content_valid(
        self,
        state: HCCLContinualDyadState,
        prepared: HCCLContinualDyadThroughMemoryTransaction,
        *,
        _work: _OuterValidationWork | None = None,
    ) -> bool:
        """Validate the sealed split without repeating a transition donor."""

        self._state_contract(state)
        self._through_memory_contract(prepared)
        if _contains_tracer((state, prepared)):
            raise TypeError("through-memory validation is host/eager-only")
        if not _tree_exact_equal(state, prepared.source_state):
            return False
        masks = self._hard_action_masks(
            prepared.next_hard_action_masks,
            name="prepared.next_hard_action_masks",
        )
        if not np.array_equal(
            np.asarray(prepared.content_tag_words),
            np.asarray(self._through_memory_tag(prepared)),
        ):
            return False

        source_valid = self.state_valid(state, _work=_work)
        event_valid = self._hccl.world.event_receipt_valid(
            state.hccl_state.world_state,
            prepared.event,
        )
        binding = prepared.binding
        source_agents = (state.agent_0_state, state.agent_1_state)
        source_contexts = (state.context_0_state, state.context_1_state)
        planner_agents = (state.planner_state.agent_0, state.planner_state.agent_1)
        expected_source_words = _tree_digest("continual-dyad-action-source-v2", state)
        expected_event_words = _tree_digest("continual-dyad-event-v2", prepared.event)
        expected_action_stack_words = jnp.stack(
            tuple(
                _tree_digest("continual-dyad-action-stack-v2", agent)
                for agent in source_agents
            )
        ).astype(jnp.uint32)
        expected_planner_agent_words = jnp.stack(
            tuple(
                _tree_digest("continual-dyad-planner-agent-v2", agent)
                for agent in planner_agents
            )
        ).astype(jnp.uint32)
        expected_context_tokens = jnp.stack(
            tuple(item.content_token for item in source_contexts)
        ).astype(jnp.uint8)
        expected_base = jnp.stack(
            tuple(item.action_binding.base_action for item in source_agents)
        ).astype(jnp.int32)
        expected_memory_before = jnp.stack(
            tuple(
                item.action_binding.memory_action_before_mask
                for item in source_agents
            )
        ).astype(jnp.int32)
        expected_memory = jnp.stack(
            tuple(item.action_binding.memory_action for item in source_agents)
        ).astype(jnp.int32)
        expected_planner_before = jnp.stack(
            tuple(
                item.action_binding.planner_action_before_mask
                for item in source_agents
            )
        ).astype(jnp.int32)
        expected_final = jnp.stack(
            tuple(item.action_binding.final_action for item in source_agents)
        ).astype(jnp.int32)
        expected_current_masks = jnp.stack(
            tuple(item.action_binding.hard_action_mask for item in source_agents)
        ).astype(jnp.bool_)
        binding_valid = jnp.asarray(
            all(
                (
                    np.array_equal(
                        np.asarray(binding.content_tag_words),
                        np.asarray(self._binding_tag(binding)),
                    ),
                    np.array_equal(
                        np.asarray(binding.source_state_words),
                        np.asarray(expected_source_words),
                    ),
                    np.array_equal(
                        np.asarray(binding.event_words),
                        np.asarray(expected_event_words),
                    ),
                    np.array_equal(
                        np.asarray(binding.action_stack_words),
                        np.asarray(expected_action_stack_words),
                    ),
                    np.array_equal(
                        np.asarray(binding.planner_agent_words),
                        np.asarray(expected_planner_agent_words),
                    ),
                    np.array_equal(
                        np.asarray(binding.context_content_tokens),
                        np.asarray(expected_context_tokens),
                    ),
                    np.array_equal(
                        np.asarray(binding.base_actions),
                        np.asarray(expected_base),
                    ),
                    np.array_equal(
                        np.asarray(binding.memory_actions_before_mask),
                        np.asarray(expected_memory_before),
                    ),
                    np.array_equal(
                        np.asarray(binding.memory_actions),
                        np.asarray(expected_memory),
                    ),
                    np.array_equal(
                        np.asarray(binding.planner_actions_before_mask),
                        np.asarray(expected_planner_before),
                    ),
                    np.array_equal(
                        np.asarray(binding.final_actions),
                        np.asarray(expected_final),
                    ),
                    np.array_equal(
                        np.asarray(binding.hard_action_masks),
                        np.asarray(expected_current_masks),
                    ),
                )
            ),
            dtype=jnp.bool_,
        )

        hccl = prepared.hccl_result
        hccl_valid = jnp.asarray(
            all(
                (
                    _bool(hccl.update_applied),
                    _bool(self._hccl.state_valid(hccl.state)),
                    np.array_equal(
                        np.asarray(hccl.pre_transaction_words),
                        np.asarray(state.hccl_state.world_state.step_words),
                    ),
                    np.array_equal(
                        np.asarray(hccl.post_transaction_words),
                        np.asarray(hccl.state.world_state.step_words),
                    ),
                    _bool(hccl.event_receipt_valid),
                    _bool(hccl.action_receipt_identities_bound),
                    int(hccl.work.world_proposal_calls) == 8,
                    int(hccl.work.attribution_proposal_calls) == 8,
                    self._hccl_result_structurally_bound(
                        state.hccl_state,
                        prepared.event,
                        binding,
                        hccl,
                    ),
                )
            ),
            dtype=jnp.bool_,
        )
        panel = derive_hccl_memory_credit_estimands(
            mm=_signals_at(hccl.world_proposals, _MM_SLOT),
            b0m1=_signals_at(hccl.world_proposals, _B0M1_SLOT),
            m0b1=_signals_at(hccl.world_proposals, _M0B1_SLOT),
            bb=_signals_at(hccl.world_proposals, _BB_SLOT),
        )
        credit_valid = jnp.asarray(
            _tree_exact_equal(panel, prepared.memory_credit_panel)
            and _bool(panel.algebra.all_identities_hold),
            dtype=jnp.bool_,
        )
        pp = _signals_at(hccl.world_proposals, _PP_SLOT)
        pp_proposal = cast(
            HCCLCausalCoreProposal,
            jax.tree.map(lambda leaf: leaf[_PP_SLOT], hccl.world_proposals),
        )
        horde_cumulants, horde_discounts = self._horde_targets(
            pp_proposal,
            pp,
            binding.final_actions,
        )
        expected_credits = (
            panel.baseline_context_direct_effect.net_reward[0, 0],
            panel.baseline_context_direct_effect.net_reward[1, 1],
        )
        next_physical = hccl.world_proposals.next_observation[_PP_SLOT]
        agents = (prepared.agent_0, prepared.agent_1)
        adapters = (self._agent_0, self._agent_1)
        pre_context_bound: list[bool] = []
        context_bound: list[bool] = []
        memory_bound: list[bool] = []
        for index, agent in enumerate(agents):
            memory = agent.memory_preparation
            context_preparation = agent.context_preparation
            context_result = agent.context_result
            expected_partner = jax.nn.one_hot(
                binding.final_actions[1 - index],
                _N_ACTIONS,
                dtype=jnp.float32,
            )
            preparation_bound = all(
                (
                    int(agent.agent_index) == index,
                    self._context._preparation_integrity_valid(context_preparation),
                    np.array_equal(
                        np.asarray(context_preparation.source_content_token),
                        np.asarray(source_contexts[index].content_token),
                    ),
                    np.array_equal(
                        np.asarray(context_preparation.observation),
                        np.asarray(expected_partner),
                    ),
                    int(context_preparation.action)
                    == int(binding.final_actions[index]),
                )
            )
            pre_context_bound.append(preparation_bound)
            context_bound.append(
                preparation_bound
                and all(
                    (
                        _tree_exact_equal(
                            context_result.preparation,
                            context_preparation,
                        ),
                        _bool(context_result.update_applied),
                        _bool(context_result.context_owner_committed),
                        _bool(context_result.lineage_owner_committed),
                        _bool(context_result.protection_snapshotted_before_outcome),
                        not _bool(
                            context_result
                            .current_outcome_changed_current_eviction_protection
                        ),
                        np.array_equal(
                            np.asarray(context_result.lineage_event.reward),
                            np.asarray(pp.task_score),
                        ),
                        np.array_equal(
                            np.asarray(context_result.lineage_event.observation),
                            np.asarray(context_preparation.observation),
                        ),
                        int(context_result.lineage_event.action)
                        == int(context_preparation.action),
                        np.array_equal(
                            np.asarray(context_result.context_result.pre_step_words),
                            np.asarray(source_contexts[index].context.step_words),
                        ),
                        np.array_equal(
                            np.asarray(context_result.context_result.post_step_words),
                            np.asarray(context_result.state.context.step_words),
                        ),
                        _bool(context_result.context_result.update_applied),
                        _bool(self._context.state_is_valid(context_result.state)),
                        np.array_equal(
                            np.asarray(context_result.state.context.step_words),
                            np.asarray(hccl.post_transaction_words),
                        ),
                    )
                )
            )
            next_raw = self._composed_observation(
                next_physical[index],
                context_result.state,
            )
            expected_transition = self._transition(
                source_agents[index],
                executed_action=binding.final_actions[index],
                reward=pp.net_reward[index],
                next_observation=next_raw,
                horde_cumulants=horde_cumulants[index],
                horde_discounts=horde_discounts[index],
            )
            expected_feedback = self._memory_feedback(
                source_agents[index],
                expected_credits[index],
            )
            feedback_bound = (
                expected_feedback is None
                and agent.memory_feedback is None
                and not _bool(memory.feedback_supplied)
            ) or (
                expected_feedback is not None
                and agent.memory_feedback is not None
                and _tree_exact_equal(agent.memory_feedback, expected_feedback)
                and _tree_exact_equal(memory.feedback, expected_feedback)
                and _bool(memory.feedback_supplied)
            )
            memory_bound.append(
                all(
                    (
                        np.array_equal(
                            np.asarray(agent.content_tag_words),
                            np.asarray(self._through_memory_agent_tag(agent)),
                        ),
                        np.array_equal(
                            np.asarray(agent.memory_credit),
                            np.asarray(expected_credits[index]),
                        ),
                        _tree_exact_equal(agent.transition, expected_transition),
                        _tree_exact_equal(memory.source_state, source_agents[index]),
                        _tree_exact_equal(memory.transition, expected_transition),
                        feedback_bound,
                        np.array_equal(
                            np.asarray(memory.hard_action_mask),
                            np.asarray(masks[index]),
                        ),
                        np.array_equal(
                            np.asarray(memory.content_tag_words),
                            np.asarray(adapters[index]._memory_preparation_tag(memory)),
                        ),
                        _bool(memory.preparation_valid),
                    )
                )
            )
        pre_context_flags = jnp.asarray(pre_context_bound, dtype=jnp.bool_)
        context_flags = jnp.asarray(context_bound, dtype=jnp.bool_)
        memory_flags = jnp.asarray(memory_bound, dtype=jnp.bool_)
        all_valid = (
            source_valid
            & event_valid
            & binding_valid
            & jnp.all(pre_context_flags)
            & hccl_valid
            & credit_valid
            & jnp.all(memory_flags)
            & jnp.all(context_flags)
            & jnp.asarray(self._through_memory_work_valid(prepared), dtype=jnp.bool_)
        )
        stored_flags = all(
            (
                _tree_exact_equal(prepared.source_state_valid, source_valid),
                _tree_exact_equal(prepared.event_valid, event_valid),
                _tree_exact_equal(prepared.binding_valid, binding_valid),
                _tree_exact_equal(prepared.binding_matches_source, binding_valid),
                _tree_exact_equal(
                    prepared.pre_outcome_context_bound,
                    pre_context_flags,
                ),
                _tree_exact_equal(prepared.hccl_staged_once, hccl_valid),
                _tree_exact_equal(prepared.credit_algebra_valid, credit_valid),
                _tree_exact_equal(
                    prepared.memory_preparations_valid,
                    memory_flags,
                ),
                _tree_exact_equal(
                    prepared.context_candidates_valid,
                    context_flags,
                ),
                _tree_exact_equal(prepared.through_memory_valid, all_valid),
            )
        )
        return stored_flags and _bool(all_valid)

    def _prepared_tag(
        self,
        prepared: HCCLContinualDyadPreparedTransaction,
    ) -> UInt[Array, " 8"]:
        bare = cast(
            HCCLContinualDyadPreparedTransaction,
            prepared.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(HCCL_CONTINUAL_DYAD_PREPARED_SCHEMA, bare)

    def _seal_prepared(
        self,
        prepared: HCCLContinualDyadPreparedTransaction,
    ) -> HCCLContinualDyadPreparedTransaction:
        return cast(
            HCCLContinualDyadPreparedTransaction,
            prepared.replace(content_tag_words=self._prepared_tag(prepared)),
        )

    def prepare_through_memory(
        self,
        state: HCCLContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
        binding: HCCLContinualDyadActionBinding,
        agent_0_event_input: ExternalLearnedStateLiveMemoryEventInput,
        agent_1_event_input: ExternalLearnedStateLiveMemoryEventInput,
        next_hard_action_masks: Array,
        *,
        candidate_evidence: tuple[
            ExternalBuilderCandidateAuditEvidence | None,
            ExternalBuilderCandidateAuditEvidence | None,
        ] = (None, None),
        partner_policy_fusion_input: tuple[
            PrototypePartnerPolicyFusionInput | None,
            PrototypePartnerPolicyFusionInput | None,
        ] = (None, None),
        partner_policy_fusion_feedback: tuple[
            PrototypePartnerPolicyFusionFeedback | None,
            PrototypePartnerPolicyFusionFeedback | None,
        ] = (None, None),
        extended_action_masks: tuple[Array | None, Array | None] = (None, None),
    ) -> HCCLContinualDyadThroughMemoryTransaction:
        """Evaluate the HCCL, context, and memory phases exactly once."""

        self._state_contract(state)
        self._hccl.world._require_event_contract(event)
        self._binding_contract(binding)
        masks = self._hard_action_masks(
            next_hard_action_masks,
            name="next_hard_action_masks",
        )
        inputs = (agent_0_event_input, agent_1_event_input)
        for index, item in enumerate(inputs):
            if type(item) is not ExternalLearnedStateLiveMemoryEventInput:
                raise TypeError(f"agent_{index}_event_input must be an exact event input")
        evidence = self._pair_option(candidate_evidence, name="candidate_evidence")
        fusion_input = self._pair_option(
            partner_policy_fusion_input,
            name="partner_policy_fusion_input",
        )
        fusion_feedback = self._pair_option(
            partner_policy_fusion_feedback,
            name="partner_policy_fusion_feedback",
        )
        extended_masks = self._pair_option(
            extended_action_masks,
            name="extended_action_masks",
        )
        all_inputs = (
            state,
            event,
            binding,
            inputs,
            masks,
            evidence,
            fusion_input,
            fusion_feedback,
            extended_masks,
        )
        if _contains_tracer(all_inputs):
            raise TypeError("continual-dyad transaction preparation is host/eager-only")

        validation_work = _OuterValidationWork()
        source_valid = self.state_valid(state, _work=validation_work)
        event_valid = self._hccl.world.event_receipt_valid(
            state.hccl_state.world_state,
            event,
        )
        binding_valid = self.binding_valid(
            state,
            event,
            binding,
            _work=validation_work,
        )
        if not all((_bool(source_valid), _bool(event_valid), _bool(binding_valid))):
            raise ValueError(
                "continual-dyad source, event, or action binding failed preflight"
            )
        expected_inputs = self._causal_core_memory_event_inputs(event)
        if not all(
            _tree_exact_equal(item, expected)
            for item, expected in zip(inputs, expected_inputs, strict=True)
        ):
            raise ValueError(
                "causal-core memory event input must be the exact event/agent-bound "
                "neutral metadata record"
            )
        if any(
            item is not None
            for pair in (evidence, fusion_input, fusion_feedback, extended_masks)
            for item in pair
        ):
            raise ValueError(
                "causal-core candidate, fusion, and extended-option sidecars are "
                "unsupported"
            )
        agents = (state.agent_0_state, state.agent_1_state)
        contexts = (state.context_0_state, state.context_1_state)
        adapters = (self._agent_0, self._agent_1)

        context_preparations = tuple(
            self._context.prepare(
                contexts[index],
                jax.nn.one_hot(
                    binding.final_actions[1 - index],
                    _N_ACTIONS,
                    dtype=jnp.float32,
                ),
                binding.final_actions[index],
            )
            for index in range(_N_AGENTS)
        )
        pre_context_bound = jnp.asarray(
            tuple(
                np.array_equal(
                    np.asarray(context_preparations[index].source_content_token),
                    np.asarray(contexts[index].content_token),
                )
                and int(context_preparations[index].action)
                == int(binding.final_actions[index])
                and np.array_equal(
                    np.asarray(context_preparations[index].observation),
                    np.asarray(
                        jax.nn.one_hot(
                            binding.final_actions[1 - index],
                            _N_ACTIONS,
                            dtype=jnp.float32,
                        )
                    ),
                )
                for index in range(_N_AGENTS)
            ),
            dtype=jnp.bool_,
        )

        hccl_result = self._hccl.stage(
            state.hccl_state,
            event,
            binding.base,
            binding.memory,
            binding.planner,
            downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
        )
        mm = _signals_at(hccl_result.world_proposals, _MM_SLOT)
        b0m1 = _signals_at(hccl_result.world_proposals, _B0M1_SLOT)
        m0b1 = _signals_at(hccl_result.world_proposals, _M0B1_SLOT)
        bb = _signals_at(hccl_result.world_proposals, _BB_SLOT)
        pp = _signals_at(hccl_result.world_proposals, _PP_SLOT)
        credit_panel = derive_hccl_memory_credit_estimands(
            mm=mm,
            b0m1=b0m1,
            m0b1=m0b1,
            bb=bb,
        )
        credits = jnp.asarray(
            (
                credit_panel.baseline_context_direct_effect.net_reward[0, 0],
                credit_panel.baseline_context_direct_effect.net_reward[1, 1],
            ),
            dtype=jnp.float32,
        )
        horde_cumulants, horde_discounts = self._horde_targets(
            cast(
                HCCLCausalCoreProposal,
                jax.tree.map(lambda leaf: leaf[_PP_SLOT], hccl_result.world_proposals),
            ),
            pp,
            binding.final_actions,
        )
        context_results = tuple(
            self._context.step(
                contexts[index],
                context_preparations[index],
                pp.task_score,
            )
            for index in range(_N_AGENTS)
        )
        pp_next_physical = hccl_result.world_proposals.next_observation[_PP_SLOT]
        next_raw = jnp.stack(
            tuple(
                self._composed_observation(
                    pp_next_physical[index],
                    context_results[index].state,
                )
                for index in range(_N_AGENTS)
            )
        ).astype(jnp.float32)
        feedback = tuple(
            self._memory_feedback(agents[index], credits[index])
            for index in range(_N_AGENTS)
        )
        transitions = tuple(
            self._transition(
                agents[index],
                executed_action=binding.final_actions[index],
                reward=pp.net_reward[index],
                next_observation=next_raw[index],
                horde_cumulants=horde_cumulants[index],
                horde_discounts=horde_discounts[index],
            )
            for index in range(_N_AGENTS)
        )
        memory_preparations = tuple(
            adapters[index].prepare_memory_transition(
                agents[index],
                transitions[index],
                inputs[index],
                masks[index],
                feedback[index],
                evidence[index],
                partner_policy_fusion_input=fusion_input[index],
                partner_policy_fusion_feedback=fusion_feedback[index],
                extended_action_mask=extended_masks[index],
            )
            for index in range(_N_AGENTS)
        )
        memory_valid = jnp.asarray(
            tuple(_bool(item.preparation_valid) for item in memory_preparations),
            dtype=jnp.bool_,
        )
        context_valid = jnp.asarray(
            tuple(
                _bool(item.update_applied)
                and _bool(self._context.state_is_valid(item.state))
                for item in context_results
            ),
            dtype=jnp.bool_,
        )
        credit_valid = credit_panel.algebra.all_identities_hold
        hccl_staged = (
            hccl_result.update_applied
            & (hccl_result.work.world_proposal_calls == 8)
            & (hccl_result.work.attribution_proposal_calls == 8)
        )
        through_valid = (
            source_valid
            & event_valid
            & binding_valid
            & jnp.all(pre_context_bound)
            & hccl_staged
            & credit_valid
            & jnp.all(memory_valid)
            & jnp.all(context_valid)
        )
        through_agents = tuple(
            self._seal_through_memory_agent(
                HCCLContinualDyadThroughMemoryAgent(
                    agent_index=jnp.asarray(index, dtype=jnp.int32),
                    context_preparation=context_preparations[index],
                    context_result=context_results[index],
                    memory_credit=credits[index],
                    memory_feedback=feedback[index],
                    transition=transitions[index],
                    memory_preparation=memory_preparations[index],
                    content_tag_words=jnp.zeros(
                        (_DIGEST_WORDS,),
                        dtype=jnp.uint32,
                    ),
                )
            )
            for index in range(_N_AGENTS)
        )
        bare = HCCLContinualDyadThroughMemoryTransaction(
            source_state=state,
            event=event,
            binding=binding,
            hccl_result=hccl_result,
            memory_credit_panel=credit_panel,
            next_hard_action_masks=masks,
            agent_0=through_agents[0],
            agent_1=through_agents[1],
            work=self._make_through_memory_work(
                hccl_result,
                memory_preparations,
                validation_work,
            ),
            source_state_valid=source_valid,
            event_valid=event_valid,
            binding_valid=binding_valid,
            binding_matches_source=binding_valid,
            pre_outcome_context_bound=pre_context_bound,
            hccl_staged_once=hccl_staged,
            credit_algebra_valid=credit_valid,
            memory_preparations_valid=memory_valid,
            context_candidates_valid=context_valid,
            through_memory_valid=through_valid,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return self._seal_through_memory(bare)

    def complete_with_factorized_planner(
        self,
        state: HCCLContinualDyadState,
        through_memory: HCCLContinualDyadThroughMemoryTransaction,
    ) -> HCCLContinualDyadPreparedTransaction:
        """Consume one exact through-memory proposal with one paired planner call."""

        if type(through_memory) is not HCCLContinualDyadThroughMemoryTransaction:
            raise TypeError(
                "through_memory must be an exact through-memory transaction"
            )
        validation_work = _OuterValidationWork()
        if not self._through_memory_content_valid(
            state,
            through_memory,
            _work=validation_work,
        ):
            raise ValueError(
                "refusing a tampered, foreign, replayed, or invalid through-memory "
                "transaction"
            )
        event = through_memory.event
        binding = through_memory.binding
        masks = through_memory.next_hard_action_masks
        hccl_result = through_memory.hccl_result
        credit_panel = through_memory.memory_credit_panel
        through_agents = (through_memory.agent_0, through_memory.agent_1)
        agents = (state.agent_0_state, state.agent_1_state)
        adapters = (self._agent_0, self._agent_1)
        context_preparations = tuple(
            item.context_preparation for item in through_agents
        )
        context_results = tuple(item.context_result for item in through_agents)
        credits = tuple(item.memory_credit for item in through_agents)
        feedback = tuple(item.memory_feedback for item in through_agents)
        transitions = tuple(item.transition for item in through_agents)
        memory_preparations = tuple(
            item.memory_preparation for item in through_agents
        )
        source_valid = through_memory.source_state_valid
        event_valid = through_memory.event_valid
        binding_valid = through_memory.binding_valid
        pre_context_bound = through_memory.pre_outcome_context_bound
        pp = _signals_at(hccl_result.world_proposals, _PP_SLOT)
        post_memory_prototypes = tuple(
            self._prototype(item.memory_candidate_state)
            for item in memory_preparations
        )
        planner_result = self._planner.completed_transition(
            state.planner_state,
            self._prototype(agents[0]),
            self._prototype(agents[1]),
            post_memory_prototypes[0],
            post_memory_prototypes[1],
            binding.final_actions,
            pp.net_reward,
            jnp.stack(
                tuple(item.current_raw_observation for item in post_memory_prototypes)
            ).astype(jnp.float32),
            jnp.asarray(self._config.discount, dtype=jnp.float32),
            masks,
        )
        planner_words = self._planner_binding_words(
            planner_result.state,
            planner_result.prototype_agent_0,
            planner_result.prototype_agent_1,
        )
        planner_agents = (planner_result.state.agent_0, planner_result.state.agent_1)
        selected_prototypes = (
            planner_result.prototype_agent_0,
            planner_result.prototype_agent_1,
        )
        planner_before = tuple(
            jnp.where(
                planner_agents[index].cache.planner_consumed,
                planner_result.diagnostics.next_prepare.proposed_actions[index],
                memory_preparations[index]
                .memory_candidate_state.action_binding.memory_action,
            ).astype(jnp.int32)
            for index in range(_N_AGENTS)
        )
        finalizations = tuple(
            adapters[index].bind_final_action(
                memory_preparations[index],
                selected_prototypes[index],
                planner_action_before_mask=planner_before[index],
                planner_candidate_words=planner_words,
                planner_consumed=planner_agents[index].cache.planner_consumed,
            )
            for index in range(_N_AGENTS)
        )
        provisional_agents = tuple(
            HCCLContinualDyadPreparedAgent(
                agent_index=jnp.asarray(index, dtype=jnp.int32),
                context_preparation=context_preparations[index],
                context_result=context_results[index],
                memory_credit=credits[index],
                memory_feedback=feedback[index],
                transition=transitions[index],
                memory_preparation=memory_preparations[index],
                finalization=finalizations[index],
                integrity_receipt=None,
            )
            for index in range(_N_AGENTS)
        )
        finalization_bound = tuple(
            self._child_finalization_bound(
                adapters[index],
                provisional_agents[index],
                selected_prototypes[index],
                planner_before[index],
                planner_agents[index].cache.planner_consumed,
                planner_words,
            )
            for index in range(_N_AGENTS)
        )
        child_receipts = tuple(
            adapters[index].integrity_receipt(finalizations[index])
            if finalization_bound[index]
            else None
            for index in range(_N_AGENTS)
        )
        prepared_agents = tuple(
            cast(
                HCCLContinualDyadPreparedAgent,
                provisional_agents[index].replace(
                    integrity_receipt=child_receipts[index]
                ),
            )
            for index in range(_N_AGENTS)
        )
        unsigned_candidate = HCCLContinualDyadState(
            config_token=state.config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            hccl_state=hccl_result.state,
            agent_0_state=finalizations[0].candidate_state,
            agent_1_state=finalizations[1].candidate_state,
            planner_state=planner_result.state,
            context_0_state=context_results[0].state,
            context_1_state=context_results[1].state,
        )
        candidate_state = self._seal_state(unsigned_candidate)
        memory_valid = jnp.asarray(
            tuple(_bool(item.preparation_valid) for item in memory_preparations),
            dtype=jnp.bool_,
        )
        context_valid = jnp.asarray(
            tuple(
                _bool(item.update_applied)
                and _bool(self._context.state_is_valid(item.state))
                for item in context_results
            ),
            dtype=jnp.bool_,
        )
        final_valid = jnp.asarray(finalization_bound, dtype=jnp.bool_)
        shared_binding = all(
            np.array_equal(
                np.asarray(item.final_action_binding.planner_candidate_words),
                np.asarray(planner_words),
            )
            for item in finalizations
        )
        planner_valid = planner_result.diagnostics.transaction_committed
        candidate_valid = self.state_valid(candidate_state, _work=validation_work)
        credit_valid = credit_panel.algebra.all_identities_hold
        hccl_staged = (
            hccl_result.update_applied
            & (hccl_result.work.world_proposal_calls == 8)
            & (hccl_result.work.attribution_proposal_calls == 8)
        )
        valid = (
            source_valid
            & event_valid
            & binding_valid
            & jnp.all(pre_context_bound)
            & hccl_staged
            & credit_valid
            & jnp.all(memory_valid)
            & jnp.all(context_valid)
            & planner_valid
            & jnp.all(final_valid)
            & jnp.asarray(shared_binding, dtype=jnp.bool_)
            & candidate_valid
        )
        planner_work = self._planner.completed_transition_work_budget()
        validation_pair_calls = (
            int(
                through_memory.work.planner_validation_pair_authentication_calls
            )
            + validation_work.planner_pair_authentication_calls
        )
        transition_pair_calls = (
            planner_work.cache_authentication_evaluations // _N_AGENTS
        )
        total_pair_calls = validation_pair_calls + transition_pair_calls
        validation_agent_evaluations = _N_AGENTS * validation_pair_calls
        validation_joint_cells = _N_AGENTS * _N_ACTIONS**2 * validation_pair_calls
        lifecycle_work = tuple(
            self._feature_lifecycle_work(item) for item in memory_preparations
        )
        final_donor_reevaluations = jnp.asarray(
            tuple(
                sum(
                    int(jax.device_get(getattr(finalizations[index].bind_work, name)))
                    for name in (
                        "prototype_replacement_evaluations",
                        "coordinator_update_evaluations",
                        "planner_model_evaluations",
                        "learned_memory_evaluations",
                    )
                )
                for index in range(_N_AGENTS)
            ),
            dtype=jnp.int32,
        )
        work = HCCLContinualDyadPrepareWork(
            supplied_event_receipts=jnp.asarray(1, dtype=jnp.int32),
            supplied_action_binding_bundles=jnp.asarray(1, dtype=jnp.int32),
            event_receipt_preparations=jnp.asarray(0, dtype=jnp.int32),
            event_random_draws=jnp.asarray(0, dtype=jnp.int32),
            action_receipt_validation_rebindings=jnp.asarray(3, dtype=jnp.int32),
            action_identity_validation_recomputations=jnp.asarray(
                6,
                dtype=jnp.int32,
            ),
            context_preparations=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            hccl_stage_calls=jnp.asarray(1, dtype=jnp.int32),
            world_proposal_calls=hccl_result.work.world_proposal_calls,
            attribution_proposal_calls=hccl_result.work.attribution_proposal_calls,
            designated_counterfactual_slots=(
                hccl_result.work.designated_counterfactual_world_slots
            ),
            inner_discarded_world_proposal_calls=(
                hccl_result.work.discarded_world_proposal_calls
            ),
            inner_selected_pp_world_successors=(
                hccl_result.work.committed_pp_world_successors
            ),
            outer_committed_pp_world_successors=jnp.asarray(0, dtype=jnp.int32),
            world_duplicate_mm_checks=(
                hccl_result.work.duplicate_mm_world_equality_checks
            ),
            attribution_duplicate_mm_checks=(
                hccl_result.attribution.work.duplicate_mm_equality_checks
            ),
            memory_credit_panel_derivations=jnp.asarray(1, dtype=jnp.int32),
            memory_credit_readouts=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            context_steps=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            lineage_proposals=jnp.ones((_N_AGENTS,), dtype=jnp.int32),
            action_stack_memory_preparations=jnp.ones(
                (_N_AGENTS,),
                dtype=jnp.int32,
            ),
            feedback_settlement_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.feedback_settlement_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            coordinator_update_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.coordinator_update_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            memory_action_replacement_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.memory_action_replacement_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            fast_state_transition_attempts=jnp.asarray(
                tuple(
                    item.prepare_work.coordinator_update_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            prototype_transition_attempts=jnp.asarray(
                tuple(
                    item.prepare_work.coordinator_update_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            feature_lifecycle_route_attempts=jnp.asarray(
                tuple(item[1] for item in lifecycle_work),
                dtype=jnp.int32,
            ),
            feature_lifecycle_arithmetic_count_available=jnp.asarray(
                tuple(item[0] for item in lifecycle_work),
                dtype=jnp.bool_,
            ),
            active_pair_value_materializations=jnp.asarray(
                tuple(5 * _ACTIVE_PAIR_SLOTS if item[0] else 0 for item in lifecycle_work),
                dtype=jnp.int32,
            ),
            candidate_pair_product_materializations=jnp.asarray(
                tuple(_PAIR_CANDIDATE_SLOTS if item[0] else 0 for item in lifecycle_work),
                dtype=jnp.int32,
            ),
            lifecycle_router_candidate_evaluations=jnp.asarray(
                tuple(2 if item[0] else 0 for item in lifecycle_work),
                dtype=jnp.int32,
            ),
            active_pair_slot_capacity=jnp.full(
                (_N_AGENTS,),
                _ACTIVE_PAIR_SLOTS,
                dtype=jnp.int32,
            ),
            pair_candidate_capacity=jnp.full(
                (_N_AGENTS,),
                _PAIR_CANDIDATE_SLOTS,
                dtype=jnp.int32,
            ),
            routed_representation_width=jnp.full(
                (_N_AGENTS,),
                _ROUTED_DIM,
                dtype=jnp.int32,
            ),
            coordinator_base_action_candidates=jnp.asarray(
                tuple(
                    item.prepare_work.coordinator_update_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            memory_action_candidates=jnp.asarray(
                tuple(
                    item.prepare_work.memory_action_replacement_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            learned_memory_query_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.learned_memory_query_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            learned_memory_write_evaluations=jnp.asarray(
                tuple(
                    item.prepare_work.learned_memory_write_evaluations
                    for item in memory_preparations
                ),
                dtype=jnp.int32,
            ),
            learned_memory_reencode_evaluations=jnp.zeros(
                (_N_AGENTS,),
                dtype=jnp.int32,
            ),
            learned_memory_reencode_count_available=jnp.zeros(
                (_N_AGENTS,),
                dtype=jnp.bool_,
            ),
            planner_completed_transition_calls=jnp.asarray(1, dtype=jnp.int32),
            behavior_update_attempts=jnp.asarray(
                planner_work.behavior_parameter_update_attempts,
                dtype=jnp.int32,
            ),
            grounded_update_attempts=jnp.asarray(
                planner_work.grounded_parameter_update_attempts,
                dtype=jnp.int32,
            ),
            planner_pair_authentication_calls=jnp.asarray(
                total_pair_calls,
                dtype=jnp.int32,
            ),
            planner_validation_pair_authentication_calls=jnp.asarray(
                validation_pair_calls,
                dtype=jnp.int32,
            ),
            planner_transition_pair_authentication_calls=jnp.asarray(
                transition_pair_calls,
                dtype=jnp.int32,
            ),
            planner_cache_authentication_evaluations=jnp.asarray(
                planner_work.cache_authentication_evaluations
                + validation_agent_evaluations,
                dtype=jnp.int32,
            ),
            planner_behavior_probability_vector_evaluations=jnp.asarray(
                planner_work.behavior_probability_vector_evaluations
                + validation_agent_evaluations,
                dtype=jnp.int32,
            ),
            planner_grounded_joint_cell_prediction_equivalents=jnp.asarray(
                planner_work.grounded_joint_cell_prediction_equivalents
                + validation_joint_cells,
                dtype=jnp.int32,
            ),
            planner_expected_reward_marginalization_products=jnp.asarray(
                planner_work.expected_reward_marginalization_products
                + validation_joint_cells,
                dtype=jnp.int32,
            ),
            planner_replacement_candidates=jnp.asarray(
                planner_work.prototype_replacement_candidates,
                dtype=jnp.int32,
            ),
            planner_atomic_pair_commit_decisions=jnp.asarray(
                planner_work.atomic_pair_commit_decisions,
                dtype=jnp.int32,
            ),
            planner_decision_evaluations=jnp.asarray(
                _N_AGENTS,
                dtype=jnp.int32,
            ),
            planner_decision_joint_cells=jnp.asarray(
                _N_AGENTS * _N_ACTIONS**2,
                dtype=jnp.int32,
            ),
            planner_environment_transition_proposals=jnp.asarray(
                planner_work.environment_transition_proposals,
                dtype=jnp.int32,
            ),
            planner_replay_updates=jnp.asarray(
                planner_work.replay_updates,
                dtype=jnp.int32,
            ),
            planner_post_init_random_draws=jnp.asarray(
                planner_work.post_init_random_draws,
                dtype=jnp.int32,
            ),
            final_action_bindings=jnp.asarray(
                tuple(
                    item.bind_work.final_action_binding_evaluations
                    for item in finalizations
                ),
                dtype=jnp.int32,
            ),
            final_binding_donor_reevaluations=final_donor_reevaluations,
            child_finalization_structural_recomputations=(
                jnp.ones((_N_AGENTS,), dtype=jnp.int32)
                + jnp.asarray(
                    tuple(item is not None for item in child_receipts),
                    dtype=jnp.int32,
                )
            ),
            child_integrity_receipts=jnp.asarray(
                tuple(item is not None for item in child_receipts),
                dtype=jnp.int32,
            ),
            prepared_content_digest_evaluations=jnp.asarray(1, dtype=jnp.int32),
        )
        bare = HCCLContinualDyadPreparedTransaction(
            source_state=state,
            event=event,
            binding=binding,
            hccl_result=hccl_result,
            memory_credit_panel=credit_panel,
            planner_result=planner_result,
            planner_candidate_words=planner_words,
            agent_0=prepared_agents[0],
            agent_1=prepared_agents[1],
            candidate_state=candidate_state,
            work=work,
            source_state_valid=source_valid,
            event_valid=event_valid,
            binding_valid=binding_valid,
            binding_matches_source=binding_valid,
            pre_outcome_context_bound=pre_context_bound,
            hccl_staged_once=hccl_staged,
            credit_algebra_valid=credit_valid,
            memory_preparations_valid=memory_valid,
            context_candidates_valid=context_valid,
            planner_transition_valid=planner_valid,
            finalizations_valid=final_valid,
            shared_planner_binding_valid=jnp.asarray(
                shared_binding,
                dtype=jnp.bool_,
            ),
            candidate_state_valid=candidate_valid,
            preparation_valid=valid,
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return self._seal_prepared(bare)

    def prepare_transaction(
        self,
        state: HCCLContinualDyadState,
        event: HCCLCausalCoreEventReceipt,
        binding: HCCLContinualDyadActionBinding,
        agent_0_event_input: ExternalLearnedStateLiveMemoryEventInput,
        agent_1_event_input: ExternalLearnedStateLiveMemoryEventInput,
        next_hard_action_masks: Array,
        *,
        candidate_evidence: tuple[
            ExternalBuilderCandidateAuditEvidence | None,
            ExternalBuilderCandidateAuditEvidence | None,
        ] = (None, None),
        partner_policy_fusion_input: tuple[
            PrototypePartnerPolicyFusionInput | None,
            PrototypePartnerPolicyFusionInput | None,
        ] = (None, None),
        partner_policy_fusion_feedback: tuple[
            PrototypePartnerPolicyFusionFeedback | None,
            PrototypePartnerPolicyFusionFeedback | None,
        ] = (None, None),
        extended_action_masks: tuple[Array | None, Array | None] = (None, None),
    ) -> HCCLContinualDyadPreparedTransaction:
        """Preserve the complete preparation API by composing the exact split."""

        through_memory = self.prepare_through_memory(
            state,
            event,
            binding,
            agent_0_event_input,
            agent_1_event_input,
            next_hard_action_masks,
            candidate_evidence=candidate_evidence,
            partner_policy_fusion_input=partner_policy_fusion_input,
            partner_policy_fusion_feedback=partner_policy_fusion_feedback,
            extended_action_masks=extended_action_masks,
        )
        return self.complete_with_factorized_planner(state, through_memory)

    def _prepared_contract(
        self,
        prepared: HCCLContinualDyadPreparedTransaction,
    ) -> None:
        if type(prepared) is not HCCLContinualDyadPreparedTransaction:
            raise TypeError("prepared must be an exact continual-dyad preparation")
        if type(prepared.source_state) is not HCCLContinualDyadState:
            raise TypeError("prepared.source_state has the wrong type")
        if type(prepared.candidate_state) is not HCCLContinualDyadState:
            raise TypeError("prepared.candidate_state has the wrong type")
        if type(prepared.event) is not HCCLCausalCoreEventReceipt:
            raise TypeError("prepared.event has the wrong type")
        self._binding_contract(prepared.binding)
        if type(prepared.hccl_result) is not HCCLWorldAttributionAdapterResult:
            raise TypeError("prepared.hccl_result has the wrong type")
        if type(prepared.memory_credit_panel) is not HCCLMemoryCreditEstimandPanel:
            raise TypeError("prepared.memory_credit_panel has the wrong type")
        if type(prepared.planner_result) is not PrototypeFactorizedPartnerTransitionResult:
            raise TypeError("prepared.planner_result has the wrong type")
        _require_array(
            prepared.planner_candidate_words,
            name="prepared.planner_candidate_words",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )
        for index, agent in enumerate((prepared.agent_0, prepared.agent_1)):
            if type(agent) is not HCCLContinualDyadPreparedAgent:
                raise TypeError(f"prepared.agent_{index} has the wrong type")
            _require_array(
                agent.agent_index,
                name=f"prepared.agent_{index}.agent_index",
                shape=(),
                dtype=jnp.int32,
            )
            _require_array(
                agent.memory_credit,
                name=f"prepared.agent_{index}.memory_credit",
                shape=(),
                dtype=jnp.float32,
            )
            if type(agent.context_preparation) is not ContextLineageRetentionPreparation:
                raise TypeError(f"prepared.agent_{index} context preparation is malformed")
            if type(agent.context_result) is not ContextLineageRetentionStepResult:
                raise TypeError(f"prepared.agent_{index} context result is malformed")
            if type(agent.transition) is not ExternalLearnedStateTransition:
                raise TypeError(f"prepared.agent_{index} transition is malformed")
            if type(agent.memory_preparation) is not (
                ExternalLearnedStateLiveMemoryActionStackMemoryPreparation
            ):
                raise TypeError(f"prepared.agent_{index} memory preparation is malformed")
            if type(agent.finalization) is not (
                ExternalLearnedStateLiveMemoryActionStackFinalizedTransition
            ):
                raise TypeError(f"prepared.agent_{index} finalization is malformed")
            if agent.memory_feedback is not None and type(agent.memory_feedback) is not (
                ExternalLearnedStateLiveMemoryActionStackFeedback
            ):
                raise TypeError(f"prepared.agent_{index} feedback is malformed")
            if agent.integrity_receipt is not None and type(agent.integrity_receipt) is not (
                ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt
            ):
                raise TypeError(f"prepared.agent_{index} receipt is malformed")
        if type(prepared.work) is not HCCLContinualDyadPrepareWork:
            raise TypeError("prepared.work has the wrong type")
        vector_bool = {
            "feature_lifecycle_arithmetic_count_available",
            "learned_memory_reencode_count_available",
        }
        vector_int = {
            field.name
            for field in dataclasses.fields(HCCLContinualDyadPrepareWork)
            if field.name
            in {
                "context_preparations",
                "memory_credit_readouts",
                "context_steps",
                "lineage_proposals",
                "action_stack_memory_preparations",
                "feedback_settlement_evaluations",
                "coordinator_update_evaluations",
                "memory_action_replacement_evaluations",
                "fast_state_transition_attempts",
                "prototype_transition_attempts",
                "feature_lifecycle_route_attempts",
                "active_pair_value_materializations",
                "candidate_pair_product_materializations",
                "lifecycle_router_candidate_evaluations",
                "active_pair_slot_capacity",
                "pair_candidate_capacity",
                "routed_representation_width",
                "coordinator_base_action_candidates",
                "memory_action_candidates",
                "learned_memory_query_evaluations",
                "learned_memory_write_evaluations",
                "learned_memory_reencode_evaluations",
                "final_action_bindings",
                "final_binding_donor_reevaluations",
                "child_finalization_structural_recomputations",
                "child_integrity_receipts",
            }
        }
        for field in dataclasses.fields(HCCLContinualDyadPrepareWork):
            name = field.name
            shape = (_N_AGENTS,) if name in vector_int | vector_bool else ()
            dtype = jnp.bool_ if name in vector_bool else jnp.int32
            _require_array(
                getattr(prepared.work, name),
                name=f"prepared.work.{name}",
                shape=shape,
                dtype=dtype,
            )
        vector_flags = {
            "pre_outcome_context_bound",
            "memory_preparations_valid",
            "context_candidates_valid",
            "finalizations_valid",
        }
        for name in (
            "source_state_valid",
            "event_valid",
            "binding_valid",
            "binding_matches_source",
            "pre_outcome_context_bound",
            "hccl_staged_once",
            "credit_algebra_valid",
            "memory_preparations_valid",
            "context_candidates_valid",
            "planner_transition_valid",
            "finalizations_valid",
            "shared_planner_binding_valid",
            "candidate_state_valid",
            "preparation_valid",
        ):
            _require_array(
                getattr(prepared, name),
                name=f"prepared.{name}",
                shape=(_N_AGENTS,) if name in vector_flags else (),
                dtype=jnp.bool_,
            )
        _require_array(
            prepared.content_tag_words,
            name="prepared.content_tag_words",
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )

    def _prepare_work_valid(
        self,
        prepared: HCCLContinualDyadPreparedTransaction,
    ) -> bool:
        work = prepared.work
        agents = (prepared.agent_0, prepared.agent_1)
        planner = self._planner.completed_transition_work_budget()
        validation_pair_calls = 4
        validation_agent_evaluations = _N_AGENTS * validation_pair_calls
        validation_joint_cells = _N_AGENTS * _N_ACTIONS**2 * validation_pair_calls
        transition_pair_calls = planner.cache_authentication_evaluations // _N_AGENTS
        exact_scalars = {
            "supplied_event_receipts": 1,
            "supplied_action_binding_bundles": 1,
            "event_receipt_preparations": 0,
            "event_random_draws": 0,
            "action_receipt_validation_rebindings": 3,
            "action_identity_validation_recomputations": 6,
            "hccl_stage_calls": 1,
            "memory_credit_panel_derivations": 1,
            "planner_completed_transition_calls": 1,
            "behavior_update_attempts": planner.behavior_parameter_update_attempts,
            "grounded_update_attempts": planner.grounded_parameter_update_attempts,
            "planner_pair_authentication_calls": (
                validation_pair_calls + transition_pair_calls
            ),
            "planner_validation_pair_authentication_calls": validation_pair_calls,
            "planner_transition_pair_authentication_calls": transition_pair_calls,
            "planner_cache_authentication_evaluations": (
                planner.cache_authentication_evaluations
                + validation_agent_evaluations
            ),
            "planner_behavior_probability_vector_evaluations": (
                planner.behavior_probability_vector_evaluations
                + validation_agent_evaluations
            ),
            "planner_grounded_joint_cell_prediction_equivalents": (
                planner.grounded_joint_cell_prediction_equivalents
                + validation_joint_cells
            ),
            "planner_expected_reward_marginalization_products": (
                planner.expected_reward_marginalization_products
                + validation_joint_cells
            ),
            "planner_replacement_candidates": planner.prototype_replacement_candidates,
            "planner_atomic_pair_commit_decisions": planner.atomic_pair_commit_decisions,
            "planner_decision_evaluations": _N_AGENTS,
            "planner_decision_joint_cells": _N_AGENTS * _N_ACTIONS**2,
            "planner_environment_transition_proposals": (
                planner.environment_transition_proposals
            ),
            "planner_replay_updates": planner.replay_updates,
            "planner_post_init_random_draws": planner.post_init_random_draws,
            "prepared_content_digest_evaluations": 1,
        }
        if any(
            int(jax.device_get(getattr(work, name))) != value
            for name, value in exact_scalars.items()
        ):
            return False
        if not all(
            np.array_equal(
                np.asarray(getattr(work, name)),
                np.asarray(expected, dtype=np.int32),
            )
            for name, expected in (
                ("context_preparations", (1, 1)),
                ("memory_credit_readouts", (1, 1)),
                ("context_steps", (1, 1)),
                ("lineage_proposals", (1, 1)),
                ("action_stack_memory_preparations", (1, 1)),
                ("active_pair_slot_capacity", (12, 12)),
                ("pair_candidate_capacity", (120, 120)),
                ("routed_representation_width", (35, 35)),
            )
        ):
            return False
        if np.any(np.asarray(work.learned_memory_reencode_count_available)):
            return False
        child_prepare_fields = (
            "feedback_settlement_evaluations",
            "coordinator_update_evaluations",
            "memory_action_replacement_evaluations",
            "learned_memory_query_evaluations",
            "learned_memory_write_evaluations",
        )
        for name in child_prepare_fields:
            child_expected: NDArray[np.int32] = np.asarray(
                tuple(
                    int(jax.device_get(getattr(agent.memory_preparation.prepare_work, name)))
                    for agent in agents
                ),
                dtype=np.int32,
            )
            if not np.array_equal(np.asarray(getattr(work, name)), child_expected):
                return False
        coordinator_counts: NDArray[np.int32] = np.asarray(
            tuple(
                int(
                    jax.device_get(
                        agent.memory_preparation.prepare_work.coordinator_update_evaluations
                    )
                )
                for agent in agents
            ),
            dtype=np.int32,
        )
        replacement_counts: NDArray[np.int32] = np.asarray(
            tuple(
                int(
                    jax.device_get(
                        agent.memory_preparation.prepare_work
                        .memory_action_replacement_evaluations
                    )
                )
                for agent in agents
            ),
            dtype=np.int32,
        )
        for name, expected in (
            ("fast_state_transition_attempts", coordinator_counts),
            ("prototype_transition_attempts", coordinator_counts),
            ("coordinator_base_action_candidates", coordinator_counts),
            ("memory_action_candidates", replacement_counts),
            ("learned_memory_reencode_evaluations", np.zeros(2, dtype=np.int32)),
            (
                "final_action_bindings",
                np.asarray(
                    tuple(
                        int(
                            jax.device_get(
                                agent.finalization.bind_work
                                .final_action_binding_evaluations
                            )
                        )
                        for agent in agents
                    ),
                    dtype=np.int32,
                ),
            ),
            (
                "final_binding_donor_reevaluations",
                np.asarray(
                    tuple(
                        sum(
                            int(jax.device_get(getattr(agent.finalization.bind_work, field)))
                            for field in (
                                "prototype_replacement_evaluations",
                                "coordinator_update_evaluations",
                                "planner_model_evaluations",
                                "learned_memory_evaluations",
                            )
                        )
                        for agent in agents
                    ),
                    dtype=np.int32,
                ),
            ),
            (
                "child_integrity_receipts",
                np.asarray(
                    tuple(agent.integrity_receipt is not None for agent in agents),
                    dtype=np.int32,
                ),
            ),
            (
                "child_finalization_structural_recomputations",
                np.ones(2, dtype=np.int32)
                + np.asarray(
                    tuple(agent.integrity_receipt is not None for agent in agents),
                    dtype=np.int32,
                ),
            ),
        ):
            if not np.array_equal(np.asarray(getattr(work, name)), expected):
                return False
        lifecycle = tuple(
            self._feature_lifecycle_work(agent.memory_preparation) for agent in agents
        )
        if not np.array_equal(
            np.asarray(work.feature_lifecycle_arithmetic_count_available),
            np.asarray(tuple(item[0] for item in lifecycle), dtype=np.bool_),
        ):
            return False
        for name, lifecycle_expected in (
            ("feature_lifecycle_route_attempts", tuple(item[1] for item in lifecycle)),
            (
                "active_pair_value_materializations",
                tuple(5 * _ACTIVE_PAIR_SLOTS if item[0] else 0 for item in lifecycle),
            ),
            (
                "candidate_pair_product_materializations",
                tuple(_PAIR_CANDIDATE_SLOTS if item[0] else 0 for item in lifecycle),
            ),
            (
                "lifecycle_router_candidate_evaluations",
                tuple(2 if item[0] else 0 for item in lifecycle),
            ),
        ):
            if not np.array_equal(
                np.asarray(getattr(work, name)),
                np.asarray(lifecycle_expected, dtype=np.int32),
            ):
                return False
        return all(
            (
                int(work.world_proposal_calls)
                == int(prepared.hccl_result.work.world_proposal_calls),
                int(work.attribution_proposal_calls)
                == int(prepared.hccl_result.work.attribution_proposal_calls),
                int(work.world_duplicate_mm_checks)
                == int(
                    prepared.hccl_result.work.duplicate_mm_world_equality_checks
                ),
                int(work.attribution_duplicate_mm_checks)
                == int(
                    prepared.hccl_result.attribution.work.duplicate_mm_equality_checks
                ),
                int(work.designated_counterfactual_slots)
                == int(
                    prepared.hccl_result.work.designated_counterfactual_world_slots
                ),
                int(work.inner_discarded_world_proposal_calls)
                == int(prepared.hccl_result.work.discarded_world_proposal_calls),
                int(work.inner_selected_pp_world_successors)
                == int(prepared.hccl_result.work.committed_pp_world_successors),
                int(work.outer_committed_pp_world_successors) == 0,
            )
        )

    def _hccl_result_structurally_bound(
        self,
        source: HCCLWorldAttributionAdapterState,
        event: HCCLCausalCoreEventReceipt,
        binding: HCCLContinualDyadActionBinding,
        result: HCCLWorldAttributionAdapterResult,
    ) -> bool:
        """Bind a supplied HCCL result without re-evaluating its world donor.

        This is a structural/content-integrity check, not caller
        authentication.  In particular, it verifies that every stored world
        and attribution row names the exact vertex implied by the supplied
        B/M/P receipts and that the committed composite destination is the PP
        row.  It deliberately does not call ``world.propose`` or
        ``attribution.stage`` a second time.
        """

        if type(result) is not HCCLWorldAttributionAdapterResult:
            return False
        try:
            vertices = self._hccl.attribution._vertices(
                binding.base,
                binding.memory,
                binding.planner,
            )
            world_rows = tuple(
                cast(
                    HCCLCausalCoreProposal,
                    jax.tree.map(lambda leaf, slot=slot: leaf[slot], result.world_proposals),
                )
                for slot in range(8)
            )
            attribution_rows = tuple(
                jax.tree.map(
                    lambda leaf, slot=slot: leaf[slot],
                    result.attribution.proposals,
                )
                for slot in range(8)
            )
            source_receipt = self._hccl._source(source)
            exogenous_receipt = self._hccl._exogenous(source_receipt, event)
            destination_words, destination_available = (
                _increment_hccl_attribution_words(
                    source.attribution_state.transaction_words
                )
            )
            row_valid = True
            for slot, (vertex, world_row, attribution_row) in enumerate(
                zip(vertices, world_rows, attribution_rows, strict=True)
            ):
                self._hccl.world._require_proposal_contract(world_row)
                self._hccl.attribution._require_proposal_contract(attribution_row)
                signals = world_row.signals
                expected_signs = (
                    jnp.float32(2.0)
                    * world_row.joint_action_ids.astype(jnp.float32)
                    - jnp.float32(1.0)
                ).astype(jnp.float32)
                row_valid = row_valid and all(
                    (
                        _bool(world_row.valid),
                        _bool(self._hccl.world.state_valid(world_row.candidate_state)),
                        np.array_equal(
                            np.asarray(world_row.source_state_tag_words),
                            np.asarray(source.world_state.content_tag_words),
                        ),
                        np.array_equal(
                            np.asarray(world_row.source_step_words),
                            np.asarray(source.world_state.step_words),
                        ),
                        np.array_equal(
                            np.asarray(world_row.event_content_tag_words),
                            np.asarray(event.content_tag_words),
                        ),
                        np.array_equal(
                            np.asarray(world_row.joint_action_ids),
                            np.asarray(vertex.actions),
                        ),
                        np.array_equal(
                            np.asarray(world_row.action_signs),
                            np.asarray(expected_signs),
                        ),
                        np.array_equal(
                            np.asarray(world_row.observation),
                            np.asarray(self._hccl.world.observe(source.world_state)),
                        ),
                        np.array_equal(
                            np.asarray(world_row.next_observation),
                            np.asarray(
                                self._hccl.world.observe(world_row.candidate_state)
                            ),
                        ),
                        np.array_equal(
                            np.asarray(
                                world_row.candidate_state.previous_action_signs
                            ),
                            np.asarray(world_row.action_signs),
                        ),
                        np.array_equal(
                            np.asarray(
                                world_row.candidate_state.previous_task_score
                            ),
                            np.asarray(signals.task_score),
                        ),
                        np.array_equal(
                            np.asarray(
                                world_row.candidate_state.previous_net_reward
                            ),
                            np.asarray(signals.net_reward),
                        ),
                        np.array_equal(
                            np.asarray(world_row.content_tag_words),
                            np.asarray(_hccl_world_proposal_tag(world_row)),
                        ),
                        np.array_equal(
                            np.asarray(attribution_row.vertex.actions),
                            np.asarray(vertex.actions),
                        ),
                        _tree_exact_equal(attribution_row.vertex, vertex),
                        _bool(
                            self._hccl.attribution._proposal_valid(
                                attribution_row,
                                source_receipt,
                                exogenous_receipt,
                                vertex,
                                destination_words,
                            )
                        ),
                        np.array_equal(
                            np.asarray(attribution_row.candidate_transition),
                            np.asarray(world_row.next_observation).reshape((-1,)),
                        ),
                        np.array_equal(
                            np.asarray(attribution_row.signals.task_score),
                            np.asarray(signals.task_score),
                        ),
                        np.array_equal(
                            np.asarray(attribution_row.signals.net_reward),
                            np.asarray(signals.net_reward),
                        ),
                        np.array_equal(
                            np.asarray(attribution_row.signals.safety_cost),
                            np.asarray(signals.safety_cost),
                        ),
                        np.array_equal(
                            np.asarray(attribution_row.signals.message_charge),
                            np.asarray(signals.message_charge),
                        ),
                        _bool(attribution_row.accepted),
                        slot != _PP_SLOT
                        or np.array_equal(
                            np.asarray(world_row.joint_action_ids),
                            np.asarray(binding.final_actions),
                        ),
                    )
                )
            pp_world = world_rows[_PP_SLOT]
            pp_attribution = attribution_rows[_PP_SLOT]
            attribution = result.attribution
            world_duplicate_mm = _tree_exact_equal(world_rows[0], world_rows[7])
            attribution_duplicate_mm = _tree_exact_equal(
                attribution_rows[0],
                attribution_rows[7],
            )
            equal_action_payloads = _bool(
                self._hccl._equal_action_payloads_valid(vertices, world_rows)
            )
            causal_core_signals = _bool(
                self._hccl._causal_core_signals_valid(world_rows)
            )
            return row_valid and all(
                (
                    world_duplicate_mm,
                    attribution_duplicate_mm,
                    equal_action_payloads,
                    causal_core_signals,
                    _bool(destination_available),
                    _tree_exact_equal(result.state.world_state, pp_world.candidate_state),
                    _tree_exact_equal(
                        result.state.attribution_state,
                        attribution.state,
                    ),
                    _tree_exact_equal(
                        attribution.state.last_committed_pp,
                        pp_attribution,
                    ),
                    _tree_exact_equal(
                        attribution.state.last_contrasts,
                        attribution.contrasts,
                    ),
                    _tree_exact_equal(
                        attribution.contrasts,
                        _derive_hccl_contrasts(attribution.proposals),
                    ),
                    int(attribution.committed_slot) == _PP_SLOT,
                    _bool(attribution.update_applied),
                    _bool(attribution.preflight_valid),
                    _bool(attribution.all_child_proposals_valid),
                    _bool(attribution.duplicate_mm_bit_exact),
                    _bool(attribution.typed_signals_valid),
                    _bool(attribution.telescoping_valid),
                    _bool(attribution.candidate_state_valid),
                    _bool(result.update_applied),
                    _bool(result.source_state_valid),
                    _bool(result.world_source_clock_bound),
                    _bool(result.event_receipt_valid),
                    _bool(result.event_receipt_identity_bound),
                    _bool(result.action_receipt_identities_bound),
                    _bool(result.all_world_proposals_valid),
                    _bool(result.equal_action_world_payloads_bit_exact),
                    _bool(result.causal_core_signal_contract_valid),
                    _bool(result.world_duplicate_mm_bit_exact),
                    _bool(result.downstream_candidate_valid),
                    _bool(result.candidate_state_valid),
                )
            )
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return False

    def _prepared_content_valid(
        self,
        prepared: HCCLContinualDyadPreparedTransaction,
        *,
        _work: _OuterValidationWork | None = None,
    ) -> bool:
        """Recompute integrity/cross-owner semantics without learner/model proposals."""

        self._prepared_contract(prepared)
        if _contains_tracer(prepared):
            raise TypeError("continual-dyad preparation validation is host/eager-only")
        if not np.array_equal(
            np.asarray(prepared.content_tag_words),
            np.asarray(self._prepared_tag(prepared)),
        ):
            return False
        source = prepared.source_state
        binding = prepared.binding
        source_valid = self.state_valid(source, _work=_work)
        event_valid = self._hccl.world.event_receipt_valid(
            source.hccl_state.world_state,
            prepared.event,
        )
        binding_valid = self.binding_valid(
            source,
            prepared.event,
            binding,
            _work=_work,
        )
        hccl = prepared.hccl_result
        hccl_valid = all(
            (
                _bool(hccl.update_applied),
                _bool(self._hccl.state_valid(hccl.state)),
                np.array_equal(
                    np.asarray(hccl.pre_transaction_words),
                    np.asarray(source.hccl_state.world_state.step_words),
                ),
                np.array_equal(
                    np.asarray(hccl.post_transaction_words),
                    np.asarray(hccl.state.world_state.step_words),
                ),
                _bool(hccl.event_receipt_valid),
                _bool(hccl.action_receipt_identities_bound),
                int(hccl.work.world_proposal_calls) == 8,
                int(hccl.work.attribution_proposal_calls) == 8,
                self._hccl_result_structurally_bound(
                    source.hccl_state,
                    prepared.event,
                    binding,
                    hccl,
                ),
            )
        )
        panel = derive_hccl_memory_credit_estimands(
            mm=_signals_at(hccl.world_proposals, _MM_SLOT),
            b0m1=_signals_at(hccl.world_proposals, _B0M1_SLOT),
            m0b1=_signals_at(hccl.world_proposals, _M0B1_SLOT),
            bb=_signals_at(hccl.world_proposals, _BB_SLOT),
        )
        panel_valid = _tree_exact_equal(panel, prepared.memory_credit_panel) and _bool(
            panel.algebra.all_identities_hold
        )
        pp = _signals_at(hccl.world_proposals, _PP_SLOT)
        pp_proposal = cast(
            HCCLCausalCoreProposal,
            jax.tree.map(lambda leaf: leaf[_PP_SLOT], hccl.world_proposals),
        )
        horde_cumulants, horde_discounts = self._horde_targets(
            pp_proposal,
            pp,
            binding.final_actions,
        )
        source_agents = (source.agent_0_state, source.agent_1_state)
        source_contexts = (source.context_0_state, source.context_1_state)
        prepared_agents = (prepared.agent_0, prepared.agent_1)
        adapters = (self._agent_0, self._agent_1)
        expected_credits = (
            panel.baseline_context_direct_effect.net_reward[0, 0],
            panel.baseline_context_direct_effect.net_reward[1, 1],
        )
        next_physical = hccl.world_proposals.next_observation[_PP_SLOT]
        context_bound: list[bool] = []
        memory_bound: list[bool] = []
        next_raw: list[Array] = []
        for index in range(_N_AGENTS):
            agent = prepared_agents[index]
            context_preparation = agent.context_preparation
            context_result = agent.context_result
            expected_partner = jax.nn.one_hot(
                binding.final_actions[1 - index],
                _N_ACTIONS,
                dtype=jnp.float32,
            )
            context_ok = all(
                (
                    int(agent.agent_index) == index,
                    self._context._preparation_integrity_valid(context_preparation),
                    np.array_equal(
                        np.asarray(context_preparation.source_content_token),
                        np.asarray(source_contexts[index].content_token),
                    ),
                    np.array_equal(
                        np.asarray(context_preparation.observation),
                        np.asarray(expected_partner),
                    ),
                    int(context_preparation.action)
                    == int(binding.final_actions[index]),
                    _tree_exact_equal(context_result.preparation, context_preparation),
                    _bool(context_result.update_applied),
                    _bool(context_result.context_owner_committed),
                    _bool(context_result.lineage_owner_committed),
                    _bool(context_result.protection_snapshotted_before_outcome),
                    not _bool(
                        context_result.current_outcome_changed_current_eviction_protection
                    ),
                    np.array_equal(
                        np.asarray(context_result.lineage_event.reward),
                        np.asarray(pp.task_score),
                    ),
                    np.array_equal(
                        np.asarray(context_result.lineage_event.observation),
                        np.asarray(context_preparation.observation),
                    ),
                    int(context_result.lineage_event.action)
                    == int(context_preparation.action),
                    np.array_equal(
                        np.asarray(context_result.context_result.pre_step_words),
                        np.asarray(source_contexts[index].context.step_words),
                    ),
                    np.array_equal(
                        np.asarray(context_result.context_result.post_step_words),
                        np.asarray(context_result.state.context.step_words),
                    ),
                    _bool(context_result.context_result.update_applied),
                    _bool(self._context.state_is_valid(context_result.state)),
                    np.array_equal(
                        np.asarray(context_result.state.context.step_words),
                        np.asarray(hccl.post_transaction_words),
                    ),
                )
            )
            context_bound.append(context_ok)
            next_value = self._composed_observation(
                next_physical[index],
                context_result.state,
            )
            next_raw.append(next_value)
            expected_transition = self._transition(
                source_agents[index],
                executed_action=binding.final_actions[index],
                reward=pp.net_reward[index],
                next_observation=next_value,
                horde_cumulants=horde_cumulants[index],
                horde_discounts=horde_discounts[index],
            )
            expected_feedback = self._memory_feedback(
                source_agents[index],
                expected_credits[index],
            )
            memory_preparation = agent.memory_preparation
            feedback_ok = (
                expected_feedback is None
                and agent.memory_feedback is None
                and not _bool(memory_preparation.feedback_supplied)
            ) or (
                expected_feedback is not None
                and agent.memory_feedback is not None
                and _tree_exact_equal(agent.memory_feedback, expected_feedback)
                and _tree_exact_equal(memory_preparation.feedback, expected_feedback)
                and _bool(memory_preparation.feedback_supplied)
            )
            memory_bound.append(
                all(
                    (
                        np.array_equal(
                            np.asarray(agent.memory_credit),
                            np.asarray(expected_credits[index]),
                        ),
                        _tree_exact_equal(agent.transition, expected_transition),
                        _tree_exact_equal(
                            memory_preparation.source_state,
                            source_agents[index],
                        ),
                        _tree_exact_equal(
                            memory_preparation.transition,
                            expected_transition,
                        ),
                        feedback_ok,
                        _bool(memory_preparation.preparation_valid),
                        np.array_equal(
                            np.asarray(memory_preparation.content_tag_words),
                            np.asarray(
                                adapters[index]._memory_preparation_tag(
                                    memory_preparation
                                )
                            ),
                        ),
                    )
                )
            )
        planner = prepared.planner_result
        expected_planner_words = self._planner_binding_words(
            planner.state,
            planner.prototype_agent_0,
            planner.prototype_agent_1,
        )
        post_memory_prototypes = tuple(
            self._prototype(agent.memory_preparation.memory_candidate_state)
            for agent in prepared_agents
        )
        expected_grounded_targets = jnp.stack(
            tuple(
                jnp.concatenate(
                    (
                        post_memory_prototypes[index].current_raw_observation,
                        jnp.reshape(pp.net_reward[index], (1,)),
                        jnp.asarray((self._config.discount,), dtype=jnp.float32),
                    )
                ).astype(jnp.float32)
                for index in range(_N_AGENTS)
            )
        )
        planner_valid = all(
            (
                _bool(planner.diagnostics.transaction_committed),
                bool(np.all(np.asarray(planner.diagnostics.source_cache_valid))),
                bool(np.all(np.asarray(planner.diagnostics.behavior_update_applied))),
                bool(np.all(np.asarray(planner.diagnostics.grounded_update_applied))),
                bool(np.all(np.asarray(planner.diagnostics.prediction_matches_cache))),
                bool(np.all(np.asarray(planner.diagnostics.candidate_clock_aligned))),
                bool(np.all(np.asarray(planner.diagnostics.candidate_generation_aligned))),
                bool(np.all(np.asarray(planner.diagnostics.next_observations_match))),
                _bool(planner.diagnostics.candidate_valid),
                _bool(planner.diagnostics.next_prepare.pair_committed),
                np.array_equal(
                    np.asarray(planner.diagnostics.executed_actions),
                    np.asarray(binding.final_actions),
                ),
                np.array_equal(
                    np.asarray(planner.diagnostics.grounded_targets),
                    np.asarray(expected_grounded_targets),
                ),
                np.array_equal(
                    np.asarray(planner.diagnostics.next_prepare.base_actions),
                    np.asarray(
                        tuple(
                            item.current_action for item in post_memory_prototypes
                        ),
                        dtype=np.int32,
                    ),
                ),
                np.array_equal(
                    np.asarray(planner.diagnostics.next_prepare.effective_actions),
                    np.asarray(
                        (
                            planner.prototype_agent_0.current_action,
                            planner.prototype_agent_1.current_action,
                        ),
                        dtype=np.int32,
                    ),
                ),
                np.array_equal(
                    np.asarray(prepared.planner_candidate_words),
                    np.asarray(expected_planner_words),
                ),
                bool(
                    np.all(
                        np.asarray(
                            self._authenticate_planner_pair(
                                planner.state,
                                planner.prototype_agent_0,
                                planner.prototype_agent_1,
                                _work,
                            )
                        )
                    )
                ),
            )
        )
        planner_agents = (planner.state.agent_0, planner.state.agent_1)
        selected = (planner.prototype_agent_0, planner.prototype_agent_1)
        planner_before = tuple(
            jnp.where(
                planner_agents[index].cache.planner_consumed,
                planner.diagnostics.next_prepare.proposed_actions[index],
                prepared_agents[index]
                .memory_preparation.memory_candidate_state.action_binding.memory_action,
            ).astype(jnp.int32)
            for index in range(_N_AGENTS)
        )
        final_bound_items: list[bool] = []
        for index in range(_N_AGENTS):
            if _work is not None:
                _work.child_finalization_structural_recomputations[index] += 1
            final_bound_items.append(
                self._child_finalization_bound(
                    adapters[index],
                    prepared_agents[index],
                    selected[index],
                    planner_before[index],
                    planner_agents[index].cache.planner_consumed,
                    expected_planner_words,
                )
            )
        final_bound = tuple(final_bound_items)
        receipt_bound = tuple(
            prepared_agents[index].integrity_receipt is not None
            and _tree_exact_equal(
                prepared_agents[index].integrity_receipt,
                adapters[index]._make_receipt(prepared_agents[index].finalization),
            )
            for index in range(_N_AGENTS)
        )
        expected_candidate = self._seal_state(
            HCCLContinualDyadState(
                config_token=source.config_token,
                content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
                hccl_state=hccl.state,
                agent_0_state=prepared.agent_0.finalization.candidate_state,
                agent_1_state=prepared.agent_1.finalization.candidate_state,
                planner_state=planner.state,
                context_0_state=prepared.agent_0.context_result.state,
                context_1_state=prepared.agent_1.context_result.state,
            )
        )
        candidate_valid = _tree_exact_equal(
            prepared.candidate_state,
            expected_candidate,
        ) and _bool(self.state_valid(expected_candidate, _work=_work))
        pre_context = jnp.asarray(context_bound, dtype=jnp.bool_)
        memory_flags = jnp.asarray(memory_bound, dtype=jnp.bool_)
        context_flags = jnp.asarray(context_bound, dtype=jnp.bool_)
        final_flags = jnp.asarray(final_bound, dtype=jnp.bool_)
        hccl_staged = jnp.asarray(hccl_valid, dtype=jnp.bool_)
        shared_planner = jnp.asarray(
            planner_valid and all(receipt_bound),
            dtype=jnp.bool_,
        )
        all_valid = (
            source_valid
            & event_valid
            & binding_valid
            & jnp.all(pre_context)
            & hccl_staged
            & jnp.asarray(panel_valid, dtype=jnp.bool_)
            & jnp.all(memory_flags)
            & jnp.all(context_flags)
            & jnp.asarray(planner_valid, dtype=jnp.bool_)
            & jnp.all(final_flags)
            & shared_planner
            & jnp.asarray(candidate_valid, dtype=jnp.bool_)
            & jnp.asarray(self._prepare_work_valid(prepared), dtype=jnp.bool_)
        )
        stored_flags = all(
            (
                _tree_exact_equal(prepared.source_state_valid, source_valid),
                _tree_exact_equal(prepared.event_valid, event_valid),
                _tree_exact_equal(prepared.binding_valid, binding_valid),
                _tree_exact_equal(prepared.binding_matches_source, binding_valid),
                _tree_exact_equal(prepared.pre_outcome_context_bound, pre_context),
                _tree_exact_equal(prepared.hccl_staged_once, hccl_staged),
                _tree_exact_equal(
                    prepared.credit_algebra_valid,
                    panel.algebra.all_identities_hold,
                ),
                _tree_exact_equal(prepared.memory_preparations_valid, memory_flags),
                _tree_exact_equal(prepared.context_candidates_valid, context_flags),
                _tree_exact_equal(
                    prepared.planner_transition_valid,
                    planner.diagnostics.transaction_committed,
                ),
                _tree_exact_equal(prepared.finalizations_valid, final_flags),
                _tree_exact_equal(
                    prepared.shared_planner_binding_valid,
                    shared_planner,
                ),
                _tree_exact_equal(
                    prepared.candidate_state_valid,
                    jnp.asarray(candidate_valid, dtype=jnp.bool_),
                ),
                _tree_exact_equal(prepared.preparation_valid, all_valid),
            )
        )
        return stored_flags and _bool(all_valid)

    def _prepared_static_contract_valid(self, prepared: object) -> bool:
        if type(prepared) is not HCCLContinualDyadPreparedTransaction:
            return False
        try:
            self._prepared_contract(prepared)
            return True
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return False

    def _receipt_tag(
        self,
        receipt: HCCLContinualDyadPreparationReceipt,
    ) -> UInt[Array, " 8"]:
        bare = cast(
            HCCLContinualDyadPreparationReceipt,
            receipt.replace(
                content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32)
            ),
        )
        return _tree_digest(HCCL_CONTINUAL_DYAD_RECEIPT_SCHEMA, bare)

    def _make_receipt(
        self,
        prepared: HCCLContinualDyadPreparedTransaction,
    ) -> HCCLContinualDyadPreparationReceipt:
        bare = HCCLContinualDyadPreparationReceipt(
            source_state_words=_tree_digest(
                "continual-dyad-prepared-source-v2",
                prepared.source_state,
            ),
            event_words=_tree_digest("continual-dyad-prepared-event-v2", prepared.event),
            binding_words=_tree_digest(
                "continual-dyad-prepared-binding-v2",
                prepared.binding,
            ),
            prepared_content_tag_words=prepared.content_tag_words,
            config_words=self._config_words,
            integrity_bound=jnp.asarray(True, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((_DIGEST_WORDS,), dtype=jnp.uint32),
        )
        return cast(
            HCCLContinualDyadPreparationReceipt,
            bare.replace(content_tag_words=self._receipt_tag(bare)),
        )

    def integrity_receipt(
        self,
        prepared: HCCLContinualDyadPreparedTransaction,
    ) -> HCCLContinualDyadPreparationReceipt:
        """Bind one fully reconstructed valid preparation; never authorize execution."""

        if type(prepared) is not HCCLContinualDyadPreparedTransaction:
            raise TypeError("prepared must be an exact continual-dyad preparation")
        if not self._prepared_static_contract_valid(prepared):
            raise ValueError("refusing a malformed continual-dyad preparation")
        try:
            valid = self._prepared_content_valid(prepared)
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("refusing to bind an invalid continual-dyad preparation")
        return self._make_receipt(prepared)

    @staticmethod
    def _receipt_static_contract_valid(receipt: object) -> bool:
        if type(receipt) is not HCCLContinualDyadPreparationReceipt:
            return False
        exact = receipt
        for name in (
            "source_state_words",
            "event_words",
            "binding_words",
            "prepared_content_tag_words",
            "config_words",
            "content_tag_words",
        ):
            value = getattr(exact, name, None)
            if getattr(value, "shape", None) != (_DIGEST_WORDS,) or getattr(
                value,
                "dtype",
                None,
            ) != jnp.dtype(jnp.uint32):
                return False
        return getattr(exact.integrity_bound, "shape", None) == () and getattr(
            exact.integrity_bound,
            "dtype",
            None,
        ) == jnp.dtype(jnp.bool_)

    def _receipt_valid(
        self,
        prepared: HCCLContinualDyadPreparedTransaction,
        receipt: HCCLContinualDyadPreparationReceipt,
    ) -> bool:
        if type(receipt) is not HCCLContinualDyadPreparationReceipt:
            raise TypeError("receipt must be an exact continual-dyad receipt")
        if not self._receipt_static_contract_valid(receipt):
            return False
        try:
            return (
                _bool(receipt.integrity_bound)
                and np.array_equal(
                    np.asarray(receipt.content_tag_words),
                    np.asarray(self._receipt_tag(receipt)),
                )
                and _tree_exact_equal(receipt, self._make_receipt(prepared))
            )
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def _validated_child_adoption_result(
        result: object,
        *,
        source_state: ExternalLearnedStateLiveMemoryActionStackState,
        finalized: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
        receipt: ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt,
    ) -> tuple[ExternalLearnedStateLiveMemoryActionStackResult | None, bool, int]:
        """Normalize one child result and safely expose its reconstruction work."""

        if type(result) is not ExternalLearnedStateLiveMemoryActionStackResult:
            return None, False, 0
        exact = result
        try:
            if type(exact.state) is not ExternalLearnedStateLiveMemoryActionStackState:
                return exact, False, 0
            if type(exact.finalized) is not (
                ExternalLearnedStateLiveMemoryActionStackFinalizedTransition
            ):
                return exact, False, 0
            if type(exact.receipt) is not (
                ExternalLearnedStateLiveMemoryActionStackIntegrityReceipt
            ):
                return exact, False, 0
            if type(exact.diagnostics) is not (
                ExternalLearnedStateLiveMemoryActionStackDiagnostics
            ):
                return exact, False, 0
            if type(exact.adoption_work) is not (
                ExternalLearnedStateLiveMemoryActionStackAdoptionWork
            ):
                return exact, False, 0
            for field in dataclasses.fields(
                ExternalLearnedStateLiveMemoryActionStackDiagnostics
            ):
                _require_array(
                    getattr(exact.diagnostics, field.name),
                    name=f"child.diagnostics.{field.name}",
                    shape=(),
                    dtype=jnp.bool_,
                )
            for field in dataclasses.fields(
                ExternalLearnedStateLiveMemoryActionStackAdoptionWork
            ):
                _require_array(
                    getattr(exact.adoption_work, field.name),
                    name=f"child.adoption_work.{field.name}",
                    shape=(),
                    dtype=jnp.int32,
                )
            reconstruction_count = int(
                jax.device_get(
                    exact.adoption_work.final_action_binding_reconstructions
                )
            )
            if reconstruction_count not in (0, 1):
                return exact, False, 0
            applied = _bool(exact.diagnostics.transaction_applied)
            selected = finalized.candidate_state if applied else source_state
            validity_flags = tuple(
                _bool(getattr(exact.diagnostics, name))
                for name in (
                    "source_state_matches",
                    "source_state_valid",
                    "finalized_content_matches",
                    "receipt_static_contract_valid",
                    "receipt_content_tag_valid",
                    "receipt_matches",
                    "receipt_integrity_bound",
                    "memory_preparation_valid",
                    "final_action_binding_valid",
                    "candidate_state_valid",
                    "transition_final_action_exact",
                    "feedback_memory_action_bound",
                    "completed_entry_final_action_exact",
                )
            )
            work = exact.adoption_work
            work_contract = (
                int(work.integrity_evaluations) == 1
                and int(work.donor_evaluations) == 0
                and int(work.coordinator_update_evaluations) == 0
                and int(work.prototype_replacement_evaluations) == 0
                and int(work.planner_model_evaluations) == 0
                and int(work.learned_memory_evaluations) == 0
            )
            contract_valid = all(
                (
                    _tree_exact_equal(exact.finalized, finalized),
                    _tree_exact_equal(exact.receipt, receipt),
                    _tree_exact_equal(exact.state, selected),
                    _bool(exact.diagnostics.complete_source_returned) == (not applied),
                    _bool(exact.diagnostics.rejected) == (not applied),
                    not applied or all(validity_flags),
                    work_contract,
                )
            )
            return exact, contract_valid, reconstruction_count
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return exact, False, 0

    def adopt_prepared_transaction(
        self,
        state: HCCLContinualDyadState,
        prepared: HCCLContinualDyadPreparedTransaction,
        receipt: HCCLContinualDyadPreparationReceipt,
    ) -> HCCLContinualDyadResult:
        """Adopt every child destination together or return ``state`` bit-for-bit."""

        self._state_contract(state)
        if type(prepared) is not HCCLContinualDyadPreparedTransaction:
            raise TypeError("prepared must be an exact continual-dyad preparation")
        if type(receipt) is not HCCLContinualDyadPreparationReceipt:
            raise TypeError("receipt must be an exact continual-dyad receipt")
        if _contains_tracer((state, prepared, receipt)):
            raise TypeError("continual-dyad adoption is host/eager-only")
        validation_work = _OuterValidationWork()
        source_matches = _tree_exact_equal(state, prepared.source_state)
        try:
            source_valid = _bool(self.state_valid(state, _work=validation_work))
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            source_valid = False
        prepared_valid = False
        if self._prepared_static_contract_valid(prepared):
            try:
                prepared_valid = self._prepared_content_valid(
                    prepared,
                    _work=validation_work,
                )
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                prepared_valid = False
        receipt_valid = self._receipt_valid(prepared, receipt)
        preflight = source_matches and source_valid and prepared_valid and receipt_valid
        prepared_agents = (prepared.agent_0, prepared.agent_1)
        adapters = (self._agent_0, self._agent_1)
        adoptions: list[ExternalLearnedStateLiveMemoryActionStackResult | None] = []
        adoption_calls = [0, 0]
        adoption_contracts = [False, False]
        child_reconstruction_counts = [0, 0]
        if preflight:
            for index in range(_N_AGENTS):
                child_receipt = prepared_agents[index].integrity_receipt
                if child_receipt is None:
                    adoptions.append(None)
                    continue
                adoption_calls[index] += 1
                try:
                    child_result = adapters[index].adopt_finalized_transition(
                        (state.agent_0_state, state.agent_1_state)[index],
                        prepared_agents[index].finalization,
                        child_receipt,
                    )
                except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                    child_result = None
                normalized, contract_valid, reconstruction_count = (
                    self._validated_child_adoption_result(
                        child_result,
                        source_state=(
                            state.agent_0_state,
                            state.agent_1_state,
                        )[index],
                        finalized=prepared_agents[index].finalization,
                        receipt=child_receipt,
                    )
                )
                adoptions.append(normalized)
                adoption_contracts[index] = contract_valid
                child_reconstruction_counts[index] = reconstruction_count
        else:
            adoptions.extend((None, None))
        child_valid = jnp.asarray(
            tuple(
                item is not None
                and adoption_contracts[index]
                and child_reconstruction_counts[index] == 1
                and _bool(item.diagnostics.transaction_applied)
                for index, item in enumerate(adoptions)
            ),
            dtype=jnp.bool_,
        )
        if bool(np.all(np.asarray(child_valid))):
            adopted_candidate = self._seal_state(
                HCCLContinualDyadState(
                    config_token=state.config_token,
                    content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
                    hccl_state=prepared.hccl_result.state,
                    agent_0_state=cast(
                        ExternalLearnedStateLiveMemoryActionStackResult,
                        adoptions[0],
                    ).state,
                    agent_1_state=cast(
                        ExternalLearnedStateLiveMemoryActionStackResult,
                        adoptions[1],
                    ).state,
                    planner_state=prepared.planner_result.state,
                    context_0_state=prepared.agent_0.context_result.state,
                    context_1_state=prepared.agent_1.context_result.state,
                )
            )
            candidate_valid = _bool(
                self.state_valid(adopted_candidate, _work=validation_work)
            ) and (
                _tree_exact_equal(adopted_candidate, prepared.candidate_state)
            )
        else:
            adopted_candidate = state
            candidate_valid = False
        commit = preflight and bool(np.all(np.asarray(child_valid))) and candidate_valid
        selected = adopted_candidate if commit else state
        applied = jnp.asarray(commit, dtype=jnp.bool_)
        rejected = jnp.asarray(not commit, dtype=jnp.bool_)
        called = jnp.asarray(
            tuple(adoption_calls),
            dtype=jnp.int32,
        )
        child_reconstructions = jnp.asarray(
            tuple(child_reconstruction_counts),
            dtype=jnp.int32,
        )
        adoption_work = HCCLContinualDyadAdoptionWork(
            source_state_integrity_checks=jnp.asarray(1, dtype=jnp.int32),
            preparation_integrity_checks=jnp.asarray(1, dtype=jnp.int32),
            receipt_integrity_checks=jnp.asarray(1, dtype=jnp.int32),
            outer_child_finalization_structural_recomputations=jnp.asarray(
                validation_work.child_finalization_structural_recomputations,
                dtype=jnp.int32,
            ),
            action_stack_integrity_adoptions=called,
            child_adoption_structural_recomputations=child_reconstructions,
            outer_commit_decisions=jnp.asarray(1, dtype=jnp.int32),
            outer_committed_pp_world_successors=jnp.asarray(
                int(commit),
                dtype=jnp.int32,
            ),
            outer_discarded_world_proposals=jnp.asarray(
                7 if commit else 8,
                dtype=jnp.int32,
            ),
            world_reevaluations=jnp.asarray(0, dtype=jnp.int32),
            context_reevaluations=jnp.zeros((_N_AGENTS,), dtype=jnp.int32),
            planner_reevaluations=jnp.asarray(0, dtype=jnp.int32),
            planner_validation_pair_authentication_calls=jnp.asarray(
                validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            planner_validation_agent_cache_authentication_evaluations=jnp.asarray(
                _N_AGENTS * validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            planner_validation_behavior_probability_vector_evaluations=jnp.asarray(
                _N_AGENTS * validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            planner_validation_grounded_joint_cell_prediction_equivalents=jnp.asarray(
                _N_AGENTS
                * _N_ACTIONS**2
                * validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            planner_validation_expected_reward_marginalization_products=jnp.asarray(
                _N_AGENTS
                * _N_ACTIONS**2
                * validation_work.planner_pair_authentication_calls,
                dtype=jnp.int32,
            ),
            coordinator_reevaluations=jnp.zeros((_N_AGENTS,), dtype=jnp.int32),
            prototype_reevaluations=jnp.zeros((_N_AGENTS,), dtype=jnp.int32),
            learned_memory_reevaluations=jnp.zeros((_N_AGENTS,), dtype=jnp.int32),
        )
        return HCCLContinualDyadResult(
            state=selected,
            prepared=prepared,
            receipt=receipt,
            agent_0_adoption=adoptions[0],
            agent_1_adoption=adoptions[1],
            adoption_work=adoption_work,
            source_state_matches=jnp.asarray(source_matches, dtype=jnp.bool_),
            source_state_valid=jnp.asarray(source_valid, dtype=jnp.bool_),
            prepared_content_valid=jnp.asarray(prepared_valid, dtype=jnp.bool_),
            receipt_valid=jnp.asarray(receipt_valid, dtype=jnp.bool_),
            child_adoptions_valid=child_valid,
            candidate_state_valid=jnp.asarray(candidate_valid, dtype=jnp.bool_),
            hccl_owner_committed=applied,
            action_stack_owners_committed=jnp.full(
                (_N_AGENTS,),
                applied,
                dtype=jnp.bool_,
            ),
            planner_owner_committed=applied,
            context_owners_committed=jnp.full(
                (_N_AGENTS,),
                applied,
                dtype=jnp.bool_,
            ),
            lineage_owners_committed=jnp.full(
                (_N_AGENTS,),
                applied,
                dtype=jnp.bool_,
            ),
            update_applied=applied,
            complete_source_returned=rejected,
        )

    def step(
        self,
        state: HCCLContinualDyadState,
        next_hard_action_masks: Array,
    ) -> HCCLContinualDyadResult:
        """Own one complete ordinary event from source state through adoption.

        The caller supplies only the persistent source and the external hard
        action masks.  Event identity, the B/M/P action binding, both canonical
        causal-core memory metadata records, the prepared transaction, and its
        integrity receipt are issued internally.  The lower two-phase methods
        remain public for explicit integrity and fault-injection protocols.
        """

        event = self.prepare_event(state)
        binding = self.bind_current_actions(state, event)
        event_inputs = self.causal_core_memory_event_inputs(state, event)
        prepared = self.prepare_transaction(
            state,
            event,
            binding,
            event_inputs[0],
            event_inputs[1],
            next_hard_action_masks,
        )
        receipt = self.integrity_receipt(prepared)
        result = self.adopt_prepared_transaction(state, prepared, receipt)
        if result.prepared is not prepared or result.receipt is not receipt:
            raise RuntimeError("continual-dyad step lost its prepared transaction identity")
        return result

    def resource_record(
        self,
        state: HCCLContinualDyadState,
        *,
        event: HCCLCausalCoreEventReceipt | None = None,
        binding: HCCLContinualDyadActionBinding | None = None,
        prepared: HCCLContinualDyadPreparedTransaction | None = None,
        receipt: HCCLContinualDyadPreparationReceipt | None = None,
    ) -> HCCLContinualDyadResourceRecord:
        """Measure the persistent owner tree and any supplied transient records.

        Nested component measurements are diagnostic partitions and are not
        added to the persistent total a second time.  Optional transient
        records are measured only when the caller supplies the exact record.
        Planner cache-authentication inference required by structural validity
        is counted explicitly below; no learner update or world donor runs.
        """

        self._state_contract(state)
        if _contains_tracer((state, event, binding, prepared, receipt)):
            raise TypeError("continual-dyad resource measurement is host/eager-only")
        validation_work = _OuterValidationWork()
        if not _bool(self.state_valid(state, _work=validation_work)):
            raise ValueError("resource measurement requires a valid continual-dyad state")

        if event is not None:
            self._hccl.world._require_event_contract(event)
            if not _bool(
                self._hccl.world.event_receipt_valid(
                    state.hccl_state.world_state,
                    event,
                )
            ):
                raise ValueError("supplied event does not belong to the measured state")
        if binding is not None:
            if event is None:
                raise ValueError("a measured action binding requires its event")
            self._binding_contract(binding)
            if not _bool(
                self.binding_valid(
                    state,
                    event,
                    binding,
                    _work=validation_work,
                )
            ):
                raise ValueError("supplied action binding is invalid for the measured state")
        if prepared is not None:
            if not self._prepared_static_contract_valid(prepared):
                raise ValueError("supplied preparation has a malformed static contract")
            if not _tree_exact_equal(prepared.source_state, state):
                raise ValueError("supplied preparation does not belong to the measured state")
            if event is not None and not _tree_exact_equal(prepared.event, event):
                raise ValueError("supplied preparation and event differ")
            if binding is not None and not _tree_exact_equal(prepared.binding, binding):
                raise ValueError("supplied preparation and action binding differ")
            if not self._prepared_content_valid(
                prepared,
                _work=validation_work,
            ):
                raise ValueError("supplied preparation content is invalid")
        if receipt is not None:
            if prepared is None:
                raise ValueError("a measured receipt requires its preparation")
            if not self._receipt_static_contract_valid(receipt):
                raise ValueError("supplied preparation receipt is malformed")
            if not self._receipt_valid(prepared, receipt):
                raise ValueError("supplied preparation receipt content is invalid")

        hccl_budget = self._hccl.resource_budget(state.hccl_state)
        planner_budget = self._planner.resource_budget(state.planner_state)
        context_0_budget = self._context.resource_record(state.context_0_state)
        context_1_budget = self._context.resource_record(state.context_1_state)

        hccl_bytes = _tree_nbytes(state.hccl_state)
        agent_0_bytes = _tree_nbytes(state.agent_0_state)
        agent_1_bytes = _tree_nbytes(state.agent_1_state)
        planner_bytes = _tree_nbytes(state.planner_state)
        context_pair_bytes = _tree_nbytes(
            (state.context_0_state, state.context_1_state)
        )
        outer_integrity_bytes = _tree_nbytes(
            (state.config_token, state.content_token)
        )
        formula_total = (
            hccl_bytes
            + agent_0_bytes
            + agent_1_bytes
            + planner_bytes
            + context_pair_bytes
            + outer_integrity_bytes
        )
        measured_total = _tree_nbytes(state)
        if formula_total != measured_total:
            raise AssertionError("continual-dyad owner partition double-counted state")
        if hccl_bytes != hccl_budget.total_persistent_state_nbytes:
            raise AssertionError("HCCL child resource formula disagrees with its state")
        if planner_bytes != planner_budget.measured_pair_nbytes:
            raise AssertionError("planner child resource measurement disagrees")
        if (
            context_pair_bytes
            != context_0_budget.measured_total_persistent_state_nbytes
            + context_1_budget.measured_total_persistent_state_nbytes
        ):
            raise AssertionError("context child resource measurements disagree")

        agents = (state.agent_0_state, state.agent_1_state)
        prototypes = tuple(self._prototype(agent) for agent in agents)
        prototype_implementations = (
            self._agent_0.coordinator.inner.prototype,
            self._agent_1.coordinator.inner.prototype,
        )
        feature_states = tuple(
            prototype_implementations[index]._feature_lifecycle_component_state(
                prototypes[index].state_builder_state
            )
            for index in range(_N_AGENTS)
        )
        fast_state_pair_bytes = _tree_nbytes(
            tuple(agent.coordinator_state.builder_state.hidden for agent in agents)
        )
        prototype_pair_bytes = _tree_nbytes(prototypes)
        feature_pair_bytes = _tree_nbytes(feature_states)
        action_binding_pair_bytes = _tree_nbytes(
            tuple(agent.action_binding for agent in agents)
        )
        learned_memory_pair_bytes = _tree_nbytes(
            tuple(agent.learned_memory_state for agent in agents)
        )

        binding_available = binding is not None
        prepared_available = prepared is not None
        receipt_available = receipt is not None
        event_bytes = (
            _tree_nbytes(event)
            if event is not None
            else hccl_budget.event_receipt_nbytes
        )
        return HCCLContinualDyadResourceRecord(
            schema=HCCL_CONTINUAL_DYAD_RESOURCE_SCHEMA,
            hccl_state_owners=1,
            action_stack_state_owners=2,
            planner_pair_state_owners=1,
            context_state_owners=2,
            lineage_state_owners=2,
            outer_integrity_owners=1,
            hccl_state_nbytes=hccl_bytes,
            agent_0_action_stack_nbytes=agent_0_bytes,
            agent_1_action_stack_nbytes=agent_1_bytes,
            planner_pair_state_nbytes=planner_bytes,
            context_pair_state_nbytes=context_pair_bytes,
            outer_integrity_nbytes=outer_integrity_bytes,
            fast_state_pair_nbytes=fast_state_pair_bytes,
            prototype_state_pair_nbytes=prototype_pair_bytes,
            feature_lifecycle_pair_nbytes=feature_pair_bytes,
            action_binding_pair_nbytes=action_binding_pair_bytes,
            learned_memory_pair_nbytes=learned_memory_pair_bytes,
            nested_breakdowns_excluded_from_total=True,
            total_persistent_state_nbytes=formula_total,
            measured_total_persistent_state_nbytes=measured_total,
            event_receipt_nbytes=event_bytes,
            outer_action_binding_nbytes=(
                0 if binding is None else _tree_nbytes(binding)
            ),
            outer_action_binding_measurement_available=binding_available,
            prepared_transaction_nbytes=(
                0 if prepared is None else _tree_nbytes(prepared)
            ),
            prepared_transaction_measurement_available=prepared_available,
            preparation_receipt_nbytes=(
                0 if receipt is None else _tree_nbytes(receipt)
            ),
            preparation_receipt_measurement_available=receipt_available,
            planner_validation_pair_authentication_calls=(
                validation_work.planner_pair_authentication_calls
            ),
            planner_validation_agent_cache_authentication_evaluations=(
                _N_AGENTS * validation_work.planner_pair_authentication_calls
            ),
            planner_validation_behavior_probability_vector_evaluations=(
                _N_AGENTS * validation_work.planner_pair_authentication_calls
            ),
            planner_validation_grounded_joint_cell_prediction_equivalents=(
                _N_AGENTS
                * _N_ACTIONS**2
                * validation_work.planner_pair_authentication_calls
            ),
            planner_validation_expected_reward_marginalization_products=(
                _N_AGENTS
                * _N_ACTIONS**2
                * validation_work.planner_pair_authentication_calls
            ),
            child_finalization_structural_recomputations=tuple(
                validation_work.child_finalization_structural_recomputations
            ),
            maximum_transient_world_proposal_stack_nbytes=(
                hccl_budget.max_transient_world_proposal_stack_nbytes
            ),
            planner=planner_budget,
            context=context_0_budget,
            hccl=hccl_budget,
            physical_observation_dim=_PHYSICAL_DIM,
            external_recurrent_input_dim=_EXTERNAL_RAW_DIM,
            stable_base_dim=_BASE_DIM,
            pair_source_dim=_PHYSICAL_DIM,
            active_pair_slots=_ACTIVE_PAIR_SLOTS,
            pair_candidate_slots=_PAIR_CANDIDATE_SLOTS,
            routed_representation_dim=_ROUTED_DIM,
            preparation_persisted=False,
            prepared_checkpoint_supported=False,
            full_generated_feature_consumer_routing=False,
            planner_generated_feature_tail_consumed=False,
            learned_memory_rows_feature_generation_bound=False,
            learned_memory_reencode_count_available=False,
            composite_jit_supported=False,
            output_write_calls=0,
            artifact_bytes_written=0,
        )
