# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Atomic HCCL causal feedback for two live learned-memory adapters.

This L0 bridge owns one HCCL world/attribution state and exactly two
``ExternalLearnedStateLiveMemoryAdapterState`` values.  The outer state also
persists the two hard-action masks that governed the currently cached
decisions.  Current B and M actions come only from each live adapter's pending
binding when one exists; an abstaining child has B=M=current action.  P=M is an
explicit no-planner rung.

After the exact eight world proposals, agent 0 receives only
``M0B1 - BB`` and agent 1 receives only ``B0M1 - BB``.  The dyad interaction is
retained as an audit fact and is never used as either controller's feedback.
Each child then advances exactly once from its own executed M action, PP net
reward, and PP next observation.  The next-decision masks enter those child
transactions but replace the outer current masks only when HCCL and both live
adapters commit together.  Any refusal returns all three owners and the current
mask binding bit-exactly.

The composite is host/eager-only.  Its deterministic receipts bind integrity,
not caller authentication.  It adds no planner, dispatch, safety, schedule,
seed, output, artifact, threshold, evidence, benefit, or promotion authority.
``delight_or_actor_backward=False`` is a protocol fact, not a judgment that a
gradient did or did not spark joy.
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

from alberta_framework.core.external_learned_state_live_memory_adapter import (
    ExternalLearnedStateLiveMemoryAdapter,
    ExternalLearnedStateLiveMemoryAdapterConfig,
    ExternalLearnedStateLiveMemoryAdapterState,
    ExternalLearnedStateLiveMemoryEventInput,
    ExternalLearnedStateLiveMemoryFeedback,
    ExternalLearnedStateLiveMemoryPendingBinding,
    ExternalLearnedStateLiveMemoryResult,
    measure_external_learned_state_live_memory_adapter_state_nbytes,
)
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateTransition,
)
from alberta_framework.core.hccl_causal_attribution import (
    HCCLActionLayer,
    HCCLActionReceipt,
    HCCLSignalContrast,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapter,
    HCCLWorldAttributionAdapterConfig,
    HCCLWorldAttributionAdapterResult,
    HCCLWorldAttributionAdapterState,
    measure_hccl_world_attribution_state_nbytes,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCLCausalCoreEventReceipt,
    HCCLCausalCoreProposal,
)

HCCL_TWO_LIVE_MEMORY_CONFIG_SCHEMA = "alberta.hccl-two-live-memory-config.v1"
HCCL_TWO_LIVE_MEMORY_STATE_SCHEMA = "alberta.hccl-two-live-memory-state.v1"
HCCL_TWO_LIVE_MEMORY_BINDING_SCHEMA = "alberta.hccl-two-live-memory-binding.v1"
HCCL_TWO_LIVE_MEMORY_CHECKPOINT_SCHEMA = (
    "alberta.hccl-two-live-memory-checkpoint.v1"
)
HCCL_TWO_LIVE_MEMORY_RESOURCE_SCHEMA = "alberta.hccl-two-live-memory-resource.v1"
HCCL_TWO_LIVE_MEMORY_STATUS = (
    "l0-development-hccl-two-live-memory-causal-feedback"
)
HCCL_TWO_LIVE_MEMORY_EVIDENCE_LEVEL = "L0"
HCCL_TWO_LIVE_MEMORY_LIMITATIONS = (
    "agent-0-feedback-is-M0B1-minus-BB-only",
    "agent-1-feedback-is-B0M1-minus-BB-only",
    "memory-interaction-is-audit-only",
    "P-equals-M-no-planner-rung",
    "current-and-next-decision-hard-masks-are-distinct",
    "delight-or-actor-backward-false-is-protocol-only",
    "integrity-binding-is-not-caller-authentication",
    "composite-stage-is-host-eager-only",
    "no-schedule-seed-output-artifact-threshold-evidence-or-promotion",
)

_N_AGENTS = 2
_N_ACTIONS = 2
_RAW_OBSERVATION_DIM = 16
_B0M1_SLOT = 1
_M0B1_SLOT = 2
_BB_SLOT = 3
_PP_SLOT = 4
_UINT32_MAX = 2**32 - 1


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_python_equal(left: object, right: object) -> bool:
    """Compare JSON-shaped values without Python's bool/int or tuple/list aliases."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_dict = cast(dict[object, object], left)
        right_dict = cast(dict[object, object], right)
        if len(left_dict) != len(right_dict):
            return False
        for key, value in left_dict.items():
            if key not in right_dict or not _canonical_python_equal(
                value, right_dict[key]
            ):
                return False
        return True
    if type(left) in {list, tuple}:
        left_items = cast(list[object] | tuple[object, ...], left)
        right_items = cast(list[object] | tuple[object, ...], right)
        return len(left_items) == len(right_items) and all(
            _canonical_python_equal(left_item, right_item)
            for left_item, right_item in zip(left_items, right_items, strict=True)
        )
    return bool(left == right)


def _require_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    label: str,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    return jnp.asarray(value)


def _contains_tracer(value: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _tree_select(condition: Array, yes: Any, no: Any) -> Any:
    return jax.tree.map(lambda left, right: jnp.where(condition, left, right), yes, no)


def _rotate_left(value: Array, distance: Array) -> Array:
    right = (jnp.asarray(32, dtype=jnp.uint32) - distance) & jnp.uint32(31)
    return jnp.asarray((value << distance) | (value >> right), dtype=jnp.uint32)


def _content_tag(owner: Array, *values: Array) -> UInt[Array, " 4"]:
    words: list[Array] = [jnp.reshape(owner, (-1,))]
    for value in values:
        array = jax.lax.stop_gradient(jnp.asarray(value))
        if array.dtype in {jnp.dtype(jnp.float32), jnp.dtype(jnp.int32)}:
            converted = jax.lax.bitcast_convert_type(array, jnp.uint32)
        elif array.dtype == jnp.dtype(jnp.bool_):
            converted = array.astype(jnp.uint32)
        elif array.dtype == jnp.dtype(jnp.uint32):
            converted = array
        else:
            raise TypeError("binding tags support float32/int32/bool/uint32 only")
        words.append(jnp.reshape(converted, (-1,)))
    payload = jnp.concatenate(tuple(words)).astype(jnp.uint32)
    indices = jnp.arange(payload.shape[0], dtype=jnp.uint32)
    mixed = _rotate_left(
        payload ^ (indices * jnp.uint32(0x9E3779B9)),
        (indices % jnp.uint32(31)) + jnp.uint32(1),
    )
    return jnp.stack(
        (
            jnp.bitwise_xor.reduce(mixed),
            jnp.sum(mixed * jnp.uint32(0x85EBCA6B), dtype=jnp.uint32),
            jnp.bitwise_xor.reduce(mixed * (indices + jnp.uint32(0xC2B2AE35))),
            jnp.sum(
                _rotate_left(
                    mixed,
                    ((indices * jnp.uint32(7)) % jnp.uint32(31)) + jnp.uint32(1),
                ),
                dtype=jnp.uint32,
            ),
        )
    ).astype(jnp.uint32)


def _array_words(value: Array) -> Array:
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        return jr.key_data(value)
    if value.dtype in {jnp.dtype(jnp.float32), jnp.dtype(jnp.int32)}:
        return jax.lax.bitcast_convert_type(value, jnp.uint32)
    if value.dtype == jnp.dtype(jnp.bool_):
        return value.astype(jnp.uint32)
    if value.dtype in {jnp.dtype(jnp.uint32), jnp.dtype(jnp.uint8)}:
        return value.astype(jnp.uint32)
    raise TypeError("exact equality supports typed keys, float32, int32, bool, and uints")


def _tree_exact_equal(left: Any, right: Any) -> Bool[Array, ""]:
    if type(left) is not type(right):
        return jnp.asarray(False, dtype=jnp.bool_)
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(
        right_leaves
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = cast(Array, left_leaf)
        right_array = cast(Array, right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        equal = equal & jnp.all(_array_words(left_array) == _array_words(right_array))
    return equal


def _float_bits_equal(left: Array, right: Array) -> Array:
    return jnp.all(
        jax.lax.bitcast_convert_type(left, jnp.uint32)
        == jax.lax.bitcast_convert_type(right, jnp.uint32)
    )


def _tree_nbytes(value: Any) -> int:
    total = 0
    for leaf in jax.tree.leaves(value):
        if not hasattr(leaf, "dtype"):
            continue
        array = cast(Array, leaf)
        material = (
            jr.key_data(array)
            if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
            else array
        )
        total += int(material.size) * int(material.dtype.itemsize)
    return total


def _state_host_payload(
    state: HCCLTwoLiveMemoryBridgeState,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for leaf in jax.tree.leaves(state):
        if not hasattr(leaf, "dtype"):
            continue
        array = cast(Array, leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            host = np.asarray(jr.key_data(array), dtype=np.uint32)
            dtype = "typed-threefry-key-uint32"
        else:
            host = np.asarray(array)
            dtype = str(host.dtype)
        payload.append(
            {
                "shape": list(host.shape),
                "dtype": dtype,
                "bytes_hex": np.ascontiguousarray(host).tobytes().hex(),
            }
        )
    return payload


def _contrast_exact_zero(contrast: HCCLSignalContrast) -> Bool[Array, ""]:
    return (
        jnp.all(jax.lax.bitcast_convert_type(contrast.task_score, jnp.uint32) == 0)
        & jnp.all(jax.lax.bitcast_convert_type(contrast.net_reward, jnp.uint32) == 0)
        & jnp.all(jax.lax.bitcast_convert_type(contrast.safety_cost, jnp.uint32) == 0)
        & jnp.all(jax.lax.bitcast_convert_type(contrast.message_charge, jnp.uint32) == 0)
    )


def _validate_live_config(
    config: ExternalLearnedStateLiveMemoryAdapterConfig,
    *,
    label: str,
) -> None:
    if type(config) is not ExternalLearnedStateLiveMemoryAdapterConfig:
        raise TypeError(f"{label} must be an exact live-memory adapter config")
    coordinator = config.coordinator
    if coordinator.builder.observation_dim != _RAW_OBSERVATION_DIM:
        raise ValueError(f"{label} raw observation width must be 16")
    if coordinator.builder.n_actions != _N_ACTIONS:
        raise ValueError(f"{label} must expose exactly two primitive actions")
    prototype = coordinator.inner.prototype
    if prototype.option_search_control is not None:
        raise ValueError(f"{label} option search would add planner authority")
    if prototype.oak.stomp.option_planning_backups_per_step != 0:
        raise ValueError(f"{label} STOMP planner backups must remain zero")


@dataclasses.dataclass(frozen=True)
class HCCLTwoLiveMemoryBridgeConfig:
    hccl: HCCLWorldAttributionAdapterConfig
    agent_0: ExternalLearnedStateLiveMemoryAdapterConfig
    agent_1: ExternalLearnedStateLiveMemoryAdapterConfig
    binding_owner_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.hccl) is not HCCLWorldAttributionAdapterConfig:
            raise TypeError("hccl must be exact HCCLWorldAttributionAdapterConfig")
        _validate_live_config(self.agent_0, label="agent_0")
        _validate_live_config(self.agent_1, label="agent_1")
        value = self.binding_owner_digest
        if type(value) is not tuple or len(value) != 8:
            raise ValueError("binding_owner_digest must be an exact eight-word tuple")
        for index, word in enumerate(value):
            if type(word) is not int or not 0 <= word <= _UINT32_MAX:
                raise ValueError(f"binding_owner_digest[{index}] must be uint32")
        if not any(value[:4]) or not any(value[4:]) or value[:4] == value[4:]:
            raise ValueError("binding owner must contain two distinct nonzero identities")


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryBridgeState:
    hccl_state: HCCLWorldAttributionAdapterState
    agent_0_state: ExternalLearnedStateLiveMemoryAdapterState
    agent_1_state: ExternalLearnedStateLiveMemoryAdapterState
    current_hard_action_masks: Bool[Array, "2 2"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryActionBinding:
    source_world_words: UInt[Array, " 2"]
    source_world_tag_words: UInt[Array, " 4"]
    event_content_tag_words: UInt[Array, " 4"]
    feedback_binding_available: Bool[Array, " 2"]
    live_memory_transaction_words: UInt[Array, "2 2"]
    prototype_decision_words: UInt[Array, "2 4"]
    base_actions: Int[Array, " 2"]
    memory_actions_before_mask: Int[Array, " 2"]
    memory_actions: Int[Array, " 2"]
    current_hard_action_masks: Bool[Array, "2 2"]
    base: HCCLActionReceipt
    memory: HCCLActionReceipt
    planner: HCCLActionReceipt
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryWork:
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    live_adapter_step_calls: Int[Array, " 2"]
    committed_composite_transactions: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLTwoLiveMemoryResult:
    state: HCCLTwoLiveMemoryBridgeState
    binding: HCCLTwoLiveMemoryActionBinding
    hccl_result: HCCLWorldAttributionAdapterResult
    agent_0_transition: ExternalLearnedStateTransition
    agent_1_transition: ExternalLearnedStateTransition
    agent_0_feedback: ExternalLearnedStateLiveMemoryFeedback
    agent_1_feedback: ExternalLearnedStateLiveMemoryFeedback
    agent_0_result: ExternalLearnedStateLiveMemoryResult
    agent_1_result: ExternalLearnedStateLiveMemoryResult
    agent_unilateral_counterfactual_delta: Float[Array, " 2"]
    memory_interaction_audit: HCCLSignalContrast
    work: HCCLTwoLiveMemoryWork
    source_state_valid: Bool[Array, ""]
    event_receipt_valid: Bool[Array, ""]
    binding_integrity_valid: Bool[Array, ""]
    binding_matches_source: Bool[Array, ""]
    feedback_bindings_complete: Bool[Array, ""]
    feedback_bindings_match_children: Bool[Array, ""]
    prior_feedback_required: Bool[Array, " 2"]
    prior_feedback_supplied: Bool[Array, " 2"]
    current_event_masks_bound: Bool[Array, ""]
    planner_equals_memory: Bool[Array, ""]
    pp_executes_memory_actions: Bool[Array, ""]
    no_planner_rung_valid: Bool[Array, ""]
    agent_0_feedback_is_m0b1_minus_bb: Bool[Array, ""]
    agent_1_feedback_is_b0m1_minus_bb: Bool[Array, ""]
    mm_minus_bb_broadcast_to_both_agents: Bool[Array, ""]
    memory_interaction_used_for_agent_feedback: Bool[Array, ""]
    delight_or_actor_backward: Bool[Array, ""]
    downstream_candidate_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    hccl_commit_applied: Bool[Array, ""]
    agent_0_update_applied: Bool[Array, ""]
    agent_1_update_applied: Bool[Array, ""]
    next_decision_masks_installed: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class HCCLTwoLiveMemoryResourceBudget:
    schema: str
    hccl_state_owners: int
    live_memory_adapter_state_owners: int
    external_coordinator_state_owners: int
    learned_memory_controller_state_owners: int
    prototype_state_owners: int
    current_hard_action_mask_bindings: int
    hccl_state_nbytes: int
    agent_0_state_nbytes: int
    agent_1_state_nbytes: int
    current_hard_action_mask_nbytes: int
    total_persistent_state_nbytes: int
    max_world_proposal_calls_per_transaction: int
    max_attribution_proposal_calls_per_transaction: int
    live_adapter_step_calls_per_transaction: int
    maximum_feedback_settlements_per_transaction: int
    maximum_cached_action_replacements_per_transaction: int
    planner_calls_per_transaction: int
    delight_or_actor_backward_calls_per_transaction: int
    maximum_composite_transactions: int
    composite_jit_supported: bool
    scan_supported: bool
    output_write_calls: int
    artifact_bytes_written: int

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class HCCLTwoLiveMemoryCheckpoint:
    schema: str
    mechanism_status: str
    evidence_level: str
    output_writes_authorized: bool
    artifact_authorized: bool
    evidence_authorized: bool
    config: dict[str, object]
    config_sha256: str
    resource_budget: dict[str, object]
    state: HCCLTwoLiveMemoryBridgeState
    state_nbytes: int
    state_sha256: str
    checkpoint_sha256: str


class HCCLTwoLiveMemoryBridge:
    """Host-only all-or-none owner over HCCL and two live memory adapters."""

    def __init__(self, config: HCCLTwoLiveMemoryBridgeConfig):
        if type(config) is not HCCLTwoLiveMemoryBridgeConfig:
            raise TypeError("config must be exact HCCLTwoLiveMemoryBridgeConfig")
        self._config = config
        self._hccl = HCCLWorldAttributionAdapter(config.hccl)
        self._agent_0 = ExternalLearnedStateLiveMemoryAdapter(config.agent_0)
        self._agent_1 = ExternalLearnedStateLiveMemoryAdapter(config.agent_1)
        self._owner = jnp.asarray(config.binding_owner_digest, dtype=jnp.uint32)

    @property
    def config(self) -> HCCLTwoLiveMemoryBridgeConfig:
        return self._config

    @property
    def hccl(self) -> HCCLWorldAttributionAdapter:
        return self._hccl

    @property
    def agent_0(self) -> ExternalLearnedStateLiveMemoryAdapter:
        return self._agent_0

    @property
    def agent_1(self) -> ExternalLearnedStateLiveMemoryAdapter:
        return self._agent_1

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": HCCL_TWO_LIVE_MEMORY_CONFIG_SCHEMA,
            "state_schema": HCCL_TWO_LIVE_MEMORY_STATE_SCHEMA,
            "binding_schema": HCCL_TWO_LIVE_MEMORY_BINDING_SCHEMA,
            "checkpoint_schema": HCCL_TWO_LIVE_MEMORY_CHECKPOINT_SCHEMA,
            "resource_schema": HCCL_TWO_LIVE_MEMORY_RESOURCE_SCHEMA,
            "mechanism_status": HCCL_TWO_LIVE_MEMORY_STATUS,
            "evidence_level": HCCL_TWO_LIVE_MEMORY_EVIDENCE_LEVEL,
            "hccl": self._hccl.to_config(),
            "agent_0": self._agent_0.to_config(),
            "agent_1": self._agent_1.to_config(),
            "binding_owner_digest": list(self._config.binding_owner_digest),
            "hccl_state_owners": 1,
            "live_memory_adapter_state_owners": 2,
            "external_coordinator_state_owners": 2,
            "learned_memory_controller_state_owners": 2,
            "prototype_state_owners": 2,
            "additional_coordinator_state_owners": 0,
            "additional_memory_controller_state_owners": 0,
            "additional_prototype_state_owners": 0,
            "current_hard_action_mask_bindings": 1,
            "planner_action_relation": "P=M-no-planner-rung",
            "per_agent_memory_feedback": ["M0B1-BB", "B0M1-BB"],
            "memory_interaction_usage": "separate-audit-fact-only",
            "current_next_mask_semantics": "distinct-install-next-only-on-outer-commit",
            "delight_or_actor_backward": False,
            "delight_interpretation": "protocol-fact-only-not-evaluated",
            "composite_jit_supported": False,
            "scan_supported": False,
            "caller_identity_authenticated": False,
            "planner_authority": False,
            "dispatch_authority": False,
            "safety_authority": False,
            "schedule_execution_authorized": False,
            "seed_authority": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "threshold_authorized": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_TWO_LIVE_MEMORY_LIMITATIONS),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLTwoLiveMemoryBridge:
        if type(payload) is not dict:
            raise TypeError("config payload must be an exact dict")
        hccl_raw = payload.get("hccl")
        agent_0_raw = payload.get("agent_0")
        agent_1_raw = payload.get("agent_1")
        owner_raw = payload.get("binding_owner_digest")
        if type(hccl_raw) is not dict:
            raise TypeError("hccl config must be an exact dict")
        if type(agent_0_raw) is not dict or type(agent_1_raw) is not dict:
            raise TypeError("live-memory configs must be exact dicts")
        if type(owner_raw) is not list:
            raise TypeError("binding_owner_digest must serialize as a list")
        candidate = cls(
            HCCLTwoLiveMemoryBridgeConfig(
                hccl=HCCLWorldAttributionAdapter.from_config(hccl_raw).config,
                agent_0=ExternalLearnedStateLiveMemoryAdapterConfig.from_config(
                    agent_0_raw
                ),
                agent_1=ExternalLearnedStateLiveMemoryAdapterConfig.from_config(
                    agent_1_raw
                ),
                binding_owner_digest=tuple(owner_raw),
            )
        )
        if not _canonical_python_equal(candidate.to_config(), dict(payload)):
            raise ValueError("HCCL two-live-memory config is unsupported")
        return candidate

    def _require_state_contract(self, state: HCCLTwoLiveMemoryBridgeState) -> None:
        if type(state) is not HCCLTwoLiveMemoryBridgeState:
            raise TypeError("state must be exact HCCLTwoLiveMemoryBridgeState")
        self._hccl._require_state_contract(state.hccl_state)
        if type(state.agent_0_state) is not ExternalLearnedStateLiveMemoryAdapterState:
            raise TypeError("agent_0_state must be an exact live-memory state")
        if type(state.agent_1_state) is not ExternalLearnedStateLiveMemoryAdapterState:
            raise TypeError("agent_1_state must be an exact live-memory state")
        _require_array(
            state.current_hard_action_masks,
            shape=(_N_AGENTS, _N_ACTIONS),
            dtype=jnp.dtype(jnp.bool_),
            label="state.current_hard_action_masks",
        )

    def _child_states(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
    ) -> tuple[
        ExternalLearnedStateLiveMemoryAdapterState,
        ExternalLearnedStateLiveMemoryAdapterState,
    ]:
        return state.agent_0_state, state.agent_1_state

    def _pending_mask_relation(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
    ) -> Bool[Array, ""]:
        valid = jnp.asarray(True, dtype=jnp.bool_)
        for agent, child in enumerate(self._child_states(state)):
            pending = child.pending_binding
            base = pending.base_action_before_retrieval
            memory = pending.effective_action
            base_in_range = (base >= 0) & (base < _N_ACTIONS)
            memory_in_range = (memory >= 0) & (memory < _N_ACTIONS)
            safe_base = jnp.clip(base, 0, _N_ACTIONS - 1)
            safe_memory = jnp.clip(memory, 0, _N_ACTIONS - 1)
            valid = valid & jnp.where(
                pending.available,
                jnp.all(pending.hard_action_mask == state.current_hard_action_masks[agent])
                & base_in_range
                & memory_in_range
                & pending.hard_action_mask[safe_base]
                & pending.hard_action_mask[safe_memory],
                jnp.asarray(True, dtype=jnp.bool_),
            )
        return valid

    def state_valid(self, state: HCCLTwoLiveMemoryBridgeState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        observations = self._hccl.world.observe(state.hccl_state.world_state)
        children = self._child_states(state)
        actions = jnp.stack(
            tuple(child.coordinator_state.current_action for child in children)
        ).astype(jnp.int32)
        safe = jnp.clip(actions, 0, _N_ACTIONS - 1)
        return (
            self._hccl.state_valid(state.hccl_state)
            & self._agent_0.state_valid(state.agent_0_state)
            & self._agent_1.state_valid(state.agent_1_state)
            & state.agent_0_state.coordinator_state.started
            & state.agent_1_state.coordinator_state.started
            & jnp.all(
                state.agent_0_state.coordinator_state.event_words
                == state.hccl_state.world_state.step_words
            )
            & jnp.all(
                state.agent_1_state.coordinator_state.event_words
                == state.hccl_state.world_state.step_words
            )
            & _float_bits_equal(
                state.agent_0_state.coordinator_state.current_raw_observation,
                observations[0],
            )
            & _float_bits_equal(
                state.agent_1_state.coordinator_state.current_raw_observation,
                observations[1],
            )
            & jnp.all((actions >= 0) & (actions < _N_ACTIONS))
            & state.current_hard_action_masks[0, safe[0]]
            & state.current_hard_action_masks[1, safe[1]]
            & self._pending_mask_relation(state)
        )

    def init(
        self,
        key: Array,
        *,
        initial_hard_action_masks: Array | None = None,
    ) -> HCCLTwoLiveMemoryBridgeState:
        if not (
            hasattr(key, "shape")
            and hasattr(key, "dtype")
            and key.shape == ()
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        ):
            raise TypeError("key must be a scalar typed JAX PRNG key")
        masks = (
            jnp.ones((_N_AGENTS, _N_ACTIONS), dtype=jnp.bool_)
            if initial_hard_action_masks is None
            else _require_array(
                initial_hard_action_masks,
                shape=(_N_AGENTS, _N_ACTIONS),
                dtype=jnp.dtype(jnp.bool_),
                label="initial_hard_action_masks",
            )
        )
        hccl_key, agent_0_key, agent_1_key = jr.split(key, 3)
        hccl_state = self._hccl.init(hccl_key)
        observations = self._hccl.world.observe(hccl_state.world_state)
        agent_0 = self._agent_0.start(
            self._agent_0.init(agent_0_key),
            observations[0],
        )
        agent_1 = self._agent_1.start(
            self._agent_1.init(agent_1_key),
            observations[1],
        )
        state = HCCLTwoLiveMemoryBridgeState(
            hccl_state=hccl_state,
            agent_0_state=agent_0,
            agent_1_state=agent_1,
            current_hard_action_masks=masks,
        )
        if not bool(self.state_valid(state)):
            raise ValueError("initial masks must admit both initialized cached actions")
        return state

    def prepare_event(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
    ) -> HCCLCausalCoreEventReceipt:
        self._require_state_contract(state)
        if not bool(self.state_valid(state)):
            raise ValueError("cannot prepare an event from an invalid composite state")
        return self._hccl.world.prepare_event(state.hccl_state.world_state)

    def _binding_components(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        available: list[Array] = []
        transactions: list[Array] = []
        decisions: list[Array] = []
        base_actions: list[Array] = []
        memory_before: list[Array] = []
        memory_actions: list[Array] = []
        for child in self._child_states(state):
            pending = child.pending_binding
            coordinator = child.coordinator_state
            available.append(pending.available)
            transactions.append(
                jnp.where(
                    pending.available,
                    pending.memory_transaction_words,
                    jnp.zeros((2,), dtype=jnp.uint32),
                )
            )
            decisions.append(
                jnp.where(
                    pending.available,
                    pending.prototype_decision_id,
                    coordinator.current_decision_id,
                )
            )
            base_actions.append(
                jnp.where(
                    pending.available,
                    pending.base_action_before_retrieval,
                    coordinator.current_action,
                ).astype(jnp.int32)
            )
            memory_actions.append(coordinator.current_action.astype(jnp.int32))
            memory_before.append(
                jnp.where(
                    pending.available & pending.categorical_retrieval,
                    pending.retrieval_action,
                    coordinator.current_action,
                ).astype(jnp.int32)
            )
        return (
            jnp.stack(tuple(available)).astype(jnp.bool_),
            jnp.stack(tuple(transactions)).astype(jnp.uint32),
            jnp.stack(tuple(decisions)).astype(jnp.uint32),
            jnp.stack(tuple(base_actions)).astype(jnp.int32),
            jnp.stack(tuple(memory_before)).astype(jnp.int32),
            jnp.stack(tuple(memory_actions)).astype(jnp.int32),
        )

    def _receipt_identity_rows(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
        event: HCCLCausalCoreEventReceipt,
        layer: HCCLActionLayer,
        transactions: Array,
        decisions: Array,
        base_actions: Array,
        memory_actions: Array,
    ) -> Array:
        rows: list[Array] = []
        for agent, child in enumerate(self._child_states(state)):
            coordinator = child.coordinator_state
            rows.append(
                _content_tag(
                    self._owner,
                    jnp.asarray(int(layer), dtype=jnp.int32),
                    jnp.asarray(agent, dtype=jnp.int32),
                    state.hccl_state.world_state.step_words,
                    event.content_tag_words,
                    coordinator.event_words,
                    coordinator.cached_builder_step_words,
                    coordinator.cached_prototype_step_words,
                    coordinator.cached_feature_generation_words,
                    transactions[agent],
                    decisions[agent, :2],
                    decisions[agent],
                    base_actions[agent],
                    memory_actions[agent],
                    state.current_hard_action_masks[agent],
                )
            )
        return jnp.stack(tuple(rows)).astype(jnp.uint32)

    def _binding_tag(self, binding: HCCLTwoLiveMemoryActionBinding) -> Array:
        return _content_tag(
            self._owner,
            binding.source_world_words,
            binding.source_world_tag_words,
            binding.event_content_tag_words,
            binding.feedback_binding_available,
            binding.live_memory_transaction_words,
            binding.prototype_decision_words,
            binding.base_actions,
            binding.memory_actions_before_mask,
            binding.memory_actions,
            binding.current_hard_action_masks,
            binding.base.action_receipt_identity_words,
            binding.memory.action_receipt_identity_words,
            binding.planner.action_receipt_identity_words,
            binding.base.content_tag_words,
            binding.memory.content_tag_words,
            binding.planner.content_tag_words,
        )

    def _make_binding(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
        event: HCCLCausalCoreEventReceipt,
    ) -> HCCLTwoLiveMemoryActionBinding:
        available, transactions, decisions, base_actions, memory_before, memory = (
            self._binding_components(state)
        )
        receipts: list[HCCLActionReceipt] = []
        for layer, before, after in (
            (HCCLActionLayer.BASE, base_actions, base_actions),
            (HCCLActionLayer.MEMORY, memory_before, memory),
            (HCCLActionLayer.PLANNER, memory, memory),
        ):
            receipts.append(
                self._hccl.bind_action_receipt(
                    state.hccl_state,
                    event,
                    layer=layer,
                    actions_before_mask=before,
                    actions_after_mask=after,
                    hard_action_masks=state.current_hard_action_masks,
                    action_receipt_identity_words=self._receipt_identity_rows(
                        state,
                        event,
                        layer,
                        transactions,
                        decisions,
                        base_actions,
                        memory,
                    ),
                )
            )
        bare = HCCLTwoLiveMemoryActionBinding(
            source_world_words=state.hccl_state.world_state.step_words,
            source_world_tag_words=state.hccl_state.world_state.content_tag_words,
            event_content_tag_words=event.content_tag_words,
            feedback_binding_available=available,
            live_memory_transaction_words=transactions,
            prototype_decision_words=decisions,
            base_actions=base_actions,
            memory_actions_before_mask=memory_before,
            memory_actions=memory,
            current_hard_action_masks=state.current_hard_action_masks,
            base=receipts[0],
            memory=receipts[1],
            planner=receipts[2],
            content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
        )
        return cast(
            HCCLTwoLiveMemoryActionBinding,
            cast(Any, bare).replace(content_tag_words=self._binding_tag(bare)),
        )

    def _identities_distinct(self, binding: HCCLTwoLiveMemoryActionBinding) -> Array:
        identities = jnp.concatenate(
            (
                binding.base.action_receipt_identity_words,
                binding.memory.action_receipt_identity_words,
                binding.planner.action_receipt_identity_words,
            ),
            axis=0,
        )
        distinct = jnp.asarray(True, dtype=jnp.bool_)
        for left in range(6):
            for right in range(left):
                distinct = distinct & (~jnp.all(identities[left] == identities[right]))
        return distinct

    def bind_live_memory_actions(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
        event: HCCLCausalCoreEventReceipt,
    ) -> HCCLTwoLiveMemoryActionBinding:
        self._require_state_contract(state)
        self._hccl.world._require_event_contract(event)
        if not bool(self.state_valid(state)):
            raise ValueError("cannot bind actions from an invalid composite state")
        if not bool(self._hccl.world.event_receipt_valid(state.hccl_state.world_state, event)):
            raise ValueError("cannot bind actions to a stale or invalid HCCL event")
        binding = self._make_binding(state, event)
        if not bool(self._identities_distinct(binding)):
            raise RuntimeError("deterministic action receipt identities collided")
        return binding

    def _require_binding_contract(self, binding: HCCLTwoLiveMemoryActionBinding) -> None:
        if type(binding) is not HCCLTwoLiveMemoryActionBinding:
            raise TypeError("binding must be exact HCCLTwoLiveMemoryActionBinding")
        for name, shape, dtype in (
            ("source_world_words", (2,), jnp.uint32),
            ("source_world_tag_words", (4,), jnp.uint32),
            ("event_content_tag_words", (4,), jnp.uint32),
            ("feedback_binding_available", (2,), jnp.bool_),
            ("live_memory_transaction_words", (2, 2), jnp.uint32),
            ("prototype_decision_words", (2, 4), jnp.uint32),
            ("base_actions", (2,), jnp.int32),
            ("memory_actions_before_mask", (2,), jnp.int32),
            ("memory_actions", (2,), jnp.int32),
            ("current_hard_action_masks", (2, 2), jnp.bool_),
            ("content_tag_words", (4,), jnp.uint32),
        ):
            _require_array(
                getattr(binding, name),
                shape=shape,
                dtype=jnp.dtype(dtype),
                label=f"binding.{name}",
            )
        for receipt in (binding.base, binding.memory, binding.planner):
            self._hccl.attribution._require_action_contract(receipt)

    def _binding_integrity_valid(self, binding: HCCLTwoLiveMemoryActionBinding) -> Array:
        return (
            jnp.all(binding.base.actions_before_mask == binding.base_actions)
            & jnp.all(binding.base.actions_after_mask == binding.base_actions)
            & jnp.all(
                binding.memory.actions_before_mask == binding.memory_actions_before_mask
            )
            & jnp.all(binding.memory.actions_after_mask == binding.memory_actions)
            & jnp.all(binding.planner.actions_before_mask == binding.memory_actions)
            & jnp.all(binding.planner.actions_after_mask == binding.memory_actions)
            & jnp.all(
                binding.base.hard_action_masks == binding.current_hard_action_masks
            )
            & jnp.all(
                binding.memory.hard_action_masks == binding.current_hard_action_masks
            )
            & jnp.all(
                binding.planner.hard_action_masks == binding.current_hard_action_masks
            )
            & self._identities_distinct(binding)
            & jnp.all(binding.content_tag_words == self._binding_tag(binding))
        )

    def _feedback_binding_relations(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
        binding: HCCLTwoLiveMemoryActionBinding,
    ) -> tuple[Array, Array, Array]:
        required = jnp.stack(
            tuple(child.pending_binding.available for child in self._child_states(state))
        ).astype(jnp.bool_)
        complete = jnp.all(binding.feedback_binding_available == required)
        matches = jnp.asarray(True, dtype=jnp.bool_)
        for agent, child in enumerate(self._child_states(state)):
            pending = child.pending_binding
            coordinator = child.coordinator_state
            active = (
                binding.feedback_binding_available[agent]
                & jnp.all(
                    binding.live_memory_transaction_words[agent]
                    == pending.memory_transaction_words
                )
                & jnp.all(
                    binding.prototype_decision_words[agent]
                    == pending.prototype_decision_id
                )
                & (
                    binding.base_actions[agent]
                    == pending.base_action_before_retrieval
                )
                & (binding.memory_actions[agent] == pending.effective_action)
                & jnp.all(
                    binding.current_hard_action_masks[agent]
                    == pending.hard_action_mask
                )
            )
            inactive = (
                ~binding.feedback_binding_available[agent]
                & jnp.all(binding.live_memory_transaction_words[agent] == 0)
                & jnp.all(
                    binding.prototype_decision_words[agent]
                    == coordinator.current_decision_id
                )
                & (binding.base_actions[agent] == coordinator.current_action)
                & (binding.memory_actions[agent] == coordinator.current_action)
            )
            matches = matches & jnp.where(pending.available, active, inactive)
        return required, complete, matches

    def _blank_feedback(self) -> ExternalLearnedStateLiveMemoryFeedback:
        return ExternalLearnedStateLiveMemoryFeedback(
            memory_transaction_words=jnp.zeros((2,), dtype=jnp.uint32),
            prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32),
            base_action_before_retrieval=jnp.asarray(-1, dtype=jnp.int32),
            effective_action=jnp.asarray(-1, dtype=jnp.int32),
            hard_action_mask=jnp.zeros((_N_ACTIONS,), dtype=jnp.bool_),
            retrieval_used=jnp.asarray(False, dtype=jnp.bool_),
            counterfactual_available=jnp.asarray(False, dtype=jnp.bool_),
            counterfactual_delta=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _feedback(
        self,
        pending: ExternalLearnedStateLiveMemoryPendingBinding,
        delta: Array,
    ) -> ExternalLearnedStateLiveMemoryFeedback:
        if not bool(pending.available):
            return self._blank_feedback()
        used = pending.retrieval_used_expected
        return ExternalLearnedStateLiveMemoryFeedback(
            memory_transaction_words=pending.memory_transaction_words,
            prototype_decision_id=pending.prototype_decision_id,
            base_action_before_retrieval=pending.base_action_before_retrieval,
            effective_action=pending.effective_action,
            hard_action_mask=pending.hard_action_mask,
            retrieval_used=used,
            counterfactual_available=used,
            counterfactual_delta=jnp.where(used, delta, 0.0).astype(jnp.float32),
        )

    def _transition(
        self,
        child: ExternalLearnedStateLiveMemoryAdapterState,
        proposal: HCCLCausalCoreProposal,
        *,
        agent: int,
        discount: float,
    ) -> ExternalLearnedStateTransition:
        coordinator = child.coordinator_state
        next_observation = proposal.next_observation[agent]
        return ExternalLearnedStateTransition(
            source_event_words=coordinator.event_words,
            source_builder_step_words=coordinator.cached_builder_step_words,
            source_prototype_step_words=coordinator.cached_prototype_step_words,
            source_feature_generation_words=coordinator.cached_feature_generation_words,
            observation=coordinator.current_raw_observation,
            representation=coordinator.current_representation,
            action=coordinator.current_action,
            decision_id=coordinator.current_decision_id,
            reward=proposal.signals.net_reward[agent],
            discount=jnp.asarray(discount, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            next_observation=next_observation,
            next_decision_observation=next_observation,
        )

    def stage(
        self,
        state: HCCLTwoLiveMemoryBridgeState,
        event: HCCLCausalCoreEventReceipt,
        binding: HCCLTwoLiveMemoryActionBinding,
        agent_0_event_input: ExternalLearnedStateLiveMemoryEventInput,
        agent_1_event_input: ExternalLearnedStateLiveMemoryEventInput,
        *,
        next_decision_hard_action_masks: Array,
        downstream_candidate_valid: Array,
    ) -> HCCLTwoLiveMemoryResult:
        """Stage HCCL plus both live adapters and adopt every owner or none."""

        self._require_state_contract(state)
        self._hccl.world._require_event_contract(event)
        self._require_binding_contract(binding)
        next_masks = _require_array(
            next_decision_hard_action_masks,
            shape=(_N_AGENTS, _N_ACTIONS),
            dtype=jnp.dtype(jnp.bool_),
            label="next_decision_hard_action_masks",
        )
        downstream = _require_array(
            downstream_candidate_valid,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            label="downstream_candidate_valid",
        )
        if _contains_tracer(
            (
                state,
                event,
                binding,
                agent_0_event_input,
                agent_1_event_input,
                next_masks,
                downstream,
            )
        ):
            raise TypeError("HCCL two-live-memory composite stage is host/eager-only JIT")
        self._agent_0._validate_event_input_static(agent_0_event_input)
        self._agent_1._validate_event_input_static(agent_1_event_input)
        source_valid = self.state_valid(state)
        event_valid = self._hccl.world.event_receipt_valid(
            state.hccl_state.world_state, event
        )
        binding_integrity = self._binding_integrity_valid(binding)
        expected_binding = self._make_binding(state, event)
        binding_matches = _tree_exact_equal(binding, expected_binding)
        required, bindings_complete, bindings_match_children = (
            self._feedback_binding_relations(state, binding)
        )
        current_masks_bound = (
            jnp.all(binding.current_hard_action_masks == state.current_hard_action_masks)
            & jnp.all(
                binding.base.hard_action_masks == state.current_hard_action_masks
            )
            & jnp.all(
                binding.memory.hard_action_masks == state.current_hard_action_masks
            )
            & jnp.all(
                binding.planner.hard_action_masks == state.current_hard_action_masks
            )
        )
        hccl_result = self._hccl.stage(
            state.hccl_state,
            event,
            binding.base,
            binding.memory,
            binding.planner,
            downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
        )
        proposals = hccl_result.world_proposals
        unilateral = jnp.stack(
            (
                proposals.signals.net_reward[_M0B1_SLOT, 0]
                - proposals.signals.net_reward[_BB_SLOT, 0],
                proposals.signals.net_reward[_B0M1_SLOT, 1]
                - proposals.signals.net_reward[_BB_SLOT, 1],
            )
        ).astype(jnp.float32)
        feedback_0 = self._feedback(state.agent_0_state.pending_binding, unilateral[0])
        feedback_1 = self._feedback(state.agent_1_state.pending_binding, unilateral[1])
        pp = cast(
            HCCLCausalCoreProposal,
            jax.tree.map(lambda leaf: leaf[_PP_SLOT], proposals),
        )
        transition_0 = self._transition(
            state.agent_0_state,
            pp,
            agent=0,
            discount=self._config.agent_0.coordinator.inner.ensemble.world_model.gamma,
        )
        transition_1 = self._transition(
            state.agent_1_state,
            pp,
            agent=1,
            discount=self._config.agent_1.coordinator.inner.ensemble.world_model.gamma,
        )
        result_0 = self._agent_0.step(
            state.agent_0_state,
            transition_0,
            agent_0_event_input,
            next_masks[0],
            feedback_0 if bool(required[0]) else None,
        )
        result_1 = self._agent_1.step(
            state.agent_1_state,
            transition_1,
            agent_1_event_input,
            next_masks[1],
            feedback_1 if bool(required[1]) else None,
        )
        planner_equals_memory = (
            jnp.all(binding.planner.actions_before_mask == binding.memory_actions)
            & jnp.all(binding.planner.actions_after_mask == binding.memory_actions)
        )
        pp_executes_memory = jnp.all(pp.joint_action_ids == binding.memory_actions)
        planner_zero = _contrast_exact_zero(
            hccl_result.attribution.contrasts.planner_total
        ) & _contrast_exact_zero(hccl_result.attribution.contrasts.planner_interaction)
        no_planner = planner_equals_memory & pp_executes_memory & planner_zero
        candidate = HCCLTwoLiveMemoryBridgeState(
            hccl_state=hccl_result.state,
            agent_0_state=result_0.state,
            agent_1_state=result_1.state,
            current_hard_action_masks=next_masks,
        )
        candidate_valid = self.state_valid(candidate)
        applied = (
            source_valid
            & event_valid
            & binding_integrity
            & binding_matches
            & bindings_complete
            & bindings_match_children
            & current_masks_bound
            & hccl_result.update_applied
            & result_0.diagnostics.transaction_applied
            & result_1.diagnostics.transaction_applied
            & no_planner
            & downstream
            & candidate_valid
        )
        final_state = cast(
            HCCLTwoLiveMemoryBridgeState,
            _tree_select(applied, candidate, state),
        )
        return HCCLTwoLiveMemoryResult(
            state=final_state,
            binding=binding,
            hccl_result=hccl_result,
            agent_0_transition=transition_0,
            agent_1_transition=transition_1,
            agent_0_feedback=feedback_0,
            agent_1_feedback=feedback_1,
            agent_0_result=result_0,
            agent_1_result=result_1,
            agent_unilateral_counterfactual_delta=unilateral,
            memory_interaction_audit=hccl_result.attribution.contrasts.memory_interaction,
            work=HCCLTwoLiveMemoryWork(
                world_proposal_calls=hccl_result.work.world_proposal_calls,
                attribution_proposal_calls=hccl_result.work.attribution_proposal_calls,
                live_adapter_step_calls=jnp.ones((2,), dtype=jnp.int32),
                committed_composite_transactions=applied.astype(jnp.int32),
            ),
            source_state_valid=source_valid,
            event_receipt_valid=event_valid,
            binding_integrity_valid=binding_integrity,
            binding_matches_source=binding_matches,
            feedback_bindings_complete=bindings_complete,
            feedback_bindings_match_children=bindings_match_children,
            prior_feedback_required=required,
            prior_feedback_supplied=required,
            current_event_masks_bound=current_masks_bound,
            planner_equals_memory=planner_equals_memory,
            pp_executes_memory_actions=pp_executes_memory,
            no_planner_rung_valid=no_planner,
            agent_0_feedback_is_m0b1_minus_bb=jnp.asarray(True, dtype=jnp.bool_),
            agent_1_feedback_is_b0m1_minus_bb=jnp.asarray(True, dtype=jnp.bool_),
            mm_minus_bb_broadcast_to_both_agents=jnp.asarray(False, dtype=jnp.bool_),
            memory_interaction_used_for_agent_feedback=jnp.asarray(
                False, dtype=jnp.bool_
            ),
            delight_or_actor_backward=jnp.asarray(False, dtype=jnp.bool_),
            downstream_candidate_valid=downstream,
            candidate_state_valid=candidate_valid,
            hccl_commit_applied=hccl_result.update_applied & applied,
            agent_0_update_applied=(
                result_0.diagnostics.transaction_applied & applied
            ),
            agent_1_update_applied=(
                result_1.diagnostics.transaction_applied & applied
            ),
            next_decision_masks_installed=applied,
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: HCCLTwoLiveMemoryBridgeState | None = None,
    ) -> HCCLTwoLiveMemoryResourceBudget:
        reference = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(reference)
        if not bool(self.state_valid(reference)):
            raise ValueError("resource measurement requires a valid composite state")
        hccl_bytes = measure_hccl_world_attribution_state_nbytes(reference.hccl_state)
        agent_0_bytes = measure_external_learned_state_live_memory_adapter_state_nbytes(
            reference.agent_0_state
        )
        agent_1_bytes = measure_external_learned_state_live_memory_adapter_state_nbytes(
            reference.agent_1_state
        )
        mask_bytes = _tree_nbytes(reference.current_hard_action_masks)
        hccl_budget = self._hccl.resource_budget(reference.hccl_state)
        return HCCLTwoLiveMemoryResourceBudget(
            schema=HCCL_TWO_LIVE_MEMORY_RESOURCE_SCHEMA,
            hccl_state_owners=1,
            live_memory_adapter_state_owners=2,
            external_coordinator_state_owners=2,
            learned_memory_controller_state_owners=2,
            prototype_state_owners=2,
            current_hard_action_mask_bindings=1,
            hccl_state_nbytes=hccl_bytes,
            agent_0_state_nbytes=agent_0_bytes,
            agent_1_state_nbytes=agent_1_bytes,
            current_hard_action_mask_nbytes=mask_bytes,
            total_persistent_state_nbytes=(
                hccl_bytes + agent_0_bytes + agent_1_bytes + mask_bytes
            ),
            max_world_proposal_calls_per_transaction=(
                hccl_budget.max_world_proposal_calls_per_transaction
            ),
            max_attribution_proposal_calls_per_transaction=(
                hccl_budget.max_attribution_proposal_calls_per_transaction
            ),
            live_adapter_step_calls_per_transaction=2,
            maximum_feedback_settlements_per_transaction=2,
            maximum_cached_action_replacements_per_transaction=2,
            planner_calls_per_transaction=0,
            delight_or_actor_backward_calls_per_transaction=0,
            maximum_composite_transactions=min(
                hccl_budget.maximum_committed_transactions,
                self._config.agent_0.coordinator.max_events,
                self._config.agent_1.coordinator.max_events,
            ),
            composite_jit_supported=False,
            scan_supported=False,
            output_write_calls=0,
            artifact_bytes_written=0,
        )


def measure_hccl_two_live_memory_state_nbytes(
    state: HCCLTwoLiveMemoryBridgeState,
) -> int:
    if type(state) is not HCCLTwoLiveMemoryBridgeState:
        raise TypeError("state must be exact HCCLTwoLiveMemoryBridgeState")
    return _tree_nbytes(state)


def _checkpoint_digest(checkpoint: HCCLTwoLiveMemoryCheckpoint) -> str:
    return _canonical_digest(
        {
            "schema": checkpoint.schema,
            "mechanism_status": checkpoint.mechanism_status,
            "evidence_level": checkpoint.evidence_level,
            "output_writes_authorized": checkpoint.output_writes_authorized,
            "artifact_authorized": checkpoint.artifact_authorized,
            "evidence_authorized": checkpoint.evidence_authorized,
            "config": checkpoint.config,
            "config_sha256": checkpoint.config_sha256,
            "resource_budget": checkpoint.resource_budget,
            "state_nbytes": checkpoint.state_nbytes,
            "state_sha256": checkpoint.state_sha256,
        }
    )


def save_hccl_two_live_memory_checkpoint(
    bridge: HCCLTwoLiveMemoryBridge,
    state: HCCLTwoLiveMemoryBridgeState,
) -> HCCLTwoLiveMemoryCheckpoint:
    """Return a strict in-memory checkpoint and perform no output write."""

    if type(bridge) is not HCCLTwoLiveMemoryBridge:
        raise TypeError("bridge must be exact HCCLTwoLiveMemoryBridge")
    bridge._require_state_contract(state)
    if not bool(bridge.state_valid(state)):
        raise ValueError("cannot checkpoint an invalid two-live-memory state")
    copied = cast(HCCLTwoLiveMemoryBridgeState, jax.tree.map(jnp.array, state))
    config = bridge.to_config()
    bare = HCCLTwoLiveMemoryCheckpoint(
        schema=HCCL_TWO_LIVE_MEMORY_CHECKPOINT_SCHEMA,
        mechanism_status=HCCL_TWO_LIVE_MEMORY_STATUS,
        evidence_level=HCCL_TWO_LIVE_MEMORY_EVIDENCE_LEVEL,
        output_writes_authorized=False,
        artifact_authorized=False,
        evidence_authorized=False,
        config=config,
        config_sha256=_canonical_digest(config),
        resource_budget=bridge.resource_budget(copied).to_config(),
        state=copied,
        state_nbytes=measure_hccl_two_live_memory_state_nbytes(copied),
        state_sha256=_canonical_digest(_state_host_payload(copied)),
        checkpoint_sha256="",
    )
    return dataclasses.replace(bare, checkpoint_sha256=_checkpoint_digest(bare))


def load_hccl_two_live_memory_checkpoint(
    checkpoint: HCCLTwoLiveMemoryCheckpoint,
) -> tuple[HCCLTwoLiveMemoryBridge, HCCLTwoLiveMemoryBridgeState]:
    """Restore only the canonical in-memory two-live-memory checkpoint."""

    if type(checkpoint) is not HCCLTwoLiveMemoryCheckpoint:
        raise TypeError("checkpoint must be exact HCCLTwoLiveMemoryCheckpoint")
    fixed = {
        "schema": HCCL_TWO_LIVE_MEMORY_CHECKPOINT_SCHEMA,
        "mechanism_status": HCCL_TWO_LIVE_MEMORY_STATUS,
        "evidence_level": HCCL_TWO_LIVE_MEMORY_EVIDENCE_LEVEL,
        "output_writes_authorized": False,
        "artifact_authorized": False,
        "evidence_authorized": False,
    }
    for name, expected in fixed.items():
        actual = getattr(checkpoint, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"checkpoint {name} differs")
    if type(checkpoint.config) is not dict:
        raise ValueError("checkpoint config must be an exact dict")
    if type(checkpoint.config_sha256) is not str:
        raise ValueError("checkpoint config digest must be an exact string")
    if checkpoint.config_sha256 != _canonical_digest(checkpoint.config):
        raise ValueError("checkpoint config digest differs")
    bridge = HCCLTwoLiveMemoryBridge.from_config(checkpoint.config)
    bridge._require_state_contract(checkpoint.state)
    if type(checkpoint.state_nbytes) is not int or checkpoint.state_nbytes != (
        measure_hccl_two_live_memory_state_nbytes(checkpoint.state)
    ):
        raise ValueError("checkpoint state bytes differ")
    if type(checkpoint.state_sha256) is not str:
        raise ValueError("checkpoint state digest must be an exact string")
    if checkpoint.state_sha256 != _canonical_digest(_state_host_payload(checkpoint.state)):
        raise ValueError("checkpoint state digest differs")
    if type(checkpoint.resource_budget) is not dict or not _canonical_python_equal(
        checkpoint.resource_budget,
        bridge.resource_budget(checkpoint.state).to_config(),
    ):
        raise ValueError("checkpoint resource budget differs")
    if type(checkpoint.checkpoint_sha256) is not str:
        raise ValueError("checkpoint digest must be an exact string")
    if checkpoint.checkpoint_sha256 != _checkpoint_digest(checkpoint):
        raise ValueError("checkpoint digest differs")
    if not bool(bridge.state_valid(checkpoint.state)):
        raise ValueError("checkpoint state is invalid")
    restored = cast(
        HCCLTwoLiveMemoryBridgeState,
        jax.tree.map(jnp.array, checkpoint.state),
    )
    return bridge, restored


__all__ = [
    "HCCL_TWO_LIVE_MEMORY_BINDING_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_CHECKPOINT_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_CONFIG_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_EVIDENCE_LEVEL",
    "HCCL_TWO_LIVE_MEMORY_LIMITATIONS",
    "HCCL_TWO_LIVE_MEMORY_RESOURCE_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_STATE_SCHEMA",
    "HCCL_TWO_LIVE_MEMORY_STATUS",
    "HCCLTwoLiveMemoryActionBinding",
    "HCCLTwoLiveMemoryBridge",
    "HCCLTwoLiveMemoryBridgeConfig",
    "HCCLTwoLiveMemoryBridgeState",
    "HCCLTwoLiveMemoryCheckpoint",
    "HCCLTwoLiveMemoryResourceBudget",
    "HCCLTwoLiveMemoryResult",
    "HCCLTwoLiveMemoryWork",
    "load_hccl_two_live_memory_checkpoint",
    "measure_hccl_two_live_memory_state_nbytes",
    "save_hccl_two_live_memory_checkpoint",
]
