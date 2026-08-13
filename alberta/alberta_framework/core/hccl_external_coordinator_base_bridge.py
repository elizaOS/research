# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Atomic base-only HCCL transactions for two external coordinators.

This L0 bridge owns exactly one :class:`HCCLWorldAttributionAdapterState` and
two independently initialized
:class:`ExternalLearnedStateRouterAuditCoordinatorState` values.  Both
coordinators start from their own raw 16-channel HCCL observation.  For each
event, their exact cached primitive actions are bound as equal B, M, and P
receipts under one caller-declared hard-mask matrix.  Each receipt identity is
deterministically derived from the corresponding coordinator's full decision
identity, lifecycle identity, owner clocks, and the exact HCCL event.

The bridge evaluates the eight HCCL proposals, takes the PP proposal as the
base-only realized event, and attempts exactly one coordinator transition per
agent using that agent's cached action, net reward, and next raw observation.
The HCCL state and both coordinator states are adopted only when all children
and the outer downstream gate succeed.  Any refusal returns all three source
owners bit-exactly.  A hard mask that excludes either cached action is rejected;
the bridge has no authority to invent a fallback.

B=M=P equality and exact zero adjacent-layer contrasts are facts about this
ablation only.  They are not delight or "no joy" judgments and do not execute
an actor backward pass.  The composite is host/eager-only, binds integrity but
does not authenticate callers, and creates no memory/planner, schedule, seed,
output, artifact, threshold, evidence, benefit, or promotion authority.
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
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateRouterAuditCoordinator,
    ExternalLearnedStateRouterAuditCoordinatorConfig,
    ExternalLearnedStateRouterAuditCoordinatorResult,
    ExternalLearnedStateRouterAuditCoordinatorState,
    ExternalLearnedStateTransition,
    measure_external_learned_state_router_audit_coordinator_state_nbytes,
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

HCCL_EXTERNAL_COORDINATOR_BASE_CONFIG_SCHEMA = (
    "alberta.hccl-external-coordinator-base-bridge-config.v1"
)
HCCL_EXTERNAL_COORDINATOR_BASE_STATE_SCHEMA = (
    "alberta.hccl-external-coordinator-base-bridge-state.v1"
)
HCCL_EXTERNAL_COORDINATOR_BASE_BINDING_SCHEMA = (
    "alberta.hccl-external-coordinator-base-action-binding.v1"
)
HCCL_EXTERNAL_COORDINATOR_BASE_CHECKPOINT_SCHEMA = (
    "alberta.hccl-external-coordinator-base-checkpoint.v1"
)
HCCL_EXTERNAL_COORDINATOR_BASE_RESOURCE_SCHEMA = (
    "alberta.hccl-external-coordinator-base-resource.v1"
)
HCCL_EXTERNAL_COORDINATOR_BASE_STATUS = (
    "l0-development-hccl-two-coordinator-base-only"
)
HCCL_EXTERNAL_COORDINATOR_BASE_EVIDENCE_LEVEL = "L0"
HCCL_EXTERNAL_COORDINATOR_BASE_LIMITATIONS = (
    "base-only-B-equals-M-equals-P-ablation",
    "zero-memory-planner-contrasts-are-ablation-facts-only",
    "delight-or-actor-backward-is-false-not-a-no-joy-inference",
    "cached-actions-must-pass-the-common-hard-masks-without-fallback",
    "receipt-identities-bind-exact-decision-and-lifecycle-identities",
    "integrity-binding-is-not-caller-authentication",
    "composite-stage-is-host-eager-only",
    "no-memory-planner-schedule-seed-output-artifact-threshold-evidence-or-promotion",
)

_N_AGENTS = 2
_N_ACTIONS = 2
_RAW_OBSERVATION_DIM = 16
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
    state: HCCLExternalCoordinatorBaseBridgeState,
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


def _validate_coordinator_config(
    config: ExternalLearnedStateRouterAuditCoordinatorConfig,
    *,
    label: str,
) -> None:
    if type(config) is not ExternalLearnedStateRouterAuditCoordinatorConfig:
        raise TypeError(f"{label} must be an exact external coordinator config")
    if config.builder.observation_dim != _RAW_OBSERVATION_DIM:
        raise ValueError(f"{label} raw observation width must be 16")
    if config.builder.n_actions != _N_ACTIONS:
        raise ValueError(f"{label} must expose exactly two primitive actions")
    prototype = config.inner.prototype
    if prototype.experiential_memory is not None:
        raise ValueError(f"{label} experiential memory is forbidden on the base-only rung")
    if prototype.option_search_control is not None:
        raise ValueError(f"{label} option search would add planner authority")
    if prototype.oak.stomp.option_planning_backups_per_step != 0:
        raise ValueError(f"{label} STOMP planner backups must remain zero")
    if config.inner.ensemble.world_model.n_actions != _N_ACTIONS:
        raise ValueError(f"{label} ensemble must model exactly two primitive actions")


@dataclasses.dataclass(frozen=True)
class HCCLExternalCoordinatorBaseBridgeConfig:
    hccl: HCCLWorldAttributionAdapterConfig
    agent_0: ExternalLearnedStateRouterAuditCoordinatorConfig
    agent_1: ExternalLearnedStateRouterAuditCoordinatorConfig
    binding_owner_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.hccl) is not HCCLWorldAttributionAdapterConfig:
            raise TypeError("hccl must be exact HCCLWorldAttributionAdapterConfig")
        _validate_coordinator_config(self.agent_0, label="agent_0")
        _validate_coordinator_config(self.agent_1, label="agent_1")
        value = self.binding_owner_digest
        if type(value) is not tuple or len(value) != 8:
            raise ValueError("binding_owner_digest must be an exact eight-word tuple")
        for index, word in enumerate(value):
            if type(word) is not int or not 0 <= word <= _UINT32_MAX:
                raise ValueError(f"binding_owner_digest[{index}] must be uint32")
        if not any(value[:4]) or not any(value[4:]) or value[:4] == value[4:]:
            raise ValueError("binding owner must contain two distinct nonzero identities")


@chex.dataclass(frozen=True)
class HCCLExternalCoordinatorBaseBridgeState:
    hccl_state: HCCLWorldAttributionAdapterState
    agent_0_state: ExternalLearnedStateRouterAuditCoordinatorState
    agent_1_state: ExternalLearnedStateRouterAuditCoordinatorState


@chex.dataclass(frozen=True)
class HCCLExternalCoordinatorBaseActionBinding:
    source_world_words: UInt[Array, " 2"]
    source_world_tag_words: UInt[Array, " 4"]
    event_content_tag_words: UInt[Array, " 4"]
    coordinator_event_words: UInt[Array, "2 2"]
    coordinator_builder_words: UInt[Array, "2 2"]
    coordinator_prototype_words: UInt[Array, "2 2"]
    coordinator_feature_generation_words: UInt[Array, "2 2"]
    coordinator_lifecycle_words: UInt[Array, "2 2"]
    coordinator_decision_words: UInt[Array, "2 4"]
    cached_actions: Int[Array, " 2"]
    hard_action_masks: Bool[Array, "2 2"]
    base: HCCLActionReceipt
    memory: HCCLActionReceipt
    planner: HCCLActionReceipt
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLExternalCoordinatorBaseWork:
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    coordinator_update_calls: Int[Array, " 2"]
    committed_composite_transactions: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLExternalCoordinatorBaseResult:
    state: HCCLExternalCoordinatorBaseBridgeState
    binding: HCCLExternalCoordinatorBaseActionBinding
    hccl_result: HCCLWorldAttributionAdapterResult
    agent_0_transition: ExternalLearnedStateTransition
    agent_1_transition: ExternalLearnedStateTransition
    agent_0_result: ExternalLearnedStateRouterAuditCoordinatorResult
    agent_1_result: ExternalLearnedStateRouterAuditCoordinatorResult
    work: HCCLExternalCoordinatorBaseWork
    source_state_valid: Bool[Array, ""]
    event_receipt_valid: Bool[Array, ""]
    binding_integrity_valid: Bool[Array, ""]
    binding_matches_source: Bool[Array, ""]
    cached_actions_admitted: Bool[Array, ""]
    action_receipt_identities_distinct: Bool[Array, ""]
    memory_contrasts_zero: Bool[Array, ""]
    planner_contrasts_zero: Bool[Array, ""]
    total_stack_contrast_zero: Bool[Array, ""]
    base_only_ablation_valid: Bool[Array, ""]
    delight_or_actor_backward: Bool[Array, ""]
    downstream_candidate_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    hccl_commit_applied: Bool[Array, ""]
    agent_0_update_applied: Bool[Array, ""]
    agent_1_update_applied: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class HCCLExternalCoordinatorBaseResourceBudget:
    schema: str
    hccl_state_owners: int
    external_coordinator_state_owners: int
    hccl_state_nbytes: int
    agent_0_state_nbytes: int
    agent_1_state_nbytes: int
    total_persistent_state_nbytes: int
    max_world_proposal_calls_per_transaction: int
    max_attribution_proposal_calls_per_transaction: int
    coordinator_update_calls_per_transaction: int
    maximum_composite_transactions: int
    composite_jit_supported: bool
    memory_layer_authority: int
    planner_layer_authority: int
    output_write_calls: int
    artifact_bytes_written: int

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class HCCLExternalCoordinatorBaseCheckpoint:
    schema: str
    mechanism_status: str
    evidence_level: str
    output_writes_authorized: bool
    artifact_authorized: bool
    evidence_authorized: bool
    config: dict[str, object]
    config_sha256: str
    resource_budget: dict[str, object]
    state: HCCLExternalCoordinatorBaseBridgeState
    state_nbytes: int
    state_sha256: str
    checkpoint_sha256: str


class HCCLExternalCoordinatorBaseBridge:
    """Host-only all-or-none owner over one HCCL and two coordinator states."""

    def __init__(self, config: HCCLExternalCoordinatorBaseBridgeConfig):
        if type(config) is not HCCLExternalCoordinatorBaseBridgeConfig:
            raise TypeError("config must be exact HCCLExternalCoordinatorBaseBridgeConfig")
        self._config = config
        self._hccl = HCCLWorldAttributionAdapter(config.hccl)
        self._agent_0 = ExternalLearnedStateRouterAuditCoordinator(config.agent_0)
        self._agent_1 = ExternalLearnedStateRouterAuditCoordinator(config.agent_1)
        self._owner = jnp.asarray(config.binding_owner_digest, dtype=jnp.uint32)

    @property
    def config(self) -> HCCLExternalCoordinatorBaseBridgeConfig:
        return self._config

    @property
    def hccl(self) -> HCCLWorldAttributionAdapter:
        return self._hccl

    @property
    def agent_0(self) -> ExternalLearnedStateRouterAuditCoordinator:
        return self._agent_0

    @property
    def agent_1(self) -> ExternalLearnedStateRouterAuditCoordinator:
        return self._agent_1

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": HCCL_EXTERNAL_COORDINATOR_BASE_CONFIG_SCHEMA,
            "state_schema": HCCL_EXTERNAL_COORDINATOR_BASE_STATE_SCHEMA,
            "binding_schema": HCCL_EXTERNAL_COORDINATOR_BASE_BINDING_SCHEMA,
            "checkpoint_schema": HCCL_EXTERNAL_COORDINATOR_BASE_CHECKPOINT_SCHEMA,
            "resource_schema": HCCL_EXTERNAL_COORDINATOR_BASE_RESOURCE_SCHEMA,
            "mechanism_status": HCCL_EXTERNAL_COORDINATOR_BASE_STATUS,
            "evidence_level": HCCL_EXTERNAL_COORDINATOR_BASE_EVIDENCE_LEVEL,
            "hccl": self._hccl.to_config(),
            "agent_0": self._agent_0.to_config(),
            "agent_1": self._agent_1.to_config(),
            "binding_owner_digest": list(self._config.binding_owner_digest),
            "hccl_state_owners": 1,
            "external_coordinator_state_owners": 2,
            "base_only_ablation": True,
            "base_memory_planner_action_relation": "B=M=P",
            "hard_mask_semantics": "common-across-layers-reject-excluded-cached-action",
            "receipt_identity_semantics": (
                "deterministic-exact-coordinator-decision-lifecycle-and-clock-binding"
            ),
            "zero_contrast_scope": "this-base-only-ablation-only",
            "delight_or_actor_backward": False,
            "memory_layer_authority": False,
            "planner_layer_authority": False,
            "composite_jit_supported": False,
            "composite_execution": "host-eager-only",
            "caller_identity_authenticated": False,
            "schedule_execution_authorized": False,
            "seed_authority": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "threshold_authorized": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_EXTERNAL_COORDINATOR_BASE_LIMITATIONS),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLExternalCoordinatorBaseBridge:
        if type(payload) is not dict:
            raise TypeError("config payload must be an exact dict")
        hccl_raw = payload.get("hccl")
        agent_0_raw = payload.get("agent_0")
        agent_1_raw = payload.get("agent_1")
        owner_raw = payload.get("binding_owner_digest")
        if type(hccl_raw) is not dict:
            raise TypeError("hccl config must be an exact dict")
        if type(agent_0_raw) is not dict or type(agent_1_raw) is not dict:
            raise TypeError("coordinator configs must be exact dicts")
        if type(owner_raw) is not list:
            raise TypeError("binding_owner_digest must serialize as a list")
        candidate = cls(
            HCCLExternalCoordinatorBaseBridgeConfig(
                hccl=HCCLWorldAttributionAdapter.from_config(hccl_raw).config,
                agent_0=ExternalLearnedStateRouterAuditCoordinatorConfig.from_config(
                    agent_0_raw
                ),
                agent_1=ExternalLearnedStateRouterAuditCoordinatorConfig.from_config(
                    agent_1_raw
                ),
                binding_owner_digest=tuple(owner_raw),
            )
        )
        if _canonical_digest(candidate.to_config()) != _canonical_digest(dict(payload)):
            raise ValueError("HCCL external-coordinator base config is unsupported")
        return candidate

    def _require_state_contract(self, state: HCCLExternalCoordinatorBaseBridgeState) -> None:
        if type(state) is not HCCLExternalCoordinatorBaseBridgeState:
            raise TypeError("state must be exact HCCLExternalCoordinatorBaseBridgeState")
        self._hccl._require_state_contract(state.hccl_state)
        if type(state.agent_0_state) is not ExternalLearnedStateRouterAuditCoordinatorState:
            raise TypeError("agent_0_state must be an exact coordinator state")
        if type(state.agent_1_state) is not ExternalLearnedStateRouterAuditCoordinatorState:
            raise TypeError("agent_1_state must be an exact coordinator state")

    def state_valid(self, state: HCCLExternalCoordinatorBaseBridgeState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        observations = self._hccl.world.observe(state.hccl_state.world_state)
        return (
            self._hccl.state_valid(state.hccl_state)
            & self._agent_0.state_valid(state.agent_0_state)
            & self._agent_1.state_valid(state.agent_1_state)
            & state.agent_0_state.started
            & state.agent_1_state.started
            & jnp.all(
                state.agent_0_state.event_words == state.hccl_state.world_state.step_words
            )
            & jnp.all(
                state.agent_1_state.event_words == state.hccl_state.world_state.step_words
            )
            & _float_bits_equal(state.agent_0_state.current_raw_observation, observations[0])
            & _float_bits_equal(state.agent_1_state.current_raw_observation, observations[1])
            & (state.agent_0_state.current_action >= 0)
            & (state.agent_0_state.current_action < _N_ACTIONS)
            & (state.agent_1_state.current_action >= 0)
            & (state.agent_1_state.current_action < _N_ACTIONS)
        )

    def init(self, key: Array) -> HCCLExternalCoordinatorBaseBridgeState:
        if not (
            hasattr(key, "shape")
            and hasattr(key, "dtype")
            and key.shape == ()
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        ):
            raise TypeError("key must be a scalar typed JAX PRNG key")
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
        state = HCCLExternalCoordinatorBaseBridgeState(
            hccl_state=hccl_state,
            agent_0_state=agent_0,
            agent_1_state=agent_1,
        )
        if not bool(self.state_valid(state)):
            raise RuntimeError("initial HCCL external-coordinator base state is invalid")
        return state

    def prepare_event(
        self,
        state: HCCLExternalCoordinatorBaseBridgeState,
    ) -> HCCLCausalCoreEventReceipt:
        self._require_state_contract(state)
        if not bool(self.state_valid(state)):
            raise ValueError("cannot prepare an event from an invalid composite state")
        return self._hccl.world.prepare_event(state.hccl_state.world_state)

    def _coordinator_states(
        self,
        state: HCCLExternalCoordinatorBaseBridgeState,
    ) -> tuple[
        ExternalLearnedStateRouterAuditCoordinatorState,
        ExternalLearnedStateRouterAuditCoordinatorState,
    ]:
        return state.agent_0_state, state.agent_1_state

    def _receipt_identity_rows(
        self,
        state: HCCLExternalCoordinatorBaseBridgeState,
        event: HCCLCausalCoreEventReceipt,
        layer: HCCLActionLayer,
    ) -> Array:
        rows: list[Array] = []
        for agent, coordinator in enumerate(self._coordinator_states(state)):
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
                    coordinator.current_decision_id[:2],
                    coordinator.current_decision_id,
                    coordinator.current_action,
                )
            )
        return jnp.stack(tuple(rows)).astype(jnp.uint32)

    def _binding_tag(self, binding: HCCLExternalCoordinatorBaseActionBinding) -> Array:
        return _content_tag(
            self._owner,
            binding.source_world_words,
            binding.source_world_tag_words,
            binding.event_content_tag_words,
            binding.coordinator_event_words,
            binding.coordinator_builder_words,
            binding.coordinator_prototype_words,
            binding.coordinator_feature_generation_words,
            binding.coordinator_lifecycle_words,
            binding.coordinator_decision_words,
            binding.cached_actions,
            binding.hard_action_masks,
            binding.base.action_receipt_identity_words,
            binding.memory.action_receipt_identity_words,
            binding.planner.action_receipt_identity_words,
            binding.base.content_tag_words,
            binding.memory.content_tag_words,
            binding.planner.content_tag_words,
        )

    def _make_binding(
        self,
        state: HCCLExternalCoordinatorBaseBridgeState,
        event: HCCLCausalCoreEventReceipt,
        masks: Array,
    ) -> HCCLExternalCoordinatorBaseActionBinding:
        coordinators = self._coordinator_states(state)
        actions = jnp.stack(tuple(item.current_action for item in coordinators)).astype(
            jnp.int32
        )
        receipts: list[HCCLActionReceipt] = []
        for layer in (
            HCCLActionLayer.BASE,
            HCCLActionLayer.MEMORY,
            HCCLActionLayer.PLANNER,
        ):
            receipts.append(
                self._hccl.bind_action_receipt(
                    state.hccl_state,
                    event,
                    layer=layer,
                    actions_before_mask=actions,
                    actions_after_mask=actions,
                    hard_action_masks=masks,
                    action_receipt_identity_words=self._receipt_identity_rows(
                        state, event, layer
                    ),
                )
            )
        bare = HCCLExternalCoordinatorBaseActionBinding(
            source_world_words=state.hccl_state.world_state.step_words,
            source_world_tag_words=state.hccl_state.world_state.content_tag_words,
            event_content_tag_words=event.content_tag_words,
            coordinator_event_words=jnp.stack(
                tuple(item.event_words for item in coordinators)
            ),
            coordinator_builder_words=jnp.stack(
                tuple(item.cached_builder_step_words for item in coordinators)
            ),
            coordinator_prototype_words=jnp.stack(
                tuple(item.cached_prototype_step_words for item in coordinators)
            ),
            coordinator_feature_generation_words=jnp.stack(
                tuple(item.cached_feature_generation_words for item in coordinators)
            ),
            coordinator_lifecycle_words=jnp.stack(
                tuple(item.current_decision_id[:2] for item in coordinators)
            ),
            coordinator_decision_words=jnp.stack(
                tuple(item.current_decision_id for item in coordinators)
            ),
            cached_actions=actions,
            hard_action_masks=masks,
            base=receipts[0],
            memory=receipts[1],
            planner=receipts[2],
            content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
        )
        return cast(
            HCCLExternalCoordinatorBaseActionBinding,
            cast(Any, bare).replace(content_tag_words=self._binding_tag(bare)),
        )

    def _identities_distinct(
        self,
        binding: HCCLExternalCoordinatorBaseActionBinding,
    ) -> Bool[Array, ""]:
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

    def _cached_actions_admitted(
        self,
        binding: HCCLExternalCoordinatorBaseActionBinding,
    ) -> Bool[Array, ""]:
        safe = jnp.clip(binding.cached_actions, 0, _N_ACTIONS - 1)
        return (
            jnp.all((binding.cached_actions >= 0) & (binding.cached_actions < _N_ACTIONS))
            & binding.hard_action_masks[0, safe[0]]
            & binding.hard_action_masks[1, safe[1]]
        )

    def bind_base_actions(
        self,
        state: HCCLExternalCoordinatorBaseBridgeState,
        event: HCCLCausalCoreEventReceipt,
        *,
        hard_action_masks: Array | None = None,
    ) -> HCCLExternalCoordinatorBaseActionBinding:
        """Bind B=M=P to the two exact cached primitives without fallback."""

        self._require_state_contract(state)
        self._hccl.world._require_event_contract(event)
        masks = (
            jnp.ones((_N_AGENTS, _N_ACTIONS), dtype=jnp.bool_)
            if hard_action_masks is None
            else _require_array(
                hard_action_masks,
                shape=(_N_AGENTS, _N_ACTIONS),
                dtype=jnp.dtype(jnp.bool_),
                label="hard_action_masks",
            )
        )
        if not bool(self.state_valid(state)):
            raise ValueError("cannot bind actions from an invalid composite state")
        if not bool(self._hccl.world.event_receipt_valid(state.hccl_state.world_state, event)):
            raise ValueError("cannot bind actions to a stale or invalid HCCL event")
        cached_actions = jnp.stack(
            (
                state.agent_0_state.current_action,
                state.agent_1_state.current_action,
            )
        ).astype(jnp.int32)
        if not bool(
            masks[0, cached_actions[0]] & masks[1, cached_actions[1]]
        ):
            raise ValueError("hard_action_masks excludes a cached coordinator action")
        binding = self._make_binding(state, event, masks)
        if not bool(self._identities_distinct(binding)):
            raise RuntimeError("deterministic action receipt identities collided")
        return binding

    def _require_binding_contract(
        self,
        binding: HCCLExternalCoordinatorBaseActionBinding,
    ) -> None:
        if type(binding) is not HCCLExternalCoordinatorBaseActionBinding:
            raise TypeError("binding must be exact HCCLExternalCoordinatorBaseActionBinding")
        for name, shape, dtype in (
            ("source_world_words", (2,), jnp.uint32),
            ("source_world_tag_words", (4,), jnp.uint32),
            ("event_content_tag_words", (4,), jnp.uint32),
            ("coordinator_event_words", (2, 2), jnp.uint32),
            ("coordinator_builder_words", (2, 2), jnp.uint32),
            ("coordinator_prototype_words", (2, 2), jnp.uint32),
            ("coordinator_feature_generation_words", (2, 2), jnp.uint32),
            ("coordinator_lifecycle_words", (2, 2), jnp.uint32),
            ("coordinator_decision_words", (2, 4), jnp.uint32),
            ("cached_actions", (2,), jnp.int32),
            ("hard_action_masks", (2, 2), jnp.bool_),
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

    def _binding_integrity_valid(
        self,
        binding: HCCLExternalCoordinatorBaseActionBinding,
    ) -> Bool[Array, ""]:
        actions_equal = (
            jnp.all(binding.base.actions_before_mask == binding.cached_actions)
            & jnp.all(binding.base.actions_after_mask == binding.cached_actions)
            & jnp.all(binding.memory.actions_before_mask == binding.cached_actions)
            & jnp.all(binding.memory.actions_after_mask == binding.cached_actions)
            & jnp.all(binding.planner.actions_before_mask == binding.cached_actions)
            & jnp.all(binding.planner.actions_after_mask == binding.cached_actions)
        )
        masks_equal = (
            jnp.all(binding.base.hard_action_masks == binding.hard_action_masks)
            & jnp.all(binding.memory.hard_action_masks == binding.hard_action_masks)
            & jnp.all(binding.planner.hard_action_masks == binding.hard_action_masks)
        )
        return (
            actions_equal
            & masks_equal
            & self._cached_actions_admitted(binding)
            & self._identities_distinct(binding)
            & jnp.all(binding.content_tag_words == self._binding_tag(binding))
        )

    def _transition(
        self,
        coordinator: ExternalLearnedStateRouterAuditCoordinatorState,
        proposal: HCCLCausalCoreProposal,
        *,
        agent: int,
        discount: float,
    ) -> ExternalLearnedStateTransition:
        next_observation = proposal.next_observation[agent]
        return ExternalLearnedStateTransition(
            source_event_words=coordinator.event_words,
            source_builder_step_words=coordinator.cached_builder_step_words,
            source_prototype_step_words=coordinator.cached_prototype_step_words,
            source_feature_generation_words=(
                coordinator.cached_feature_generation_words
            ),
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
        state: HCCLExternalCoordinatorBaseBridgeState,
        event: HCCLCausalCoreEventReceipt,
        binding: HCCLExternalCoordinatorBaseActionBinding,
        *,
        downstream_candidate_valid: Array,
    ) -> HCCLExternalCoordinatorBaseResult:
        """Attempt HCCL PP and both coordinator transitions, then adopt all or none."""

        self._require_state_contract(state)
        self._hccl.world._require_event_contract(event)
        self._require_binding_contract(binding)
        downstream = _require_array(
            downstream_candidate_valid,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            label="downstream_candidate_valid",
        )
        if _contains_tracer((state, event, binding, downstream)):
            raise TypeError(
                "HCCL external-coordinator base composite stage is host/eager only"
            )
        source_valid = self.state_valid(state)
        event_valid = self._hccl.world.event_receipt_valid(
            state.hccl_state.world_state, event
        )
        binding_integrity = self._binding_integrity_valid(binding)
        expected_binding = self._make_binding(state, event, binding.hard_action_masks)
        binding_matches = _tree_exact_equal(binding, expected_binding)
        cached_admitted = self._cached_actions_admitted(binding)
        identities_distinct = self._identities_distinct(binding)
        hccl_result = self._hccl.stage(
            state.hccl_state,
            event,
            binding.base,
            binding.memory,
            binding.planner,
            downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
        )
        pp = cast(
            HCCLCausalCoreProposal,
            jax.tree.map(lambda leaf: leaf[_PP_SLOT], hccl_result.world_proposals),
        )
        transition_0 = self._transition(
            state.agent_0_state,
            pp,
            agent=0,
            discount=self._config.agent_0.inner.ensemble.world_model.gamma,
        )
        transition_1 = self._transition(
            state.agent_1_state,
            pp,
            agent=1,
            discount=self._config.agent_1.inner.ensemble.world_model.gamma,
        )
        result_0 = self._agent_0.step(state.agent_0_state, transition_0)
        result_1 = self._agent_1.step(state.agent_1_state, transition_1)
        contrasts = hccl_result.attribution.contrasts
        memory_zero = _contrast_exact_zero(contrasts.memory_total) & (
            _contrast_exact_zero(contrasts.memory_interaction)
        )
        planner_zero = _contrast_exact_zero(contrasts.planner_total) & (
            _contrast_exact_zero(contrasts.planner_interaction)
        )
        total_zero = _contrast_exact_zero(contrasts.pp_minus_bb)
        proposal_actions_base_only = jnp.all(
            hccl_result.world_proposals.joint_action_ids
            == binding.cached_actions[None, :]
        )
        ablation = (
            binding_integrity
            & binding_matches
            & proposal_actions_base_only
            & memory_zero
            & planner_zero
            & total_zero
        )
        candidate = HCCLExternalCoordinatorBaseBridgeState(
            hccl_state=hccl_result.state,
            agent_0_state=result_0.state,
            agent_1_state=result_1.state,
        )
        candidate_valid = self.state_valid(candidate)
        applied = (
            source_valid
            & event_valid
            & binding_integrity
            & binding_matches
            & cached_admitted
            & identities_distinct
            & hccl_result.update_applied
            & result_0.diagnostics.transaction_applied
            & result_1.diagnostics.transaction_applied
            & ablation
            & downstream
            & candidate_valid
        )
        final_state = cast(
            HCCLExternalCoordinatorBaseBridgeState,
            _tree_select(applied, candidate, state),
        )
        return HCCLExternalCoordinatorBaseResult(
            state=final_state,
            binding=binding,
            hccl_result=hccl_result,
            agent_0_transition=transition_0,
            agent_1_transition=transition_1,
            agent_0_result=result_0,
            agent_1_result=result_1,
            work=HCCLExternalCoordinatorBaseWork(
                world_proposal_calls=hccl_result.work.world_proposal_calls,
                attribution_proposal_calls=hccl_result.work.attribution_proposal_calls,
                coordinator_update_calls=jnp.ones((2,), dtype=jnp.int32),
                committed_composite_transactions=applied.astype(jnp.int32),
            ),
            source_state_valid=source_valid,
            event_receipt_valid=event_valid,
            binding_integrity_valid=binding_integrity,
            binding_matches_source=binding_matches,
            cached_actions_admitted=cached_admitted,
            action_receipt_identities_distinct=identities_distinct,
            memory_contrasts_zero=memory_zero,
            planner_contrasts_zero=planner_zero,
            total_stack_contrast_zero=total_zero,
            base_only_ablation_valid=ablation,
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
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: HCCLExternalCoordinatorBaseBridgeState | None = None,
    ) -> HCCLExternalCoordinatorBaseResourceBudget:
        reference = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(reference)
        if not bool(self.state_valid(reference)):
            raise ValueError("resource measurement requires a valid composite state")
        hccl_bytes = measure_hccl_world_attribution_state_nbytes(reference.hccl_state)
        agent_0_bytes = (
            measure_external_learned_state_router_audit_coordinator_state_nbytes(
                reference.agent_0_state
            )
        )
        agent_1_bytes = (
            measure_external_learned_state_router_audit_coordinator_state_nbytes(
                reference.agent_1_state
            )
        )
        hccl_budget = self._hccl.resource_budget(reference.hccl_state)
        return HCCLExternalCoordinatorBaseResourceBudget(
            schema=HCCL_EXTERNAL_COORDINATOR_BASE_RESOURCE_SCHEMA,
            hccl_state_owners=1,
            external_coordinator_state_owners=2,
            hccl_state_nbytes=hccl_bytes,
            agent_0_state_nbytes=agent_0_bytes,
            agent_1_state_nbytes=agent_1_bytes,
            total_persistent_state_nbytes=(
                hccl_bytes + agent_0_bytes + agent_1_bytes
            ),
            max_world_proposal_calls_per_transaction=(
                hccl_budget.max_world_proposal_calls_per_transaction
            ),
            max_attribution_proposal_calls_per_transaction=(
                hccl_budget.max_attribution_proposal_calls_per_transaction
            ),
            coordinator_update_calls_per_transaction=2,
            maximum_composite_transactions=min(
                hccl_budget.maximum_committed_transactions,
                self._config.agent_0.max_events,
                self._config.agent_1.max_events,
            ),
            composite_jit_supported=False,
            memory_layer_authority=0,
            planner_layer_authority=0,
            output_write_calls=0,
            artifact_bytes_written=0,
        )


def measure_hccl_external_coordinator_base_state_nbytes(
    state: HCCLExternalCoordinatorBaseBridgeState,
) -> int:
    if type(state) is not HCCLExternalCoordinatorBaseBridgeState:
        raise TypeError("state must be exact HCCLExternalCoordinatorBaseBridgeState")
    return _tree_nbytes(state)


def _checkpoint_digest(checkpoint: HCCLExternalCoordinatorBaseCheckpoint) -> str:
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


def save_hccl_external_coordinator_base_checkpoint(
    bridge: HCCLExternalCoordinatorBaseBridge,
    state: HCCLExternalCoordinatorBaseBridgeState,
) -> HCCLExternalCoordinatorBaseCheckpoint:
    """Return a strict in-memory checkpoint and perform no output write."""

    if type(bridge) is not HCCLExternalCoordinatorBaseBridge:
        raise TypeError("bridge must be exact HCCLExternalCoordinatorBaseBridge")
    bridge._require_state_contract(state)
    if not bool(bridge.state_valid(state)):
        raise ValueError("cannot checkpoint an invalid base composite state")
    copied = cast(
        HCCLExternalCoordinatorBaseBridgeState,
        jax.tree.map(jnp.array, state),
    )
    config = bridge.to_config()
    bare = HCCLExternalCoordinatorBaseCheckpoint(
        schema=HCCL_EXTERNAL_COORDINATOR_BASE_CHECKPOINT_SCHEMA,
        mechanism_status=HCCL_EXTERNAL_COORDINATOR_BASE_STATUS,
        evidence_level=HCCL_EXTERNAL_COORDINATOR_BASE_EVIDENCE_LEVEL,
        output_writes_authorized=False,
        artifact_authorized=False,
        evidence_authorized=False,
        config=config,
        config_sha256=_canonical_digest(config),
        resource_budget=bridge.resource_budget(copied).to_config(),
        state=copied,
        state_nbytes=measure_hccl_external_coordinator_base_state_nbytes(copied),
        state_sha256=_canonical_digest(_state_host_payload(copied)),
        checkpoint_sha256="",
    )
    return dataclasses.replace(bare, checkpoint_sha256=_checkpoint_digest(bare))


def load_hccl_external_coordinator_base_checkpoint(
    checkpoint: HCCLExternalCoordinatorBaseCheckpoint,
) -> tuple[HCCLExternalCoordinatorBaseBridge, HCCLExternalCoordinatorBaseBridgeState]:
    """Restore only the canonical in-memory base-composite checkpoint."""

    if type(checkpoint) is not HCCLExternalCoordinatorBaseCheckpoint:
        raise TypeError("checkpoint must be exact HCCLExternalCoordinatorBaseCheckpoint")
    fixed = {
        "schema": HCCL_EXTERNAL_COORDINATOR_BASE_CHECKPOINT_SCHEMA,
        "mechanism_status": HCCL_EXTERNAL_COORDINATOR_BASE_STATUS,
        "evidence_level": HCCL_EXTERNAL_COORDINATOR_BASE_EVIDENCE_LEVEL,
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
    bridge = HCCLExternalCoordinatorBaseBridge.from_config(checkpoint.config)
    bridge._require_state_contract(checkpoint.state)
    if type(checkpoint.state_nbytes) is not int or checkpoint.state_nbytes != (
        measure_hccl_external_coordinator_base_state_nbytes(checkpoint.state)
    ):
        raise ValueError("checkpoint state bytes differ")
    if type(checkpoint.state_sha256) is not str:
        raise ValueError("checkpoint state digest must be an exact string")
    if checkpoint.state_sha256 != _canonical_digest(_state_host_payload(checkpoint.state)):
        raise ValueError("checkpoint state digest differs")
    if type(checkpoint.resource_budget) is not dict or _canonical_digest(
        checkpoint.resource_budget
    ) != _canonical_digest(bridge.resource_budget(checkpoint.state).to_config()):
        raise ValueError("checkpoint resource budget differs")
    if type(checkpoint.checkpoint_sha256) is not str:
        raise ValueError("checkpoint digest must be an exact string")
    if checkpoint.checkpoint_sha256 != _checkpoint_digest(checkpoint):
        raise ValueError("checkpoint digest differs")
    if not bool(bridge.state_valid(checkpoint.state)):
        raise ValueError("checkpoint state is invalid")
    restored = cast(
        HCCLExternalCoordinatorBaseBridgeState,
        jax.tree.map(jnp.array, checkpoint.state),
    )
    return bridge, restored


__all__ = [
    "HCCL_EXTERNAL_COORDINATOR_BASE_BINDING_SCHEMA",
    "HCCL_EXTERNAL_COORDINATOR_BASE_CHECKPOINT_SCHEMA",
    "HCCL_EXTERNAL_COORDINATOR_BASE_CONFIG_SCHEMA",
    "HCCL_EXTERNAL_COORDINATOR_BASE_EVIDENCE_LEVEL",
    "HCCL_EXTERNAL_COORDINATOR_BASE_LIMITATIONS",
    "HCCL_EXTERNAL_COORDINATOR_BASE_RESOURCE_SCHEMA",
    "HCCL_EXTERNAL_COORDINATOR_BASE_STATE_SCHEMA",
    "HCCL_EXTERNAL_COORDINATOR_BASE_STATUS",
    "HCCLExternalCoordinatorBaseActionBinding",
    "HCCLExternalCoordinatorBaseBridge",
    "HCCLExternalCoordinatorBaseBridgeConfig",
    "HCCLExternalCoordinatorBaseBridgeState",
    "HCCLExternalCoordinatorBaseCheckpoint",
    "HCCLExternalCoordinatorBaseResourceBudget",
    "HCCLExternalCoordinatorBaseResult",
    "HCCLExternalCoordinatorBaseWork",
    "load_hccl_external_coordinator_base_checkpoint",
    "measure_hccl_external_coordinator_base_state_nbytes",
    "save_hccl_external_coordinator_base_checkpoint",
]
