# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Source-bound HCCL causal feedback for learned experiential memory.

This v1 bridge owns exactly one :class:`HCCLWorldAttributionAdapterState`, one
:class:`LearnedExperientialMemoryControllerState`, and one fixed pending
binding.  Preparation uses the unchanged controller to admit a categorical
retrieval and binds its pending receipt to one agent, one HCCL source/decision,
the exact B/M action-receipt identities, and the hard-mask/routing outcome.
Settlement uses the unchanged eight-proposal HCCL adapter and derives only that
agent's immediate ``memory_total.net_reward``.  The matching controller receipt
is then settled inside the same composite commit.

The unbound agent must have the same effective B and M action.  This prevents a
dyad-total MM-minus-BB contrast from being mislabeled as unilateral retrieval
credit.  Masked, unrouted, or otherwise unused retrievals clear only through
the controller's existing matching no-learning settlement.  Any failed child,
identity, bound, or downstream gate selects the complete source state so the
same prepared event can be retried.

Composite preparation, settlement, and bounded prebound scan are host/eager
only.  The donor kernels retain their smaller JIT boundaries; this bridge
rejects full-composite tracing before proposal work.  Its deterministic tags
bind integrity but do not authenticate the caller.  It implements no agent,
schedule run, seed authority, output, artifact, threshold, evidence, promotion,
or memory-benefit claim.  The scan only replays rows whose later receipts were
already prepared from their exact predecessor states; it generates no event or
action and is not an online orchestrator.
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

from alberta_framework.core.experiential_memory import ExperientialMemoryEntry
from alberta_framework.core.hccl_causal_attribution import (
    HCCLActionLayer,
    HCCLActionReceipt,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapter,
    HCCLWorldAttributionAdapterConfig,
    HCCLWorldAttributionAdapterResult,
    HCCLWorldAttributionAdapterState,
    measure_hccl_world_attribution_state_nbytes,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryController,
    LearnedExperientialMemoryControllerConfig,
    LearnedExperientialMemoryControllerState,
    LearnedExperientialMemoryFeedback,
    LearnedExperientialMemoryFeedbackResult,
    LearnedExperientialMemoryStepResult,
)
from alberta_framework.streams.hccl_causal_core import HCCLCausalCoreEventReceipt

HCCL_LEARNED_MEMORY_FEEDBACK_CONFIG_SCHEMA = (
    "alberta.hccl-learned-memory-feedback-bridge-config.v1"
)
HCCL_LEARNED_MEMORY_FEEDBACK_STATE_SCHEMA = (
    "alberta.hccl-learned-memory-feedback-bridge-state.v1"
)
HCCL_LEARNED_MEMORY_FEEDBACK_BINDING_SCHEMA = (
    "alberta.hccl-learned-memory-feedback-binding.v1"
)
HCCL_LEARNED_MEMORY_FEEDBACK_CHECKPOINT_SCHEMA = (
    "alberta.hccl-learned-memory-feedback-checkpoint.v1"
)
HCCL_LEARNED_MEMORY_FEEDBACK_RESOURCE_SCHEMA = (
    "alberta.hccl-learned-memory-feedback-resource.v1"
)
HCCL_LEARNED_MEMORY_FEEDBACK_STATUS = (
    "l0-development-hccl-learned-memory-feedback-only"
)
HCCL_LEARNED_MEMORY_FEEDBACK_EVIDENCE_LEVEL = "L0"
HCCL_LEARNED_MEMORY_FEEDBACK_LIMITATIONS = (
    "immediate-same-event-memory-action-effect-only",
    "memory-utility-feedback-not-delight-and-no-actor-backward",
    "unbound-agent-effective-B-and-M-actions-must-match",
    "categorical-two-action-controller-retrievals-only",
    "integrity-binding-is-not-caller-authentication",
    "composite-prepare-settle-and-scan-are-host-eager-only",
    "prebound-scan-replays-prepared-rows-and-generates-no-events-or-actions",
    "no-agent-schedule-run-output-artifact-threshold-evidence-or-promotion-authority",
)

_N_AGENTS = 2
_N_ACTIONS = 2
_IDENTITY_WORDS = 4
_OWNER_WORDS = 8
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
            raise TypeError("binding tag values must be float32/int32/bool/uint32")
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
    if value.dtype == jnp.dtype(jnp.uint32):
        return value
    raise TypeError("exact equality supports typed keys, float32, int32, bool, and uint32")


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


def _tree_nbytes(value: Any) -> int:
    total = 0
    for leaf in jax.tree.leaves(value):
        if not hasattr(leaf, "dtype"):
            continue
        array = cast(Array, leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            material = jr.key_data(array)
            total += int(material.size) * int(material.dtype.itemsize)
        else:
            total += int(array.size) * int(array.dtype.itemsize)
    return total


def _state_host_payload(state: HCCLLearnedMemoryFeedbackBridgeState) -> list[dict[str, object]]:
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


def _identities_distinct(base: HCCLActionReceipt, memory: HCCLActionReceipt) -> Array:
    identities = jnp.reshape(
        jnp.stack(
            (base.action_receipt_identity_words, memory.action_receipt_identity_words)
        ),
        (4, _IDENTITY_WORDS),
    )
    distinct = jnp.asarray(True, dtype=jnp.bool_)
    for left in range(4):
        for right in range(left):
            distinct = distinct & (~jnp.all(identities[left] == identities[right]))
    return distinct


@dataclasses.dataclass(frozen=True)
class HCCLLearnedMemoryFeedbackBridgeConfig:
    hccl: HCCLWorldAttributionAdapterConfig
    controller: LearnedExperientialMemoryControllerConfig
    binding_owner_digest: tuple[int, ...]
    max_host_scan_steps: int = 64

    def __post_init__(self) -> None:
        if type(self.hccl) is not HCCLWorldAttributionAdapterConfig:
            raise TypeError("hccl must be exact HCCLWorldAttributionAdapterConfig")
        if type(self.controller) is not LearnedExperientialMemoryControllerConfig:
            raise TypeError("controller must be exact LearnedExperientialMemoryControllerConfig")
        if self.controller.memory.action_dim != _N_ACTIONS:
            raise ValueError("bridge requires categorical two-action memory payloads")
        digest = self.binding_owner_digest
        if type(digest) is not tuple or len(digest) != _OWNER_WORDS:
            raise ValueError("binding_owner_digest must be an exact eight-word tuple")
        for index, word in enumerate(digest):
            if type(word) is not int or not 0 <= word <= _UINT32_MAX:
                raise ValueError(f"binding_owner_digest[{index}] must be uint32")
        if not any(digest):
            raise ValueError("binding_owner_digest must be nonzero")
        if type(self.max_host_scan_steps) is not int or not 1 <= self.max_host_scan_steps <= 1024:
            raise ValueError("max_host_scan_steps must be an exact integer in [1, 1024]")


@chex.dataclass(frozen=True)
class HCCLLearnedMemoryPendingBinding:
    available: Bool[Array, ""]
    agent_index: Int[Array, ""]
    controller_transaction_words: UInt[Array, " 2"]
    hccl_source_state_tag_words: UInt[Array, " 4"]
    hccl_source_words: UInt[Array, " 2"]
    hccl_decision_words: UInt[Array, " 2"]
    event_content_tag_words: UInt[Array, " 4"]
    base_action_receipt_identity_words: UInt[Array, "2 4"]
    memory_action_receipt_identity_words: UInt[Array, "2 4"]
    base_action_content_tag_words: UInt[Array, " 4"]
    memory_action_content_tag_words: UInt[Array, " 4"]
    hard_action_mask: Bool[Array, " 2"]
    base_action_id: Int[Array, ""]
    memory_action_before_mask_id: Int[Array, ""]
    memory_action_after_mask_id: Int[Array, ""]
    retrieved_action_id: Int[Array, ""]
    retrieval_routed: Bool[Array, ""]
    retrieved_action_safe: Bool[Array, ""]
    retrieval_used: Bool[Array, ""]
    content_tag_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLLearnedMemoryFeedbackBridgeState:
    hccl_state: HCCLWorldAttributionAdapterState
    controller_state: LearnedExperientialMemoryControllerState
    pending_binding: HCCLLearnedMemoryPendingBinding


@chex.dataclass(frozen=True)
class HCCLLearnedMemoryPrepareResult:
    state: HCCLLearnedMemoryFeedbackBridgeState
    controller_step: LearnedExperientialMemoryStepResult
    source_state_valid: Bool[Array, ""]
    pending_slot_free: Bool[Array, ""]
    event_receipt_valid: Bool[Array, ""]
    action_receipts_valid: Bool[Array, ""]
    action_receipt_identities_distinct: Bool[Array, ""]
    controller_retrieval_admitted: Bool[Array, ""]
    retrieved_action_categorical: Bool[Array, ""]
    categorical_action_timing_valid: Bool[Array, ""]
    hard_mask_result_bound: Bool[Array, ""]
    unbound_agent_unchanged: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLLearnedMemoryFeedbackWork:
    controller_prepare_calls: Int[Array, ""]
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    controller_settlement_calls: Int[Array, ""]
    committed_composite_transactions: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLLearnedMemoryFeedbackResult:
    state: HCCLLearnedMemoryFeedbackBridgeState
    hccl_result: HCCLWorldAttributionAdapterResult
    controller_feedback: LearnedExperientialMemoryFeedbackResult
    counterfactual_delta: Float[Array, ""]
    work: HCCLLearnedMemoryFeedbackWork
    source_state_valid: Bool[Array, ""]
    pending_binding_available: Bool[Array, ""]
    binding_matches_event: Bool[Array, ""]
    binding_matches_action_receipts: Bool[Array, ""]
    binding_matches_controller_pending: Bool[Array, ""]
    attribution_source_bound_and_committed: Bool[Array, ""]
    counterfactual_within_controller_bound: Bool[Array, ""]
    controller_settlement_applied: Bool[Array, ""]
    downstream_candidate_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLLearnedMemoryFeedbackScanResult:
    state: HCCLLearnedMemoryFeedbackBridgeState
    counterfactual_delta: Float[Array, " steps"]
    prepare_applied: Bool[Array, " steps"]
    update_applied: Bool[Array, " steps"]
    learning_eligible: Bool[Array, " steps"]
    hccl_post_transaction_words: UInt[Array, "steps 2"]
    controller_transaction_words: UInt[Array, "steps 2"]


@dataclasses.dataclass(frozen=True)
class HCCLLearnedMemoryFeedbackResourceBudget:
    schema: str
    hccl_state_owners: int
    controller_state_owners: int
    fixed_pending_bindings: int
    hccl_state_nbytes: int
    controller_state_nbytes: int
    pending_binding_nbytes: int
    total_persistent_state_nbytes: int
    max_controller_prepare_calls_per_binding: int
    max_world_proposal_calls_per_feedback: int
    max_attribution_proposal_calls_per_feedback: int
    max_controller_settlements_per_feedback: int
    max_host_scan_steps: int
    random_draws_per_feedback: int
    output_write_calls: int
    artifact_bytes_written: int

    def __post_init__(self) -> None:
        if (
            type(self.schema) is not str
            or self.schema != HCCL_LEARNED_MEMORY_FEEDBACK_RESOURCE_SCHEMA
        ):
            raise ValueError("resource schema differs")
        integer_fields = tuple(
            field.name for field in dataclasses.fields(self) if field.name != "schema"
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"resource {name} must be an exact int")
            if value < 0:
                raise ValueError(f"resource {name} must be nonnegative")
        if (
            self.hccl_state_owners != 1
            or self.controller_state_owners != 1
            or self.fixed_pending_bindings != 1
        ):
            raise ValueError("resource owner counts must remain exactly one")
        if min(
            self.hccl_state_nbytes,
            self.controller_state_nbytes,
            self.pending_binding_nbytes,
        ) <= 0:
            raise ValueError("resource state byte components must be positive")
        if self.total_persistent_state_nbytes != (
            self.hccl_state_nbytes
            + self.controller_state_nbytes
            + self.pending_binding_nbytes
        ):
            raise ValueError("resource state byte total differs from its components")
        if (
            self.max_controller_prepare_calls_per_binding != 1
            or self.max_world_proposal_calls_per_feedback < 1
            or self.max_attribution_proposal_calls_per_feedback < 1
            or self.max_controller_settlements_per_feedback != 1
            or not 1 <= self.max_host_scan_steps <= 1024
        ):
            raise ValueError("resource bounded-call contract differs")
        if (
            self.random_draws_per_feedback != 0
            or self.output_write_calls != 0
            or self.artifact_bytes_written != 0
        ):
            raise ValueError("resource no-randomness/no-output contract differs")

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class HCCLLearnedMemoryFeedbackCheckpoint:
    schema: str
    mechanism_status: str
    evidence_level: str
    output_writes_authorized: bool
    artifact_authorized: bool
    evidence_authorized: bool
    config: dict[str, object]
    config_sha256: str
    resource_budget: dict[str, object]
    state: HCCLLearnedMemoryFeedbackBridgeState
    state_nbytes: int
    state_sha256: str
    checkpoint_sha256: str


class HCCLLearnedMemoryFeedbackBridge:
    """Atomic host owner joining unchanged HCCL and learned-memory donors."""

    def __init__(self, config: HCCLLearnedMemoryFeedbackBridgeConfig):
        if type(config) is not HCCLLearnedMemoryFeedbackBridgeConfig:
            raise TypeError("config must be exact HCCLLearnedMemoryFeedbackBridgeConfig")
        self._config = config
        self._hccl = HCCLWorldAttributionAdapter(config.hccl)
        self._controller = LearnedExperientialMemoryController(config.controller)
        self._owner = jnp.asarray(config.binding_owner_digest, dtype=jnp.uint32)

    @property
    def config(self) -> HCCLLearnedMemoryFeedbackBridgeConfig:
        return self._config

    @property
    def hccl(self) -> HCCLWorldAttributionAdapter:
        return self._hccl

    @property
    def controller(self) -> LearnedExperientialMemoryController:
        return self._controller

    def to_config(self) -> dict[str, object]:
        return {
            "type": "HCCLLearnedMemoryFeedbackBridge",
            "schema": HCCL_LEARNED_MEMORY_FEEDBACK_CONFIG_SCHEMA,
            "state_schema": HCCL_LEARNED_MEMORY_FEEDBACK_STATE_SCHEMA,
            "binding_schema": HCCL_LEARNED_MEMORY_FEEDBACK_BINDING_SCHEMA,
            "checkpoint_schema": HCCL_LEARNED_MEMORY_FEEDBACK_CHECKPOINT_SCHEMA,
            "resource_schema": HCCL_LEARNED_MEMORY_FEEDBACK_RESOURCE_SCHEMA,
            "mechanism_status": HCCL_LEARNED_MEMORY_FEEDBACK_STATUS,
            "evidence_level": HCCL_LEARNED_MEMORY_FEEDBACK_EVIDENCE_LEVEL,
            "hccl_config": self._hccl.to_config(),
            "controller_config": self._controller.to_config(),
            "binding_owner_digest": list(self._config.binding_owner_digest),
            "max_host_scan_steps": self._config.max_host_scan_steps,
            "feedback_signal": "selected-agent-memory_total.net_reward",
            "delight_or_actor_backward": False,
            "required_action_payload": "exact-categorical-two-action-one-hot",
            "unbound_agent_effective_B_M_action_must_match": True,
            "uint64_clock_encoding": "big-endian-two-uint32-words",
            "hccl_state_owners": 1,
            "controller_state_owners": 1,
            "fixed_pending_bindings": 1,
            "composite_jit_supported": False,
            "prebound_scan_execution": "bounded-host-eager-python-loop",
            "caller_identity_authenticated": False,
            "agent_implementation_present": False,
            "schedule_execution_authorized": False,
            "output_writes_authorized": False,
            "artifact_authorized": False,
            "threshold_authorized": False,
            "seed_authority": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_LEARNED_MEMORY_FEEDBACK_LIMITATIONS),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLLearnedMemoryFeedbackBridge:
        if type(payload) is not dict:
            raise TypeError("config payload must be an exact dict")
        fields = dict(payload)
        hccl_raw = fields.get("hccl_config")
        controller_raw = fields.get("controller_config")
        digest = fields.get("binding_owner_digest")
        max_steps = fields.get("max_host_scan_steps")
        if type(hccl_raw) is not dict or type(controller_raw) is not dict:
            raise TypeError("nested bridge configs must be exact dicts")
        if type(digest) is not list or type(max_steps) is not int:
            raise TypeError("bridge digest/scan config fields have invalid types")
        candidate = cls(
            HCCLLearnedMemoryFeedbackBridgeConfig(
                hccl=HCCLWorldAttributionAdapter.from_config(hccl_raw).config,
                controller=LearnedExperientialMemoryControllerConfig.from_config(
                    controller_raw
                ),
                binding_owner_digest=tuple(digest),
                max_host_scan_steps=max_steps,
            )
        )
        if _canonical_digest(fields) != _canonical_digest(candidate.to_config()):
            raise ValueError("HCCL learned-memory feedback config is unsupported")
        return candidate

    def _empty_binding(self) -> HCCLLearnedMemoryPendingBinding:
        return HCCLLearnedMemoryPendingBinding(
            available=jnp.asarray(False, dtype=jnp.bool_),
            agent_index=jnp.asarray(-1, dtype=jnp.int32),
            controller_transaction_words=jnp.zeros((2,), dtype=jnp.uint32),
            hccl_source_state_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
            hccl_source_words=jnp.zeros((2,), dtype=jnp.uint32),
            hccl_decision_words=jnp.zeros((2,), dtype=jnp.uint32),
            event_content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
            base_action_receipt_identity_words=jnp.zeros((2, 4), dtype=jnp.uint32),
            memory_action_receipt_identity_words=jnp.zeros((2, 4), dtype=jnp.uint32),
            base_action_content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
            memory_action_content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
            hard_action_mask=jnp.zeros((2,), dtype=jnp.bool_),
            base_action_id=jnp.asarray(-1, dtype=jnp.int32),
            memory_action_before_mask_id=jnp.asarray(-1, dtype=jnp.int32),
            memory_action_after_mask_id=jnp.asarray(-1, dtype=jnp.int32),
            retrieved_action_id=jnp.asarray(-1, dtype=jnp.int32),
            retrieval_routed=jnp.asarray(False, dtype=jnp.bool_),
            retrieved_action_safe=jnp.asarray(False, dtype=jnp.bool_),
            retrieval_used=jnp.asarray(False, dtype=jnp.bool_),
            content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
        )

    def init(
        self,
        key: Array,
        *,
        controller_state: LearnedExperientialMemoryControllerState | None = None,
    ) -> HCCLLearnedMemoryFeedbackBridgeState:
        controller = self._controller.init() if controller_state is None else controller_state
        self._controller._validate_state_static(controller)
        if not bool(self._controller.state_valid(controller)):
            raise ValueError("controller_state must be valid for this bridge config")
        if bool(controller.pending.available):
            raise ValueError("initial controller_state must not own pending feedback")
        state = HCCLLearnedMemoryFeedbackBridgeState(
            hccl_state=self._hccl.init(key),
            controller_state=cast(
                LearnedExperientialMemoryControllerState,
                jax.tree.map(jnp.array, controller),
            ),
            pending_binding=self._empty_binding(),
        )
        if not bool(self.state_valid(state)):
            raise RuntimeError("initial bridge state is invalid")
        return state

    def _binding_tag(self, binding: HCCLLearnedMemoryPendingBinding) -> Array:
        return _content_tag(
            self._owner,
            binding.available,
            binding.agent_index,
            binding.controller_transaction_words,
            binding.hccl_source_state_tag_words,
            binding.hccl_source_words,
            binding.hccl_decision_words,
            binding.event_content_tag_words,
            binding.base_action_receipt_identity_words,
            binding.memory_action_receipt_identity_words,
            binding.base_action_content_tag_words,
            binding.memory_action_content_tag_words,
            binding.hard_action_mask,
            binding.base_action_id,
            binding.memory_action_before_mask_id,
            binding.memory_action_after_mask_id,
            binding.retrieved_action_id,
            binding.retrieval_routed,
            binding.retrieved_action_safe,
            binding.retrieval_used,
        )

    def _require_binding_contract(self, binding: HCCLLearnedMemoryPendingBinding) -> None:
        if type(binding) is not HCCLLearnedMemoryPendingBinding:
            raise TypeError("pending_binding must be exact HCCLLearnedMemoryPendingBinding")
        for name, shape, dtype in (
            ("available", (), jnp.bool_),
            ("agent_index", (), jnp.int32),
            ("controller_transaction_words", (2,), jnp.uint32),
            ("hccl_source_state_tag_words", (4,), jnp.uint32),
            ("hccl_source_words", (2,), jnp.uint32),
            ("hccl_decision_words", (2,), jnp.uint32),
            ("event_content_tag_words", (4,), jnp.uint32),
            ("base_action_receipt_identity_words", (2, 4), jnp.uint32),
            ("memory_action_receipt_identity_words", (2, 4), jnp.uint32),
            ("base_action_content_tag_words", (4,), jnp.uint32),
            ("memory_action_content_tag_words", (4,), jnp.uint32),
            ("hard_action_mask", (2,), jnp.bool_),
            ("base_action_id", (), jnp.int32),
            ("memory_action_before_mask_id", (), jnp.int32),
            ("memory_action_after_mask_id", (), jnp.int32),
            ("retrieved_action_id", (), jnp.int32),
            ("retrieval_routed", (), jnp.bool_),
            ("retrieved_action_safe", (), jnp.bool_),
            ("retrieval_used", (), jnp.bool_),
            ("content_tag_words", (4,), jnp.uint32),
        ):
            _require_array(
                getattr(binding, name),
                shape=shape,
                dtype=jnp.dtype(dtype),
                label=f"pending_binding.{name}",
            )

    def _require_state_contract(self, state: HCCLLearnedMemoryFeedbackBridgeState) -> None:
        if type(state) is not HCCLLearnedMemoryFeedbackBridgeState:
            raise TypeError("state must be exact HCCLLearnedMemoryFeedbackBridgeState")
        self._hccl._require_state_contract(state.hccl_state)
        self._controller._validate_state_static(state.controller_state)
        self._require_binding_contract(state.pending_binding)

    def _binding_valid(
        self,
        state: HCCLLearnedMemoryFeedbackBridgeState,
    ) -> Array:
        binding = state.pending_binding
        empty = _tree_exact_equal(binding, self._empty_binding())
        safe_index = jnp.clip(binding.retrieved_action_id, 0, _N_ACTIONS - 1)
        active = (
            binding.available
            & ((binding.agent_index == 0) | (binding.agent_index == 1))
            & jnp.all(
                binding.controller_transaction_words
                == state.controller_state.pending.transaction_words
            )
            & jnp.all(
                binding.hccl_source_state_tag_words
                == state.hccl_state.world_state.content_tag_words
            )
            & jnp.all(binding.hccl_source_words == state.hccl_state.world_state.step_words)
            & jnp.all(
                binding.hccl_decision_words == state.hccl_state.attribution_state.decision_words
            )
            & (binding.base_action_id >= 0)
            & (binding.base_action_id < _N_ACTIONS)
            & (binding.memory_action_before_mask_id >= 0)
            & (binding.memory_action_before_mask_id < _N_ACTIONS)
            & (binding.memory_action_after_mask_id >= 0)
            & (binding.memory_action_after_mask_id < _N_ACTIONS)
            & (binding.retrieved_action_id >= 0)
            & (binding.retrieved_action_id < _N_ACTIONS)
            & (binding.retrieved_action_safe == binding.hard_action_mask[safe_index])
            & (
                binding.retrieval_used
                == (
                    binding.retrieval_routed
                    & binding.retrieved_action_safe
                    & (binding.memory_action_after_mask_id == binding.retrieved_action_id)
                )
            )
            & jnp.all(binding.content_tag_words == self._binding_tag(binding))
        )
        return ((~binding.available) & empty) | active

    def state_valid(self, state: HCCLLearnedMemoryFeedbackBridgeState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return (
            self._hccl.state_valid(state.hccl_state)
            & self._controller.state_valid(state.controller_state)
            & self._binding_valid(state)
            & (state.pending_binding.available == state.controller_state.pending.available)
        )

    def _make_binding(
        self,
        state: HCCLLearnedMemoryFeedbackBridgeState,
        controller_state: LearnedExperientialMemoryControllerState,
        event: HCCLCausalCoreEventReceipt,
        base: HCCLActionReceipt,
        memory: HCCLActionReceipt,
        *,
        agent: Array,
        retrieved_action: Array,
        routed: Array,
        retrieved_safe: Array,
        retrieval_used: Array,
    ) -> HCCLLearnedMemoryPendingBinding:
        hard_mask = memory.hard_action_masks[agent]
        bare = HCCLLearnedMemoryPendingBinding(
            available=jnp.asarray(True, dtype=jnp.bool_),
            agent_index=agent,
            controller_transaction_words=controller_state.pending.transaction_words,
            hccl_source_state_tag_words=state.hccl_state.world_state.content_tag_words,
            hccl_source_words=state.hccl_state.world_state.step_words,
            hccl_decision_words=state.hccl_state.attribution_state.decision_words,
            event_content_tag_words=event.content_tag_words,
            base_action_receipt_identity_words=base.action_receipt_identity_words,
            memory_action_receipt_identity_words=memory.action_receipt_identity_words,
            base_action_content_tag_words=base.content_tag_words,
            memory_action_content_tag_words=memory.content_tag_words,
            hard_action_mask=hard_mask,
            base_action_id=base.actions_after_mask[agent],
            memory_action_before_mask_id=memory.actions_before_mask[agent],
            memory_action_after_mask_id=memory.actions_after_mask[agent],
            retrieved_action_id=retrieved_action,
            retrieval_routed=routed,
            retrieved_action_safe=retrieved_safe,
            retrieval_used=retrieval_used,
            content_tag_words=jnp.zeros((4,), dtype=jnp.uint32),
        )
        return cast(
            HCCLLearnedMemoryPendingBinding,
            cast(Any, bare).replace(content_tag_words=self._binding_tag(bare)),
        )

    def prepare_retrieval(
        self,
        state: HCCLLearnedMemoryFeedbackBridgeState,
        event: HCCLCausalCoreEventReceipt,
        base: HCCLActionReceipt,
        memory: HCCLActionReceipt,
        *,
        agent_index: Array,
        retrieval_routed: Array,
        query_key: Array,
        representation_version: Array,
        query_uncertainty: Array,
        query_uncertainty_available: Array,
        entry: ExperientialMemoryEntry,
    ) -> HCCLLearnedMemoryPrepareResult:
        """Admit one retrieval and bind it to the later exact HCCL event."""

        self._require_state_contract(state)
        self._hccl.world._require_event_contract(event)
        self._hccl.attribution._require_action_contract(base)
        self._hccl.attribution._require_action_contract(memory)
        agent = _require_array(
            agent_index,
            shape=(),
            dtype=jnp.dtype(jnp.int32),
            label="agent_index",
        )
        routed = _require_array(
            retrieval_routed,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            label="retrieval_routed",
        )
        self._controller._validate_step_inputs(
            state.controller_state,
            query_key,
            representation_version,
            query_uncertainty,
            query_uncertainty_available,
            entry,
        )
        if _contains_tracer(
            (
                state,
                event,
                base,
                memory,
                agent,
                routed,
                query_key,
                representation_version,
                query_uncertainty,
                query_uncertainty_available,
                entry,
            )
        ):
            raise TypeError(
                "HCCL learned-memory feedback composite prepare is host/eager only"
            )
        source_valid = self.state_valid(state)
        pending_free = (~state.pending_binding.available) & (
            ~state.controller_state.pending.available
        )
        event_valid = self._hccl.world.event_receipt_valid(state.hccl_state.world_state, event)
        source = self._hccl._source(state.hccl_state)
        exogenous = self._hccl._exogenous(source, event)
        action_valid = self._hccl.attribution._action_valid(
            source, exogenous, base, HCCLActionLayer.BASE
        ) & self._hccl.attribution._action_valid(
            source, exogenous, memory, HCCLActionLayer.MEMORY
        )
        identities_distinct = _identities_distinct(base, memory)
        step = self._controller.step(
            state.controller_state,
            query_key,
            representation_version,
            query_uncertainty,
            query_uncertainty_available,
            entry,
        )
        retrieval = step.retrieval
        action = retrieval.action
        categorical = (
            retrieval.accepted
            & jnp.all((action == 0.0) | (action == 1.0))
            & (jnp.sum(action) == 1.0)
        )
        retrieved_action = jnp.argmax(action).astype(jnp.int32)
        agent_valid = (agent == 0) | (agent == 1)
        safe_agent = jnp.clip(agent, 0, _N_AGENTS - 1)
        other = jnp.asarray(1, dtype=jnp.int32) - safe_agent
        retrieved_safe = memory.hard_action_masks[safe_agent, retrieved_action]
        expected_after = jnp.where(
            routed & retrieved_safe,
            retrieved_action,
            base.actions_after_mask[safe_agent],
        )
        routed_timing = memory.actions_before_mask[safe_agent] == retrieved_action
        unrouted_timing = (
            (memory.actions_before_mask[safe_agent] == base.actions_before_mask[safe_agent])
            & (memory.actions_after_mask[safe_agent] == base.actions_after_mask[safe_agent])
        )
        timing = agent_valid & jnp.where(routed, routed_timing, unrouted_timing)
        mask_bound = agent_valid & (
            memory.actions_after_mask[safe_agent] == expected_after
        )
        other_unchanged = agent_valid & (
            memory.actions_after_mask[other] == base.actions_after_mask[other]
        )
        retrieval_used = routed & retrieved_safe & (
            memory.actions_after_mask[safe_agent] == retrieved_action
        )
        admitted = (
            step.diagnostics.transaction_applied
            & step.diagnostics.pending_created
            & retrieval.accepted
            & step.state.pending.available
        )
        binding = self._make_binding(
            state,
            step.state,
            event,
            base,
            memory,
            agent=safe_agent,
            retrieved_action=retrieved_action,
            routed=routed,
            retrieved_safe=retrieved_safe,
            retrieval_used=retrieval_used,
        )
        candidate = HCCLLearnedMemoryFeedbackBridgeState(
            hccl_state=state.hccl_state,
            controller_state=step.state,
            pending_binding=binding,
        )
        candidate_valid = self.state_valid(candidate)
        applied = (
            source_valid
            & pending_free
            & event_valid
            & action_valid
            & identities_distinct
            & admitted
            & categorical
            & timing
            & mask_bound
            & other_unchanged
            & candidate_valid
        )
        final_state = cast(
            HCCLLearnedMemoryFeedbackBridgeState,
            _tree_select(applied, candidate, state),
        )
        return HCCLLearnedMemoryPrepareResult(
            state=final_state,
            controller_step=step,
            source_state_valid=source_valid,
            pending_slot_free=pending_free,
            event_receipt_valid=event_valid,
            action_receipts_valid=action_valid,
            action_receipt_identities_distinct=identities_distinct,
            controller_retrieval_admitted=admitted,
            retrieved_action_categorical=categorical,
            categorical_action_timing_valid=timing,
            hard_mask_result_bound=mask_bound,
            unbound_agent_unchanged=other_unchanged,
            candidate_state_valid=candidate_valid,
            update_applied=applied,
        )

    def _binding_matches_event_and_actions(
        self,
        binding: HCCLLearnedMemoryPendingBinding,
        event: HCCLCausalCoreEventReceipt,
        base: HCCLActionReceipt,
        memory: HCCLActionReceipt,
    ) -> tuple[Array, Array]:
        agent = jnp.clip(binding.agent_index, 0, _N_AGENTS - 1)
        event_match = jnp.all(
            binding.event_content_tag_words == event.content_tag_words
        ) & jnp.all(binding.hccl_source_words == event.source_step_words)
        actions_match = (
            jnp.all(
                binding.base_action_receipt_identity_words
                == base.action_receipt_identity_words
            )
            & jnp.all(
                binding.memory_action_receipt_identity_words
                == memory.action_receipt_identity_words
            )
            & jnp.all(binding.base_action_content_tag_words == base.content_tag_words)
            & jnp.all(binding.memory_action_content_tag_words == memory.content_tag_words)
            & jnp.all(binding.hard_action_mask == memory.hard_action_masks[agent])
            & (binding.base_action_id == base.actions_after_mask[agent])
            & (
                binding.memory_action_before_mask_id
                == memory.actions_before_mask[agent]
            )
            & (
                binding.memory_action_after_mask_id
                == memory.actions_after_mask[agent]
            )
        )
        return event_match, actions_match

    def stage_feedback(
        self,
        state: HCCLLearnedMemoryFeedbackBridgeState,
        event: HCCLCausalCoreEventReceipt,
        base: HCCLActionReceipt,
        memory: HCCLActionReceipt,
        planner: HCCLActionReceipt,
        *,
        downstream_candidate_valid: Array,
    ) -> HCCLLearnedMemoryFeedbackResult:
        """Commit HCCL PP and settle its exact memory contrast atomically."""

        self._require_state_contract(state)
        self._hccl.world._require_event_contract(event)
        for receipt in (base, memory, planner):
            self._hccl.attribution._require_action_contract(receipt)
        downstream = _require_array(
            downstream_candidate_valid,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            label="downstream_candidate_valid",
        )
        if _contains_tracer((state, event, base, memory, planner, downstream)):
            raise TypeError(
                "HCCL learned-memory feedback composite settlement is host/eager only"
            )
        source_valid = self.state_valid(state)
        binding = state.pending_binding
        event_match, actions_match = self._binding_matches_event_and_actions(
            binding, event, base, memory
        )
        controller_match = (
            binding.available
            & state.controller_state.pending.available
            & jnp.all(
                binding.controller_transaction_words
                == state.controller_state.pending.transaction_words
            )
        )
        hccl_result = self._hccl.stage(
            state.hccl_state,
            event,
            base,
            memory,
            planner,
            downstream_candidate_valid=downstream,
        )
        agent = jnp.clip(binding.agent_index, 0, _N_AGENTS - 1)
        derived_delta = hccl_result.attribution.contrasts.memory_total.net_reward[agent]
        feedback_delta = jnp.where(binding.retrieval_used, derived_delta, 0.0).astype(
            jnp.float32
        )
        attribution_bound = (
            hccl_result.update_applied
            & jnp.all(hccl_result.pre_transaction_words == binding.hccl_source_words)
            & jnp.all(
                hccl_result.attribution.pre_transaction_words
                == binding.hccl_decision_words
            )
            & hccl_result.attribution.typed_signals_valid
            & hccl_result.attribution.duplicate_mm_bit_exact
        )
        within_bound = jnp.isfinite(feedback_delta) & (
            jnp.abs(feedback_delta)
            <= jnp.asarray(
                self._config.controller.max_abs_counterfactual_delta,
                dtype=jnp.float32,
            )
        )
        feedback = LearnedExperientialMemoryFeedback(
            transaction_words=binding.controller_transaction_words,
            retrieval_used=binding.retrieval_used,
            counterfactual_available=binding.retrieval_used,
            counterfactual_delta=feedback_delta,
        )
        controller_feedback = self._controller.settle(state.controller_state, feedback)
        candidate = HCCLLearnedMemoryFeedbackBridgeState(
            hccl_state=hccl_result.state,
            controller_state=controller_feedback.state,
            pending_binding=self._empty_binding(),
        )
        candidate_valid = self.state_valid(candidate)
        applied = (
            source_valid
            & binding.available
            & event_match
            & actions_match
            & controller_match
            & attribution_bound
            & within_bound
            & controller_feedback.diagnostics.transaction_applied
            & downstream
            & candidate_valid
        )
        final_state = cast(
            HCCLLearnedMemoryFeedbackBridgeState,
            _tree_select(applied, candidate, state),
        )
        return HCCLLearnedMemoryFeedbackResult(
            state=final_state,
            hccl_result=hccl_result,
            controller_feedback=controller_feedback,
            counterfactual_delta=feedback_delta,
            work=HCCLLearnedMemoryFeedbackWork(
                controller_prepare_calls=jnp.asarray(0, dtype=jnp.int32),
                world_proposal_calls=hccl_result.work.world_proposal_calls,
                attribution_proposal_calls=hccl_result.work.attribution_proposal_calls,
                controller_settlement_calls=jnp.asarray(1, dtype=jnp.int32),
                committed_composite_transactions=applied.astype(jnp.int32),
            ),
            source_state_valid=source_valid,
            pending_binding_available=binding.available,
            binding_matches_event=event_match,
            binding_matches_action_receipts=actions_match,
            binding_matches_controller_pending=controller_match,
            attribution_source_bound_and_committed=attribution_bound & applied,
            counterfactual_within_controller_bound=within_bound,
            controller_settlement_applied=(
                controller_feedback.diagnostics.transaction_applied & applied
            ),
            downstream_candidate_valid=downstream,
            candidate_state_valid=candidate_valid,
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: HCCLLearnedMemoryFeedbackBridgeState | None = None,
    ) -> HCCLLearnedMemoryFeedbackResourceBudget:
        reference = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(reference)
        if not bool(self.state_valid(reference)):
            raise ValueError("resource measurement requires a valid bridge state")
        hccl_bytes = measure_hccl_world_attribution_state_nbytes(reference.hccl_state)
        controller_budget = self._controller.resource_budget(reference.controller_state)
        controller_bytes = controller_budget.owned_persistent_state_bytes
        binding_bytes = _tree_nbytes(reference.pending_binding)
        hccl_budget = self._hccl.resource_budget(reference.hccl_state)
        return HCCLLearnedMemoryFeedbackResourceBudget(
            schema=HCCL_LEARNED_MEMORY_FEEDBACK_RESOURCE_SCHEMA,
            hccl_state_owners=1,
            controller_state_owners=1,
            fixed_pending_bindings=1,
            hccl_state_nbytes=hccl_bytes,
            controller_state_nbytes=controller_bytes,
            pending_binding_nbytes=binding_bytes,
            total_persistent_state_nbytes=hccl_bytes + controller_bytes + binding_bytes,
            max_controller_prepare_calls_per_binding=1,
            max_world_proposal_calls_per_feedback=(
                hccl_budget.max_world_proposal_calls_per_transaction
            ),
            max_attribution_proposal_calls_per_feedback=(
                hccl_budget.max_attribution_proposal_calls_per_transaction
            ),
            max_controller_settlements_per_feedback=1,
            max_host_scan_steps=self._config.max_host_scan_steps,
            random_draws_per_feedback=0,
            output_write_calls=0,
            artifact_bytes_written=0,
        )


def measure_hccl_learned_memory_feedback_state_nbytes(
    state: HCCLLearnedMemoryFeedbackBridgeState,
) -> int:
    if type(state) is not HCCLLearnedMemoryFeedbackBridgeState:
        raise TypeError("state must be exact HCCLLearnedMemoryFeedbackBridgeState")
    return _tree_nbytes(state)


def _leading_steps(tree: Any, *, label: str) -> int:
    leaves = jax.tree.leaves(tree)
    if not leaves or getattr(leaves[0], "ndim", 0) < 1:
        raise ValueError(f"{label} must have a leading step dimension")
    steps = int(leaves[0].shape[0])
    if steps < 1:
        raise ValueError(f"{label} must contain at least one row")
    for leaf in leaves:
        if getattr(leaf, "ndim", 0) < 1 or leaf.shape[0] != steps:
            raise ValueError(f"{label} leaves must share one leading step dimension")
    return steps


def _tree_at(tree: Any, index: int) -> Any:
    return jax.tree.map(lambda leaf: leaf[index], tree)


def run_hccl_learned_memory_feedback_scan(
    bridge: HCCLLearnedMemoryFeedbackBridge,
    state: HCCLLearnedMemoryFeedbackBridgeState,
    events: HCCLCausalCoreEventReceipt,
    base: HCCLActionReceipt,
    memory: HCCLActionReceipt,
    planner: HCCLActionReceipt,
    agent_indices: Array,
    retrieval_routed: Array,
    query_keys: Array,
    representation_versions: Array,
    query_uncertainties: Array,
    query_uncertainty_available: Array,
    entries: ExperientialMemoryEntry,
    downstream_candidate_valid: Array,
) -> HCCLLearnedMemoryFeedbackScanResult:
    """Replay a bounded prebound trace; generate no online event, receipt, or action.

    Every row, including later-row receipts derived from exact predecessor
    states, must already exist before this host/eager verifier is called.
    """

    if type(bridge) is not HCCLLearnedMemoryFeedbackBridge:
        raise TypeError("bridge must be exact HCCLLearnedMemoryFeedbackBridge")
    bridge._require_state_contract(state)
    steps = _leading_steps(events, label="events")
    if steps > bridge.config.max_host_scan_steps:
        raise ValueError("scan exceeds configured max_host_scan_steps")
    for label, value in (
        ("base", base),
        ("memory", memory),
        ("planner", planner),
        ("entries", entries),
    ):
        if _leading_steps(value, label=label) != steps:
            raise ValueError(f"{label} step count differs")
    memory_config = bridge.config.controller.memory
    for array_label, array_value, shape, dtype in (
        ("agent_indices", agent_indices, (steps,), jnp.int32),
        ("retrieval_routed", retrieval_routed, (steps,), jnp.bool_),
        ("query_keys", query_keys, (steps, memory_config.key_dim), jnp.float32),
        ("representation_versions", representation_versions, (steps,), jnp.int32),
        ("query_uncertainties", query_uncertainties, (steps,), jnp.float32),
        (
            "query_uncertainty_available",
            query_uncertainty_available,
            (steps,),
            jnp.bool_,
        ),
        (
            "downstream_candidate_valid",
            downstream_candidate_valid,
            (steps,),
            jnp.bool_,
        ),
    ):
        _require_array(
            array_value,
            shape=shape,
            dtype=jnp.dtype(dtype),
            label=array_label,
        )
    if _contains_tracer(
        (
            state,
            events,
            base,
            memory,
            planner,
            agent_indices,
            retrieval_routed,
            query_keys,
            representation_versions,
            query_uncertainties,
            query_uncertainty_available,
            entries,
            downstream_candidate_valid,
        )
    ):
        raise TypeError("HCCL learned-memory feedback scan is host/eager only")
    carry = state
    deltas: list[Array] = []
    prepared_rows: list[Array] = []
    applied_rows: list[Array] = []
    learning_rows: list[Array] = []
    hccl_words: list[Array] = []
    controller_words: list[Array] = []
    for index in range(steps):
        event = cast(HCCLCausalCoreEventReceipt, _tree_at(events, index))
        base_row = cast(HCCLActionReceipt, _tree_at(base, index))
        memory_row = cast(HCCLActionReceipt, _tree_at(memory, index))
        planner_row = cast(HCCLActionReceipt, _tree_at(planner, index))
        entry = cast(ExperientialMemoryEntry, _tree_at(entries, index))
        prepared = bridge.prepare_retrieval(
            carry,
            event,
            base_row,
            memory_row,
            agent_index=agent_indices[index],
            retrieval_routed=retrieval_routed[index],
            query_key=query_keys[index],
            representation_version=representation_versions[index],
            query_uncertainty=query_uncertainties[index],
            query_uncertainty_available=query_uncertainty_available[index],
            entry=entry,
        )
        settled = bridge.stage_feedback(
            prepared.state,
            event,
            base_row,
            memory_row,
            planner_row,
            downstream_candidate_valid=downstream_candidate_valid[index],
        )
        carry = settled.state
        deltas.append(settled.counterfactual_delta)
        prepared_rows.append(prepared.update_applied)
        applied_rows.append(settled.update_applied)
        learning_rows.append(settled.controller_feedback.diagnostics.learning_eligible)
        hccl_words.append(settled.state.hccl_state.world_state.step_words)
        controller_words.append(settled.state.controller_state.transaction_words)
    return HCCLLearnedMemoryFeedbackScanResult(
        state=carry,
        counterfactual_delta=jnp.stack(tuple(deltas)),
        prepare_applied=jnp.stack(tuple(prepared_rows)),
        update_applied=jnp.stack(tuple(applied_rows)),
        learning_eligible=jnp.stack(tuple(learning_rows)),
        hccl_post_transaction_words=jnp.stack(tuple(hccl_words)),
        controller_transaction_words=jnp.stack(tuple(controller_words)),
    )


def _checkpoint_digest(checkpoint: HCCLLearnedMemoryFeedbackCheckpoint) -> str:
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


def save_hccl_learned_memory_feedback_checkpoint(
    bridge: HCCLLearnedMemoryFeedbackBridge,
    state: HCCLLearnedMemoryFeedbackBridgeState,
) -> HCCLLearnedMemoryFeedbackCheckpoint:
    """Return an in-memory checkpoint; perform no output writes."""

    if type(bridge) is not HCCLLearnedMemoryFeedbackBridge:
        raise TypeError("bridge must be exact HCCLLearnedMemoryFeedbackBridge")
    bridge._require_state_contract(state)
    if not bool(bridge.state_valid(state)):
        raise ValueError("cannot checkpoint invalid HCCL learned-memory bridge state")
    copied = cast(HCCLLearnedMemoryFeedbackBridgeState, jax.tree.map(jnp.array, state))
    config = bridge.to_config()
    budget = bridge.resource_budget(copied).to_config()
    bare = HCCLLearnedMemoryFeedbackCheckpoint(
        schema=HCCL_LEARNED_MEMORY_FEEDBACK_CHECKPOINT_SCHEMA,
        mechanism_status=HCCL_LEARNED_MEMORY_FEEDBACK_STATUS,
        evidence_level=HCCL_LEARNED_MEMORY_FEEDBACK_EVIDENCE_LEVEL,
        output_writes_authorized=False,
        artifact_authorized=False,
        evidence_authorized=False,
        config=config,
        config_sha256=_canonical_digest(config),
        resource_budget=budget,
        state=copied,
        state_nbytes=measure_hccl_learned_memory_feedback_state_nbytes(copied),
        state_sha256=_canonical_digest(_state_host_payload(copied)),
        checkpoint_sha256="",
    )
    return dataclasses.replace(bare, checkpoint_sha256=_checkpoint_digest(bare))


def load_hccl_learned_memory_feedback_checkpoint(
    checkpoint: HCCLLearnedMemoryFeedbackCheckpoint,
) -> tuple[HCCLLearnedMemoryFeedbackBridge, HCCLLearnedMemoryFeedbackBridgeState]:
    """Restore only a canonical in-memory bridge checkpoint."""

    if type(checkpoint) is not HCCLLearnedMemoryFeedbackCheckpoint:
        raise TypeError("checkpoint must be exact HCCLLearnedMemoryFeedbackCheckpoint")
    fixed = {
        "schema": HCCL_LEARNED_MEMORY_FEEDBACK_CHECKPOINT_SCHEMA,
        "mechanism_status": HCCL_LEARNED_MEMORY_FEEDBACK_STATUS,
        "evidence_level": HCCL_LEARNED_MEMORY_FEEDBACK_EVIDENCE_LEVEL,
        "output_writes_authorized": False,
        "artifact_authorized": False,
        "evidence_authorized": False,
    }
    for name, expected in fixed.items():
        actual = getattr(checkpoint, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"checkpoint {name} differs")
    if type(checkpoint.config) is not dict:
        raise TypeError("checkpoint config must be an exact dict")
    if type(checkpoint.resource_budget) is not dict:
        raise TypeError("checkpoint resource budget must be an exact dict")
    for name in ("config_sha256", "state_sha256", "checkpoint_sha256"):
        if type(getattr(checkpoint, name)) is not str:
            raise TypeError(f"checkpoint {name} must be an exact str")
    bridge = HCCLLearnedMemoryFeedbackBridge.from_config(checkpoint.config)
    if checkpoint.config_sha256 != _canonical_digest(checkpoint.config):
        raise ValueError("checkpoint config digest differs")
    bridge._require_state_contract(checkpoint.state)
    if type(checkpoint.state_nbytes) is not int:
        raise TypeError("checkpoint state_nbytes must be an exact int")
    if checkpoint.state_nbytes != measure_hccl_learned_memory_feedback_state_nbytes(
        checkpoint.state
    ):
        raise ValueError("checkpoint state bytes differ")
    if checkpoint.state_sha256 != _canonical_digest(_state_host_payload(checkpoint.state)):
        raise ValueError("checkpoint state digest differs")
    expected_budget = bridge.resource_budget(checkpoint.state).to_config()
    if _canonical_digest(checkpoint.resource_budget) != _canonical_digest(expected_budget):
        raise ValueError("checkpoint resource budget differs")
    if checkpoint.checkpoint_sha256 != _checkpoint_digest(checkpoint):
        raise ValueError("checkpoint digest differs")
    if not bool(bridge.state_valid(checkpoint.state)):
        raise ValueError("checkpoint state is invalid")
    restored = cast(
        HCCLLearnedMemoryFeedbackBridgeState,
        jax.tree.map(jnp.array, checkpoint.state),
    )
    return bridge, restored


__all__ = [
    "HCCL_LEARNED_MEMORY_FEEDBACK_BINDING_SCHEMA",
    "HCCL_LEARNED_MEMORY_FEEDBACK_CHECKPOINT_SCHEMA",
    "HCCL_LEARNED_MEMORY_FEEDBACK_CONFIG_SCHEMA",
    "HCCL_LEARNED_MEMORY_FEEDBACK_EVIDENCE_LEVEL",
    "HCCL_LEARNED_MEMORY_FEEDBACK_LIMITATIONS",
    "HCCL_LEARNED_MEMORY_FEEDBACK_RESOURCE_SCHEMA",
    "HCCL_LEARNED_MEMORY_FEEDBACK_STATE_SCHEMA",
    "HCCL_LEARNED_MEMORY_FEEDBACK_STATUS",
    "HCCLLearnedMemoryFeedbackBridge",
    "HCCLLearnedMemoryFeedbackBridgeConfig",
    "HCCLLearnedMemoryFeedbackBridgeState",
    "HCCLLearnedMemoryFeedbackCheckpoint",
    "HCCLLearnedMemoryFeedbackResourceBudget",
    "HCCLLearnedMemoryFeedbackResult",
    "HCCLLearnedMemoryFeedbackScanResult",
    "HCCLLearnedMemoryFeedbackWork",
    "HCCLLearnedMemoryPendingBinding",
    "HCCLLearnedMemoryPrepareResult",
    "load_hccl_learned_memory_feedback_checkpoint",
    "measure_hccl_learned_memory_feedback_state_nbytes",
    "run_hccl_learned_memory_feedback_scan",
    "save_hccl_learned_memory_feedback_checkpoint",
]
