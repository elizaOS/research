# mypy: disable-error-code="call-arg"
"""Causal closed-loop Prototype partner-fusion development lane.

This is a deliberately consumed, threshold-free L0 development protocol.  It
uses the real :class:`PrototypeAgent` action cache and its opt-in
:class:`PartnerPolicyFusion` consumer.  Each arm owns a distinct Prototype
lifecycle, fusion state, environment state, hard-mask authority, feedback
receipt chain, and trace hash chain.  The only paired values are the frozen
exogenous context, signal-noise, drift-noise, availability, cost, and mask
schedule.

The three arms differ only in the information made available to fusion:

``learned_fusion``
    Applies that arm's realized action-relative net assistance value.
``outcome_blind_fusion``
    Executes the same messages and feedback calls but replaces every realized
    assistance target with the fixed value zero.
``base_only``
    Executes the same Prototype/fusion kernels with a canonical empty message
    batch and unavailable feedback.

Partner reliability reverses without a task/regime identifier.  Observations,
rewards, messages, and feedback are generated from each arm's own executed
history.  The evaluator retains hidden correctness/reliability only as raw
scoring annotations.  A separate caller-owned receipt is the authority for
the hard action mask; fusion never owns or relaxes it.

There is no artifact writer, output path, threshold, winner, held-out seed,
acceptance decision, or promotion hook in this module.  A finite run is not
evidence of intelligence amplification or Alberta Plan completion.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.partner_policy_fusion import (
    ROUTE_IGNORE,
    ROUTE_QUERY,
    PartnerMessageBatch,
    PartnerPolicyFusionConfig,
    PartnerPolicyFusionFeedback,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeInteractionState,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
    PrototypeUpdateResult,
)

SCHEMA = "alberta.prototype-partner-fusion-closed-loop-development.v1"
CHECKPOINT_SCHEMA = "alberta.prototype-partner-fusion-closed-loop-development.checkpoint.v1"
PROTOCOL_NAMESPACE = "prototype-hidden-partner-causal-closed-loop-consumed-v1"
ASSESSMENT = "not_assessed"
EVIDENCE_LEVEL = "L0_mechanism_and_development_diagnostic_only"
DEVELOPMENT_ONLY = True
DEVELOPMENT_PROTOCOL_CONSUMED = True
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES_ALLOWED = False
THRESHOLDS_DEFINED = False
WINNER_DECLARED = False

LEARNED_FUSION: Literal["learned_fusion"] = "learned_fusion"
OUTCOME_BLIND_FUSION: Literal["outcome_blind_fusion"] = "outcome_blind_fusion"
BASE_ONLY: Literal["base_only"] = "base_only"
Condition = Literal["learned_fusion", "outcome_blind_fusion", "base_only"]
CONDITIONS: tuple[Condition, ...] = (
    LEARNED_FUSION,
    OUTCOME_BLIND_FUSION,
    BASE_ONLY,
)

N_ACTIONS = 3
OBSERVATION_DIM = 3
MAX_PARTNERS = 2
_INT32_MAX = int(np.iinfo(np.int32).max)
_UINT32_MAX = int(np.iinfo(np.uint32).max)


def _canonical_json_bytes(value: object) -> bytes:
    """Return the exact finite JSON encoding used by every digest."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _tree_payload(tree: object) -> dict[str, object]:
    """Encode one PyTree exactly, including typed PRNG key data."""

    leaves, structure = jax.tree_util.tree_flatten(tree)
    encoded: list[dict[str, object]] = []
    for leaf in leaves:
        value = leaf
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        array = np.asarray(value)
        encoded.append(
            {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "bytes_hex": array.tobytes(order="C").hex(),
            }
        )
    return {"structure": str(structure), "leaves": encoded}


def _tree_digest(tree: object) -> str:
    return _digest(_tree_payload(tree))


def _tree_nbytes(tree: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        value = leaf
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        total += int(np.asarray(value).nbytes)
    return total


def _condition_resource_entry(
    initial_arm: ClosedLoopArmState,
    arm: ClosedLoopArmState,
    *,
    event_clock: int,
) -> dict[str, object]:
    """Return one arm's shape/call budget without retaining sibling states."""

    wrapper = cast(PrototypeInteractionState, arm.prototype_state.ia_state)
    return {
        "prototype_state_bytes_initial": _tree_nbytes(initial_arm.prototype_state),
        "prototype_state_bytes_final": _tree_nbytes(arm.prototype_state),
        "fusion_state_bytes": _tree_nbytes(wrapper.partner_policy_fusion_state),
        "prototype_update_calls": arm.prototype_update_calls,
        "fusion_decision_calls": arm.fusion_decision_calls,
        "fusion_feedback_calls": arm.fusion_feedback_calls,
        "fixed_message_slots_per_call": MAX_PARTNERS,
        "paired_exogenous_events_read": event_clock,
        "committed_prototype_transitions": event_clock,
        "discarded_base_action_previews": event_clock,
        "runtime_evaluator_rng_draws": 0,
        "shared_mutable_agent_state": False,
        "shared_mutable_environment_state": False,
    }


def _tree_finite(tree: object) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        value = leaf
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
            value = jr.key_data(value)
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating) and not bool(np.all(np.isfinite(array))):
            return False
    return True


def _canonicalize_prototype_host_timing_metadata(
    state: PrototypeAgentState,
) -> PrototypeAgentState:
    """Remove nonpersistent wall-clock metadata from the exact replay state."""

    oak_state = cast(Any, state.oak_state)
    stomp_state = cast(Any, oak_state.stomp_state)
    base_learner_state = cast(Any, stomp_state.base_learner_state)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    base_learner_state = base_learner_state.replace(
        birth_timestamp=zero,
        uptime_s=zero,
    )
    stomp_state = stomp_state.replace(base_learner_state=base_learner_state)
    oak_state = oak_state.replace(stomp_state=stomp_state)
    return cast(
        PrototypeAgentState,
        state.replace(oak_state=oak_state),  # type: ignore[attr-defined]
    )


def _prototype_host_timing_metadata_canonical(state: PrototypeAgentState) -> bool:
    """Return whether host-only learner timing leaves have the exact zero encoding."""

    try:
        base_learner_state = cast(Any, state.oak_state).stomp_state.base_learner_state
        expected = np.asarray(0.0, dtype=np.float32).tobytes()
        for value in (base_learner_state.birth_timestamp, base_learner_state.uptime_s):
            array = np.asarray(value)
            if array.shape != () or array.dtype != np.dtype(np.float32):
                return False
            if array.tobytes() != expected:
                return False
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return True


def _words_tuple(value: Array) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(value, dtype=np.uint32).reshape(-1))


def _words_array(value: tuple[int, ...], *, length: int) -> Array:
    if len(value) != length or any(not 0 <= item <= _UINT32_MAX for item in value):
        raise ValueError("exact identity words are invalid")
    return jnp.asarray(value, dtype=jnp.uint32)


def _words_to_int(words: tuple[int, int]) -> int:
    return (words[0] << 32) | words[1]


def _increment_words(value: Array, amount: int = 1) -> Array:
    words = cast(tuple[int, int], _words_tuple(value))
    advanced = _words_to_int(words) + amount
    if not 0 <= advanced <= np.iinfo(np.uint64).max:
        raise ValueError("exact identity capacity exhausted")
    return jnp.asarray((advanced >> 32, advanced & _UINT32_MAX), dtype=jnp.uint32)


def _increment_prototype_decision_id(value: Array) -> Array:
    words = _words_tuple(value)
    if len(words) != 4:
        raise ValueError("Prototype decision identity must have four words")
    suffix = _increment_words(jnp.asarray(words[2:], dtype=jnp.uint32))
    return jnp.concatenate((jnp.asarray(words[:2], dtype=jnp.uint32), suffix))


def _identity_telemetry(value: Array) -> int:
    high, low = cast(tuple[int, int], _words_tuple(value))
    return low if high == 0 and low <= _INT32_MAX else _INT32_MAX


def _float32(value: float) -> float:
    normalized = float(np.float32(value))
    if not math.isfinite(normalized):
        raise ValueError("development protocol produced a non-finite float32 value")
    return normalized


def _owner_digest(condition: Condition, role: str) -> str:
    return _digest(
        {
            "namespace": PROTOCOL_NAMESPACE,
            "condition": condition,
            "role": role,
        }
    )


def _owner_lifecycle_words(condition: Condition) -> Array:
    raw = bytes.fromhex(_owner_digest(condition, "prototype_agent"))[:8]
    high = int.from_bytes(raw[:4], "big")
    low = int.from_bytes(raw[4:], "big")
    return jnp.asarray((high, low), dtype=jnp.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class ClosedLoopPartnerFusionConfig:
    """Frozen, already-consumed development protocol dimensions."""

    horizon: int = 12
    reversal_event: int = 6
    n_actions: int = N_ACTIONS
    observation_dim: int = OBSERVATION_DIM
    max_partners: int = MAX_PARTNERS
    declared_confidence: float = 0.9
    base_partner_costs: tuple[float, float] = (0.05, 0.20)
    cost_spike: float = 0.80

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "horizon": 12,
            "reversal_event": 6,
            "n_actions": 3,
            "observation_dim": 3,
            "max_partners": 2,
            "declared_confidence": 0.9,
            "base_partner_costs": (0.05, 0.20),
            "cost_spike": 0.80,
        }
        for name, expected_value in expected.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"{name} is frozen at {expected_value!r} for {SCHEMA}")

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


CONFIG = ClosedLoopPartnerFusionConfig()


@dataclasses.dataclass(frozen=True, slots=True)
class ClosedLoopExogenousEvent:
    """Paired exogenous values; no arm action or feedback appears here."""

    index: int
    after_reversal: bool
    context_bit: int
    reward_noise: float
    history_drift: float
    partner_signal_flip: tuple[bool, bool]
    partner_available: tuple[bool, bool]
    communication_cost: tuple[float, float]
    hard_action_mask: tuple[bool, bool, bool]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def build_closed_loop_exogenous_schedule(
    config: ClosedLoopPartnerFusionConfig = CONFIG,
) -> tuple[ClosedLoopExogenousEvent, ...]:
    """Return the frozen paired schedule, including one bootstrap event."""

    events: list[ClosedLoopExogenousEvent] = []
    for index in range(config.horizon + 1):
        phase_step = index % 6
        available = [True, True]
        if phase_step == 2:
            available[0] = False
        elif phase_step == 3:
            available[1] = False
        elif phase_step == 4:
            available = [False, False]
        costs = list(config.base_partner_costs)
        if phase_step == 1:
            costs[0] = config.cost_spike
        elif phase_step == 5:
            costs[1] = config.cost_spike
        mask = [True, True, True]
        if index % 6 == 2:
            mask[1] = False
        elif index % 6 == 5:
            mask[2] = False
        events.append(
            ClosedLoopExogenousEvent(
                index=index,
                after_reversal=index >= config.reversal_event,
                context_bit=index % 2,
                reward_noise=_float32((((index * 7) % 5) - 2) * 0.025),
                history_drift=_float32((((index * 11) % 7) - 3) * 0.01),
                partner_signal_flip=(index % 11 == 4, index % 13 == 8),
                partner_available=(available[0], available[1]),
                communication_cost=(_float32(costs[0]), _float32(costs[1])),
                hard_action_mask=(mask[0], mask[1], mask[2]),
            )
        )
    return tuple(events)


EXOGENOUS_SCHEDULE = build_closed_loop_exogenous_schedule()


def _partner_fusion_config() -> PartnerPolicyFusionConfig:
    return PartnerPolicyFusionConfig(
        max_partners=MAX_PARTNERS,
        context_dim=OBSERVATION_DIM,
        n_actions=N_ACTIONS,
        max_message_horizon=1,
        min_feedback_for_learned_routing=1,
        counter_cap=10_000,
        learning_rate=0.5,
        max_abs_weight=8.0,
        max_abs_context=2.0,
        max_communication_cost=1.0,
        communication_cost_weight=1.0,
        assistance_value_bound=2.0,
        safety_target_weight=0.0,
        accept_net_value_threshold=0.30,
        blend_net_value_threshold=0.05,
        clarification_confidence_threshold=0.20,
        max_query_cost=0.25,
        base_blend_weight=1.0,
        option_blend_weight=1.0,
        partner_blend_weight=1.0,
        max_abs_declared_score=2.0,
    )


def _prototype_config() -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(
                        feature_index=0,
                        threshold=1_000.0,
                        max_option_steps=8,
                    ),
                ),
                observation_dim=OBSERVATION_DIM,
                n_primitive_actions=N_ACTIONS,
                base_hidden_sizes=(),
                epsilon_base=0.0,
                epsilon_option=0.0,
            )
        ),
        partner_policy_fusion=_partner_fusion_config(),
    )


_AGENT = PrototypeAgent(_prototype_config())
_JIT_UPDATE = jax.jit(_AGENT.update_transition)


@dataclasses.dataclass(frozen=True, slots=True)
class ClosedLoopEnvironmentState:
    """Arm-owned endogenous environment state."""

    owner_digest: str
    clock: int
    observation: tuple[float, float, float]
    history_score: float
    last_action: int
    last_net_reward: float

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class HardMaskAuthorityReceipt:
    """Independent caller authority for one dispatch mask."""

    owner_digest: str
    condition: Condition
    execution_clock: int
    exogenous_candidate_mask: tuple[bool, bool, bool]
    mask: tuple[bool, bool, bool]
    source_event_digest: str
    receipt_digest: str

    def body(self) -> dict[str, object]:
        return {
            "owner_digest": self.owner_digest,
            "condition": self.condition,
            "execution_clock": self.execution_clock,
            "exogenous_candidate_mask": self.exogenous_candidate_mask,
            "mask": self.mask,
            "source_event_digest": self.source_event_digest,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "receipt_digest": self.receipt_digest}


def _hard_mask_receipt(
    condition: Condition,
    event: ClosedLoopExogenousEvent,
    *,
    counterfactual_base_action: int,
) -> HardMaskAuthorityReceipt:
    if not 0 <= counterfactual_base_action < N_ACTIONS:
        raise ValueError("hard-mask authority requires an in-range safe fallback")
    actual_mask = list(event.hard_action_mask)
    # Prototype requires the independently computed counterfactual base to be
    # admissible.  The caller owns this conservative fallback augmentation;
    # fusion neither sees nor controls how it was derived.
    actual_mask[counterfactual_base_action] = True
    provisional = HardMaskAuthorityReceipt(
        owner_digest=_owner_digest(condition, "caller_hard_mask"),
        condition=condition,
        execution_clock=event.index,
        exogenous_candidate_mask=event.hard_action_mask,
        mask=cast(tuple[bool, bool, bool], tuple(actual_mask)),
        source_event_digest=_digest(event.to_dict()),
        receipt_digest="",
    )
    return dataclasses.replace(provisional, receipt_digest=_digest(provisional.body()))


@dataclasses.dataclass(frozen=True, slots=True)
class PendingDecisionReceipt:
    """Exact owner-bound action/feedback handoff across one environment step."""

    condition: Condition
    evaluator_owner_digest: str
    prototype_owner_digest: str
    fusion_owner_digest: str
    execution_clock: int
    prototype_decision_id: tuple[int, int, int, int]
    fusion_decision_words: tuple[int, int]
    fusion_event_words: tuple[int, int]
    effective_action: int
    counterfactual_base_action: int
    selected_partner_id: int
    route: int
    feedback_armed: bool
    quoted_communication_cost: float
    charged_communication_cost: float
    message_batch_digest: str
    decision_environment_digest: str
    hard_mask_receipt: HardMaskAuthorityReceipt
    receipt_digest: str

    def body(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "evaluator_owner_digest": self.evaluator_owner_digest,
            "prototype_owner_digest": self.prototype_owner_digest,
            "fusion_owner_digest": self.fusion_owner_digest,
            "execution_clock": self.execution_clock,
            "prototype_decision_id": self.prototype_decision_id,
            "fusion_decision_words": self.fusion_decision_words,
            "fusion_event_words": self.fusion_event_words,
            "effective_action": self.effective_action,
            "counterfactual_base_action": self.counterfactual_base_action,
            "selected_partner_id": self.selected_partner_id,
            "route": self.route,
            "feedback_armed": self.feedback_armed,
            "quoted_communication_cost": self.quoted_communication_cost,
            "charged_communication_cost": self.charged_communication_cost,
            "message_batch_digest": self.message_batch_digest,
            "decision_environment_digest": self.decision_environment_digest,
            "hard_mask_receipt": self.hard_mask_receipt.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "receipt_digest": self.receipt_digest}


def _seal_pending(receipt: PendingDecisionReceipt) -> PendingDecisionReceipt:
    return dataclasses.replace(receipt, receipt_digest=_digest(receipt.body()))


@dataclasses.dataclass(frozen=True, slots=True)
class ClosedLoopTraceRecord:
    """One raw causal execution record in an arm-specific hash chain."""

    condition: Condition
    evaluator_owner_digest: str
    execution_clock: int
    exogenous_event: dict[str, object]
    environment_before: dict[str, object]
    environment_after: dict[str, object]
    pending_receipt: dict[str, object]
    prototype_state_before_digest: str
    prototype_state_after_digest: str
    executed_prototype_decision_id: tuple[int, int, int, int]
    executed_action: int
    hidden_correct_action: int
    hidden_reliable_partner: int
    task_reward: float
    base_counterfactual_reward: float
    charged_communication_cost: float
    net_reward: float
    realized_assistance_value: float
    feedback_target_kind: str
    feedback_target_supplied: float
    feedback_applied: bool
    next_observation: tuple[float, float, float]
    next_message_available: tuple[bool, bool]
    next_message_suggestions: tuple[int, int]
    next_message_provenance: tuple[int, int]
    next_message_batch_digest: str
    next_hard_mask_receipt: dict[str, object]
    next_counterfactual_base_action: int
    next_effective_action: int
    next_selected_partner_id: int
    next_route: int
    next_partner_influenced: bool
    next_decision_applied: bool
    next_action_allowed_by_caller: bool
    transition_valid: bool
    transaction_applied: bool
    previous_record_hash: str
    record_hash: str

    def body(self) -> dict[str, object]:
        value = dataclasses.asdict(self)
        value.pop("previous_record_hash")
        value.pop("record_hash")
        return value

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ClosedLoopArmState:
    """Complete independently owned state for one comparator arm."""

    condition: Condition
    evaluator_owner_digest: str
    prototype_owner_digest: str
    fusion_owner_digest: str
    environment: ClosedLoopEnvironmentState
    prototype_state: PrototypeAgentState
    pending_decision: PendingDecisionReceipt
    trace: tuple[ClosedLoopTraceRecord, ...]
    trace_head: str
    prototype_update_calls: int
    fusion_decision_calls: int
    fusion_feedback_calls: int
    state_seal: str


@dataclasses.dataclass(frozen=True, slots=True)
class ClosedLoopRunState:
    """All arms at one exact shared exogenous clock."""

    schema: str
    namespace: str
    protocol_digest: str
    event_clock: int
    arms: tuple[ClosedLoopArmState, ...]
    run_seal: str


def _environment_payload(state: ClosedLoopEnvironmentState) -> dict[str, object]:
    return state.to_dict()


def _arm_body(state: ClosedLoopArmState) -> dict[str, object]:
    return {
        "condition": state.condition,
        "evaluator_owner_digest": state.evaluator_owner_digest,
        "prototype_owner_digest": state.prototype_owner_digest,
        "fusion_owner_digest": state.fusion_owner_digest,
        "environment": state.environment.to_dict(),
        "prototype_state_digest": _tree_digest(state.prototype_state),
        "pending_decision": state.pending_decision.to_dict(),
        "trace": [record.to_dict() for record in state.trace],
        "trace_head": state.trace_head,
        "prototype_update_calls": state.prototype_update_calls,
        "fusion_decision_calls": state.fusion_decision_calls,
        "fusion_feedback_calls": state.fusion_feedback_calls,
    }


def _seal_arm(state: ClosedLoopArmState) -> ClosedLoopArmState:
    return dataclasses.replace(state, state_seal=_digest(_arm_body(state)))


def _run_body(state: ClosedLoopRunState) -> dict[str, object]:
    return {
        "schema": state.schema,
        "namespace": state.namespace,
        "protocol_digest": state.protocol_digest,
        "event_clock": state.event_clock,
        "arms": [{**_arm_body(arm), "state_seal": arm.state_seal} for arm in state.arms],
    }


def _seal_run(state: ClosedLoopRunState) -> ClosedLoopRunState:
    return dataclasses.replace(state, run_seal=_digest(_run_body(state)))


def _initial_observation(event: ClosedLoopExogenousEvent) -> tuple[float, float, float]:
    context = 1.0 if event.context_bit == 0 else -1.0
    return (_float32(context), 0.0, 0.0)


def _hidden_correct_action(
    environment: ClosedLoopEnvironmentState,
    event: ClosedLoopExogenousEvent,
) -> int:
    history_bit = int(environment.history_score > 0.25)
    return 1 + ((event.context_bit + history_bit) % 2)


def _hidden_reliable_partner(event: ClosedLoopExogenousEvent) -> int:
    before = event.context_bit
    return 1 - before if event.after_reversal else before


def _task_reward(action: int, correct_action: int, reward_noise: float) -> float:
    outcome = 1.0 if action == correct_action else (0.0 if action == 0 else -1.0)
    return _float32(outcome + reward_noise)


def _next_environment(
    state: ClosedLoopEnvironmentState,
    *,
    action: int,
    correct_action: int,
    net_reward: float,
    event: ClosedLoopExogenousEvent,
    next_event: ClosedLoopExogenousEvent,
) -> ClosedLoopEnvironmentState:
    outcome = 1.0 if action == correct_action else (0.0 if action == 0 else -1.0)
    history = _float32(
        np.clip(0.65 * state.history_score + 0.35 * outcome + event.history_drift, -1.0, 1.0)
    )
    context = 1.0 if next_event.context_bit == 0 else -1.0
    observation = (
        _float32(context),
        history,
        _float32(np.clip(net_reward / 2.0, -1.0, 1.0)),
    )
    return ClosedLoopEnvironmentState(
        owner_digest=state.owner_digest,
        clock=state.clock + 1,
        observation=observation,
        history_score=history,
        last_action=action,
        last_net_reward=_float32(net_reward),
    )


def _message_suggestions(
    environment: ClosedLoopEnvironmentState,
    event: ClosedLoopExogenousEvent,
) -> tuple[int, int]:
    correct = _hidden_correct_action(environment, event)
    wrong = 1 if correct == 2 else 2
    reliable = _hidden_reliable_partner(event)
    suggestions: list[int] = []
    for partner in range(MAX_PARTNERS):
        suggestion = correct if partner == reliable else wrong
        if event.partner_signal_flip[partner]:
            suggestion = wrong if suggestion == correct else correct
        suggestions.append(suggestion)
    return cast(tuple[int, int], tuple(suggestions))


def _message_batch_payload(batch: PartnerMessageBatch) -> dict[str, object]:
    return _tree_payload(batch)


def _runtime_manifest() -> dict[str, object]:
    devices = jax.devices()
    first = devices[0] if devices else None
    try:
        jaxlib_version = version("jaxlib")
    except PackageNotFoundError:
        jaxlib_version = "unavailable"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "byteorder": sys.byteorder,
        "jax": str(jax.__version__),
        "jaxlib": jaxlib_version,
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kind": None if first is None else str(first.device_kind),
    }


_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).parents[1] / "core" / "prototype_agent.py",
    Path(__file__).parents[1] / "core" / "partner_policy_fusion.py",
    Path(__file__).parents[1] / "core" / "oak.py",
    Path(__file__).parents[1] / "core" / "options.py",
)


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).parents[2].resolve()
    result: dict[str, str] = {}
    for path in _SOURCE_PATHS:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(root).as_posix()
        except ValueError:
            label = resolved.name
        result[label] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


class PrototypePartnerFusionClosedLoopDevelopmentEvaluator:
    """Strict host orchestrator for the consumed causal development lane."""

    def __init__(self, config: ClosedLoopPartnerFusionConfig = CONFIG) -> None:
        if config != CONFIG:
            raise ValueError("the closed-loop development protocol is frozen")
        self.config = config
        self.agent = _AGENT
        self.protocol_digest = _digest(
            {
                "schema": SCHEMA,
                "namespace": PROTOCOL_NAMESPACE,
                "config": config.to_dict(),
                "agent_config": self.agent.to_config(),
                "schedule": [event.to_dict() for event in EXOGENOUS_SCHEDULE],
            }
        )

    @staticmethod
    def _condition_index(condition: Condition) -> int:
        return CONDITIONS.index(condition)

    def _empty_batch(self) -> PartnerMessageBatch:
        fusion = self.agent.partner_policy_fusion
        if fusion is None:  # pragma: no cover - construction invariant
            raise RuntimeError("closed-loop evaluator requires partner fusion")
        return fusion.empty_messages()

    def _messages(
        self,
        arm: ClosedLoopArmState,
        environment: ClosedLoopEnvironmentState,
        event: ClosedLoopExogenousEvent,
        *,
        decision_words: Array,
        event_words: Array,
        observation_id: int,
        context_id: int,
        empty: bool,
    ) -> tuple[PartnerMessageBatch, tuple[int, int], tuple[int, int]]:
        suggestions = _message_suggestions(environment, event)
        provenance = cast(
            tuple[int, int],
            tuple(
                1_000 + 10 * self._condition_index(arm.condition) + partner
                for partner in range(MAX_PARTNERS)
            ),
        )
        if empty:
            return self._empty_batch(), suggestions, provenance
        decision_id = _identity_telemetry(decision_words)
        event_id = _identity_telemetry(event_words)
        ids = jnp.full((MAX_PARTNERS,), decision_id, dtype=jnp.int32)
        event_ids = jnp.full((MAX_PARTNERS,), event_id, dtype=jnp.int32)
        decision_matrix = jnp.broadcast_to(decision_words, (MAX_PARTNERS, 2))
        event_matrix = jnp.broadcast_to(event_words, (MAX_PARTNERS, 2))
        return (
            PartnerMessageBatch(
                available=jnp.asarray(event.partner_available, dtype=jnp.bool_),
                partner_id=jnp.arange(MAX_PARTNERS, dtype=jnp.int32),
                observation_id=jnp.full((MAX_PARTNERS,), observation_id, dtype=jnp.int32),
                context_id=jnp.full((MAX_PARTNERS,), context_id, dtype=jnp.int32),
                suggested_action=jnp.asarray(suggestions, dtype=jnp.int32),
                declared_confidence=jnp.full(
                    (MAX_PARTNERS,),
                    self.config.declared_confidence,
                    dtype=jnp.float32,
                ),
                rationale_reference=jnp.asarray((100, 101), dtype=jnp.int32),
                provenance_reference=jnp.asarray(provenance, dtype=jnp.int32),
                communication_cost=jnp.asarray(event.communication_cost, dtype=jnp.float32),
                issued_decision_id=ids,
                issued_event_id=event_ids,
                valid_through_event_id=event_ids,
                issued_decision_words=decision_matrix,
                issued_event_words=event_matrix,
                valid_through_event_words=event_matrix,
            ),
            suggestions,
            provenance,
        )

    def _initial_arm(self, condition: Condition) -> ClosedLoopArmState:
        event = EXOGENOUS_SCHEDULE[0]
        observation = _initial_observation(event)
        prototype_state = self.agent.start(
            self.agent.init(
                jr.key(10_000 + self._condition_index(condition)),
                lifecycle_id=_owner_lifecycle_words(condition),
            ),
            jnp.asarray(observation, dtype=jnp.float32),
        )
        prototype_state = _canonicalize_prototype_host_timing_metadata(prototype_state)
        environment = ClosedLoopEnvironmentState(
            owner_digest=_owner_digest(condition, "environment"),
            clock=0,
            observation=observation,
            history_score=0.0,
            last_action=-1,
            last_net_reward=0.0,
        )
        mask = _hard_mask_receipt(
            condition,
            event,
            counterfactual_base_action=int(prototype_state.current_action),
        )
        empty_digest = _digest(_message_batch_payload(self._empty_batch()))
        initial_receipt = _seal_pending(
            PendingDecisionReceipt(
                condition=condition,
                evaluator_owner_digest=_owner_digest(condition, "evaluator"),
                prototype_owner_digest=_owner_digest(condition, "prototype_agent"),
                fusion_owner_digest=_owner_digest(condition, "partner_fusion"),
                execution_clock=0,
                prototype_decision_id=cast(
                    tuple[int, int, int, int],
                    _words_tuple(prototype_state.current_decision_id),
                ),
                fusion_decision_words=(0, 0),
                fusion_event_words=(0, 0),
                effective_action=int(prototype_state.current_action),
                counterfactual_base_action=int(prototype_state.current_action),
                selected_partner_id=-1,
                route=ROUTE_IGNORE,
                feedback_armed=False,
                quoted_communication_cost=0.0,
                charged_communication_cost=0.0,
                message_batch_digest=empty_digest,
                decision_environment_digest=_digest(environment.to_dict()),
                hard_mask_receipt=mask,
                receipt_digest="",
            )
        )
        trace_head = _digest(
            {
                "namespace": PROTOCOL_NAMESPACE,
                "condition": condition,
                "owner": _owner_digest(condition, "evaluator"),
                "initial_environment": environment.to_dict(),
                "initial_prototype_state": _tree_digest(prototype_state),
            }
        )
        arm = ClosedLoopArmState(
            condition=condition,
            evaluator_owner_digest=_owner_digest(condition, "evaluator"),
            prototype_owner_digest=_owner_digest(condition, "prototype_agent"),
            fusion_owner_digest=_owner_digest(condition, "partner_fusion"),
            environment=environment,
            prototype_state=prototype_state,
            pending_decision=initial_receipt,
            trace=(),
            trace_head=trace_head,
            prototype_update_calls=0,
            fusion_decision_calls=0,
            fusion_feedback_calls=0,
            state_seal="",
        )
        return _seal_arm(arm)

    def initial_state(self) -> ClosedLoopRunState:
        state = ClosedLoopRunState(
            schema=SCHEMA,
            namespace=PROTOCOL_NAMESPACE,
            protocol_digest=self.protocol_digest,
            event_clock=0,
            arms=tuple(self._initial_arm(condition) for condition in CONDITIONS),
            run_seal="",
        )
        return _seal_run(state)

    def _validate_mask_receipt(
        self,
        receipt: HardMaskAuthorityReceipt,
        condition: Condition,
        event: ClosedLoopExogenousEvent,
        counterfactual_base_action: int,
    ) -> bool:
        expected_mask = list(event.hard_action_mask)
        if not 0 <= counterfactual_base_action < N_ACTIONS:
            return False
        expected_mask[counterfactual_base_action] = True
        return (
            receipt.owner_digest == _owner_digest(condition, "caller_hard_mask")
            and receipt.condition == condition
            and receipt.execution_clock == event.index
            and receipt.exogenous_candidate_mask == event.hard_action_mask
            and receipt.mask == tuple(expected_mask)
            and receipt.source_event_digest == _digest(event.to_dict())
            and receipt.receipt_digest == _digest(receipt.body())
        )

    def _validate_pending(
        self,
        arm: ClosedLoopArmState,
        receipt: PendingDecisionReceipt,
        event_clock: int,
    ) -> bool:
        event = EXOGENOUS_SCHEDULE[event_clock]
        action = int(arm.prototype_state.current_action)
        return (
            receipt.condition == arm.condition
            and receipt.evaluator_owner_digest == arm.evaluator_owner_digest
            and receipt.prototype_owner_digest == arm.prototype_owner_digest
            and receipt.fusion_owner_digest == arm.fusion_owner_digest
            and receipt.execution_clock == event_clock
            and receipt.prototype_decision_id
            == _words_tuple(arm.prototype_state.current_decision_id)
            and receipt.effective_action == action
            and 0 <= action < N_ACTIONS
            and receipt.hard_mask_receipt.mask[action]
            and self._validate_mask_receipt(
                receipt.hard_mask_receipt,
                arm.condition,
                event,
                receipt.counterfactual_base_action,
            )
            and math.isfinite(receipt.quoted_communication_cost)
            and 0.0 <= receipt.quoted_communication_cost <= _float32(self.config.cost_spike)
            and math.isfinite(receipt.charged_communication_cost)
            and 0.0 <= receipt.charged_communication_cost <= _float32(self.config.cost_spike)
            and receipt.receipt_digest == _digest(receipt.body())
        )

    def _trace_chain_valid(self, arm: ClosedLoopArmState) -> bool:
        expected = _digest(
            {
                "namespace": PROTOCOL_NAMESPACE,
                "condition": arm.condition,
                "owner": arm.evaluator_owner_digest,
                "initial_environment": {**self._initial_arm(arm.condition).environment.to_dict()},
                "initial_prototype_state": _tree_digest(
                    self._initial_arm(arm.condition).prototype_state
                ),
            }
        )
        for index, record in enumerate(arm.trace):
            if record.condition != arm.condition or record.execution_clock != index:
                return False
            if record.previous_record_hash != expected:
                return False
            expected = _digest({"previous_record_hash": expected, "record": record.body()})
            if record.record_hash != expected:
                return False
        return arm.trace_head == expected

    def _arm_structure_valid(self, arm: ClosedLoopArmState, event_clock: int) -> bool:
        try:
            valid = (
                arm.condition in CONDITIONS
                and arm.evaluator_owner_digest == _owner_digest(arm.condition, "evaluator")
                and arm.prototype_owner_digest == _owner_digest(arm.condition, "prototype_agent")
                and arm.fusion_owner_digest == _owner_digest(arm.condition, "partner_fusion")
                and arm.environment.owner_digest == _owner_digest(arm.condition, "environment")
                and arm.environment.clock == event_clock
                and len(arm.trace) == event_clock
                and arm.prototype_update_calls == 2 * event_clock
                and arm.fusion_decision_calls == 2 * event_clock
                and arm.fusion_feedback_calls == 2 * event_clock
                and arm.state_seal == _digest(_arm_body(arm))
                and self._validate_pending(arm, arm.pending_decision, event_clock)
                and bool(self.agent.validate_state(arm.prototype_state))
                and _prototype_host_timing_metadata_canonical(arm.prototype_state)
                and _tree_finite(arm.prototype_state)
                and all(
                    math.isfinite(value)
                    for value in (
                        *arm.environment.observation,
                        arm.environment.history_score,
                        arm.environment.last_net_reward,
                    )
                )
                and -1.0 <= arm.environment.history_score <= 1.0
                and self._trace_chain_valid(arm)
            )
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(valid)

    def _same_state(self, left: ClosedLoopRunState, right: ClosedLoopRunState) -> bool:
        try:
            return _canonical_json_bytes(
                {**_run_body(left), "run_seal": left.run_seal}
            ) == _canonical_json_bytes({**_run_body(right), "run_seal": right.run_seal})
        except (TypeError, ValueError, OverflowError):
            return False

    def _structure_valid(self, state: ClosedLoopRunState) -> bool:
        try:
            return bool(
                state.schema == SCHEMA
                and state.namespace == PROTOCOL_NAMESPACE
                and state.protocol_digest == self.protocol_digest
                and type(state.event_clock) is int
                and 0 <= state.event_clock <= self.config.horizon
                and len(state.arms) == len(CONDITIONS)
                and tuple(arm.condition for arm in state.arms) == CONDITIONS
                and len({arm.evaluator_owner_digest for arm in state.arms}) == len(CONDITIONS)
                and len({arm.environment.owner_digest for arm in state.arms}) == len(CONDITIONS)
                and all(self._arm_structure_valid(arm, state.event_clock) for arm in state.arms)
                and state.run_seal == _digest(_run_body(state))
            )
        except (TypeError, ValueError, OverflowError):
            return False

    def validate_state(self, state: object, *, reconstruct: bool = True) -> bool:
        """Reject structural, numeric, owner, chain, and causal-prefix drift."""

        if not isinstance(state, ClosedLoopRunState) or not self._structure_valid(state):
            return False
        if not reconstruct:
            return True
        try:
            expected = self._reconstruct_unchecked(state.event_clock)
        except (RuntimeError, TypeError, ValueError, OverflowError):
            return False
        return self._same_state(state, expected)

    def _feedback(
        self,
        arm: ClosedLoopArmState,
        pending: PendingDecisionReceipt,
        realized_assistance: float,
    ) -> PrototypePartnerPolicyFusionFeedback | None:
        if not pending.feedback_armed:
            return None
        target = realized_assistance if arm.condition == LEARNED_FUSION else 0.0
        return PrototypePartnerPolicyFusionFeedback(
            prototype_decision_id=_words_array(pending.prototype_decision_id, length=4),
            feedback=PartnerPolicyFusionFeedback(
                available=jnp.asarray(True, dtype=jnp.bool_),
                decision_id=jnp.asarray(
                    min(_words_to_int(pending.fusion_decision_words), _INT32_MAX),
                    dtype=jnp.int32,
                ),
                executed_event_id=jnp.asarray(
                    min(_words_to_int(pending.fusion_event_words), _INT32_MAX),
                    dtype=jnp.int32,
                ),
                decision_words=_words_array(pending.fusion_decision_words, length=2),
                executed_event_words=_words_array(pending.fusion_event_words, length=2),
                executed_action=jnp.asarray(pending.effective_action, dtype=jnp.int32),
                partner_id=jnp.asarray(pending.selected_partner_id, dtype=jnp.int32),
                assistance_value_available=jnp.asarray(True, dtype=jnp.bool_),
                realized_assistance_value=jnp.asarray(target, dtype=jnp.float32),
                safety_outcome_available=jnp.asarray(True, dtype=jnp.bool_),
                safety_outcome_ok=jnp.asarray(True, dtype=jnp.bool_),
            ),
        )

    def _next_sidecar(
        self,
        arm: ClosedLoopArmState,
        next_environment: ClosedLoopEnvironmentState,
        next_event: ClosedLoopExogenousEvent,
        *,
        final_bootstrap: bool,
        counterfactual_base_action: int,
    ) -> tuple[
        PrototypePartnerPolicyFusionInput,
        PartnerMessageBatch,
        tuple[int, int],
        tuple[int, int],
        HardMaskAuthorityReceipt,
    ]:
        prototype_state = arm.prototype_state
        decision_words = _increment_words(prototype_state.step_words)
        event_words = _increment_words(prototype_state.observation_event_words)
        observation_id = 10_000 + next_event.index
        context_id = next_event.context_bit
        empty = arm.condition == BASE_ONLY or final_bootstrap
        messages, suggestions, provenance = self._messages(
            arm,
            next_environment,
            next_event,
            decision_words=decision_words,
            event_words=event_words,
            observation_id=observation_id,
            context_id=context_id,
            empty=empty,
        )
        receipt = _hard_mask_receipt(
            arm.condition,
            next_event,
            counterfactual_base_action=counterfactual_base_action,
        )
        sidecar = PrototypePartnerPolicyFusionInput(
            available=jnp.asarray(True, dtype=jnp.bool_),
            prototype_decision_id=_increment_prototype_decision_id(
                prototype_state.current_decision_id
            ),
            observation_id=jnp.asarray(observation_id, dtype=jnp.int32),
            context_id=jnp.asarray(context_id, dtype=jnp.int32),
            context_features=jnp.asarray(next_environment.observation, dtype=jnp.float32),
            safety_action_mask=jnp.asarray(receipt.mask, dtype=jnp.bool_),
            keyboard_available=jnp.asarray(False, dtype=jnp.bool_),
            keyboard_vector=jnp.zeros((1,), dtype=jnp.float32),
            messages=messages,
        )
        return sidecar, messages, suggestions, provenance, receipt

    def _pending_from_result(
        self,
        arm: ClosedLoopArmState,
        result: PrototypeUpdateResult,
        next_environment: ClosedLoopEnvironmentState,
        next_event: ClosedLoopExogenousEvent,
        messages: PartnerMessageBatch,
        mask_receipt: HardMaskAuthorityReceipt,
    ) -> PendingDecisionReceipt:
        diagnostics = result.partner_policy_fusion_diagnostics
        if diagnostics is None:  # pragma: no cover - construction invariant
            raise RuntimeError("Prototype omitted configured fusion diagnostics")
        decision = diagnostics.decision
        selected = int(decision.selected_partner_id)
        route = int(decision.route)
        quoted = next_event.communication_cost[selected] if 0 <= selected < MAX_PARTNERS else 0.0
        charged = quoted if route != ROUTE_IGNORE else 0.0
        return _seal_pending(
            PendingDecisionReceipt(
                condition=arm.condition,
                evaluator_owner_digest=arm.evaluator_owner_digest,
                prototype_owner_digest=arm.prototype_owner_digest,
                fusion_owner_digest=arm.fusion_owner_digest,
                execution_clock=next_environment.clock,
                prototype_decision_id=cast(
                    tuple[int, int, int, int],
                    _words_tuple(result.state.current_decision_id),
                ),
                fusion_decision_words=cast(tuple[int, int], _words_tuple(decision.decision_words)),
                fusion_event_words=cast(tuple[int, int], _words_tuple(decision.event_words)),
                effective_action=int(result.action),
                counterfactual_base_action=int(diagnostics.counterfactual_base_action),
                selected_partner_id=selected,
                route=route,
                feedback_armed=bool(decision.feedback_armed),
                quoted_communication_cost=_float32(quoted),
                charged_communication_cost=_float32(charged),
                message_batch_digest=_digest(_message_batch_payload(messages)),
                decision_environment_digest=_digest(next_environment.to_dict()),
                hard_mask_receipt=mask_receipt,
                receipt_digest="",
            )
        )

    def _advance_arm(
        self,
        arm: ClosedLoopArmState,
        event_clock: int,
        *,
        pending: PendingDecisionReceipt,
        compiled: bool,
    ) -> ClosedLoopArmState:
        event = EXOGENOUS_SCHEDULE[event_clock]
        next_event = EXOGENOUS_SCHEDULE[event_clock + 1]
        if not self._validate_pending(arm, pending, event_clock):
            raise ValueError("stale, tampered, or cross-owner pending receipt")
        environment_before = arm.environment
        if environment_before.clock != event_clock:
            raise ValueError("environment clock is stale")
        action = int(arm.prototype_state.current_action)
        correct_action = _hidden_correct_action(environment_before, event)
        reliable_partner = _hidden_reliable_partner(event)
        task_reward = _task_reward(action, correct_action, event.reward_noise)
        base_reward = _task_reward(
            pending.counterfactual_base_action,
            correct_action,
            event.reward_noise,
        )
        net_reward = _float32(task_reward - pending.charged_communication_cost)
        realized_assistance = _float32(net_reward - base_reward)
        next_environment = _next_environment(
            environment_before,
            action=action,
            correct_action=correct_action,
            net_reward=net_reward,
            event=event,
            next_event=next_event,
        )
        transition = PrototypeTransition(
            observation=arm.prototype_state.current_raw_observation,
            action=arm.prototype_state.current_action,
            decision_id=arm.prototype_state.current_decision_id,
            reward=jnp.asarray(net_reward, dtype=jnp.float32),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            next_observation=jnp.asarray(next_environment.observation, dtype=jnp.float32),
            next_decision_observation=jnp.asarray(next_environment.observation, dtype=jnp.float32),
        )
        feedback = self._feedback(arm, pending, realized_assistance)
        update = _JIT_UPDATE if compiled else self.agent.update_transition
        # A pure candidate evaluation obtains OaK's real counterfactual base
        # action before the independent caller constructs its hard mask.  The
        # candidate state is discarded; every arm pays the same logical call.
        preview = update(
            arm.prototype_state,
            transition,
            partner_policy_fusion_feedback=feedback,
        )
        if not bool(preview.transition_diagnostics.valid):
            raise RuntimeError("Prototype rejected the base-action preview")
        preview_base_action = int(preview.action)
        final_bootstrap = event_clock + 1 >= self.config.horizon
        sidecar, messages, suggestions, provenance, mask_receipt = self._next_sidecar(
            arm,
            next_environment,
            next_event,
            final_bootstrap=final_bootstrap,
            counterfactual_base_action=preview_base_action,
        )
        result = update(
            arm.prototype_state,
            transition,
            partner_policy_fusion_input=sidecar,
            partner_policy_fusion_feedback=feedback,
        )
        diagnostics = result.partner_policy_fusion_diagnostics
        if diagnostics is None:
            raise RuntimeError("configured fusion diagnostics are missing")
        if not bool(result.transition_diagnostics.valid):
            raise RuntimeError(f"Prototype rejected {arm.condition} transition {event_clock}")
        if (
            not bool(self.agent.validate_state(result.state))
            or not _prototype_host_timing_metadata_canonical(result.state)
            or not _tree_finite(result)
        ):
            raise RuntimeError("Prototype update failed finite checkpoint validation")
        next_action = int(result.action)
        next_allowed = 0 <= next_action < N_ACTIONS and mask_receipt.mask[next_action]
        if not next_allowed:
            raise RuntimeError("Prototype dispatch escaped caller-owned hard mask")
        if not bool(diagnostics.decision.applied):
            raise RuntimeError("valid closed-loop fusion decision was not applied")
        if int(diagnostics.counterfactual_base_action) != preview_base_action:
            raise RuntimeError("pure preview disagreed with Prototype's base action")

        pending_next = self._pending_from_result(
            arm,
            result,
            next_environment,
            next_event,
            messages,
            mask_receipt,
        )
        feedback_target = (
            realized_assistance
            if arm.condition == LEARNED_FUSION and pending.feedback_armed
            else 0.0
        )
        feedback_kind = (
            "own_realized_assistance"
            if arm.condition == LEARNED_FUSION
            else (
                "fixed_zero_outcome_blind"
                if arm.condition == OUTCOME_BLIND_FUSION
                else "unavailable_base_only"
            )
        )
        record = ClosedLoopTraceRecord(
            condition=arm.condition,
            evaluator_owner_digest=arm.evaluator_owner_digest,
            execution_clock=event_clock,
            exogenous_event=event.to_dict(),
            environment_before=environment_before.to_dict(),
            environment_after=next_environment.to_dict(),
            pending_receipt=pending.to_dict(),
            prototype_state_before_digest=_tree_digest(arm.prototype_state),
            prototype_state_after_digest=_tree_digest(result.state),
            executed_prototype_decision_id=cast(
                tuple[int, int, int, int],
                _words_tuple(arm.prototype_state.current_decision_id),
            ),
            executed_action=action,
            hidden_correct_action=correct_action,
            hidden_reliable_partner=reliable_partner,
            task_reward=task_reward,
            base_counterfactual_reward=base_reward,
            charged_communication_cost=pending.charged_communication_cost,
            net_reward=net_reward,
            realized_assistance_value=realized_assistance,
            feedback_target_kind=feedback_kind,
            feedback_target_supplied=feedback_target,
            feedback_applied=bool(diagnostics.feedback.applied),
            next_observation=next_environment.observation,
            next_message_available=cast(
                tuple[bool, bool],
                tuple(bool(value) for value in np.asarray(messages.available)),
            ),
            next_message_suggestions=suggestions,
            next_message_provenance=provenance,
            next_message_batch_digest=_digest(_message_batch_payload(messages)),
            next_hard_mask_receipt=mask_receipt.to_dict(),
            next_counterfactual_base_action=int(diagnostics.counterfactual_base_action),
            next_effective_action=next_action,
            next_selected_partner_id=int(diagnostics.decision.selected_partner_id),
            next_route=int(diagnostics.decision.route),
            next_partner_influenced=bool(diagnostics.decision.partner_influenced),
            next_decision_applied=bool(diagnostics.decision.applied),
            next_action_allowed_by_caller=next_allowed,
            transition_valid=bool(result.transition_diagnostics.valid),
            transaction_applied=bool(diagnostics.transaction_applied),
            previous_record_hash=arm.trace_head,
            record_hash="",
        )
        record = dataclasses.replace(
            record,
            record_hash=_digest({"previous_record_hash": arm.trace_head, "record": record.body()}),
        )
        next_arm = ClosedLoopArmState(
            condition=arm.condition,
            evaluator_owner_digest=arm.evaluator_owner_digest,
            prototype_owner_digest=arm.prototype_owner_digest,
            fusion_owner_digest=arm.fusion_owner_digest,
            environment=next_environment,
            prototype_state=result.state,
            pending_decision=pending_next,
            trace=(*arm.trace, record),
            trace_head=record.record_hash,
            prototype_update_calls=arm.prototype_update_calls + 2,
            fusion_decision_calls=arm.fusion_decision_calls + 2,
            fusion_feedback_calls=arm.fusion_feedback_calls + 2,
            state_seal="",
        )
        return _seal_arm(next_arm)

    def _advance_unchecked(
        self,
        state: ClosedLoopRunState,
        *,
        receipt_overrides: Mapping[Condition, PendingDecisionReceipt] | None = None,
        compiled: bool = True,
    ) -> ClosedLoopRunState:
        if state.event_clock >= self.config.horizon:
            raise ValueError("closed-loop development life is complete")
        overrides = {} if receipt_overrides is None else dict(receipt_overrides)
        unknown = set(overrides).difference(CONDITIONS)
        if unknown:
            raise ValueError("receipt override names an unknown condition")
        chosen: dict[Condition, PendingDecisionReceipt] = {}
        for arm in state.arms:
            receipt = overrides.get(arm.condition, arm.pending_decision)
            if receipt != arm.pending_decision:
                raise ValueError("receipt override is not the exact pending receipt")
            if not self._validate_pending(arm, receipt, state.event_clock):
                raise ValueError("receipt preflight failed before arm execution")
            chosen[arm.condition] = receipt
        next_arms = tuple(
            self._advance_arm(
                arm,
                state.event_clock,
                pending=chosen[arm.condition],
                compiled=compiled,
            )
            for arm in state.arms
        )
        next_state = ClosedLoopRunState(
            schema=state.schema,
            namespace=state.namespace,
            protocol_digest=state.protocol_digest,
            event_clock=state.event_clock + 1,
            arms=next_arms,
            run_seal="",
        )
        return _seal_run(next_state)

    def step(
        self,
        state: ClosedLoopRunState,
        *,
        receipt_overrides: Mapping[Condition, PendingDecisionReceipt] | None = None,
    ) -> ClosedLoopRunState:
        """Advance atomically after exact causal-prefix and receipt validation."""

        if not self.validate_state(state):
            raise ValueError("invalid or noncausal closed-loop run state")
        return self._advance_unchecked(state, receipt_overrides=receipt_overrides, compiled=True)

    def _reconstruct_unchecked(self, event_count: int) -> ClosedLoopRunState:
        if type(event_count) is not int or not 0 <= event_count <= self.config.horizon:
            raise ValueError("event_count is outside the frozen development life")
        state = self.initial_state()
        for _ in range(event_count):
            state = self._advance_unchecked(state, compiled=True)
        return state

    def reconstruct(self, event_count: int) -> ClosedLoopRunState:
        """Return an exact source/runtime-bound causal prefix."""

        state = self._reconstruct_unchecked(event_count)
        if not self.validate_state(state, reconstruct=False):
            raise RuntimeError("internally reconstructed prefix failed validation")
        return state

    def run_to_end(self, state: ClosedLoopRunState | None = None) -> ClosedLoopRunState:
        """Finish the consumed schedule from a validated in-memory prefix."""

        current = self.initial_state() if state is None else state
        if not self.validate_state(current):
            raise ValueError("run_to_end requires an exact causal prefix")
        while current.event_clock < self.config.horizon:
            current = self._advance_unchecked(current, compiled=True)
        if not self.validate_state(current, reconstruct=False):
            raise RuntimeError("completed development state is invalid")
        return current

    def eager_jit_parity(self) -> bool:
        """Check exact parity for one real action-changing Prototype boundary."""

        initial = self.initial_state()
        learned = initial.arms[0]
        eager = self._advance_arm(
            learned,
            0,
            pending=learned.pending_decision,
            compiled=False,
        )
        compiled = self._advance_arm(
            learned,
            0,
            pending=learned.pending_decision,
            compiled=True,
        )
        return _canonical_json_bytes(
            {**_arm_body(eager), "state_seal": eager.state_seal}
        ) == _canonical_json_bytes({**_arm_body(compiled), "state_seal": compiled.state_seal})

    def resource_report(self, state: ClosedLoopRunState) -> dict[str, object]:
        """Return exact matched logical allocation and invocation declarations."""

        if not self.validate_state(state, reconstruct=False):
            raise ValueError("resource report requires a valid run state")
        fusion = self.agent.partner_policy_fusion
        if fusion is None:  # pragma: no cover - construction invariant
            raise RuntimeError("fusion resource budget is unavailable")
        initial = self.initial_state()
        per_condition: dict[str, object] = {}
        for initial_arm, arm in zip(initial.arms, state.arms, strict=True):
            per_condition[arm.condition] = _condition_resource_entry(
                initial_arm, arm, event_clock=state.event_clock
            )
        comparable = list(per_condition.values())
        matched = all(value == comparable[0] for value in comparable[1:])
        return {
            "matched_logical_budgets": matched,
            "paired_randomness_scope": "frozen_exogenous_values_only",
            "learner_rng_states_are_independent": True,
            "fusion_resource_budget": fusion.resource_budget.to_config(),
            "per_condition": per_condition,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ClosedLoopCheckpoint:
    """In-memory only checkpoint with exact causal-prefix reconstruction."""

    schema: str
    namespace: str
    protocol_digest: str
    next_event: int
    source_manifest: dict[str, str]
    runtime_manifest: dict[str, object]
    state: ClosedLoopRunState
    state_digest: str
    checkpoint_digest: str

    def metadata(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "namespace": self.namespace,
            "protocol_digest": self.protocol_digest,
            "next_event": self.next_event,
            "source_manifest": self.source_manifest,
            "runtime_manifest": self.runtime_manifest,
            "state_digest": self.state_digest,
        }


def make_prototype_partner_fusion_closed_loop_checkpoint(
    next_event: int,
) -> ClosedLoopCheckpoint:
    """Build an in-memory checkpoint after exactly ``next_event`` executions."""

    evaluator = PrototypePartnerFusionClosedLoopDevelopmentEvaluator()
    state = evaluator.reconstruct(next_event)
    state_digest = _digest({**_run_body(state), "run_seal": state.run_seal})
    provisional = ClosedLoopCheckpoint(
        schema=CHECKPOINT_SCHEMA,
        namespace=PROTOCOL_NAMESPACE,
        protocol_digest=evaluator.protocol_digest,
        next_event=next_event,
        source_manifest=_source_manifest(),
        runtime_manifest=_runtime_manifest(),
        state=state,
        state_digest=state_digest,
        checkpoint_digest="",
    )
    return dataclasses.replace(
        provisional,
        checkpoint_digest=_digest(provisional.metadata()),
    )


def _validate_checkpoint(checkpoint: object) -> tuple[str, ...]:
    if not isinstance(checkpoint, ClosedLoopCheckpoint):
        return ("checkpoint has the wrong type",)
    evaluator = PrototypePartnerFusionClosedLoopDevelopmentEvaluator()
    errors: list[str] = []
    if checkpoint.schema != CHECKPOINT_SCHEMA:
        errors.append("checkpoint schema changed")
    if checkpoint.namespace != PROTOCOL_NAMESPACE:
        errors.append("checkpoint namespace changed")
    if checkpoint.protocol_digest != evaluator.protocol_digest:
        errors.append("checkpoint protocol digest changed")
    if checkpoint.source_manifest != _source_manifest():
        errors.append("checkpoint source manifest changed")
    if checkpoint.runtime_manifest != _runtime_manifest():
        errors.append("checkpoint runtime manifest changed")
    if type(checkpoint.next_event) is not int or not (0 <= checkpoint.next_event <= CONFIG.horizon):
        errors.append("checkpoint next_event is invalid")
        return tuple(errors)
    try:
        actual_state_digest = _digest(
            {**_run_body(checkpoint.state), "run_seal": checkpoint.state.run_seal}
        )
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"checkpoint state cannot be encoded: {exc}")
        return tuple(errors)
    if checkpoint.state_digest != actual_state_digest:
        errors.append("checkpoint state digest mismatch")
    if checkpoint.checkpoint_digest != _digest(checkpoint.metadata()):
        errors.append("checkpoint digest mismatch")
    if checkpoint.state.event_clock != checkpoint.next_event:
        errors.append("checkpoint state clock mismatch")
    if not evaluator.validate_state(checkpoint.state):
        errors.append("checkpoint differs from exact causal prefix reconstruction")
    return tuple(errors)


def _arm_summary(arm: ClosedLoopArmState) -> dict[str, object]:
    rewards = [record.net_reward for record in arm.trace]
    pre = [
        record.net_reward
        for record in arm.trace
        if not bool(record.exogenous_event["after_reversal"])
    ]
    post = [
        record.net_reward for record in arm.trace if bool(record.exogenous_event["after_reversal"])
    ]

    def mean(values: list[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0

    return {
        "condition": arm.condition,
        "event_count": len(arm.trace),
        "mean_net_reward": mean(rewards),
        "pre_reversal_mean_net_reward": mean(pre),
        "post_reversal_mean_net_reward": mean(post),
        "partner_influenced_count": sum(record.next_partner_influenced for record in arm.trace),
        "query_route_count": sum(record.next_route == ROUTE_QUERY for record in arm.trace),
        "feedback_applied_count": sum(record.feedback_applied for record in arm.trace),
        "communication_cost_total": float(
            np.sum(
                np.asarray(
                    [record.charged_communication_cost for record in arm.trace],
                    dtype=np.float64,
                )
            )
        ),
        "correct_action_count": sum(
            record.executed_action == record.hidden_correct_action for record in arm.trace
        ),
        "hard_mask_violation_count": sum(
            not record.next_action_allowed_by_caller for record in arm.trace
        ),
        "disconnect_event_count": sum(
            not any(cast(tuple[bool, bool], record.exogenous_event["partner_available"]))
            for record in arm.trace
        ),
        "trace_head": arm.trace_head,
        "final_environment_digest": _digest(arm.environment.to_dict()),
        "final_prototype_state_digest": _tree_digest(arm.prototype_state),
    }


def _assemble_report(
    evaluator: PrototypePartnerFusionClosedLoopDevelopmentEvaluator,
    state: ClosedLoopRunState,
    *,
    eager_jit_parity: bool | None = None,
) -> dict[str, object]:
    resource_report = evaluator.resource_report(state)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "namespace": PROTOCOL_NAMESPACE,
        "assessment": ASSESSMENT,
        "evidence_level": EVIDENCE_LEVEL,
        "development_only": DEVELOPMENT_ONLY,
        "development_protocol_consumed": DEVELOPMENT_PROTOCOL_CONSUMED,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "output_writes_allowed": OUTPUT_WRITES_ALLOWED,
        "thresholds_defined": THRESHOLDS_DEFINED,
        "winner_declared": WINNER_DECLARED,
        "claim_scope": (
            "causal closed-loop mechanism exercise only; no intelligence-"
            "amplification efficacy, confidence-calibration, safety, or Alberta "
            "Plan completion claim"
        ),
        "paired_randomness_scope": "frozen_exogenous_values_only",
        "config": CONFIG.to_dict(),
        "agent_config": evaluator.agent.to_config(),
        "protocol_digest": evaluator.protocol_digest,
        "schedule_digest": _digest([event.to_dict() for event in EXOGENOUS_SCHEDULE]),
        "source_manifest": _source_manifest(),
        "runtime_manifest": _runtime_manifest(),
        "prototype_eager_jit_parity": (
            evaluator.eager_jit_parity() if eager_jit_parity is None else eager_jit_parity
        ),
        "compiled_scope": "real PrototypeAgent.update_transition boundary only",
        "resource_report": resource_report,
        "arm_owner_receipts": {
            arm.condition: {
                "evaluator_owner_digest": arm.evaluator_owner_digest,
                "prototype_owner_digest": arm.prototype_owner_digest,
                "fusion_owner_digest": arm.fusion_owner_digest,
                "environment_owner_digest": arm.environment.owner_digest,
                "hard_mask_owner_digest": arm.pending_decision.hard_mask_receipt.owner_digest,
            }
            for arm in state.arms
        },
        "traces": {arm.condition: [record.to_dict() for record in arm.trace] for arm in state.arms},
        "summaries": {arm.condition: _arm_summary(arm) for arm in state.arms},
        "final_run_state_digest": _digest({**_run_body(state), "run_seal": state.run_seal}),
        "deterministic_payload_digest": "",
    }
    unsigned = dict(payload)
    unsigned.pop("deterministic_payload_digest")
    payload["deterministic_payload_digest"] = _digest(unsigned)
    return payload


def _run_unvalidated() -> dict[str, object]:
    evaluator = PrototypePartnerFusionClosedLoopDevelopmentEvaluator()
    state = evaluator.run_to_end()
    return _assemble_report(evaluator, state)


@functools.lru_cache(maxsize=1)
def _replay_expected_report_condition_by_condition() -> dict[str, object]:
    """Reconstruct all raw records without retaining three learner PyTrees."""

    evaluator = PrototypePartnerFusionClosedLoopDevelopmentEvaluator()
    traces: dict[str, object] = {}
    summaries: dict[str, object] = {}
    owners: dict[str, object] = {}
    resources: dict[str, object] = {}
    arm_payloads: list[dict[str, object]] = []
    for condition in CONDITIONS:
        initial = evaluator._initial_arm(condition)
        arm = initial
        for event_clock in range(CONFIG.horizon):
            arm = evaluator._advance_arm(
                arm,
                event_clock,
                pending=arm.pending_decision,
                compiled=True,
            )
        traces[condition] = [record.to_dict() for record in arm.trace]
        summaries[condition] = _arm_summary(arm)
        owners[condition] = {
            "evaluator_owner_digest": arm.evaluator_owner_digest,
            "prototype_owner_digest": arm.prototype_owner_digest,
            "fusion_owner_digest": arm.fusion_owner_digest,
            "environment_owner_digest": arm.environment.owner_digest,
            "hard_mask_owner_digest": arm.pending_decision.hard_mask_receipt.owner_digest,
        }
        resources[condition] = _condition_resource_entry(initial, arm, event_clock=CONFIG.horizon)
        arm_payloads.append({**_arm_body(arm), "state_seal": arm.state_seal})
    comparable = list(resources.values())
    fusion = evaluator.agent.partner_policy_fusion
    if fusion is None:  # pragma: no cover - construction invariant
        raise RuntimeError("fusion resource budget is unavailable")
    resource_report = {
        "matched_logical_budgets": all(value == comparable[0] for value in comparable[1:]),
        "paired_randomness_scope": "frozen_exogenous_values_only",
        "learner_rng_states_are_independent": True,
        "fusion_resource_budget": fusion.resource_budget.to_config(),
        "per_condition": resources,
    }
    run_body: dict[str, object] = {
        "schema": SCHEMA,
        "namespace": PROTOCOL_NAMESPACE,
        "protocol_digest": evaluator.protocol_digest,
        "event_clock": CONFIG.horizon,
        "arms": arm_payloads,
    }
    run_seal = _digest(run_body)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "namespace": PROTOCOL_NAMESPACE,
        "assessment": ASSESSMENT,
        "evidence_level": EVIDENCE_LEVEL,
        "development_only": DEVELOPMENT_ONLY,
        "development_protocol_consumed": DEVELOPMENT_PROTOCOL_CONSUMED,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "output_writes_allowed": OUTPUT_WRITES_ALLOWED,
        "thresholds_defined": THRESHOLDS_DEFINED,
        "winner_declared": WINNER_DECLARED,
        "claim_scope": (
            "causal closed-loop mechanism exercise only; no intelligence-"
            "amplification efficacy, confidence-calibration, safety, or Alberta "
            "Plan completion claim"
        ),
        "paired_randomness_scope": "frozen_exogenous_values_only",
        "config": CONFIG.to_dict(),
        "agent_config": evaluator.agent.to_config(),
        "protocol_digest": evaluator.protocol_digest,
        "schedule_digest": _digest([event.to_dict() for event in EXOGENOUS_SCHEDULE]),
        "source_manifest": _source_manifest(),
        "runtime_manifest": _runtime_manifest(),
        "prototype_eager_jit_parity": True,
        "compiled_scope": "real PrototypeAgent.update_transition boundary only",
        "resource_report": resource_report,
        "arm_owner_receipts": owners,
        "traces": traces,
        "summaries": summaries,
        "final_run_state_digest": _digest({**run_body, "run_seal": run_seal}),
        "deterministic_payload_digest": "",
    }
    unsigned = dict(payload)
    unsigned.pop("deterministic_payload_digest")
    payload["deterministic_payload_digest"] = _digest(unsigned)
    return payload


def run_prototype_partner_fusion_closed_loop_development() -> dict[str, object]:
    """Run and strictly validate the consumed nonpromoting development lane."""

    report = _run_unvalidated()
    errors = validate_prototype_partner_fusion_closed_loop_report(report)
    if errors:
        raise RuntimeError("invalid closed-loop partner report: " + "; ".join(errors))
    return report


def validate_prototype_partner_fusion_closed_loop_report(
    report: object,
) -> tuple[str, ...]:
    """Reject metadata, digest, raw-chain, or exact causal-replay drift."""

    if not isinstance(report, Mapping):
        return ("report must be a mapping",)
    expected_fields = {
        "schema",
        "namespace",
        "assessment",
        "evidence_level",
        "development_only",
        "development_protocol_consumed",
        "scientific_promotion_allowed",
        "output_writes_allowed",
        "thresholds_defined",
        "winner_declared",
        "claim_scope",
        "paired_randomness_scope",
        "config",
        "agent_config",
        "protocol_digest",
        "schedule_digest",
        "source_manifest",
        "runtime_manifest",
        "prototype_eager_jit_parity",
        "compiled_scope",
        "resource_report",
        "arm_owner_receipts",
        "traces",
        "summaries",
        "final_run_state_digest",
        "deterministic_payload_digest",
    }
    if set(report) != expected_fields:
        return ("report fields do not match the v1 schema",)
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("report schema changed")
    if report.get("namespace") != PROTOCOL_NAMESPACE:
        errors.append("report namespace changed")
    if report.get("assessment") != ASSESSMENT:
        errors.append("assessment must remain not_assessed")
    if report.get("evidence_level") != EVIDENCE_LEVEL:
        errors.append("evidence level changed")
    if report.get("development_only") is not True:
        errors.append("development_only must remain true")
    if report.get("development_protocol_consumed") is not True:
        errors.append("development protocol must remain consumed")
    for forbidden in (
        "scientific_promotion_allowed",
        "output_writes_allowed",
        "thresholds_defined",
        "winner_declared",
    ):
        if report.get(forbidden) is not False:
            errors.append(f"{forbidden} must remain false")
    if report.get("paired_randomness_scope") != "frozen_exogenous_values_only":
        errors.append("paired randomness scope changed")
    if report.get("prototype_eager_jit_parity") is not True:
        errors.append("Prototype eager/JIT parity did not hold")
    unsigned = dict(report)
    supplied_digest = unsigned.pop("deterministic_payload_digest", None)
    try:
        if supplied_digest != _digest(unsigned):
            errors.append("deterministic payload digest mismatch")
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"report cannot be canonically encoded: {exc}")
        return tuple(errors)
    if errors:
        return tuple(errors)
    try:
        # Parity was already required and authenticated above.  Replay each
        # causal arm separately so validation never retains a second joint set
        # of three learner PyTrees alongside the supplied raw report.
        expected = _replay_expected_report_condition_by_condition()
    except Exception as exc:  # pragma: no cover - fail-closed diagnostic
        errors.append(f"exact causal replay failed: {type(exc).__name__}: {exc}")
        return tuple(errors)
    try:
        if _canonical_json_bytes(dict(report)) != _canonical_json_bytes(expected):
            errors.append("report differs from exact causal replay")
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"report comparison failed closed: {exc}")
    return tuple(errors)


def resume_prototype_partner_fusion_closed_loop_checkpoint(
    checkpoint: object,
) -> dict[str, object]:
    """Validate, causally reconstruct, resume, and report an in-memory prefix."""

    errors = _validate_checkpoint(checkpoint)
    if errors:
        raise ValueError("invalid closed-loop checkpoint: " + "; ".join(errors))
    typed = cast(ClosedLoopCheckpoint, checkpoint)
    evaluator = PrototypePartnerFusionClosedLoopDevelopmentEvaluator()
    final_state = evaluator.run_to_end(typed.state)
    report = _assemble_report(evaluator, final_state)
    validation = validate_prototype_partner_fusion_closed_loop_report(report)
    if validation:
        raise RuntimeError("resumed report failed validation: " + "; ".join(validation))
    return report


__all__ = [
    "ASSESSMENT",
    "BASE_ONLY",
    "CHECKPOINT_SCHEMA",
    "CONDITIONS",
    "CONFIG",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_PROTOCOL_CONSUMED",
    "EVIDENCE_LEVEL",
    "EXOGENOUS_SCHEDULE",
    "LEARNED_FUSION",
    "OUTCOME_BLIND_FUSION",
    "OUTPUT_WRITES_ALLOWED",
    "PROTOCOL_NAMESPACE",
    "ROUTE_QUERY",
    "SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "THRESHOLDS_DEFINED",
    "WINNER_DECLARED",
    "ClosedLoopArmState",
    "ClosedLoopCheckpoint",
    "ClosedLoopEnvironmentState",
    "ClosedLoopExogenousEvent",
    "ClosedLoopPartnerFusionConfig",
    "ClosedLoopRunState",
    "ClosedLoopTraceRecord",
    "HardMaskAuthorityReceipt",
    "PendingDecisionReceipt",
    "PrototypePartnerFusionClosedLoopDevelopmentEvaluator",
    "build_closed_loop_exogenous_schedule",
    "make_prototype_partner_fusion_closed_loop_checkpoint",
    "resume_prototype_partner_fusion_closed_loop_checkpoint",
    "run_prototype_partner_fusion_closed_loop_development",
    "validate_prototype_partner_fusion_closed_loop_report",
]
