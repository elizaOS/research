# mypy: disable-error-code="call-arg"
"""Matched-initialization causal partner-fusion development lane.

This is a separately versioned, permanently nonpromoting L0 development
protocol.  Unlike the consumed v1 diagnostic, every comparator starts from
the exact same typed PRNG key and bit-identical Prototype, partner-fusion, and
endogenous environment state.  Separate owner receipts wrap those identical
states without perturbing the learner PyTrees.

Only the declared partner intervention differs:

``learned_feedback``
    Receives partner messages and learns from its own realized action-relative
    net assistance value.
``fixed_zero_feedback``
    Receives the same partner-message mechanism but every available feedback
    target is the fixed value zero.
``empty_message_base_only``
    Receives a canonical empty message batch and unavailable feedback while
    still executing the same Prototype boundaries and fixed message slots.

The exogenous context, reward noise, history drift, partner availability,
communication cost, and caller-owned hard-mask schedule are paired.  Inputs
are *not* forced to remain identical after actions diverge: each arm's action
causally determines its next environment, reward, later message suggestions,
and feedback.  That distinction is load-bearing for the comparison.

There is no artifact writer, output path, threshold, winner, acceptance
decision, held-out seed, or promotion hook here.  Reported action changes,
returns, and costs are descriptive mechanism diagnostics only.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.partner_policy_fusion import (
    ROUTE_IGNORE,
    ROUTE_QUERY,
    PartnerMessageBatch,
    PartnerPolicyFusionFeedback,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgentState,
    PrototypeInteractionState,
    PrototypePartnerPolicyFusionFeedback,
    PrototypePartnerPolicyFusionInput,
    PrototypeTransition,
    PrototypeUpdateResult,
)
from alberta_framework.evaluation import (
    prototype_partner_fusion_closed_loop_development as _v1,
)

SCHEMA = "alberta.prototype-partner-fusion-matched-initialization-development.v2"
CHECKPOINT_SCHEMA = (
    "alberta.prototype-partner-fusion-matched-initialization-development.checkpoint.v2"
)
PROTOCOL_NAMESPACE = "prototype-hidden-partner-matched-initialization-causal-closed-loop-v2"
ASSESSMENT = "not_assessed"
EVIDENCE_LEVEL = "L0_mechanism_and_development_diagnostic_only"
DEVELOPMENT_ONLY = True
# The deterministic development seed is exercised by this module's contract;
# it can never become an untouched promotion seed.
DEVELOPMENT_PROTOCOL_CONSUMED = True
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES_ALLOWED = False
THRESHOLDS_DEFINED = False
WINNER_DECLARED = False
PAIRING_SCOPE = (
    "bit-identical initialization plus paired exogenous context/noise/drift/"
    "availability/cost/hard-mask schedule"
)
IDENTICAL_POST_DIVERGENCE_INPUTS = False

LEARNED_FEEDBACK: Literal["learned_feedback"] = "learned_feedback"
FIXED_ZERO_FEEDBACK: Literal["fixed_zero_feedback"] = "fixed_zero_feedback"
EMPTY_MESSAGE_BASE_ONLY: Literal["empty_message_base_only"] = "empty_message_base_only"
Condition = Literal[
    "learned_feedback",
    "fixed_zero_feedback",
    "empty_message_base_only",
]
CONDITIONS: tuple[Condition, ...] = (
    LEARNED_FEEDBACK,
    FIXED_ZERO_FEEDBACK,
    EMPTY_MESSAGE_BASE_ONLY,
)
DECLARED_INTERVENTIONS: dict[Condition, str] = {
    LEARNED_FEEDBACK: "partner messages plus own realized-assistance feedback",
    FIXED_ZERO_FEEDBACK: "partner messages plus fixed-zero outcome-blind feedback",
    EMPTY_MESSAGE_BASE_ONLY: "canonical empty messages plus unavailable feedback",
}

N_ACTIONS = _v1.N_ACTIONS
OBSERVATION_DIM = _v1.OBSERVATION_DIM
MAX_PARTNERS = _v1.MAX_PARTNERS
_INT32_MAX = int(np.iinfo(np.int32).max)
_UINT32_MAX = int(np.iinfo(np.uint32).max)

# Exact encoding helpers are shared with v1 and included in this lane's source
# manifest.  Re-exporting the two digest helpers keeps adversarial tests able
# to reseal malformed candidates and prove replay, rather than hashes, is the
# final authority.
_canonical_json_bytes = _v1._canonical_json_bytes
_digest = _v1._digest
_tree_payload = _v1._tree_payload
_tree_digest = _v1._tree_digest
_tree_nbytes = _v1._tree_nbytes
_tree_finite = _v1._tree_finite
_words_tuple = _v1._words_tuple
_words_array = _v1._words_array
_words_to_int = _v1._words_to_int
_increment_words = _v1._increment_words
_increment_prototype_decision_id = _v1._increment_prototype_decision_id
_identity_telemetry = _v1._identity_telemetry
_float32 = _v1._float32
_canonicalize_prototype_host_timing_metadata = _v1._canonicalize_prototype_host_timing_metadata
_prototype_host_timing_metadata_canonical = _v1._prototype_host_timing_metadata_canonical


def _owner_digest(condition: Condition, role: str) -> str:
    return _digest(
        {
            "namespace": PROTOCOL_NAMESPACE,
            "condition": condition,
            "role": role,
        }
    )


def _shared_lifecycle_words() -> Array:
    raw = bytes.fromhex(
        _digest({"namespace": PROTOCOL_NAMESPACE, "role": "shared_initial_lifecycle"})
    )[:8]
    return jnp.asarray(
        (int.from_bytes(raw[:4], "big"), int.from_bytes(raw[4:], "big")),
        dtype=jnp.uint32,
    )


def _typed_key_payload(key: Array) -> dict[str, object]:
    data = np.asarray(jr.key_data(key), dtype=np.uint32)
    return {
        "dtype": str(key.dtype),
        "shape": list(key.shape),
        "implementation": str(jr.key_impl(key)),
        "data_dtype": str(data.dtype),
        "data_shape": list(data.shape),
        "data_bytes_hex": data.tobytes(order="C").hex(),
    }


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedV2Config:
    """Frozen dimensions for the consumed, nonpromoting v2 diagnostic."""

    horizon: int = 12
    reversal_event: int = 6
    initialization_seed: int = 24_602
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
            "initialization_seed": 24_602,
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


CONFIG = MatchedV2Config()


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedV2ExogenousEvent:
    """Arm-independent event values; no action or learned state appears here."""

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


def build_matched_v2_exogenous_schedule(
    config: MatchedV2Config = CONFIG,
) -> tuple[MatchedV2ExogenousEvent, ...]:
    """Return the frozen paired schedule, including one bootstrap event.

    Every primitive action is safe in this synthetic task, so the paired mask
    is deliberately all-true.  The separately owned, source-bound mask receipt
    is still mandatory and fail-closed; this lane makes no shielding-efficacy
    claim.
    """

    events: list[MatchedV2ExogenousEvent] = []
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
        events.append(
            MatchedV2ExogenousEvent(
                index=index,
                after_reversal=index >= config.reversal_event,
                context_bit=index % 2,
                reward_noise=_float32((((index * 7) % 5) - 2) * 0.025),
                history_drift=_float32((((index * 11) % 7) - 3) * 0.01),
                partner_signal_flip=(index % 11 == 4, index % 13 == 8),
                partner_available=(available[0], available[1]),
                communication_cost=(_float32(costs[0]), _float32(costs[1])),
                hard_action_mask=(True, True, True),
            )
        )
    return tuple(events)


EXOGENOUS_SCHEDULE = build_matched_v2_exogenous_schedule()

_AGENT = _v1._AGENT
_JIT_UPDATE = _v1._JIT_UPDATE


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedV2EnvironmentState:
    """Endogenous environment values, with ownership kept outside the payload."""

    clock: int
    observation: tuple[float, float, float]
    history_score: float
    last_action: int
    last_net_reward: float

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedV2HardMaskAuthorityReceipt:
    """Arm-owned receipt for the common exogenous dispatch mask."""

    owner_digest: str
    condition: Condition
    execution_clock: int
    mask: tuple[bool, bool, bool]
    source_event_digest: str
    receipt_digest: str

    def body(self) -> dict[str, object]:
        return {
            "owner_digest": self.owner_digest,
            "condition": self.condition,
            "execution_clock": self.execution_clock,
            "mask": self.mask,
            "source_event_digest": self.source_event_digest,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "receipt_digest": self.receipt_digest}


def _hard_mask_receipt(
    condition: Condition,
    event: MatchedV2ExogenousEvent,
) -> MatchedV2HardMaskAuthorityReceipt:
    provisional = MatchedV2HardMaskAuthorityReceipt(
        owner_digest=_owner_digest(condition, "caller_hard_mask"),
        condition=condition,
        execution_clock=event.index,
        mask=event.hard_action_mask,
        source_event_digest=_digest(event.to_dict()),
        receipt_digest="",
    )
    return dataclasses.replace(provisional, receipt_digest=_digest(provisional.body()))


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedV2PendingDecisionReceipt:
    """Exact arm-owned action and feedback handoff for one transition."""

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
    hard_mask_receipt: MatchedV2HardMaskAuthorityReceipt
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


def _seal_pending(receipt: MatchedV2PendingDecisionReceipt) -> MatchedV2PendingDecisionReceipt:
    return dataclasses.replace(receipt, receipt_digest=_digest(receipt.body()))


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedV2TraceRecord:
    """One raw causal event in an arm-specific SHA-256 hash chain."""

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
    action_changed_by_assistance: bool
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
class MatchedV2ArmState:
    """One independently owned arm around initially identical dynamic state."""

    condition: Condition
    evaluator_owner_digest: str
    prototype_owner_digest: str
    fusion_owner_digest: str
    environment_owner_digest: str
    initialization_key_digest: str
    environment: MatchedV2EnvironmentState
    prototype_state: PrototypeAgentState
    pending_decision: MatchedV2PendingDecisionReceipt
    trace: tuple[MatchedV2TraceRecord, ...]
    trace_head: str
    prototype_update_calls: int
    fusion_decision_calls: int
    fusion_feedback_opportunities: int
    state_seal: str


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedV2RunState:
    """All comparator arms at one paired exogenous event clock."""

    schema: str
    namespace: str
    protocol_digest: str
    event_clock: int
    arms: tuple[MatchedV2ArmState, ...]
    run_seal: str


def _arm_body(state: MatchedV2ArmState) -> dict[str, object]:
    return {
        "condition": state.condition,
        "evaluator_owner_digest": state.evaluator_owner_digest,
        "prototype_owner_digest": state.prototype_owner_digest,
        "fusion_owner_digest": state.fusion_owner_digest,
        "environment_owner_digest": state.environment_owner_digest,
        "initialization_key_digest": state.initialization_key_digest,
        "environment": state.environment.to_dict(),
        "prototype_state_digest": _tree_digest(state.prototype_state),
        "pending_decision": state.pending_decision.to_dict(),
        "trace": [record.to_dict() for record in state.trace],
        "trace_head": state.trace_head,
        "prototype_update_calls": state.prototype_update_calls,
        "fusion_decision_calls": state.fusion_decision_calls,
        "fusion_feedback_opportunities": state.fusion_feedback_opportunities,
    }


def _seal_arm(state: MatchedV2ArmState) -> MatchedV2ArmState:
    return dataclasses.replace(state, state_seal=_digest(_arm_body(state)))


def _run_body(state: MatchedV2RunState) -> dict[str, object]:
    return {
        "schema": state.schema,
        "namespace": state.namespace,
        "protocol_digest": state.protocol_digest,
        "event_clock": state.event_clock,
        "arms": [{**_arm_body(arm), "state_seal": arm.state_seal} for arm in state.arms],
    }


def _seal_run(state: MatchedV2RunState) -> MatchedV2RunState:
    return dataclasses.replace(state, run_seal=_digest(_run_body(state)))


def _initial_observation(event: MatchedV2ExogenousEvent) -> tuple[float, float, float]:
    context = 1.0 if event.context_bit == 0 else -1.0
    return (_float32(context), 0.0, 0.0)


def _hidden_correct_action(
    environment: MatchedV2EnvironmentState,
    event: MatchedV2ExogenousEvent,
) -> int:
    history_bit = int(environment.history_score > 0.25)
    return 1 + ((event.context_bit + history_bit) % 2)


def _hidden_reliable_partner(event: MatchedV2ExogenousEvent) -> int:
    before = event.context_bit
    return 1 - before if event.after_reversal else before


def _task_reward(action: int, correct_action: int, reward_noise: float) -> float:
    outcome = 1.0 if action == correct_action else (0.0 if action == 0 else -1.0)
    return _float32(outcome + reward_noise)


def _next_environment(
    state: MatchedV2EnvironmentState,
    *,
    action: int,
    correct_action: int,
    net_reward: float,
    event: MatchedV2ExogenousEvent,
    next_event: MatchedV2ExogenousEvent,
) -> MatchedV2EnvironmentState:
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
    return MatchedV2EnvironmentState(
        clock=state.clock + 1,
        observation=observation,
        history_score=history,
        last_action=action,
        last_net_reward=_float32(net_reward),
    )


def _message_suggestions(
    environment: MatchedV2EnvironmentState,
    event: MatchedV2ExogenousEvent,
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


_SOURCE_PATHS = (
    Path(__file__),
    Path(_v1.__file__).resolve(),
    Path(__file__).parents[1] / "core" / "initializers.py",
    Path(__file__).parents[1] / "core" / "multi_head_learner.py",
    Path(__file__).parents[1] / "core" / "normalizers.py",
    Path(__file__).parents[1] / "core" / "optimizers.py",
    Path(__file__).parents[1] / "core" / "prototype_agent.py",
    Path(__file__).parents[1] / "core" / "partner_policy_fusion.py",
    Path(__file__).parents[1] / "core" / "oak.py",
    Path(__file__).parents[1] / "core" / "options.py",
    Path(__file__).parents[1] / "core" / "state_builder.py",
    Path(__file__).parents[1] / "core" / "types.py",
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


def _runtime_manifest() -> dict[str, object]:
    return {
        **_v1._runtime_manifest(),
        "numpy": str(np.__version__),
    }


class PrototypePartnerFusionMatchedV2DevelopmentEvaluator:
    """Strict host orchestrator for matched initialization and causal rollout."""

    def __init__(self, config: MatchedV2Config = CONFIG) -> None:
        if config != CONFIG:
            raise ValueError("the matched-initialization v2 development protocol is frozen")
        self.config = config
        self.agent = _AGENT
        self.config_digest = _digest(config.to_dict())
        key = jr.key(config.initialization_seed)
        self.initialization_key_payload = _typed_key_payload(key)
        self.initialization_key_digest = _digest(self.initialization_key_payload)
        self.protocol_digest = _digest(
            {
                "schema": SCHEMA,
                "namespace": PROTOCOL_NAMESPACE,
                "config": config.to_dict(),
                "agent_config": self.agent.to_config(),
                "typed_initialization_key": self.initialization_key_payload,
                "shared_lifecycle_words": _words_tuple(_shared_lifecycle_words()),
                "declared_interventions": DECLARED_INTERVENTIONS,
                "pairing_scope": PAIRING_SCOPE,
                "identical_post_divergence_inputs": IDENTICAL_POST_DIVERGENCE_INPUTS,
                "schedule": [event.to_dict() for event in EXOGENOUS_SCHEDULE],
            }
        )

    def _empty_batch(self) -> PartnerMessageBatch:
        fusion = self.agent.partner_policy_fusion
        if fusion is None:  # pragma: no cover - construction invariant
            raise RuntimeError("matched v2 evaluator requires partner fusion")
        return fusion.empty_messages()

    def _messages(
        self,
        environment: MatchedV2EnvironmentState,
        event: MatchedV2ExogenousEvent,
        *,
        decision_words: Array,
        event_words: Array,
        observation_id: int,
        context_id: int,
        empty: bool,
    ) -> tuple[PartnerMessageBatch, tuple[int, int], tuple[int, int]]:
        suggestions = _message_suggestions(environment, event)
        # Common provenance is part of the matched intervention.  Unlike v1,
        # the condition label never leaks into the message payload.
        provenance: tuple[int, int] = (1_000, 1_001)
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

    def _initial_arm(self, condition: Condition) -> MatchedV2ArmState:
        event = EXOGENOUS_SCHEDULE[0]
        observation = _initial_observation(event)
        # The key and lifecycle are intentionally identical.  Arm ownership
        # lives only in the receipts outside the Prototype PyTree.
        prototype_state = self.agent.start(
            self.agent.init(
                jr.key(self.config.initialization_seed),
                lifecycle_id=_shared_lifecycle_words(),
            ),
            jnp.asarray(observation, dtype=jnp.float32),
        )
        prototype_state = _canonicalize_prototype_host_timing_metadata(prototype_state)
        environment = MatchedV2EnvironmentState(
            clock=0,
            observation=observation,
            history_score=0.0,
            last_action=-1,
            last_net_reward=0.0,
        )
        mask = _hard_mask_receipt(condition, event)
        initial_receipt = _seal_pending(
            MatchedV2PendingDecisionReceipt(
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
                message_batch_digest=_digest(_message_batch_payload(self._empty_batch())),
                decision_environment_digest=_digest(environment.to_dict()),
                hard_mask_receipt=mask,
                receipt_digest="",
            )
        )
        evaluator_owner = _owner_digest(condition, "evaluator")
        trace_head = _digest(
            {
                "namespace": PROTOCOL_NAMESPACE,
                "condition": condition,
                "owner": evaluator_owner,
                "matched_initialization_key_digest": self.initialization_key_digest,
                "initial_environment": environment.to_dict(),
                "initial_prototype_state": _tree_digest(prototype_state),
            }
        )
        arm = MatchedV2ArmState(
            condition=condition,
            evaluator_owner_digest=evaluator_owner,
            prototype_owner_digest=_owner_digest(condition, "prototype_agent"),
            fusion_owner_digest=_owner_digest(condition, "partner_fusion"),
            environment_owner_digest=_owner_digest(condition, "environment"),
            initialization_key_digest=self.initialization_key_digest,
            environment=environment,
            prototype_state=prototype_state,
            pending_decision=initial_receipt,
            trace=(),
            trace_head=trace_head,
            prototype_update_calls=0,
            fusion_decision_calls=0,
            fusion_feedback_opportunities=0,
            state_seal="",
        )
        return _seal_arm(arm)

    def initial_state(self) -> MatchedV2RunState:
        state = MatchedV2RunState(
            schema=SCHEMA,
            namespace=PROTOCOL_NAMESPACE,
            protocol_digest=self.protocol_digest,
            event_clock=0,
            arms=tuple(self._initial_arm(condition) for condition in CONDITIONS),
            run_seal="",
        )
        return _seal_run(state)

    def initialization_receipt(self, state: MatchedV2RunState) -> dict[str, object]:
        """Describe exact initialization matching without conflating ownership."""

        if state.event_clock != 0 or not self._structure_valid(state):
            raise ValueError("initialization receipt requires a valid event-zero state")
        prototype_digests = [_tree_digest(arm.prototype_state) for arm in state.arms]
        fusion_digests: list[str] = []
        environment_digests = [_digest(arm.environment.to_dict()) for arm in state.arms]
        for arm in state.arms:
            wrapper = cast(PrototypeInteractionState, arm.prototype_state.ia_state)
            fusion_digests.append(_tree_digest(wrapper.partner_policy_fusion_state))
        key_digests = [arm.initialization_key_digest for arm in state.arms]
        return {
            "typed_rng_key": self.initialization_key_payload,
            "typed_rng_key_digest": self.initialization_key_digest,
            "per_condition_typed_rng_key_digests": dict(zip(CONDITIONS, key_digests, strict=True)),
            "per_condition_prototype_state_digests": dict(
                zip(CONDITIONS, prototype_digests, strict=True)
            ),
            "per_condition_fusion_state_digests": dict(
                zip(CONDITIONS, fusion_digests, strict=True)
            ),
            "per_condition_environment_state_digests": dict(
                zip(CONDITIONS, environment_digests, strict=True)
            ),
            "typed_rng_keys_bit_identical": len(set(key_digests)) == 1,
            "prototype_states_bit_identical": len(set(prototype_digests)) == 1,
            "fusion_states_bit_identical": len(set(fusion_digests)) == 1,
            "environment_states_bit_identical": len(set(environment_digests)) == 1,
            "ownership_metadata_is_outside_matched_dynamic_state": True,
        }

    def _validate_mask_receipt(
        self,
        receipt: MatchedV2HardMaskAuthorityReceipt,
        condition: Condition,
        event: MatchedV2ExogenousEvent,
    ) -> bool:
        return (
            receipt.owner_digest == _owner_digest(condition, "caller_hard_mask")
            and receipt.condition == condition
            and receipt.execution_clock == event.index
            and receipt.mask == event.hard_action_mask
            and receipt.source_event_digest == _digest(event.to_dict())
            and receipt.receipt_digest == _digest(receipt.body())
        )

    def _validate_pending(
        self,
        arm: MatchedV2ArmState,
        receipt: MatchedV2PendingDecisionReceipt,
        event_clock: int,
    ) -> bool:
        event = EXOGENOUS_SCHEDULE[event_clock]
        action = int(arm.prototype_state.current_action)
        base_action = receipt.counterfactual_base_action
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
            and 0 <= base_action < N_ACTIONS
            and receipt.hard_mask_receipt.mask[action]
            and receipt.hard_mask_receipt.mask[base_action]
            and self._validate_mask_receipt(receipt.hard_mask_receipt, arm.condition, event)
            and receipt.decision_environment_digest == _digest(arm.environment.to_dict())
            and math.isfinite(receipt.quoted_communication_cost)
            and 0.0 <= receipt.quoted_communication_cost <= _float32(self.config.cost_spike)
            and math.isfinite(receipt.charged_communication_cost)
            and 0.0 <= receipt.charged_communication_cost <= _float32(self.config.cost_spike)
            and receipt.receipt_digest == _digest(receipt.body())
        )

    def _trace_chain_valid(self, arm: MatchedV2ArmState) -> bool:
        initial = self._initial_arm(arm.condition)
        expected = _digest(
            {
                "namespace": PROTOCOL_NAMESPACE,
                "condition": arm.condition,
                "owner": arm.evaluator_owner_digest,
                "matched_initialization_key_digest": self.initialization_key_digest,
                "initial_environment": initial.environment.to_dict(),
                "initial_prototype_state": _tree_digest(initial.prototype_state),
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

    def _arm_structure_valid(self, arm: MatchedV2ArmState, event_clock: int) -> bool:
        try:
            valid = (
                arm.condition in CONDITIONS
                and arm.evaluator_owner_digest == _owner_digest(arm.condition, "evaluator")
                and arm.prototype_owner_digest == _owner_digest(arm.condition, "prototype_agent")
                and arm.fusion_owner_digest == _owner_digest(arm.condition, "partner_fusion")
                and arm.environment_owner_digest == _owner_digest(arm.condition, "environment")
                and arm.initialization_key_digest == self.initialization_key_digest
                and arm.environment.clock == event_clock
                and len(arm.trace) == event_clock
                and arm.prototype_update_calls == 2 * event_clock
                and arm.fusion_decision_calls == 2 * event_clock
                and arm.fusion_feedback_opportunities == 2 * event_clock
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

    def _structure_valid(self, state: MatchedV2RunState) -> bool:
        try:
            arm_conditions = tuple(arm.condition for arm in state.arms)
            unique_owners = all(
                len({getattr(arm, field) for arm in state.arms}) == len(CONDITIONS)
                for field in (
                    "evaluator_owner_digest",
                    "prototype_owner_digest",
                    "fusion_owner_digest",
                    "environment_owner_digest",
                )
            )
            initial_dynamic_match = True
            if state.event_clock == 0:
                initial_dynamic_match = (
                    len({_tree_digest(arm.prototype_state) for arm in state.arms}) == 1
                    and len({_digest(arm.environment.to_dict()) for arm in state.arms}) == 1
                )
            return bool(
                state.schema == SCHEMA
                and state.namespace == PROTOCOL_NAMESPACE
                and state.protocol_digest == self.protocol_digest
                and type(state.event_clock) is int
                and 0 <= state.event_clock <= self.config.horizon
                and len(state.arms) == len(CONDITIONS)
                and arm_conditions == CONDITIONS
                and unique_owners
                and initial_dynamic_match
                and len({arm.initialization_key_digest for arm in state.arms}) == 1
                and all(self._arm_structure_valid(arm, state.event_clock) for arm in state.arms)
                and state.run_seal == _digest(_run_body(state))
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    def _same_state(self, left: MatchedV2RunState, right: MatchedV2RunState) -> bool:
        try:
            return _canonical_json_bytes(
                {**_run_body(left), "run_seal": left.run_seal}
            ) == _canonical_json_bytes({**_run_body(right), "run_seal": right.run_seal})
        except (TypeError, ValueError, OverflowError):
            return False

    def validate_state(self, state: object, *, reconstruct: bool = True) -> bool:
        """Reject structural, numeric, owner, chain, and causal-prefix drift."""

        if not isinstance(state, MatchedV2RunState) or not self._structure_valid(state):
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
        arm: MatchedV2ArmState,
        pending: MatchedV2PendingDecisionReceipt,
        realized_assistance: float,
    ) -> PrototypePartnerPolicyFusionFeedback | None:
        if arm.condition == EMPTY_MESSAGE_BASE_ONLY or not pending.feedback_armed:
            return None
        target = realized_assistance if arm.condition == LEARNED_FEEDBACK else 0.0
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
        arm: MatchedV2ArmState,
        next_environment: MatchedV2EnvironmentState,
        next_event: MatchedV2ExogenousEvent,
        *,
        final_bootstrap: bool,
    ) -> tuple[
        PrototypePartnerPolicyFusionInput,
        PartnerMessageBatch,
        tuple[int, int],
        tuple[int, int],
        MatchedV2HardMaskAuthorityReceipt,
    ]:
        prototype_state = arm.prototype_state
        decision_words = _increment_words(prototype_state.step_words)
        event_words = _increment_words(prototype_state.observation_event_words)
        observation_id = 10_000 + next_event.index
        context_id = next_event.context_bit
        empty = arm.condition == EMPTY_MESSAGE_BASE_ONLY or final_bootstrap
        messages, suggestions, provenance = self._messages(
            next_environment,
            next_event,
            decision_words=decision_words,
            event_words=event_words,
            observation_id=observation_id,
            context_id=context_id,
            empty=empty,
        )
        receipt = _hard_mask_receipt(arm.condition, next_event)
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
        arm: MatchedV2ArmState,
        result: PrototypeUpdateResult,
        next_environment: MatchedV2EnvironmentState,
        next_event: MatchedV2ExogenousEvent,
        messages: PartnerMessageBatch,
        mask_receipt: MatchedV2HardMaskAuthorityReceipt,
    ) -> MatchedV2PendingDecisionReceipt:
        diagnostics = result.partner_policy_fusion_diagnostics
        if diagnostics is None:  # pragma: no cover - construction invariant
            raise RuntimeError("Prototype omitted configured fusion diagnostics")
        decision = diagnostics.decision
        selected = int(decision.selected_partner_id)
        route = int(decision.route)
        quoted = next_event.communication_cost[selected] if 0 <= selected < MAX_PARTNERS else 0.0
        charged = quoted if route != ROUTE_IGNORE else 0.0
        return _seal_pending(
            MatchedV2PendingDecisionReceipt(
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
        arm: MatchedV2ArmState,
        event_clock: int,
        *,
        pending: MatchedV2PendingDecisionReceipt,
        compiled: bool,
    ) -> MatchedV2ArmState:
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
        # Preview is a pure candidate used only to obtain OaK's exact base
        # action.  It receives the same feedback opportunity as the commit so
        # every arm has a matched logical call graph; its state is discarded.
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
        )
        if not mask_receipt.mask[preview_base_action]:
            raise RuntimeError("paired caller mask rejected the counterfactual base action")
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
            if arm.condition == LEARNED_FEEDBACK and pending.feedback_armed
            else 0.0
        )
        if arm.condition == LEARNED_FEEDBACK:
            feedback_kind = "own_realized_assistance"
        elif arm.condition == FIXED_ZERO_FEEDBACK:
            feedback_kind = "fixed_zero_outcome_blind"
        else:
            feedback_kind = "unavailable_empty_message_base_only"
        action_changed = bool(
            pending.feedback_armed
            and pending.selected_partner_id >= 0
            and pending.effective_action != pending.counterfactual_base_action
        )
        record = MatchedV2TraceRecord(
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
            action_changed_by_assistance=action_changed,
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
        next_arm = MatchedV2ArmState(
            condition=arm.condition,
            evaluator_owner_digest=arm.evaluator_owner_digest,
            prototype_owner_digest=arm.prototype_owner_digest,
            fusion_owner_digest=arm.fusion_owner_digest,
            environment_owner_digest=arm.environment_owner_digest,
            initialization_key_digest=arm.initialization_key_digest,
            environment=next_environment,
            prototype_state=result.state,
            pending_decision=pending_next,
            trace=(*arm.trace, record),
            trace_head=record.record_hash,
            prototype_update_calls=arm.prototype_update_calls + 2,
            fusion_decision_calls=arm.fusion_decision_calls + 2,
            fusion_feedback_opportunities=arm.fusion_feedback_opportunities + 2,
            state_seal="",
        )
        return _seal_arm(next_arm)

    def _advance_unchecked(
        self,
        state: MatchedV2RunState,
        *,
        receipt_overrides: Mapping[Condition, MatchedV2PendingDecisionReceipt] | None = None,
        compiled: bool = True,
    ) -> MatchedV2RunState:
        if state.event_clock >= self.config.horizon:
            raise ValueError("matched v2 development life is complete")
        overrides = {} if receipt_overrides is None else dict(receipt_overrides)
        unknown = set(overrides).difference(CONDITIONS)
        if unknown:
            raise ValueError("receipt override names an unknown condition")
        chosen: dict[Condition, MatchedV2PendingDecisionReceipt] = {}
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
        return _seal_run(
            MatchedV2RunState(
                schema=state.schema,
                namespace=state.namespace,
                protocol_digest=state.protocol_digest,
                event_clock=state.event_clock + 1,
                arms=next_arms,
                run_seal="",
            )
        )

    def step(
        self,
        state: MatchedV2RunState,
        *,
        receipt_overrides: Mapping[Condition, MatchedV2PendingDecisionReceipt] | None = None,
    ) -> MatchedV2RunState:
        """Advance atomically after exact prefix and owner-receipt validation."""

        if not self.validate_state(state):
            raise ValueError("invalid or noncausal matched v2 run state")
        return self._advance_unchecked(state, receipt_overrides=receipt_overrides, compiled=True)

    def _reconstruct_unchecked(self, event_count: int) -> MatchedV2RunState:
        if type(event_count) is not int or not 0 <= event_count <= self.config.horizon:
            raise ValueError("event_count is outside the frozen development life")
        state = self.initial_state()
        for _ in range(event_count):
            state = self._advance_unchecked(state, compiled=True)
        return state

    def reconstruct(self, event_count: int) -> MatchedV2RunState:
        """Return an exact source/config/runtime-compatible causal prefix."""

        state = self._reconstruct_unchecked(event_count)
        if not self.validate_state(state, reconstruct=False):
            raise RuntimeError("internally reconstructed prefix failed validation")
        return state

    def run_to_end(self, state: MatchedV2RunState | None = None) -> MatchedV2RunState:
        """Finish the paired schedule from a validated in-memory prefix."""

        current = self.initial_state() if state is None else state
        if not self.validate_state(current):
            raise ValueError("run_to_end requires an exact causal prefix")
        while current.event_clock < self.config.horizon:
            current = self._advance_unchecked(current, compiled=True)
        if not self.validate_state(current, reconstruct=False):
            raise RuntimeError("completed development state is invalid")
        return current

    def eager_jit_parity(self) -> bool:
        """Check exact eager/JIT parity at the real Prototype boundary."""

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

    def resource_report(self, state: MatchedV2RunState) -> dict[str, object]:
        """Return exact matched logical work and initial-state allocations."""

        if not self.validate_state(state, reconstruct=False):
            raise ValueError("resource report requires a valid run state")
        fusion = self.agent.partner_policy_fusion
        if fusion is None:  # pragma: no cover - construction invariant
            raise RuntimeError("fusion resource budget is unavailable")
        initial = self.initial_state()
        entries: dict[str, object] = {}
        for initial_arm, arm in zip(initial.arms, state.arms, strict=True):
            entries[arm.condition] = {
                "prototype_state_bytes_initial": _tree_nbytes(initial_arm.prototype_state),
                "prototype_state_bytes_final": _tree_nbytes(arm.prototype_state),
                "fusion_state_bytes_initial": _tree_nbytes(
                    cast(
                        PrototypeInteractionState,
                        initial_arm.prototype_state.ia_state,
                    ).partner_policy_fusion_state
                ),
                "prototype_update_calls": arm.prototype_update_calls,
                "fusion_decision_calls": arm.fusion_decision_calls,
                "fusion_feedback_opportunities": arm.fusion_feedback_opportunities,
                "fixed_message_slots_per_call": MAX_PARTNERS,
                "paired_exogenous_events_read": state.event_clock,
                "committed_prototype_transitions": state.event_clock,
                "discarded_base_action_previews": state.event_clock,
                "shared_mutable_agent_state": False,
                "shared_mutable_environment_state": False,
            }
        comparable = list(entries.values())
        initialization = self.initialization_receipt(initial)
        return {
            "matched_logical_budgets": all(value == comparable[0] for value in comparable[1:]),
            "initial_typed_rng_keys_bit_identical": initialization["typed_rng_keys_bit_identical"],
            "initial_learner_states_bit_identical": initialization[
                "prototype_states_bit_identical"
            ],
            "runtime_evaluator_rng_draws": 0,
            "pairing_scope": PAIRING_SCOPE,
            "identical_post_divergence_inputs": IDENTICAL_POST_DIVERGENCE_INPUTS,
            "fusion_resource_budget": fusion.resource_budget.to_config(),
            "per_condition": entries,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedV2Checkpoint:
    """In-memory-only checkpoint bound to exact source, runtime, and config."""

    schema: str
    namespace: str
    protocol_digest: str
    config_digest: str
    next_event: int
    source_manifest: dict[str, str]
    runtime_manifest: dict[str, object]
    state: MatchedV2RunState
    state_digest: str
    checkpoint_digest: str

    def metadata(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "namespace": self.namespace,
            "protocol_digest": self.protocol_digest,
            "config_digest": self.config_digest,
            "next_event": self.next_event,
            "source_manifest": self.source_manifest,
            "runtime_manifest": self.runtime_manifest,
            "state_digest": self.state_digest,
        }


def make_prototype_partner_fusion_matched_v2_checkpoint(
    next_event: int,
) -> MatchedV2Checkpoint:
    """Build an in-memory v2 checkpoint after exactly ``next_event`` events."""

    evaluator = PrototypePartnerFusionMatchedV2DevelopmentEvaluator()
    state = evaluator.reconstruct(next_event)
    state_digest = _digest({**_run_body(state), "run_seal": state.run_seal})
    provisional = MatchedV2Checkpoint(
        schema=CHECKPOINT_SCHEMA,
        namespace=PROTOCOL_NAMESPACE,
        protocol_digest=evaluator.protocol_digest,
        config_digest=evaluator.config_digest,
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
    if type(checkpoint) is not MatchedV2Checkpoint:
        return ("checkpoint has the wrong type",)
    evaluator = PrototypePartnerFusionMatchedV2DevelopmentEvaluator()
    errors: list[str] = []
    if type(checkpoint.schema) is not str or checkpoint.schema != CHECKPOINT_SCHEMA:
        errors.append("checkpoint schema changed")
    if type(checkpoint.namespace) is not str or checkpoint.namespace != PROTOCOL_NAMESPACE:
        errors.append("checkpoint namespace changed")
    if (
        type(checkpoint.protocol_digest) is not str
        or checkpoint.protocol_digest != evaluator.protocol_digest
    ):
        errors.append("checkpoint protocol digest changed")
    if (
        type(checkpoint.config_digest) is not str
        or checkpoint.config_digest != evaluator.config_digest
    ):
        errors.append("checkpoint config digest changed")
    try:
        current_source_manifest = _source_manifest()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"current source manifest is unavailable: {type(exc).__name__}: {exc}")
    else:
        try:
            if _canonical_json_bytes(checkpoint.source_manifest) != _canonical_json_bytes(
                current_source_manifest
            ):
                errors.append("checkpoint source manifest changed")
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"checkpoint source manifest cannot be compared: {exc}")
    try:
        current_runtime_manifest = _runtime_manifest()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"current runtime manifest is unavailable: {type(exc).__name__}: {exc}")
    else:
        try:
            if _canonical_json_bytes(checkpoint.runtime_manifest) != _canonical_json_bytes(
                current_runtime_manifest
            ):
                errors.append("checkpoint runtime manifest changed")
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"checkpoint runtime manifest cannot be compared: {exc}")
    if type(checkpoint.next_event) is not int or not (0 <= checkpoint.next_event <= CONFIG.horizon):
        errors.append("checkpoint next_event is invalid")
        return tuple(errors)
    if type(checkpoint.state) is not MatchedV2RunState:
        errors.append("checkpoint state has the wrong type")
        return tuple(errors)
    try:
        actual_state_digest = _digest(
            {**_run_body(checkpoint.state), "run_seal": checkpoint.state.run_seal}
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"checkpoint state cannot be encoded: {exc}")
        return tuple(errors)
    if type(checkpoint.state_digest) is not str or checkpoint.state_digest != actual_state_digest:
        errors.append("checkpoint state digest mismatch")
    try:
        expected_checkpoint_digest = _digest(checkpoint.metadata())
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"checkpoint metadata cannot be encoded: {exc}")
    else:
        if (
            type(checkpoint.checkpoint_digest) is not str
            or checkpoint.checkpoint_digest != expected_checkpoint_digest
        ):
            errors.append("checkpoint digest mismatch")
    if checkpoint.state.event_clock != checkpoint.next_event:
        errors.append("checkpoint state clock mismatch")
    if not evaluator.validate_state(checkpoint.state):
        errors.append("checkpoint differs from exact causal prefix reconstruction")
    return tuple(errors)


def _arm_summary(arm: MatchedV2ArmState) -> dict[str, object]:
    records = list(arm.trace)
    rewards = [record.net_reward for record in records]
    pre = [
        record.net_reward
        for record in records
        if not bool(record.exogenous_event["after_reversal"])
    ]
    post = [
        record.net_reward for record in records if bool(record.exogenous_event["after_reversal"])
    ]

    def mean(values: list[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0

    return {
        "condition": arm.condition,
        "event_count": len(records),
        "action_changing_assistance_count": sum(
            record.action_changed_by_assistance for record in records
        ),
        "partner_influenced_next_decision_count": sum(
            record.next_partner_influenced for record in records
        ),
        "realized_task_return": float(
            np.sum(np.asarray([record.task_reward for record in records], dtype=np.float64))
        ),
        "realized_net_return": float(
            np.sum(np.asarray([record.net_reward for record in records], dtype=np.float64))
        ),
        "realized_communication_cost": float(
            np.sum(
                np.asarray(
                    [record.charged_communication_cost for record in records],
                    dtype=np.float64,
                )
            )
        ),
        "realized_assistance_value_total": float(
            np.sum(
                np.asarray(
                    [record.realized_assistance_value for record in records],
                    dtype=np.float64,
                )
            )
        ),
        "mean_net_reward": mean(rewards),
        "pre_reversal_mean_net_reward": mean(pre),
        "post_reversal_mean_net_reward": mean(post),
        "query_route_count": sum(record.next_route == ROUTE_QUERY for record in records),
        "feedback_applied_count": sum(record.feedback_applied for record in records),
        "correct_action_count": sum(
            record.executed_action == record.hidden_correct_action for record in records
        ),
        "hard_mask_violation_count": sum(
            not record.next_action_allowed_by_caller for record in records
        ),
        "trace_head": arm.trace_head,
        "final_environment_digest": _digest(arm.environment.to_dict()),
        "final_prototype_state_digest": _tree_digest(arm.prototype_state),
    }


def _claim_scope() -> str:
    return (
        "matched-initialization causal mechanism diagnostic only; descriptive action changes, "
        "realized returns, and communication costs are not an intelligence-amplification, "
        "confidence-calibration, safety-efficacy, or Alberta Plan completion claim"
    )


def _assemble_report(
    evaluator: PrototypePartnerFusionMatchedV2DevelopmentEvaluator,
    state: MatchedV2RunState,
    *,
    eager_jit_parity: bool | None = None,
) -> dict[str, object]:
    initial = evaluator.initial_state()
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
        "claim_scope": _claim_scope(),
        "pairing_scope": PAIRING_SCOPE,
        "identical_post_divergence_inputs": IDENTICAL_POST_DIVERGENCE_INPUTS,
        "post_divergence_rule": (
            "only exogenous schedule values stay paired; each executed action owns its next "
            "observation, reward, message suggestions, feedback, and subsequent trajectory"
        ),
        "declared_interventions": dict(DECLARED_INTERVENTIONS),
        "config": CONFIG.to_dict(),
        "config_digest": evaluator.config_digest,
        "agent_config": evaluator.agent.to_config(),
        "protocol_digest": evaluator.protocol_digest,
        "schedule_digest": _digest([event.to_dict() for event in EXOGENOUS_SCHEDULE]),
        "initialization_receipt": evaluator.initialization_receipt(initial),
        "source_manifest": _source_manifest(),
        "runtime_manifest": _runtime_manifest(),
        "prototype_eager_jit_parity": (
            evaluator.eager_jit_parity() if eager_jit_parity is None else eager_jit_parity
        ),
        "compiled_scope": "real PrototypeAgent.update_transition boundary only",
        "resource_report": evaluator.resource_report(state),
        "arm_owner_receipts": {
            arm.condition: {
                "evaluator_owner_digest": arm.evaluator_owner_digest,
                "prototype_owner_digest": arm.prototype_owner_digest,
                "fusion_owner_digest": arm.fusion_owner_digest,
                "environment_owner_digest": arm.environment_owner_digest,
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
    evaluator = PrototypePartnerFusionMatchedV2DevelopmentEvaluator()
    return _assemble_report(evaluator, evaluator.run_to_end())


@functools.lru_cache(maxsize=1)
def _replay_expected_report_condition_by_condition() -> dict[str, object]:
    """Replay each arm independently to bound peak validation memory."""

    evaluator = PrototypePartnerFusionMatchedV2DevelopmentEvaluator()
    initial_state = evaluator.initial_state()
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
            "environment_owner_digest": arm.environment_owner_digest,
            "hard_mask_owner_digest": arm.pending_decision.hard_mask_receipt.owner_digest,
        }
        wrapper = cast(PrototypeInteractionState, initial.prototype_state.ia_state)
        resources[condition] = {
            "prototype_state_bytes_initial": _tree_nbytes(initial.prototype_state),
            "prototype_state_bytes_final": _tree_nbytes(arm.prototype_state),
            "fusion_state_bytes_initial": _tree_nbytes(wrapper.partner_policy_fusion_state),
            "prototype_update_calls": arm.prototype_update_calls,
            "fusion_decision_calls": arm.fusion_decision_calls,
            "fusion_feedback_opportunities": arm.fusion_feedback_opportunities,
            "fixed_message_slots_per_call": MAX_PARTNERS,
            "paired_exogenous_events_read": CONFIG.horizon,
            "committed_prototype_transitions": CONFIG.horizon,
            "discarded_base_action_previews": CONFIG.horizon,
            "shared_mutable_agent_state": False,
            "shared_mutable_environment_state": False,
        }
        arm_payloads.append({**_arm_body(arm), "state_seal": arm.state_seal})
    comparable = list(resources.values())
    fusion = evaluator.agent.partner_policy_fusion
    if fusion is None:  # pragma: no cover - construction invariant
        raise RuntimeError("fusion resource budget is unavailable")
    initialization = evaluator.initialization_receipt(initial_state)
    resource_report = {
        "matched_logical_budgets": all(value == comparable[0] for value in comparable[1:]),
        "initial_typed_rng_keys_bit_identical": initialization["typed_rng_keys_bit_identical"],
        "initial_learner_states_bit_identical": initialization["prototype_states_bit_identical"],
        "runtime_evaluator_rng_draws": 0,
        "pairing_scope": PAIRING_SCOPE,
        "identical_post_divergence_inputs": IDENTICAL_POST_DIVERGENCE_INPUTS,
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
        "claim_scope": _claim_scope(),
        "pairing_scope": PAIRING_SCOPE,
        "identical_post_divergence_inputs": IDENTICAL_POST_DIVERGENCE_INPUTS,
        "post_divergence_rule": (
            "only exogenous schedule values stay paired; each executed action owns its next "
            "observation, reward, message suggestions, feedback, and subsequent trajectory"
        ),
        "declared_interventions": dict(DECLARED_INTERVENTIONS),
        "config": CONFIG.to_dict(),
        "config_digest": evaluator.config_digest,
        "agent_config": evaluator.agent.to_config(),
        "protocol_digest": evaluator.protocol_digest,
        "schedule_digest": _digest([event.to_dict() for event in EXOGENOUS_SCHEDULE]),
        "initialization_receipt": initialization,
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


def run_prototype_partner_fusion_matched_v2_development() -> dict[str, object]:
    """Run and strictly validate the nonpromoting matched v2 diagnostic."""

    report = _run_unvalidated()
    errors = validate_prototype_partner_fusion_matched_v2_report(report)
    if errors:
        raise RuntimeError("invalid matched v2 partner report: " + "; ".join(errors))
    return report


def validate_prototype_partner_fusion_matched_v2_report(
    report: object,
) -> tuple[str, ...]:
    """Reject metadata, digest, raw-chain, or exact causal-replay drift."""

    if not isinstance(report, Mapping):
        return ("report must be a mapping",)
    try:
        candidate = dict(report)
    except (KeyError, TypeError, ValueError) as exc:
        return (f"report mapping cannot be normalized: {exc}",)
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
        "pairing_scope",
        "identical_post_divergence_inputs",
        "post_divergence_rule",
        "declared_interventions",
        "config",
        "config_digest",
        "agent_config",
        "protocol_digest",
        "schedule_digest",
        "initialization_receipt",
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
    if set(candidate) != expected_fields:
        return ("report fields do not match the matched v2 schema",)
    errors: list[str] = []
    if type(candidate.get("schema")) is not str or candidate.get("schema") != SCHEMA:
        errors.append("report schema changed")
    if (
        type(candidate.get("namespace")) is not str
        or candidate.get("namespace") != PROTOCOL_NAMESPACE
    ):
        errors.append("report namespace changed")
    if (
        type(candidate.get("assessment")) is not str
        or candidate.get("assessment") != ASSESSMENT
    ):
        errors.append("assessment must remain not_assessed")
    if (
        type(candidate.get("evidence_level")) is not str
        or candidate.get("evidence_level") != EVIDENCE_LEVEL
    ):
        errors.append("evidence level changed")
    if candidate.get("development_only") is not True:
        errors.append("development_only must remain true")
    if candidate.get("development_protocol_consumed") is not True:
        errors.append("development protocol must remain consumed")
    for forbidden in (
        "scientific_promotion_allowed",
        "output_writes_allowed",
        "thresholds_defined",
        "winner_declared",
        "identical_post_divergence_inputs",
    ):
        if candidate.get(forbidden) is not False:
            errors.append(f"{forbidden} must remain false")
    if (
        type(candidate.get("pairing_scope")) is not str
        or candidate.get("pairing_scope") != PAIRING_SCOPE
    ):
        errors.append("pairing scope changed")
    if candidate.get("prototype_eager_jit_parity") is not True:
        errors.append("Prototype eager/JIT parity did not hold")
    evaluator = PrototypePartnerFusionMatchedV2DevelopmentEvaluator()
    binding_expectations: tuple[tuple[str, object, str], ...] = (
        ("config", CONFIG.to_dict(), "report config binding changed"),
        (
            "config_digest",
            evaluator.config_digest,
            "report config digest binding changed",
        ),
        (
            "agent_config",
            evaluator.agent.to_config(),
            "report agent config binding changed",
        ),
        (
            "protocol_digest",
            evaluator.protocol_digest,
            "report protocol digest binding changed",
        ),
        (
            "schedule_digest",
            _digest([event.to_dict() for event in EXOGENOUS_SCHEDULE]),
            "report schedule digest binding changed",
        ),
    )
    for field, expected, message in binding_expectations:
        try:
            if _canonical_json_bytes(candidate.get(field)) != _canonical_json_bytes(expected):
                errors.append(message)
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"{field} binding cannot be compared: {exc}")
    try:
        current_source_manifest = _source_manifest()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"current source manifest is unavailable: {type(exc).__name__}: {exc}")
    else:
        try:
            if _canonical_json_bytes(candidate.get("source_manifest")) != _canonical_json_bytes(
                current_source_manifest
            ):
                errors.append("report source manifest changed")
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"source manifest binding cannot be compared: {exc}")
    try:
        current_runtime_manifest = _runtime_manifest()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"current runtime manifest is unavailable: {type(exc).__name__}: {exc}")
    else:
        try:
            if _canonical_json_bytes(candidate.get("runtime_manifest")) != _canonical_json_bytes(
                current_runtime_manifest
            ):
                errors.append("report runtime manifest changed")
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"runtime manifest binding cannot be compared: {exc}")
    unsigned = dict(candidate)
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
        expected = _replay_expected_report_condition_by_condition()
    # This deliberately broad catch is confined to the public validation
    # boundary. Normal evaluator execution propagates programming errors; an
    # unavailable replay authority instead makes an untrusted report invalid.
    except Exception as exc:  # pragma: no cover - exercised through monkeypatching
        errors.append(f"exact causal replay failed: {type(exc).__name__}: {exc}")
        return tuple(errors)
    try:
        if _canonical_json_bytes(candidate) != _canonical_json_bytes(expected):
            errors.append("report differs from exact causal replay")
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"report comparison failed closed: {exc}")
    return tuple(errors)


def resume_prototype_partner_fusion_matched_v2_checkpoint(
    checkpoint: object,
) -> dict[str, object]:
    """Validate, reconstruct, resume, and report an in-memory v2 prefix."""

    errors = _validate_checkpoint(checkpoint)
    if errors:
        raise ValueError("invalid matched v2 checkpoint: " + "; ".join(errors))
    typed = cast(MatchedV2Checkpoint, checkpoint)
    evaluator = PrototypePartnerFusionMatchedV2DevelopmentEvaluator()
    final_state = evaluator.run_to_end(typed.state)
    report = _assemble_report(evaluator, final_state)
    validation = validate_prototype_partner_fusion_matched_v2_report(report)
    if validation:
        raise RuntimeError("resumed matched v2 report failed validation: " + "; ".join(validation))
    return report


__all__ = [
    "ASSESSMENT",
    "CHECKPOINT_SCHEMA",
    "CONDITIONS",
    "CONFIG",
    "DECLARED_INTERVENTIONS",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_PROTOCOL_CONSUMED",
    "EMPTY_MESSAGE_BASE_ONLY",
    "EVIDENCE_LEVEL",
    "EXOGENOUS_SCHEDULE",
    "FIXED_ZERO_FEEDBACK",
    "IDENTICAL_POST_DIVERGENCE_INPUTS",
    "LEARNED_FEEDBACK",
    "OUTPUT_WRITES_ALLOWED",
    "PAIRING_SCOPE",
    "PROTOCOL_NAMESPACE",
    "SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "THRESHOLDS_DEFINED",
    "WINNER_DECLARED",
    "MatchedV2ArmState",
    "MatchedV2Checkpoint",
    "MatchedV2Config",
    "MatchedV2EnvironmentState",
    "MatchedV2ExogenousEvent",
    "MatchedV2HardMaskAuthorityReceipt",
    "MatchedV2PendingDecisionReceipt",
    "MatchedV2RunState",
    "MatchedV2TraceRecord",
    "PrototypePartnerFusionMatchedV2DevelopmentEvaluator",
    "build_matched_v2_exogenous_schedule",
    "make_prototype_partner_fusion_matched_v2_checkpoint",
    "resume_prototype_partner_fusion_matched_v2_checkpoint",
    "run_prototype_partner_fusion_matched_v2_development",
    "validate_prototype_partner_fusion_matched_v2_report",
]
