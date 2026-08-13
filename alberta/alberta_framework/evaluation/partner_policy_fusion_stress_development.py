# mypy: disable-error-code="call-arg"
"""Strict development stress lane for contextual partner-policy fusion.

This module exercises the action-changing :class:`PartnerPolicyFusion` core on
one evaluator-owned, uninterrupted stream with two partners.  Partner utility
depends on observable context and reverses halfway through the life.  The same
stream also contains partner-specific disconnects, total communication
failures, communication-cost spikes, and hard-mask exclusions.

Three arms start from the same immutable zero snapshot and invoke the same
fixed-shape decision and feedback kernels once per event:

``learned_fusion``
    Applies the realized, action-relative assistance value.
``outcome_blinded_fusion``
    Clears every armed record through the same feedback kernel but supplies a
    neutral assistance target.  It therefore retains identical state
    allocation and call counts without learning which partner helped.
``base_only``
    Supplies the canonical empty message batch while retaining the same core,
    state allocation, tensor shapes, and decision/feedback call counts.

The evaluator reports complete primitive traces and descriptive summaries.  It
has no threshold, seed search, artifact writer, output path, held-out claim, or
promotion entry point.  In particular, a favorable finite trace is not
evidence that partner fusion improves a deployed Eliza agent.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.core.partner_policy_fusion import (
    PartnerMessageBatch,
    PartnerPolicyFusion,
    PartnerPolicyFusionConfig,
    PartnerPolicyFusionFeedback,
    PartnerPolicyFusionState,
    partner_policy_fusion_identity_words,
)

SCHEMA = "alberta.partner-policy-fusion-stress-development.v1"
CHECKPOINT_SCHEMA = "alberta.partner-policy-fusion-stress-development.checkpoint.v1"
PROTOCOL_NAMESPACE = "partner-policy-fusion-reversal-cost-failure-stress-v1"
ASSESSMENT = "not_assessed"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
OUTPUT_WRITES_ALLOWED = False
RNG_DRAWS_PER_EVENT = 0

LEARNED_FUSION: Literal["learned_fusion"] = "learned_fusion"
OUTCOME_BLINDED_FUSION: Literal["outcome_blinded_fusion"] = (
    "outcome_blinded_fusion"
)
BASE_ONLY: Literal["base_only"] = "base_only"
Condition = Literal[
    "learned_fusion",
    "outcome_blinded_fusion",
    "base_only",
]
CONDITIONS: tuple[Condition, ...] = (
    LEARNED_FUSION,
    OUTCOME_BLINDED_FUSION,
    BASE_ONLY,
)

_INT32_MAX = 2_147_483_647
_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).parents[1] / "core" / "partner_policy_fusion.py",
)


def _canonical_json_bytes(value: object) -> bytes:
    """Return the canonical, finite JSON representation used for binding."""

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


def _telemetry(identity: int) -> int:
    return min(identity, _INT32_MAX)


@dataclasses.dataclass(frozen=True)
class PartnerPolicyFusionStressConfig:
    """Frozen dimensions and controller settings for this development lane."""

    phase_length: int = 12
    num_phases: int = 8
    reversal_phase: int = 4
    n_actions: int = 3
    max_partners: int = 2
    context_dim: int = 2
    declared_confidence: float = 0.9
    partner_costs: tuple[float, float] = (0.05, 0.10)
    cost_spike: float = 0.75

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "phase_length": 12,
            "num_phases": 8,
            "reversal_phase": 4,
            "n_actions": 3,
            "max_partners": 2,
            "context_dim": 2,
            "declared_confidence": 0.9,
            "partner_costs": (0.05, 0.10),
            "cost_spike": 0.75,
        }
        for name, expected_value in expected.items():
            value = getattr(self, name)
            if type(value) is not type(expected_value) or value != expected_value:
                raise ValueError(
                    f"{name} is frozen at {expected_value!r} for {SCHEMA}"
                )

    @property
    def num_events(self) -> int:
        return self.phase_length * self.num_phases

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


CONFIG = PartnerPolicyFusionStressConfig()


def _fusion_config(config: PartnerPolicyFusionStressConfig) -> PartnerPolicyFusionConfig:
    return PartnerPolicyFusionConfig(
        max_partners=config.max_partners,
        context_dim=config.context_dim,
        n_actions=config.n_actions,
        max_message_horizon=1,
        min_feedback_for_learned_routing=1,
        counter_cap=10_000,
        learning_rate=0.5,
        max_abs_weight=8.0,
        max_abs_context=1.0,
        max_communication_cost=1.0,
        communication_cost_weight=1.0,
        assistance_value_bound=1.0,
        safety_target_weight=0.0,
        accept_net_value_threshold=0.25,
        blend_net_value_threshold=-1.0,
        clarification_confidence_threshold=0.1,
        max_query_cost=1.0,
        base_blend_weight=1.0,
        option_blend_weight=1.0,
        partner_blend_weight=1.0,
        max_abs_declared_score=1.0,
    )


_FUSION = PartnerPolicyFusion(_fusion_config(CONFIG))
_JIT_DECIDE = jax.jit(_FUSION.decide)
_JIT_APPLY_FEEDBACK = jax.jit(_FUSION.apply_feedback)


@dataclasses.dataclass(frozen=True)
class StressEvent:
    """One fully specified evaluator-owned event.

    ``reliable_partner`` and ``correct_action`` are scoring annotations.  They
    are never supplied as controller inputs.  The controller receives only
    ``context_features`` and the ordinary typed partner messages.
    """

    index: int
    phase: int
    phase_step: int
    after_reversal: bool
    context_id: int
    context_features: tuple[float, float]
    reliable_partner: int
    correct_action: int
    message_available: tuple[bool, bool]
    communication_cost: tuple[float, float]
    safety_action_mask: tuple[bool, bool, bool]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def build_stress_schedule(
    config: PartnerPolicyFusionStressConfig = CONFIG,
) -> tuple[StressEvent, ...]:
    """Build the single frozen, deterministic development schedule."""

    events: list[StressEvent] = []
    for index in range(config.num_events):
        phase, phase_step = divmod(index, config.phase_length)
        context_id = phase % 2
        after_reversal = phase >= config.reversal_phase
        pre_reversal_partner = context_id
        reliable_partner = (
            1 - pre_reversal_partner if after_reversal else pre_reversal_partner
        )
        correct_action = reliable_partner + 1

        available = [True, True]
        if phase_step == 8:
            available[0] = False
        elif phase_step == 9:
            available[1] = False
        elif phase_step == 10:
            available = [False, False]

        costs = list(config.partner_costs)
        if phase_step == 6:
            costs[0] = config.cost_spike

        safety = [True, True, True]
        if phase_step == 4:
            safety[(1 - reliable_partner) + 1] = False
        elif phase_step == 11:
            safety[correct_action] = False

        events.append(
            StressEvent(
                index=index,
                phase=phase,
                phase_step=phase_step,
                after_reversal=after_reversal,
                context_id=context_id,
                # Signed contrast coding keeps the two observable contexts
                # nearly orthogonal after the model's bias/confidence terms;
                # the hidden reliable-partner annotation remains unavailable.
                context_features=(1.0, -1.0) if context_id == 0 else (-1.0, 1.0),
                reliable_partner=reliable_partner,
                correct_action=correct_action,
                message_available=(available[0], available[1]),
                communication_cost=(float(costs[0]), float(costs[1])),
                safety_action_mask=(safety[0], safety[1], safety[2]),
            )
        )
    return tuple(events)


SCHEDULE = build_stress_schedule()


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).parents[2]
    manifest: dict[str, str] = {}
    for path in _SOURCE_PATHS:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            label = resolved.name
        manifest[label] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return dict(sorted(manifest.items()))


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
        "jax": str(jax.__version__),
        "jaxlib": jaxlib_version,
        "default_backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kind": None if first is None else str(first.device_kind),
        "byteorder": sys.byteorder,
    }


def _messages(
    fusion: PartnerPolicyFusion,
    event: StressEvent,
    *,
    empty: bool,
) -> PartnerMessageBatch:
    if empty:
        return fusion.empty_messages()
    decision_words = partner_policy_fusion_identity_words(event.index)
    event_words = partner_policy_fusion_identity_words(event.index)
    horizon_words = jnp.broadcast_to(event_words, (CONFIG.max_partners, 2))
    ids = jnp.full(
        (CONFIG.max_partners,), _telemetry(event.index), dtype=jnp.int32
    )
    return PartnerMessageBatch(
        available=jnp.asarray(event.message_available, dtype=jnp.bool_),
        partner_id=jnp.arange(CONFIG.max_partners, dtype=jnp.int32),
        observation_id=ids,
        context_id=jnp.full(
            (CONFIG.max_partners,), event.context_id, dtype=jnp.int32
        ),
        suggested_action=jnp.asarray((1, 2), dtype=jnp.int32),
        declared_confidence=jnp.full(
            (CONFIG.max_partners,), CONFIG.declared_confidence, dtype=jnp.float32
        ),
        rationale_reference=jnp.asarray((100, 101), dtype=jnp.int32),
        provenance_reference=jnp.asarray((200, 201), dtype=jnp.int32),
        communication_cost=jnp.asarray(event.communication_cost, dtype=jnp.float32),
        issued_decision_id=ids,
        issued_event_id=ids,
        valid_through_event_id=ids,
        issued_decision_words=jnp.broadcast_to(
            decision_words, (CONFIG.max_partners, 2)
        ),
        issued_event_words=horizon_words,
        valid_through_event_words=horizon_words,
    )


@dataclasses.dataclass(frozen=True)
class StressTraceRecord:
    """Primitive causal record for one condition and event."""

    condition: Condition
    event: StressEvent
    effective_action: int
    selected_partner_id: int
    selected_partner_slot: int
    route: int
    partner_influenced: bool
    feedback_armed: bool
    decision_applied: bool
    valid_messages: tuple[bool, bool]
    predicted_reliability: tuple[float, float]
    predicted_net_value: tuple[float, float]
    task_reward: float
    selected_communication_cost: float
    net_utility: float
    feedback_applied: bool
    feedback_target: float
    parameter_update_l2_norm: float

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> StressTraceRecord:
        expected = {
            "condition",
            "event",
            "effective_action",
            "selected_partner_id",
            "selected_partner_slot",
            "route",
            "partner_influenced",
            "feedback_armed",
            "decision_applied",
            "valid_messages",
            "predicted_reliability",
            "predicted_net_value",
            "task_reward",
            "selected_communication_cost",
            "net_utility",
            "feedback_applied",
            "feedback_target",
            "parameter_update_l2_norm",
        }
        if set(payload) != expected:
            raise ValueError("stress trace fields do not match the v1 schema")
        event_payload = _strict_mapping(payload["event"], name="trace event")
        event_fields = {
            "index",
            "phase",
            "phase_step",
            "after_reversal",
            "context_id",
            "context_features",
            "reliable_partner",
            "correct_action",
            "message_available",
            "communication_cost",
            "safety_action_mask",
        }
        if set(event_payload) != event_fields:
            raise ValueError("stress event fields do not match the v1 schema")

        def strict_int(name: str) -> int:
            value = event_payload[name]
            if type(value) is not int:
                raise ValueError(f"event {name} must be a strict integer")
            return value

        def strict_bool(name: str) -> bool:
            value = event_payload[name]
            if type(value) is not bool:
                raise ValueError(f"event {name} must be a strict boolean")
            return value

        def event_pair(
            name: str, cast_type: type[bool] | type[float]
        ) -> tuple[Any, Any]:
            value = event_payload[name]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"event {name} must be a pair")
            return (cast_type(value[0]), cast_type(value[1]))

        safety_value = event_payload["safety_action_mask"]
        if not isinstance(safety_value, (list, tuple)) or len(safety_value) != 3:
            raise ValueError("event safety_action_mask must have length three")
        if any(type(value) is not bool for value in safety_value):
            raise ValueError("event safety_action_mask must contain strict booleans")
        event = StressEvent(
            index=strict_int("index"),
            phase=strict_int("phase"),
            phase_step=strict_int("phase_step"),
            after_reversal=strict_bool("after_reversal"),
            context_id=strict_int("context_id"),
            context_features=cast(
                tuple[float, float], event_pair("context_features", float)
            ),
            reliable_partner=strict_int("reliable_partner"),
            correct_action=strict_int("correct_action"),
            message_available=cast(
                tuple[bool, bool], event_pair("message_available", bool)
            ),
            communication_cost=cast(
                tuple[float, float], event_pair("communication_cost", float)
            ),
            safety_action_mask=(
                safety_value[0],
                safety_value[1],
                safety_value[2],
            ),
        )
        condition = payload["condition"]
        if condition not in CONDITIONS:
            raise ValueError("unknown stress condition")

        def pair(name: str, cast_type: type[bool] | type[float]) -> tuple[Any, Any]:
            value = payload[name]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{name} must be a pair")
            return (cast_type(value[0]), cast_type(value[1]))

        return cls(
            condition=condition,
            event=event,
            effective_action=int(cast(int, payload["effective_action"])),
            selected_partner_id=int(cast(int, payload["selected_partner_id"])),
            selected_partner_slot=int(cast(int, payload["selected_partner_slot"])),
            route=int(cast(int, payload["route"])),
            partner_influenced=bool(payload["partner_influenced"]),
            feedback_armed=bool(payload["feedback_armed"]),
            decision_applied=bool(payload["decision_applied"]),
            valid_messages=cast(tuple[bool, bool], pair("valid_messages", bool)),
            predicted_reliability=cast(
                tuple[float, float], pair("predicted_reliability", float)
            ),
            predicted_net_value=cast(
                tuple[float, float], pair("predicted_net_value", float)
            ),
            task_reward=float(cast(float, payload["task_reward"])),
            selected_communication_cost=float(
                cast(float, payload["selected_communication_cost"])
            ),
            net_utility=float(cast(float, payload["net_utility"])),
            feedback_applied=bool(payload["feedback_applied"]),
            feedback_target=float(cast(float, payload["feedback_target"])),
            parameter_update_l2_norm=float(
                cast(float, payload["parameter_update_l2_norm"])
            ),
        )


def _step(
    fusion: PartnerPolicyFusion,
    state: PartnerPolicyFusionState,
    condition: Condition,
    event: StressEvent,
) -> tuple[PartnerPolicyFusionState, StressTraceRecord]:
    words = partner_policy_fusion_identity_words(event.index)
    messages = _messages(fusion, event, empty=condition == BASE_ONLY)
    # The fixed kernels are compiled once; host orchestration retains the raw
    # event trace and exact checkpoint split semantics.
    decision_result = _JIT_DECIDE(
        state,
        decision_id=jnp.asarray(_telemetry(event.index), dtype=jnp.int32),
        event_id=jnp.asarray(_telemetry(event.index), dtype=jnp.int32),
        decision_words=words,
        event_words=words,
        observation_id=jnp.asarray(_telemetry(event.index), dtype=jnp.int32),
        context_id=jnp.asarray(event.context_id, dtype=jnp.int32),
        context_features=jnp.asarray(event.context_features, dtype=jnp.float32),
        base_action=jnp.asarray(0, dtype=jnp.int32),
        base_declared_score=jnp.asarray(0.0, dtype=jnp.float32),
        safety_action_mask=jnp.asarray(event.safety_action_mask, dtype=jnp.bool_),
        option_proposal=fusion.empty_option_proposal(),
        messages=messages,
    )
    decision = decision_result.decision
    action = int(decision.effective_action)
    action_safe = 0 <= action < CONFIG.n_actions and event.safety_action_mask[action]
    if not action_safe:
        raise RuntimeError("partner fusion emitted an action outside the hard mask")

    task_reward = 1.0 if action == event.correct_action else (-1.0 if action != 0 else 0.0)
    selected_partner = int(decision.selected_partner_id)
    influenced = bool(decision.partner_influenced)
    selected_cost = (
        event.communication_cost[selected_partner]
        if influenced and 0 <= selected_partner < CONFIG.max_partners
        else 0.0
    )
    net_utility = task_reward - selected_cost
    feedback_available = bool(decision.feedback_armed)
    assistance = task_reward if condition == LEARNED_FUSION else 0.0
    feedback = PartnerPolicyFusionFeedback(
        available=jnp.asarray(feedback_available, dtype=jnp.bool_),
        decision_id=jnp.asarray(_telemetry(event.index), dtype=jnp.int32),
        executed_event_id=jnp.asarray(_telemetry(event.index), dtype=jnp.int32),
        decision_words=words,
        executed_event_words=words,
        executed_action=jnp.asarray(action, dtype=jnp.int32),
        partner_id=jnp.asarray(selected_partner, dtype=jnp.int32),
        assistance_value_available=jnp.asarray(feedback_available, dtype=jnp.bool_),
        realized_assistance_value=jnp.asarray(assistance, dtype=jnp.float32),
        safety_outcome_available=jnp.asarray(feedback_available, dtype=jnp.bool_),
        safety_outcome_ok=jnp.asarray(action_safe, dtype=jnp.bool_),
    )
    feedback_result = _JIT_APPLY_FEEDBACK(decision_result.state, feedback)
    record = StressTraceRecord(
        condition=condition,
        event=event,
        effective_action=action,
        selected_partner_id=selected_partner,
        selected_partner_slot=int(decision.selected_partner_slot),
        route=int(decision.route),
        partner_influenced=influenced,
        feedback_armed=feedback_available,
        decision_applied=bool(decision.applied),
        valid_messages=cast(
            tuple[bool, bool],
            tuple(
                bool(value)
                for value in np.asarray(decision.availability.messages_valid)
            ),
        ),
        predicted_reliability=cast(
            tuple[float, float],
            tuple(
                float(value)
                for value in np.asarray(decision.scores.predicted_reliability)
            ),
        ),
        predicted_net_value=cast(
            tuple[float, float],
            tuple(
                float(value)
                for value in np.asarray(decision.scores.predicted_net_value)
            ),
        ),
        task_reward=task_reward,
        selected_communication_cost=float(selected_cost),
        net_utility=float(net_utility),
        feedback_applied=bool(feedback_result.applied),
        feedback_target=float(feedback_result.realized_training_target),
        parameter_update_l2_norm=float(feedback_result.parameter_update_l2_norm),
    )
    return feedback_result.state, record


def _run_condition(
    fusion: PartnerPolicyFusion,
    condition: Condition,
    *,
    start: int = 0,
    stop: int | None = None,
    state: PartnerPolicyFusionState | None = None,
    prefix: tuple[StressTraceRecord, ...] = (),
) -> tuple[PartnerPolicyFusionState, tuple[StressTraceRecord, ...]]:
    end = CONFIG.num_events if stop is None else stop
    if not 0 <= start <= end <= CONFIG.num_events:
        raise ValueError("condition run bounds are outside the frozen schedule")
    if len(prefix) != start:
        raise ValueError("trace prefix length must equal the start event")
    current = fusion.init() if state is None else state
    records = list(prefix)
    for event in SCHEDULE[start:end]:
        current, record = _step(fusion, current, condition, event)
        records.append(record)
    fusion.validate_state(current)
    return current, tuple(records)


@dataclasses.dataclass(frozen=True)
class StressConditionSummary:
    """Descriptive condition readout; no field is an acceptance threshold."""

    condition: Condition
    event_count: int
    decision_calls: int
    feedback_calls: int
    fixed_message_slots_per_call: int
    persistent_state_bytes: int
    mean_task_reward: float
    mean_net_utility: float
    pre_reversal_mean_net_utility: float
    post_reversal_mean_net_utility: float
    partner_influenced_count: int
    feedback_applied_count: int
    correct_safe_opportunity_count: int
    correct_safe_action_count: int
    total_communication_failure_count: int
    partner_selection_counts: tuple[int, int]
    final_feedback_counts: tuple[int, int]
    final_reliability_weights: tuple[tuple[float, ...], tuple[float, ...]]

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _summarize(
    fusion: PartnerPolicyFusion,
    condition: Condition,
    state: PartnerPolicyFusionState,
    trace: tuple[StressTraceRecord, ...],
) -> StressConditionSummary:
    pre = [record.net_utility for record in trace if not record.event.after_reversal]
    post = [record.net_utility for record in trace if record.event.after_reversal]
    safe_opportunities = [
        record
        for record in trace
        if record.event.safety_action_mask[record.event.correct_action]
    ]
    selected = tuple(
        sum(record.selected_partner_id == partner for record in trace)
        for partner in range(CONFIG.max_partners)
    )
    weights = np.asarray(state.reliability_weights, dtype=np.float32)
    return StressConditionSummary(
        condition=condition,
        event_count=len(trace),
        decision_calls=len(trace),
        feedback_calls=len(trace),
        fixed_message_slots_per_call=CONFIG.max_partners,
        persistent_state_bytes=fusion.resource_budget.persistent_state_bytes,
        mean_task_reward=_mean([record.task_reward for record in trace]),
        mean_net_utility=_mean([record.net_utility for record in trace]),
        pre_reversal_mean_net_utility=_mean(pre),
        post_reversal_mean_net_utility=_mean(post),
        partner_influenced_count=sum(record.partner_influenced for record in trace),
        feedback_applied_count=sum(record.feedback_applied for record in trace),
        correct_safe_opportunity_count=len(safe_opportunities),
        correct_safe_action_count=sum(
            record.effective_action == record.event.correct_action
            for record in safe_opportunities
        ),
        total_communication_failure_count=sum(
            not any(record.event.message_available) for record in trace
        ),
        partner_selection_counts=cast(tuple[int, int], selected),
        final_feedback_counts=cast(
            tuple[int, int],
            tuple(int(value) for value in np.asarray(state.feedback_counts)),
        ),
        final_reliability_weights=cast(
            tuple[tuple[float, ...], tuple[float, ...]],
            tuple(tuple(float(value) for value in row) for row in weights),
        ),
    )


@dataclasses.dataclass(frozen=True)
class PartnerPolicyFusionStressReport:
    """Self-bound, deterministic and explicitly nonpromoting report."""

    schema: str
    namespace: str
    assessment: str
    development_only: bool
    scientific_promotion_allowed: bool
    output_writes_allowed: bool
    rng_draws_per_event: int
    config: dict[str, object]
    fusion_config: dict[str, object]
    schedule_digest: str
    source_manifest: dict[str, str]
    runtime_manifest: dict[str, object]
    initial_snapshot_digest: str
    traces: dict[Condition, tuple[StressTraceRecord, ...]]
    summaries: dict[Condition, StressConditionSummary]
    deterministic_payload_digest: str

    def payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "namespace": self.namespace,
            "assessment": self.assessment,
            "development_only": self.development_only,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "output_writes_allowed": self.output_writes_allowed,
            "rng_draws_per_event": self.rng_draws_per_event,
            "config": self.config,
            "fusion_config": self.fusion_config,
            "schedule_digest": self.schedule_digest,
            "source_manifest": self.source_manifest,
            "runtime_manifest": self.runtime_manifest,
            "initial_snapshot_digest": self.initial_snapshot_digest,
            "traces": {
                condition: [record.to_dict() for record in self.traces[condition]]
                for condition in CONDITIONS
            },
            "summaries": {
                condition: self.summaries[condition].to_dict()
                for condition in CONDITIONS
            },
        }
        if include_digest:
            payload["deterministic_payload_digest"] = self.deterministic_payload_digest
        return payload


def _assemble_report(
    fusion: PartnerPolicyFusion,
    states: Mapping[Condition, PartnerPolicyFusionState],
    traces: Mapping[Condition, tuple[StressTraceRecord, ...]],
) -> PartnerPolicyFusionStressReport:
    initial_payload = fusion.checkpoint_payload(fusion.init())
    kwargs: dict[str, object] = {
        "schema": SCHEMA,
        "namespace": PROTOCOL_NAMESPACE,
        "assessment": ASSESSMENT,
        "development_only": DEVELOPMENT_ONLY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "output_writes_allowed": OUTPUT_WRITES_ALLOWED,
        "rng_draws_per_event": RNG_DRAWS_PER_EVENT,
        "config": CONFIG.to_dict(),
        "fusion_config": fusion.to_config(),
        "schedule_digest": _digest([event.to_dict() for event in SCHEDULE]),
        "source_manifest": _source_manifest(),
        "runtime_manifest": _runtime_manifest(),
        "initial_snapshot_digest": _digest(initial_payload),
        "traces": {condition: traces[condition] for condition in CONDITIONS},
        "summaries": {
            condition: _summarize(
                fusion, condition, states[condition], traces[condition]
            )
            for condition in CONDITIONS
        },
    }
    provisional = PartnerPolicyFusionStressReport(
        **cast(dict[str, Any], kwargs), deterministic_payload_digest=""
    )
    return dataclasses.replace(
        provisional,
        deterministic_payload_digest=_digest(provisional.payload(include_digest=False)),
    )


def _run_unvalidated() -> PartnerPolicyFusionStressReport:
    fusion = _FUSION
    states: dict[Condition, PartnerPolicyFusionState] = {}
    traces: dict[Condition, tuple[StressTraceRecord, ...]] = {}
    for condition in CONDITIONS:
        states[condition], traces[condition] = _run_condition(fusion, condition)
    return _assemble_report(fusion, states, traces)


def run_partner_policy_fusion_stress_development() -> PartnerPolicyFusionStressReport:
    """Execute and strictly validate the frozen nonpromoting stress lane."""

    report = _run_unvalidated()
    errors = validate_partner_policy_fusion_stress_report(report)
    if errors:
        raise RuntimeError("invalid stress report: " + "; ".join(errors))
    return report


def validate_partner_policy_fusion_stress_report(
    report: object,
) -> tuple[str, ...]:
    """Reconstruct the full causal run and reject any report drift or tamper."""

    if not isinstance(report, PartnerPolicyFusionStressReport):
        return ("report has the wrong type",)
    errors: list[str] = []
    if report.schema != SCHEMA:
        errors.append("report schema changed")
    if report.namespace != PROTOCOL_NAMESPACE:
        errors.append("protocol namespace changed")
    if report.assessment != ASSESSMENT:
        errors.append("assessment must remain not_assessed")
    if not report.development_only:
        errors.append("development_only must remain true")
    if report.scientific_promotion_allowed:
        errors.append("scientific promotion is forbidden")
    if report.output_writes_allowed:
        errors.append("output writes are forbidden")
    if report.rng_draws_per_event != 0:
        errors.append("the frozen evaluator owns no RNG")
    actual_digest = _digest(report.payload(include_digest=False))
    if report.deterministic_payload_digest != actual_digest:
        errors.append("deterministic report digest mismatch")
    try:
        expected = _run_unvalidated()
    except Exception as exc:  # pragma: no cover - fail-closed diagnostic
        errors.append(f"causal replay failed: {type(exc).__name__}: {exc}")
        return tuple(errors)
    if report.payload() != expected.payload():
        errors.append("report differs from exact causal replay")
    return tuple(errors)


def make_partner_policy_fusion_stress_checkpoint(
    next_event: int,
) -> dict[str, object]:
    """Create a source-bound checkpoint after exactly ``next_event`` events."""

    if type(next_event) is not int or not 0 <= next_event <= CONFIG.num_events:
        raise ValueError("next_event must be a strict integer inside the frozen life")
    fusion = _FUSION
    condition_payloads: dict[str, object] = {}
    for condition in CONDITIONS:
        state, trace = _run_condition(fusion, condition, stop=next_event)
        condition_payloads[condition] = {
            "state": fusion.checkpoint_payload(state),
            "trace_prefix": [record.to_dict() for record in trace],
        }
    payload: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "namespace": PROTOCOL_NAMESPACE,
        "next_event": next_event,
        "config_digest": _digest(CONFIG.to_dict()),
        "fusion_config_digest": _digest(fusion.to_config()),
        "schedule_digest": _digest([event.to_dict() for event in SCHEDULE]),
        "source_manifest": _source_manifest(),
        "runtime_manifest": _runtime_manifest(),
        "conditions": condition_payloads,
    }
    payload["checkpoint_digest"] = _digest(payload)
    return payload


def _validate_checkpoint(payload: Mapping[str, object]) -> tuple[str, ...]:
    expected_fields = {
        "schema",
        "namespace",
        "next_event",
        "config_digest",
        "fusion_config_digest",
        "schedule_digest",
        "source_manifest",
        "runtime_manifest",
        "conditions",
        "checkpoint_digest",
    }
    errors: list[str] = []
    if set(payload) != expected_fields:
        return ("checkpoint fields do not match the v1 schema",)
    supplied_digest = payload.get("checkpoint_digest")
    unsigned = dict(payload)
    unsigned.pop("checkpoint_digest", None)
    if supplied_digest != _digest(unsigned):
        errors.append("checkpoint digest mismatch")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        errors.append("checkpoint schema changed")
    if payload.get("namespace") != PROTOCOL_NAMESPACE:
        errors.append("checkpoint namespace changed")
    next_event = payload.get("next_event")
    if type(next_event) is not int or not 0 <= next_event <= CONFIG.num_events:
        errors.append("checkpoint next_event is invalid")
        return tuple(errors)
    fusion = _FUSION
    expected_static = {
        "config_digest": _digest(CONFIG.to_dict()),
        "fusion_config_digest": _digest(fusion.to_config()),
        "schedule_digest": _digest([event.to_dict() for event in SCHEDULE]),
        "source_manifest": _source_manifest(),
        "runtime_manifest": _runtime_manifest(),
    }
    for name, expected in expected_static.items():
        if payload.get(name) != expected:
            errors.append(f"checkpoint {name} changed")
    conditions_value = payload.get("conditions")
    if not isinstance(conditions_value, Mapping) or set(conditions_value) != set(
        CONDITIONS
    ):
        errors.append("checkpoint conditions changed")
        return tuple(errors)
    for condition in CONDITIONS:
        try:
            entry = _strict_mapping(
                conditions_value[condition], name=f"{condition} checkpoint"
            )
            if set(entry) != {"state", "trace_prefix"}:
                raise ValueError("condition checkpoint fields changed")
            state_payload = _strict_mapping(entry["state"], name="fusion state")
            restored_fusion, restored_state = PartnerPolicyFusion.from_checkpoint_payload(
                state_payload
            )
            if restored_fusion.to_config() != fusion.to_config():
                raise ValueError("restored fusion configuration changed")
            trace_value = entry["trace_prefix"]
            if not isinstance(trace_value, list) or len(trace_value) != next_event:
                raise ValueError("trace prefix length changed")
            restored_trace = tuple(
                StressTraceRecord.from_dict(
                    _strict_mapping(record, name="trace prefix record")
                )
                for record in trace_value
            )
            expected_state, expected_trace = _run_condition(
                fusion, condition, stop=next_event
            )
            if fusion.checkpoint_payload(restored_state) != fusion.checkpoint_payload(
                expected_state
            ):
                raise ValueError("restored state differs from causal prefix replay")
            if tuple(record.to_dict() for record in restored_trace) != tuple(
                record.to_dict() for record in expected_trace
            ):
                raise ValueError("trace differs from causal prefix replay")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{condition}: {exc}")
    return tuple(errors)


def resume_partner_policy_fusion_stress_checkpoint(
    checkpoint: object,
) -> PartnerPolicyFusionStressReport:
    """Validate a causal prefix, restore its states, and finish the frozen life."""

    payload = _strict_mapping(checkpoint, name="checkpoint")
    errors = _validate_checkpoint(payload)
    if errors:
        raise ValueError("invalid stress checkpoint: " + "; ".join(errors))
    next_event = cast(int, payload["next_event"])
    fusion = _FUSION
    condition_payloads = _strict_mapping(payload["conditions"], name="conditions")
    states: dict[Condition, PartnerPolicyFusionState] = {}
    traces: dict[Condition, tuple[StressTraceRecord, ...]] = {}
    for condition in CONDITIONS:
        entry = _strict_mapping(condition_payloads[condition], name=condition)
        _, state = PartnerPolicyFusion.from_checkpoint_payload(
            _strict_mapping(entry["state"], name="fusion state")
        )
        prefix_value = entry["trace_prefix"]
        if not isinstance(prefix_value, list):  # already checked, keeps mypy exact
            raise ValueError("trace_prefix must be a list")
        prefix = tuple(
            StressTraceRecord.from_dict(
                _strict_mapping(record, name="trace prefix record")
            )
            for record in prefix_value
        )
        states[condition], traces[condition] = _run_condition(
            fusion,
            condition,
            start=next_event,
            state=state,
            prefix=prefix,
        )
    report = _assemble_report(fusion, states, traces)
    validation = validate_partner_policy_fusion_stress_report(report)
    if validation:
        raise RuntimeError("resumed stress report is invalid: " + "; ".join(validation))
    return report


__all__ = [
    "ASSESSMENT",
    "BASE_ONLY",
    "CHECKPOINT_SCHEMA",
    "CONDITIONS",
    "CONFIG",
    "DEVELOPMENT_ONLY",
    "LEARNED_FUSION",
    "OUTCOME_BLINDED_FUSION",
    "OUTPUT_WRITES_ALLOWED",
    "PROTOCOL_NAMESPACE",
    "PartnerPolicyFusionStressConfig",
    "PartnerPolicyFusionStressReport",
    "RNG_DRAWS_PER_EVENT",
    "SCHEMA",
    "SCHEDULE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "StressConditionSummary",
    "StressEvent",
    "StressTraceRecord",
    "build_stress_schedule",
    "make_partner_policy_fusion_stress_checkpoint",
    "resume_partner_policy_fusion_stress_checkpoint",
    "run_partner_policy_fusion_stress_development",
    "validate_partner_policy_fusion_stress_report",
]
