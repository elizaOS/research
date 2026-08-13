"""Pure-stdlib historical-state behavioral-KL retention development probe.

This L0 scaffold localizes one narrow claim: a behavioral KL constraint is
defined by both the frozen policy and the state distribution on which the two
policies are compared.  A deterministic full-information contextual-bandit
prefix trains a two-parameter Bernoulli actor on A.  The resulting actor and
one real A state are frozen, then four bit-identical copies learn the same B
events online:

* ordinary task learning;
* task learning plus behavioral KL on the retained real A state;
* task learning plus behavioral KL on the current B state; and
* task learning plus a full-parameter L2 movement anchor.

The actor logit is ``theta dot state``.  A uses state ``(1, 0)`` and rewards
``(0, 1)``; B uses state ``(1, 1)`` and rewards ``(1, 0)``.  B therefore
updates the A-sensitive coordinate but also exposes a second escape
coordinate.  There is no hidden shrink, reset, replay, sampling, or RNG.
Every B arm computes the same four candidate objective components and their
two-coordinate gradients before a fixed route selects which are applied.
Interventions occur during B learning; there is no recovery phase.

The retained state is bound directly to bytes from a consumed real A event.
It is not a dream and is not minted by a world/value snapshot.  This analytic
binary-bandit construction is not a CPO reproduction, an MRCL result, a VLM
experiment, or a generalization claim.  It has no output writer, threshold,
winner, default arm, evidence role, artifact authority, or promotion path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Final, cast

CONFIG_SCHEMA: Final = "alberta.historical-behavioral-kl-retention.config.v1"
SOURCE_SCHEMA: Final = "alberta.historical-behavioral-kl-retention.source.v1"
STATE_SCHEMA: Final = "alberta.historical-behavioral-kl-retention.actor-state.v1"
TRACE_SCHEMA: Final = "alberta.historical-behavioral-kl-retention.trace.v1"
REPORT_SCHEMA: Final = "alberta.historical-behavioral-kl-retention.development.v1"
SOURCE_GENERATOR_VERSION: Final = "deterministic-full-information-ab-bandit-v1"

DEVELOPMENT_ONLY: Final = True
ASSESSMENT_STATUS: Final = "not_assessed"
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
BENCHMARK_EXECUTION_AUTHORITY: Final = False
ARTIFACT_AUTHORITY: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_CLAIMED: Final = False
THRESHOLDS_FROZEN: Final = False
CPO_REPRODUCTION: Final = False
MRCL_REPRODUCTION: Final = False
VLM_GENERALIZATION_CLAIMED: Final = False
RNG_USED: Final = False
GLOBAL_SHRINK_APPLIED: Final = False

ARM_NAMES: Final = (
    "ordinary_control",
    "historical_a_state_kl",
    "current_b_state_kl",
    "parameter_movement_l2",
)
CANDIDATE_COMPONENT_NAMES: Final = (
    "b_task_loss",
    "historical_a_state_behavioral_kl",
    "current_b_state_behavioral_kl",
    "parameter_movement_l2",
)
ARM_ROUTINGS: Final = (
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
)
PARAMETER_COUNT: Final = 2
ACTION_COUNT: Final = 2
HARD_MAX_PHASE_STEPS: Final = 4_096
HARD_MAX_TOTAL_TRACE_RECORDS: Final = 20_480
HARD_MAX_REPORT_BYTES: Final = 16_000_000

_LIMITATIONS: Final = (
    "one analytic full-information binary contextual bandit is not sampled on-policy RL",
    "one retained real A state is not a historical state distribution estimate",
    "the two-parameter shared-feature interference geometry is deliberately small",
    "the experiment has no hidden global shrink, replay, reset, sampling, or RNG",
    "the historical anchor comes from a consumed real A event and is not a dream",
    "no world/value snapshot or synthetic world grounding mints the historical anchor",
    "the L2 arm is a parameter-movement control, not CPO's cumulative masked L1 method",
    "raw Pareto coordinates are reported without a frontier, threshold, or winner",
    "this is not a CPO or MRCL reproduction and makes no VLM generalization claim",
    "serialized bytes describe Python float64 records, not allocator or object overhead",
)


def _positive_finite_float(value: object, *, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class HistoricalBehavioralKLRetentionConfig:
    """Bounded deterministic construction; all numeric fields are exact types."""

    schema: str = CONFIG_SCHEMA
    a_prefix_steps: int = 32
    b_interference_steps: int = 48
    step_size: float = 0.40
    historical_kl_weight: float = 2.0
    current_kl_weight: float = 2.0
    movement_l2_weight: float = 0.25
    max_abs_parameter: float = 20.0
    max_report_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != CONFIG_SCHEMA:
            raise ValueError("config schema differs")
        for name in ("a_prefix_steps", "b_interference_steps", "max_report_bytes"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if not 1 <= self.a_prefix_steps <= HARD_MAX_PHASE_STEPS:
            raise ValueError("a_prefix_steps exceeds its bounded positive range")
        if not 1 <= self.b_interference_steps <= HARD_MAX_PHASE_STEPS:
            raise ValueError("b_interference_steps exceeds its bounded positive range")
        trace_records = self.a_prefix_steps + len(ARM_NAMES) * self.b_interference_steps
        if trace_records > HARD_MAX_TOTAL_TRACE_RECORDS:
            raise ValueError("configured trace records exceed the hard cap")
        if not 1_024 <= self.max_report_bytes <= HARD_MAX_REPORT_BYTES:
            raise ValueError("max_report_bytes exceeds its bounded range")
        for name in (
            "step_size",
            "historical_kl_weight",
            "current_kl_weight",
            "movement_l2_weight",
            "max_abs_parameter",
        ):
            _positive_finite_float(getattr(self, name), name=name)
        if self.step_size > 1.0:
            raise ValueError("step_size must not exceed one")
        if max(
            self.historical_kl_weight,
            self.current_kl_weight,
            self.movement_l2_weight,
        ) > 100.0:
            raise ValueError("intervention weights must not exceed 100")
        if not 1.0 <= self.max_abs_parameter <= 100.0:
            raise ValueError("max_abs_parameter must lie in [1, 100]")


@dataclasses.dataclass(frozen=True, slots=True)
class ContextualBanditEvent:
    """One evaluator event; learners receive only state and reward table."""

    ordinal: int
    state: tuple[float, float]
    rewards_by_action: tuple[float, float]


@dataclasses.dataclass(frozen=True, slots=True)
class HistoricalBehavioralKLSource:
    """Exact real A prefix and matched B interference event streams."""

    schema: str
    config: HistoricalBehavioralKLRetentionConfig
    a_events: tuple[ContextualBanditEvent, ...]
    b_events: tuple[ContextualBanditEvent, ...]
    generator_contract_sha256: str
    input_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class ActorState:
    """Two float64 parameters and one logical accepted-update counter."""

    schema: str
    parameters: tuple[float, float]
    accepted_updates: int


@dataclasses.dataclass(frozen=True, slots=True)
class RetainedRealStateAnchor:
    """Historical A state bound directly to one consumed evaluator event."""

    kind: str
    state: tuple[float, float]
    source_event_ordinal: int
    source_event_sha256: str
    world_value_snapshot_used: bool
    synthetic_world_grounding_used: bool
    is_dream: bool


@dataclasses.dataclass(frozen=True, slots=True)
class PrefixStepTrace:
    """One streamed A update from one real evaluator event."""

    schema: str
    step: int
    source_event_sha256: str
    parameters_pre: tuple[float, float]
    probability_action_one_pre: float
    expected_return_pre: float
    task_loss: float
    task_gradient: tuple[float, float]
    parameter_delta: tuple[float, float]
    parameters_post: tuple[float, float]
    update_applied: bool
    parameter_address_mask: tuple[bool, bool]


@dataclasses.dataclass(frozen=True, slots=True)
class BInterferenceStepTrace:
    """All B candidates, fixed route, update, and raw pre/post probes."""

    schema: str
    step: int
    source_event_sha256: str
    parameters_pre: tuple[float, float]
    candidate_objectives: tuple[float, float, float, float]
    candidate_gradients: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    routing: tuple[int, int, int, int]
    routed_objective: float
    routed_gradient: tuple[float, float]
    parameter_delta: tuple[float, float]
    parameters_post: tuple[float, float]
    update_applied: bool
    parameter_address_mask: tuple[bool, bool]
    a_return_pre: float
    a_return_post: float
    a_margin_pre: float
    a_margin_post: float
    b_return_pre: float
    b_return_post: float
    b_margin_pre: float
    b_margin_post: float
    historical_a_kl_pre: float
    historical_a_kl_post: float
    current_b_kl_pre: float
    current_b_kl_post: float
    movement_l2_pre: float
    movement_l2_post: float


@dataclasses.dataclass(frozen=True, slots=True)
class ArmMetrics:
    """Raw final retention, forgetting, plasticity, KL, movement, and margins."""

    a_return_before_b: float
    a_return_after_b: float
    a_return_delta: float
    a_forgetting: float
    a_margin_before_b: float
    a_margin_after_b: float
    a_margin_delta: float
    b_return_before_b: float
    b_return_after_b: float
    b_return_delta: float
    b_plasticity_gain: float
    b_margin_before_b: float
    b_margin_after_b: float
    b_margin_delta: float
    historical_a_kl_final: float
    current_b_kl_final: float
    movement_l2_final: float
    movement_norm_final: float
    mean_historical_a_kl_pre: float
    mean_current_b_kl_pre: float
    mean_movement_l2_pre: float
    mean_routed_objective: float


@dataclasses.dataclass(frozen=True, slots=True)
class ArmRun:
    """One B continuation from an exact copy of the frozen A actor."""

    name: str
    routing: tuple[int, int, int, int]
    initial_state: ActorState
    final_state: ActorState
    trace: tuple[BInterferenceStepTrace, ...]
    metrics: ArmMetrics


@dataclasses.dataclass(frozen=True, slots=True)
class ParetoCoordinate:
    """Unclassified stability/plasticity coordinate; no frontier or winner."""

    arm_name: str
    retained_a_return: float
    a_forgetting: float
    b_return: float
    b_plasticity_gain: float


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedArmAudit:
    """Exact B events, candidate computations, addresses, updates, and RNG."""

    arm_name: str
    routing: tuple[int, int, int, int]
    b_event_stream_sha256: str
    events_consumed: int
    updates_attempted: int
    updates_applied: int
    candidate_objectives_per_event: int
    candidate_gradient_float64_scalars_per_event: int
    parameters_addressed_per_event: int
    posthoc_recovery_updates: int
    rng_draws: int


@dataclasses.dataclass(frozen=True, slots=True)
class WorkSummary:
    """High-level exact logical call and scalar accounting."""

    prefix_task_objective_evaluations: int
    b_task_objective_evaluations: int
    historical_kl_objective_evaluations: int
    current_kl_objective_evaluations: int
    movement_l2_objective_evaluations: int
    total_candidate_objective_evaluations: int
    total_candidate_gradient_float64_scalars: int
    prefix_parameter_updates: int
    routed_parameter_updates: int
    total_parameter_updates: int
    addressed_parameter_float64_scalars: int
    frozen_policy_probability_evaluations: int
    b_pre_post_probe_probability_evaluations: int
    rng_draws: int
    global_shrink_evaluations: int


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceSummary:
    """Logical state sizes and canonical in-memory record byte counts."""

    parameter_count: int
    actor_state_logical_nbytes: int
    retained_anchor_float64_scalars: int
    retained_anchor_logical_nbytes: int
    frozen_actor_parameter_float64_scalars: int
    per_arm_actor_state_logical_nbytes: int
    prefix_trace_records: int
    b_trace_records_per_arm: int
    total_trace_records: int
    canonical_source_nbytes: int
    canonical_trace_nbytes: int
    max_report_bytes: int
    hard_max_report_bytes: int
    report_cap_enforced: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ScalingSummary:
    """Transparent fixed-width scaling projection, not an empirical scale claim."""

    actor_parameter_count: int
    historical_anchor_count: int
    historical_anchor_float64_scalars: int
    persistent_float64_scalars_per_added_anchor: int
    candidate_gradient_scalars_per_b_event: int
    trace_records_total: int
    trace_record_growth_formula: str
    candidate_work_growth_formula: str
    persistent_anchor_growth_formula: str
    empirical_scale_claimed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class HistoricalBehavioralKLRetentionReport:
    """Strict in-memory development report with no evidence authority."""

    schema: str
    status: str
    development_only: bool
    assessment_status: str
    scientific_promotion_allowed: bool
    benchmark_execution_authority: bool
    artifact_authority: bool
    output_writes_allowed: bool
    evidence_claimed: bool
    thresholds_frozen: bool
    cpo_reproduction: bool
    mrcl_reproduction: bool
    vlm_generalization_claimed: bool
    rng_used: bool
    global_shrink_applied: bool
    posthoc_recovery_updates: int
    arm_names: tuple[str, str, str, str]
    candidate_component_names: tuple[str, str, str, str]
    config: HistoricalBehavioralKLRetentionConfig
    source: HistoricalBehavioralKLSource
    initial_state: ActorState
    frozen_a_state: ActorState
    retained_a_anchor: RetainedRealStateAnchor
    prefix_trace: tuple[PrefixStepTrace, ...]
    arms: tuple[ArmRun, ArmRun, ArmRun, ArmRun]
    pareto_coordinates: tuple[
        ParetoCoordinate,
        ParetoCoordinate,
        ParetoCoordinate,
        ParetoCoordinate,
    ]
    matched_arm_audits: tuple[
        MatchedArmAudit,
        MatchedArmAudit,
        MatchedArmAudit,
        MatchedArmAudit,
    ]
    work: WorkSummary
    resource: ResourceSummary
    scaling: ScalingSummary
    implementation_source_sha256: str
    config_sha256: str
    source_sha256: str
    initial_state_sha256: str
    frozen_a_state_sha256: str
    retained_anchor_sha256: str
    arm_states_sha256: str
    trace_sha256: str
    work_sha256: str
    resource_sha256: str
    scaling_sha256: str
    limitations: tuple[str, ...]


def _canonical_value(value: object) -> object:
    """Return type-explicit JSON data; float hex preserves signed zero exactly."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__name__,
            "fields": [
                [field.name, _canonical_value(getattr(value, field.name))]
                for field in dataclasses.fields(value)
            ],
        }
    if type(value) is tuple:
        return {"__tuple__": [_canonical_value(item) for item in cast(tuple[object, ...], value)]}
    if type(value) is str:
        return {"__str__": value}
    if type(value) is bool:
        return {"__bool__": value}
    if type(value) is int:
        return {"__int__": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical values must be finite")
        return {"__float64_hex__": value.hex()}
    if value is None:
        return {"__none__": True}
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _implementation_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _exact(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        return all(
            _exact(getattr(left, field.name), getattr(right, field.name))
            for field in dataclasses.fields(left)
        )
    if type(left) is tuple:
        left_tuple = cast(tuple[object, ...], left)
        right_tuple = cast(tuple[object, ...], right)
        return len(left_tuple) == len(right_tuple) and all(
            _exact(a, b) for a, b in zip(left_tuple, right_tuple, strict=True)
        )
    if type(left) is float:
        return struct.pack(">d", left) == struct.pack(">d", right)
    return bool(left == right)


def _all_finite(value: object) -> bool:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return all(_all_finite(getattr(value, field.name)) for field in dataclasses.fields(value))
    if type(value) is tuple:
        return all(_all_finite(item) for item in cast(tuple[object, ...], value))
    if type(value) is float:
        return math.isfinite(value)
    return True


def _canonical_config_copy(
    config: HistoricalBehavioralKLRetentionConfig,
) -> HistoricalBehavioralKLRetentionConfig:
    if type(config) is not HistoricalBehavioralKLRetentionConfig:
        raise TypeError("config must be a HistoricalBehavioralKLRetentionConfig")
    payload = {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(HistoricalBehavioralKLRetentionConfig)
    }
    reconstructed = HistoricalBehavioralKLRetentionConfig(**payload)
    if not _exact(config, reconstructed):
        raise ValueError("config fields are not canonical exact values")
    return reconstructed


def _generator_contract(config: HistoricalBehavioralKLRetentionConfig) -> tuple[object, ...]:
    return (
        SOURCE_GENERATOR_VERSION,
        config,
        ("A", (1.0, 0.0), (0.0, 1.0)),
        ("B", (1.0, 1.0), (1.0, 0.0)),
        ARM_NAMES,
        CANDIDATE_COMPONENT_NAMES,
        ARM_ROUTINGS,
        "each-event-consumed-once",
        "exact-expected-contextual-bandit-gradient",
        "no-rng-no-replay-no-reset-no-global-shrink",
    )


def _build_source_unchecked(
    config: HistoricalBehavioralKLRetentionConfig,
) -> HistoricalBehavioralKLSource:
    a_events = tuple(
        ContextualBanditEvent(
            ordinal=step,
            state=(1.0, 0.0),
            rewards_by_action=(0.0, 1.0),
        )
        for step in range(config.a_prefix_steps)
    )
    b_events = tuple(
        ContextualBanditEvent(
            ordinal=step,
            state=(1.0, 1.0),
            rewards_by_action=(1.0, 0.0),
        )
        for step in range(config.b_interference_steps)
    )
    contract_sha = _sha256(_generator_contract(config))
    input_sha = _sha256((contract_sha, a_events, b_events))
    return HistoricalBehavioralKLSource(
        schema=SOURCE_SCHEMA,
        config=config,
        a_events=a_events,
        b_events=b_events,
        generator_contract_sha256=contract_sha,
        input_sha256=input_sha,
    )


def build_historical_behavioral_kl_source(
    config: HistoricalBehavioralKLRetentionConfig | None = None,
) -> HistoricalBehavioralKLSource:
    """Build the deterministic real A prefix and matched B event stream."""

    supplied = HistoricalBehavioralKLRetentionConfig() if config is None else config
    return _build_source_unchecked(_canonical_config_copy(supplied))


def validate_historical_behavioral_kl_source(
    source: HistoricalBehavioralKLSource,
) -> tuple[str, ...]:
    """Reconstruct every source value with exact type and float-bit equality."""

    if type(source) is not HistoricalBehavioralKLSource:
        return ("source type differs",)
    if type(source.config) is not HistoricalBehavioralKLRetentionConfig:
        return ("source config type differs",)
    try:
        config = _canonical_config_copy(source.config)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return ("source config fields are not canonical",)
    expected = _build_source_unchecked(config)
    errors: list[str] = []
    source_finite = _all_finite(source)
    if not source_finite:
        errors.append("source contains non-finite values")
    if type(source.schema) is not str or source.schema != SOURCE_SCHEMA:
        errors.append("source schema differs")
    if type(source.a_events) is not tuple or any(
        type(event) is not ContextualBanditEvent for event in source.a_events
    ):
        errors.append("source A event types differ")
    if type(source.b_events) is not tuple or any(
        type(event) is not ContextualBanditEvent for event in source.b_events
    ):
        errors.append("source B event types differ")
    if type(source.generator_contract_sha256) is not str:
        errors.append("source generator digest type differs")
    if type(source.input_sha256) is not str:
        errors.append("source input digest type differs")
    if not _exact(source, expected):
        errors.append("source does not reconstruct bit-exactly")
    if source_finite:
        try:
            reconstructed_input_sha = _sha256(
                (source.generator_contract_sha256, source.a_events, source.b_events)
            )
        except (TypeError, ValueError):
            errors.append("source digest payload is not canonical")
        else:
            if source.input_sha256 != reconstructed_input_sha:
                errors.append("source input digest does not bind source events")
    return tuple(errors)


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _dot(parameters: tuple[float, float], state: tuple[float, float]) -> float:
    return parameters[0] * state[0] + parameters[1] * state[1]


def _policy_probability_one(
    parameters: tuple[float, float],
    state: tuple[float, float],
) -> float:
    return _sigmoid(_dot(parameters, state))


def _expected_return(
    probability_one: float,
    rewards: tuple[float, float],
) -> float:
    return (1.0 - probability_one) * rewards[0] + probability_one * rewards[1]


def _task_objective_gradient(
    parameters: tuple[float, float],
    state: tuple[float, float],
    rewards: tuple[float, float],
) -> tuple[float, float, tuple[float, float]]:
    """Return probability, negative expected return, and its exact gradient."""

    probability_one = _policy_probability_one(parameters, state)
    expected_return = _expected_return(probability_one, rewards)
    reward_difference = rewards[1] - rewards[0]
    logit_gradient = -reward_difference * probability_one * (1.0 - probability_one)
    gradient = (logit_gradient * state[0], logit_gradient * state[1])
    return probability_one, -expected_return, gradient


def _softplus(logit: float) -> float:
    return max(logit, 0.0) + math.log1p(math.exp(-abs(logit)))


def _bernoulli_kl_from_logit(old_probability: float, current_logit: float) -> float:
    """Compute KL(old || sigmoid(logit)) without dividing by rounded 0/1."""

    old_negative_entropy = old_probability * math.log(old_probability) + (
        1.0 - old_probability
    ) * math.log(1.0 - old_probability)
    cross_entropy = _softplus(current_logit) - old_probability * current_logit
    return max(0.0, cross_entropy + old_negative_entropy)


def _behavioral_kl_component(
    parameters: tuple[float, float],
    state: tuple[float, float],
    frozen_probability: float,
) -> tuple[float, tuple[float, float], float]:
    current_logit = _dot(parameters, state)
    current = _sigmoid(current_logit)
    gradient_scale = current - frozen_probability
    return (
        _bernoulli_kl_from_logit(frozen_probability, current_logit),
        (gradient_scale * state[0], gradient_scale * state[1]),
        current,
    )


def _movement_component(
    parameters: tuple[float, float],
    frozen_parameters: tuple[float, float],
) -> tuple[float, tuple[float, float]]:
    gradient = (
        parameters[0] - frozen_parameters[0],
        parameters[1] - frozen_parameters[1],
    )
    return 0.5 * (gradient[0] ** 2 + gradient[1] ** 2), gradient


def _apply_update(
    state: ActorState,
    gradient: tuple[float, float],
    config: HistoricalBehavioralKLRetentionConfig,
) -> tuple[ActorState, tuple[float, float]]:
    delta = (-config.step_size * gradient[0], -config.step_size * gradient[1])
    candidate = (
        state.parameters[0] + delta[0],
        state.parameters[1] + delta[1],
    )
    if not all(math.isfinite(value) for value in candidate):
        raise RuntimeError("candidate actor parameters are non-finite")
    if any(abs(value) > config.max_abs_parameter for value in candidate):
        raise RuntimeError("candidate actor parameters exceed the configured cap")
    return (
        ActorState(
            schema=STATE_SCHEMA,
            parameters=candidate,
            accepted_updates=state.accepted_updates + 1,
        ),
        delta,
    )


def _run_a_prefix(
    initial_state: ActorState,
    events: tuple[ContextualBanditEvent, ...],
    config: HistoricalBehavioralKLRetentionConfig,
) -> tuple[ActorState, tuple[PrefixStepTrace, ...]]:
    state = initial_state
    trace: list[PrefixStepTrace] = []
    for step, event in enumerate(events):
        probability, loss, gradient = _task_objective_gradient(
            state.parameters,
            event.state,
            event.rewards_by_action,
        )
        expected_return = -loss
        next_state, delta = _apply_update(state, gradient, config)
        trace.append(
            PrefixStepTrace(
                schema=TRACE_SCHEMA,
                step=step,
                source_event_sha256=_sha256(event),
                parameters_pre=state.parameters,
                probability_action_one_pre=probability,
                expected_return_pre=expected_return,
                task_loss=loss,
                task_gradient=gradient,
                parameter_delta=delta,
                parameters_post=next_state.parameters,
                update_applied=True,
                parameter_address_mask=(True, True),
            )
        )
        state = next_state
    return state, tuple(trace)


def _candidate_components(
    parameters: tuple[float, float],
    event: ContextualBanditEvent,
    anchor: RetainedRealStateAnchor,
    frozen_parameters: tuple[float, float],
    frozen_a_probability: float,
    frozen_b_probability: float,
    config: HistoricalBehavioralKLRetentionConfig,
) -> tuple[
    tuple[float, float, float, float],
    tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    tuple[float, float, float, float, float],
]:
    b_probability, task_loss, task_gradient = _task_objective_gradient(
        parameters,
        event.state,
        event.rewards_by_action,
    )
    historical_kl, historical_gradient, a_probability = _behavioral_kl_component(
        parameters,
        anchor.state,
        frozen_a_probability,
    )
    current_kl, current_gradient, current_b_probability = _behavioral_kl_component(
        parameters,
        event.state,
        frozen_b_probability,
    )
    if struct.pack(">d", b_probability) != struct.pack(">d", current_b_probability):
        raise RuntimeError("B task and current-state KL probabilities are not bound")
    movement_l2, movement_gradient = _movement_component(parameters, frozen_parameters)
    objectives = (
        task_loss,
        config.historical_kl_weight * historical_kl,
        config.current_kl_weight * current_kl,
        config.movement_l2_weight * movement_l2,
    )
    gradients = (
        task_gradient,
        (
            config.historical_kl_weight * historical_gradient[0],
            config.historical_kl_weight * historical_gradient[1],
        ),
        (
            config.current_kl_weight * current_gradient[0],
            config.current_kl_weight * current_gradient[1],
        ),
        (
            config.movement_l2_weight * movement_gradient[0],
            config.movement_l2_weight * movement_gradient[1],
        ),
    )
    probes = (a_probability, b_probability, historical_kl, current_kl, movement_l2)
    return objectives, gradients, probes


def _route_candidates(
    objectives: tuple[float, float, float, float],
    gradients: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    routing: tuple[int, int, int, int],
) -> tuple[float, tuple[float, float]]:
    objective = math.fsum(
        route * candidate for route, candidate in zip(routing, objectives, strict=True)
    )
    gradient = tuple(
        math.fsum(
            routing[index] * gradients[index][coordinate]
            for index in range(len(CANDIDATE_COMPONENT_NAMES))
        )
        for coordinate in range(PARAMETER_COUNT)
    )
    return objective, cast(tuple[float, float], gradient)


def _probe_values(
    parameters: tuple[float, float],
    anchor: RetainedRealStateAnchor,
    b_state: tuple[float, float],
    frozen_parameters: tuple[float, float],
    frozen_a_probability: float,
    frozen_b_probability: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    a_probability = _policy_probability_one(parameters, anchor.state)
    b_probability_one = _policy_probability_one(parameters, b_state)
    b_return = 1.0 - b_probability_one
    historical_kl = _bernoulli_kl_from_logit(
        frozen_a_probability,
        _dot(parameters, anchor.state),
    )
    current_kl = _bernoulli_kl_from_logit(
        frozen_b_probability,
        _dot(parameters, b_state),
    )
    movement_l2, _ = _movement_component(parameters, frozen_parameters)
    return (
        a_probability,
        2.0 * a_probability - 1.0,
        b_probability_one,
        b_return,
        2.0 * b_return - 1.0,
        historical_kl,
        current_kl,
        movement_l2,
    )


def _run_b_arm(
    name: str,
    routing: tuple[int, int, int, int],
    initial_state: ActorState,
    events: tuple[ContextualBanditEvent, ...],
    anchor: RetainedRealStateAnchor,
    frozen_a_state: ActorState,
    config: HistoricalBehavioralKLRetentionConfig,
) -> ArmRun:
    frozen_a_probability = _policy_probability_one(
        frozen_a_state.parameters, anchor.state
    )
    frozen_b_probability = _policy_probability_one(
        frozen_a_state.parameters, events[0].state
    )
    state = initial_state
    trace: list[BInterferenceStepTrace] = []
    for step, event in enumerate(events):
        objectives, gradients, pre_components = _candidate_components(
            state.parameters,
            event,
            anchor,
            frozen_a_state.parameters,
            frozen_a_probability,
            frozen_b_probability,
            config,
        )
        routed_objective, routed_gradient = _route_candidates(
            objectives, gradients, routing
        )
        pre_probe = _probe_values(
            state.parameters,
            anchor,
            event.state,
            frozen_a_state.parameters,
            frozen_a_probability,
            frozen_b_probability,
        )
        if not _exact(
            pre_components,
            (
                pre_probe[0],
                pre_probe[2],
                pre_probe[5],
                pre_probe[6],
                pre_probe[7],
            ),
        ):
            raise RuntimeError("candidate components and pre-update probes are not bound")
        next_state, delta = _apply_update(state, routed_gradient, config)
        post_probe = _probe_values(
            next_state.parameters,
            anchor,
            event.state,
            frozen_a_state.parameters,
            frozen_a_probability,
            frozen_b_probability,
        )
        trace.append(
            BInterferenceStepTrace(
                schema=TRACE_SCHEMA,
                step=step,
                source_event_sha256=_sha256(event),
                parameters_pre=state.parameters,
                candidate_objectives=objectives,
                candidate_gradients=gradients,
                routing=routing,
                routed_objective=routed_objective,
                routed_gradient=routed_gradient,
                parameter_delta=delta,
                parameters_post=next_state.parameters,
                update_applied=True,
                parameter_address_mask=(True, True),
                a_return_pre=pre_probe[0],
                a_return_post=post_probe[0],
                a_margin_pre=pre_probe[1],
                a_margin_post=post_probe[1],
                b_return_pre=pre_probe[3],
                b_return_post=post_probe[3],
                b_margin_pre=pre_probe[4],
                b_margin_post=post_probe[4],
                historical_a_kl_pre=pre_probe[5],
                historical_a_kl_post=post_probe[5],
                current_b_kl_pre=pre_probe[6],
                current_b_kl_post=post_probe[6],
                movement_l2_pre=pre_probe[7],
                movement_l2_post=post_probe[7],
            )
        )
        state = next_state

    first = trace[0]
    last = trace[-1]
    movement_norm = math.sqrt(2.0 * last.movement_l2_post)
    metrics = ArmMetrics(
        a_return_before_b=first.a_return_pre,
        a_return_after_b=last.a_return_post,
        a_return_delta=last.a_return_post - first.a_return_pre,
        a_forgetting=first.a_return_pre - last.a_return_post,
        a_margin_before_b=first.a_margin_pre,
        a_margin_after_b=last.a_margin_post,
        a_margin_delta=last.a_margin_post - first.a_margin_pre,
        b_return_before_b=first.b_return_pre,
        b_return_after_b=last.b_return_post,
        b_return_delta=last.b_return_post - first.b_return_pre,
        b_plasticity_gain=last.b_return_post - first.b_return_pre,
        b_margin_before_b=first.b_margin_pre,
        b_margin_after_b=last.b_margin_post,
        b_margin_delta=last.b_margin_post - first.b_margin_pre,
        historical_a_kl_final=last.historical_a_kl_post,
        current_b_kl_final=last.current_b_kl_post,
        movement_l2_final=last.movement_l2_post,
        movement_norm_final=movement_norm,
        mean_historical_a_kl_pre=statistics.fmean(
            record.historical_a_kl_pre for record in trace
        ),
        mean_current_b_kl_pre=statistics.fmean(
            record.current_b_kl_pre for record in trace
        ),
        mean_movement_l2_pre=statistics.fmean(record.movement_l2_pre for record in trace),
        mean_routed_objective=statistics.fmean(
            record.routed_objective for record in trace
        ),
    )
    return ArmRun(
        name=name,
        routing=routing,
        initial_state=initial_state,
        final_state=state,
        trace=tuple(trace),
        metrics=metrics,
    )


def _copy_state(state: ActorState) -> ActorState:
    return ActorState(
        schema=state.schema,
        parameters=(state.parameters[0], state.parameters[1]),
        accepted_updates=state.accepted_updates,
    )


def _work_summary(config: HistoricalBehavioralKLRetentionConfig) -> WorkSummary:
    b_arm_steps = len(ARM_NAMES) * config.b_interference_steps
    candidate_evaluations = len(CANDIDATE_COMPONENT_NAMES) * b_arm_steps
    return WorkSummary(
        prefix_task_objective_evaluations=config.a_prefix_steps,
        b_task_objective_evaluations=b_arm_steps,
        historical_kl_objective_evaluations=b_arm_steps,
        current_kl_objective_evaluations=b_arm_steps,
        movement_l2_objective_evaluations=b_arm_steps,
        total_candidate_objective_evaluations=candidate_evaluations,
        total_candidate_gradient_float64_scalars=(
            candidate_evaluations * PARAMETER_COUNT
        ),
        prefix_parameter_updates=config.a_prefix_steps,
        routed_parameter_updates=b_arm_steps,
        total_parameter_updates=config.a_prefix_steps + b_arm_steps,
        addressed_parameter_float64_scalars=(
            (config.a_prefix_steps + b_arm_steps) * PARAMETER_COUNT
        ),
        frozen_policy_probability_evaluations=2 * len(ARM_NAMES),
        b_pre_post_probe_probability_evaluations=4 * b_arm_steps,
        rng_draws=0,
        global_shrink_evaluations=0,
    )


def _scaling_summary(config: HistoricalBehavioralKLRetentionConfig) -> ScalingSummary:
    trace_records = config.a_prefix_steps + len(ARM_NAMES) * config.b_interference_steps
    return ScalingSummary(
        actor_parameter_count=PARAMETER_COUNT,
        historical_anchor_count=1,
        historical_anchor_float64_scalars=PARAMETER_COUNT,
        persistent_float64_scalars_per_added_anchor=PARAMETER_COUNT,
        candidate_gradient_scalars_per_b_event=(
            len(CANDIDATE_COMPONENT_NAMES) * PARAMETER_COUNT
        ),
        trace_records_total=trace_records,
        trace_record_growth_formula="a_prefix_steps + arm_count * b_interference_steps",
        candidate_work_growth_formula=(
            "arm_count * b_interference_steps * candidate_component_count * parameter_count"
        ),
        persistent_anchor_growth_formula="historical_anchor_count * parameter_count",
        empirical_scale_claimed=False,
    )


def _execute_unchecked(
    config: HistoricalBehavioralKLRetentionConfig,
) -> HistoricalBehavioralKLRetentionReport:
    source = _build_source_unchecked(config)
    initial_state = ActorState(
        schema=STATE_SCHEMA,
        parameters=(0.0, 0.0),
        accepted_updates=0,
    )
    frozen_a_state, prefix_trace = _run_a_prefix(
        initial_state,
        source.a_events,
        config,
    )
    anchor_event = source.a_events[-1]
    anchor = RetainedRealStateAnchor(
        kind="retained_real_A_state_anchor",
        state=anchor_event.state,
        source_event_ordinal=anchor_event.ordinal,
        source_event_sha256=_sha256(anchor_event),
        world_value_snapshot_used=False,
        synthetic_world_grounding_used=False,
        is_dream=False,
    )
    arms = cast(
        tuple[ArmRun, ArmRun, ArmRun, ArmRun],
        tuple(
            _run_b_arm(
                name,
                routing,
                _copy_state(frozen_a_state),
                source.b_events,
                anchor,
                frozen_a_state,
                config,
            )
            for name, routing in zip(ARM_NAMES, ARM_ROUTINGS, strict=True)
        ),
    )
    pareto = cast(
        tuple[ParetoCoordinate, ParetoCoordinate, ParetoCoordinate, ParetoCoordinate],
        tuple(
            ParetoCoordinate(
                arm_name=arm.name,
                retained_a_return=arm.metrics.a_return_after_b,
                a_forgetting=arm.metrics.a_forgetting,
                b_return=arm.metrics.b_return_after_b,
                b_plasticity_gain=arm.metrics.b_plasticity_gain,
            )
            for arm in arms
        ),
    )
    b_event_stream_sha = _sha256(source.b_events)
    audits = cast(
        tuple[MatchedArmAudit, MatchedArmAudit, MatchedArmAudit, MatchedArmAudit],
        tuple(
            MatchedArmAudit(
                arm_name=arm.name,
                routing=arm.routing,
                b_event_stream_sha256=b_event_stream_sha,
                events_consumed=config.b_interference_steps,
                updates_attempted=config.b_interference_steps,
                updates_applied=config.b_interference_steps,
                candidate_objectives_per_event=len(CANDIDATE_COMPONENT_NAMES),
                candidate_gradient_float64_scalars_per_event=(
                    len(CANDIDATE_COMPONENT_NAMES) * PARAMETER_COUNT
                ),
                parameters_addressed_per_event=PARAMETER_COUNT,
                posthoc_recovery_updates=0,
                rng_draws=0,
            )
            for arm in arms
        ),
    )
    work = _work_summary(config)
    trace_payload = (prefix_trace, tuple(arm.trace for arm in arms))
    resource = ResourceSummary(
        parameter_count=PARAMETER_COUNT,
        actor_state_logical_nbytes=PARAMETER_COUNT * 8 + 8,
        retained_anchor_float64_scalars=PARAMETER_COUNT,
        retained_anchor_logical_nbytes=PARAMETER_COUNT * 8,
        frozen_actor_parameter_float64_scalars=PARAMETER_COUNT,
        per_arm_actor_state_logical_nbytes=PARAMETER_COUNT * 8 + 8,
        prefix_trace_records=config.a_prefix_steps,
        b_trace_records_per_arm=config.b_interference_steps,
        total_trace_records=config.a_prefix_steps
        + len(ARM_NAMES) * config.b_interference_steps,
        canonical_source_nbytes=len(_canonical_bytes(source)),
        canonical_trace_nbytes=len(_canonical_bytes(trace_payload)),
        max_report_bytes=config.max_report_bytes,
        hard_max_report_bytes=HARD_MAX_REPORT_BYTES,
        report_cap_enforced=True,
    )
    scaling = _scaling_summary(config)
    report = HistoricalBehavioralKLRetentionReport(
        schema=REPORT_SCHEMA,
        status="development_only_descriptive_not_assessed",
        development_only=DEVELOPMENT_ONLY,
        assessment_status=ASSESSMENT_STATUS,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        benchmark_execution_authority=BENCHMARK_EXECUTION_AUTHORITY,
        artifact_authority=ARTIFACT_AUTHORITY,
        output_writes_allowed=OUTPUT_WRITES_ALLOWED,
        evidence_claimed=EVIDENCE_CLAIMED,
        thresholds_frozen=THRESHOLDS_FROZEN,
        cpo_reproduction=CPO_REPRODUCTION,
        mrcl_reproduction=MRCL_REPRODUCTION,
        vlm_generalization_claimed=VLM_GENERALIZATION_CLAIMED,
        rng_used=RNG_USED,
        global_shrink_applied=GLOBAL_SHRINK_APPLIED,
        posthoc_recovery_updates=0,
        arm_names=ARM_NAMES,
        candidate_component_names=CANDIDATE_COMPONENT_NAMES,
        config=config,
        source=source,
        initial_state=initial_state,
        frozen_a_state=frozen_a_state,
        retained_a_anchor=anchor,
        prefix_trace=prefix_trace,
        arms=arms,
        pareto_coordinates=pareto,
        matched_arm_audits=audits,
        work=work,
        resource=resource,
        scaling=scaling,
        implementation_source_sha256=_implementation_source_sha256(),
        config_sha256=_sha256(config),
        source_sha256=_sha256(source),
        initial_state_sha256=_sha256(initial_state),
        frozen_a_state_sha256=_sha256(frozen_a_state),
        retained_anchor_sha256=_sha256(anchor),
        arm_states_sha256=_sha256(
            tuple((arm.initial_state, arm.final_state) for arm in arms)
        ),
        trace_sha256=_sha256(trace_payload),
        work_sha256=_sha256(work),
        resource_sha256=_sha256(resource),
        scaling_sha256=_sha256(scaling),
        limitations=_LIMITATIONS,
    )
    if len(_canonical_bytes(report)) > config.max_report_bytes:
        raise RuntimeError("canonical in-memory report exceeds max_report_bytes")
    if len(_canonical_bytes(report)) > HARD_MAX_REPORT_BYTES:
        raise RuntimeError("canonical in-memory report exceeds the hard byte cap")
    return report


def run_historical_behavioral_kl_retention_development(
    config: HistoricalBehavioralKLRetentionConfig | None = None,
) -> HistoricalBehavioralKLRetentionReport:
    """Run the real A prefix and four matched in-B interventions in memory."""

    supplied = HistoricalBehavioralKLRetentionConfig() if config is None else config
    return _execute_unchecked(_canonical_config_copy(supplied))


def validate_historical_behavioral_kl_retention_report(
    report: HistoricalBehavioralKLRetentionReport,
) -> tuple[str, ...]:
    """Reconstruct the complete deterministic report without recursive validation."""

    if type(report) is not HistoricalBehavioralKLRetentionReport:
        return ("report type differs",)
    errors: list[str] = []
    if type(report.config) is not HistoricalBehavioralKLRetentionConfig:
        return ("report config type differs",)
    try:
        config = _canonical_config_copy(report.config)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return ("report config fields are not canonical",)
    if type(report.source) is not HistoricalBehavioralKLSource:
        return ("report source type differs",)
    source_errors = validate_historical_behavioral_kl_source(report.source)
    errors.extend(source_errors)
    if (
        type(report.arm_names) is not tuple
        or any(type(name) is not str for name in report.arm_names)
        or report.arm_names != ARM_NAMES
    ):
        errors.append("report arm names differ")
    if (
        type(report.candidate_component_names) is not tuple
        or any(type(name) is not str for name in report.candidate_component_names)
        or report.candidate_component_names != CANDIDATE_COMPONENT_NAMES
    ):
        errors.append("report candidate component names differ")
    nested_types = (
        (report.initial_state, ActorState, "initial state type differs"),
        (report.frozen_a_state, ActorState, "frozen A state type differs"),
        (
            report.retained_a_anchor,
            RetainedRealStateAnchor,
            "retained A anchor type differs",
        ),
        (report.work, WorkSummary, "work summary type differs"),
        (report.resource, ResourceSummary, "resource summary type differs"),
        (report.scaling, ScalingSummary, "scaling summary type differs"),
    )
    nested_valid = True
    for value, expected_type, message in nested_types:
        if type(value) is not expected_type:
            errors.append(message)
            nested_valid = False
    if type(report.prefix_trace) is not tuple or any(
        type(record) is not PrefixStepTrace for record in report.prefix_trace
    ):
        errors.append("prefix trace types differ")
        nested_valid = False
    if type(report.arms) is not tuple or len(report.arms) != len(ARM_NAMES) or any(
        type(arm) is not ArmRun for arm in report.arms
    ):
        errors.append("arm types or cardinality differ")
        nested_valid = False
    elif tuple(arm.name for arm in report.arms) != ARM_NAMES or any(
        type(arm.name) is not str for arm in report.arms
    ):
        errors.append("arm names or order differ")
        nested_valid = False
    else:
        for arm in report.arms:
            if type(arm.trace) is not tuple or any(
                type(record) is not BInterferenceStepTrace for record in arm.trace
            ):
                errors.append(f"arm {arm.name} trace types differ")
                nested_valid = False
            if type(arm.initial_state) is not ActorState or type(arm.final_state) is not ActorState:
                errors.append(f"arm {arm.name} state types differ")
                nested_valid = False
            if type(arm.metrics) is not ArmMetrics:
                errors.append(f"arm {arm.name} metrics type differs")
                nested_valid = False
    tuple_contracts = (
        (report.pareto_coordinates, ParetoCoordinate, "Pareto coordinate types differ"),
        (report.matched_arm_audits, MatchedArmAudit, "matched arm audit types differ"),
    )
    for values, tuple_expected_type, message in tuple_contracts:
        if type(values) is not tuple or len(values) != len(ARM_NAMES) or any(
            type(item) is not tuple_expected_type for item in values
        ):
            errors.append(message)
            nested_valid = False
    if (
        type(report.limitations) is not tuple
        or any(type(item) is not str for item in report.limitations)
        or report.limitations != _LIMITATIONS
    ):
        errors.append("report limitations differ")
    if not nested_valid:
        return tuple(errors)
    report_finite = _all_finite(report)
    if not report_finite:
        errors.append("report contains non-finite values")

    expected = _execute_unchecked(config)
    if not _exact(report.source.config, report.config):
        errors.append("report and source configs are not bit-exactly bound")
    if not _exact(report, expected):
        errors.append("report does not reconstruct bit-exactly")
    if report.implementation_source_sha256 != _implementation_source_sha256():
        errors.append("implementation source digest differs")
    if report_finite:
        try:
            hashes = (
                (report.config_sha256, _sha256(report.config), "config digest differs"),
                (report.source_sha256, _sha256(report.source), "source digest differs"),
                (
                    report.initial_state_sha256,
                    _sha256(report.initial_state),
                    "initial state digest differs",
                ),
                (
                    report.frozen_a_state_sha256,
                    _sha256(report.frozen_a_state),
                    "frozen A state digest differs",
                ),
                (
                    report.retained_anchor_sha256,
                    _sha256(report.retained_a_anchor),
                    "retained anchor digest differs",
                ),
                (
                    report.arm_states_sha256,
                    _sha256(
                        tuple(
                            (arm.initial_state, arm.final_state) for arm in report.arms
                        )
                    ),
                    "arm state digest differs",
                ),
                (
                    report.trace_sha256,
                    _sha256(
                        (
                            report.prefix_trace,
                            tuple(arm.trace for arm in report.arms),
                        )
                    ),
                    "trace digest differs",
                ),
                (report.work_sha256, _sha256(report.work), "work digest differs"),
                (
                    report.resource_sha256,
                    _sha256(report.resource),
                    "resource digest differs",
                ),
                (
                    report.scaling_sha256,
                    _sha256(report.scaling),
                    "scaling digest differs",
                ),
            )
        except (TypeError, ValueError):
            errors.append("report digest payload is not canonical")
        else:
            for actual, recomputed, message in hashes:
                if type(actual) is not str or actual != recomputed:
                    errors.append(message)
        try:
            report_nbytes = len(_canonical_bytes(report))
        except (TypeError, ValueError):
            errors.append("report cannot be canonically serialized")
        else:
            if report_nbytes > config.max_report_bytes:
                errors.append("report exceeds configured byte cap")
            if report_nbytes > HARD_MAX_REPORT_BYTES:
                errors.append("report exceeds hard byte cap")
    return tuple(errors)


__all__ = [
    "ACTION_COUNT",
    "ARM_NAMES",
    "ARM_ROUTINGS",
    "ARTIFACT_AUTHORITY",
    "ASSESSMENT_STATUS",
    "ArmMetrics",
    "ArmRun",
    "BENCHMARK_EXECUTION_AUTHORITY",
    "CANDIDATE_COMPONENT_NAMES",
    "CONFIG_SCHEMA",
    "CPO_REPRODUCTION",
    "ContextualBanditEvent",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_CLAIMED",
    "GLOBAL_SHRINK_APPLIED",
    "HistoricalBehavioralKLRetentionConfig",
    "HistoricalBehavioralKLRetentionReport",
    "HistoricalBehavioralKLSource",
    "MRCL_REPRODUCTION",
    "MatchedArmAudit",
    "OUTPUT_WRITES_ALLOWED",
    "PARAMETER_COUNT",
    "ParetoCoordinate",
    "REPORT_SCHEMA",
    "RNG_USED",
    "RetainedRealStateAnchor",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "THRESHOLDS_FROZEN",
    "VLM_GENERALIZATION_CLAIMED",
    "build_historical_behavioral_kl_source",
    "run_historical_behavioral_kl_retention_development",
    "validate_historical_behavioral_kl_retention_report",
    "validate_historical_behavioral_kl_source",
]
