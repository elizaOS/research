"""Pure evaluator geometry for the nonexecuting HCCL continual ladder.

The objects in this module describe schedule coverage and exact integer work
opportunities.  They do not run a world or an agent, reserve a seed, select a
threshold, write an artifact, validate evidence, or authorize promotion.  In
particular, the Scale-L4 resource values are calculator results: no ``2**N``
attribution cube (and no randomized-coalition budget) is selected as an
executable Scale-L4 policy here.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
from collections.abc import Mapping
from typing import Final

from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_SCHEDULE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    hccl_causal_core_cycle_count_for_profile,
    hccl_causal_core_lifetime_for_profile,
    hccl_causal_core_schedule_for_profile,
)

HCCL_CONTINUAL_LADDER_CONFIG_SCHEMA: Final = "alberta.hccl-continual-ladder.config.v1"
HCCL_CONTINUAL_LADDER_STATUS: Final = (
    "l0-development-evaluator-geometry-only-nonexecuting-not-assessed"
)
HCCL_CONTINUAL_LADDER_MAX_EXACT_INTEGER: Final = 2**63 - 1
HCCL_CONTINUAL_LADDER_LIMITATIONS: Final = (
    "evaluator-geometry-only-no-world-or-agent-execution",
    "no-seed-reservation-or-consumption",
    "no-threshold-selection",
    "no-artifact-or-output-writer",
    "no-evidence-validation-or-promotion-authority",
    "smoke-420-is-mechanics-only-and-not-property-bearing",
    "longevity-geometry-does-not-authorize-promotion",
    "scale-l4-selects-no-executable-attribution-or-coalition-policy",
    "finite-calculators-do-not-recover-omitted-higher-order-interactions",
)

_REGIME_NAMES: Final = ("A", "B", "C", "D")
_CANONICAL_EVENTS: Final = hccl_causal_core_lifetime_for_profile(
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE
)
_SMOKE_EVENTS: Final = hccl_causal_core_lifetime_for_profile(HCCL_CAUSAL_CORE_SMOKE_PROFILE)
_CORE_L2_CYCLES: Final = hccl_causal_core_cycle_count_for_profile(
    HCCL_CAUSAL_CORE_L2_PROFILE
)
_CORE_L3_CYCLES: Final = hccl_causal_core_cycle_count_for_profile(
    HCCL_CAUSAL_CORE_L3_PROFILE
)
_SCALE_L4_AGENTS: Final = 4
_SCALE_L4_RING_DEGREE: Final = 2
_DYAD_AGENTS: Final = 2
_DYAD_NEIGHBOR_DEGREE: Final = 1
_ACTION_COUNT: Final = 2
_ADJACENT_ACTION_LAYERS: Final = 2
_BASE_FEATURE_WIDTH: Final = 16
_GENERATED_FEATURE_COUNT: Final = 12
_MEMORY_ROWS: Final = 64


class HCCLContinualLadderRung(enum.StrEnum):
    """The five frozen, development-only HCCL evaluator geometries."""

    SMOKE = "smoke-420-v1"
    SMOKE_420 = SMOKE
    CORE_L1 = "core-l1-8998-v1"
    CORE_L2 = "core-l2-71984-v1"
    CORE_L3 = "core-l3-1007776-v1"
    SCALE_L4 = "scale-l4-ring4-8998-v1"


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualLadderConfig:
    """Strict selector with fail-closed nonauthority declarations."""

    rung: HCCLContinualLadderRung = HCCLContinualLadderRung.CORE_L1
    evaluator_only: bool = True
    learner_visible_regime_labels: bool = False
    execution_authorized: bool = False
    seed_reservation_or_consumption_authorized: bool = False
    threshold_selection_authorized: bool = False
    artifact_or_output_writes_authorized: bool = False
    evidence_validation_authorized: bool = False
    promotion_authorized: bool = False
    longevity_authorizes_promotion: bool = False
    scale_exact_attribution_policy_selected: bool = False
    scale_randomized_coalition_budget_selected: bool = False
    omitted_higher_order_interactions_recovered: bool = False

    def __post_init__(self) -> None:
        if type(self.rung) is not HCCLContinualLadderRung:
            raise TypeError("rung must be an HCCLContinualLadderRung")
        fixed_flags = {
            "evaluator_only": (self.evaluator_only, True),
            "learner_visible_regime_labels": (self.learner_visible_regime_labels, False),
            "execution_authorized": (self.execution_authorized, False),
            "seed_reservation_or_consumption_authorized": (
                self.seed_reservation_or_consumption_authorized,
                False,
            ),
            "threshold_selection_authorized": (self.threshold_selection_authorized, False),
            "artifact_or_output_writes_authorized": (
                self.artifact_or_output_writes_authorized,
                False,
            ),
            "evidence_validation_authorized": (self.evidence_validation_authorized, False),
            "promotion_authorized": (self.promotion_authorized, False),
            "longevity_authorizes_promotion": (self.longevity_authorizes_promotion, False),
            "scale_exact_attribution_policy_selected": (
                self.scale_exact_attribution_policy_selected,
                False,
            ),
            "scale_randomized_coalition_budget_selected": (
                self.scale_randomized_coalition_budget_selected,
                False,
            ),
            "omitted_higher_order_interactions_recovered": (
                self.omitted_higher_order_interactions_recovered,
                False,
            ),
        }
        for name, (actual, expected) in fixed_flags.items():
            if type(actual) is not bool or actual is not expected:
                raise ValueError(f"{name} is a fixed ladder nonauthority declaration")

    @property
    def event_count(self) -> int:
        return _rung_geometry(self.rung)[0]

    @property
    def cycle_count(self) -> int:
        return _rung_geometry(self.rung)[1]

    @property
    def learning_agent_count(self) -> int:
        return _rung_geometry(self.rung)[2]

    @property
    def fixed_neighbor_degree(self) -> int:
        return _rung_geometry(self.rung)[3]

    @property
    def agent_topology(self) -> str:
        if self.rung is HCCLContinualLadderRung.SCALE_L4:
            return "fixed-undirected-four-agent-ring-v1"
        return "two-agent-dyad-v1"

    @property
    def fixed_neighbor_indices(self) -> tuple[tuple[int, ...], ...]:
        if self.rung is HCCLContinualLadderRung.SCALE_L4:
            return ((1, 3), (0, 2), (1, 3), (0, 2))
        return ((1,), (0,))

    @property
    def source_schedule_profile(self) -> str:
        if self.rung is HCCLContinualLadderRung.SMOKE_420:
            return HCCL_CAUSAL_CORE_SMOKE_PROFILE
        if self.rung is HCCLContinualLadderRung.CORE_L2:
            return HCCL_CAUSAL_CORE_L2_PROFILE
        if self.rung is HCCLContinualLadderRung.CORE_L3:
            return HCCL_CAUSAL_CORE_L3_PROFILE
        return HCCL_CAUSAL_CORE_CANONICAL_PROFILE

    @property
    def smoke_mechanics_only(self) -> bool:
        return self.rung is HCCLContinualLadderRung.SMOKE_420

    @property
    def property_bearing_development_geometry(self) -> bool:
        return not self.smoke_mechanics_only

    @property
    def longevity_geometry(self) -> bool:
        return self.rung in (
            HCCLContinualLadderRung.CORE_L2,
            HCCLContinualLadderRung.CORE_L3,
        )

    def to_config(self) -> dict[str, object]:
        """Return the complete strict JSON-compatible declaration."""

        return {
            "schema": HCCL_CONTINUAL_LADDER_CONFIG_SCHEMA,
            "status": HCCL_CONTINUAL_LADDER_STATUS,
            "rung": self.rung.value,
            "event_count": self.event_count,
            "cycle_count": self.cycle_count,
            "learning_agent_count": self.learning_agent_count,
            "fixed_neighbor_degree": self.fixed_neighbor_degree,
            "agent_topology": self.agent_topology,
            "fixed_neighbor_indices": [list(row) for row in self.fixed_neighbor_indices],
            "source_schedule_profile": self.source_schedule_profile,
            "smoke_mechanics_only": self.smoke_mechanics_only,
            "property_bearing_development_geometry": (
                self.property_bearing_development_geometry
            ),
            "longevity_geometry": self.longevity_geometry,
            "evaluator_only": self.evaluator_only,
            "learner_visible_regime_labels": self.learner_visible_regime_labels,
            "execution_authorized": self.execution_authorized,
            "seed_reservation_or_consumption_authorized": (
                self.seed_reservation_or_consumption_authorized
            ),
            "threshold_selection_authorized": self.threshold_selection_authorized,
            "artifact_or_output_writes_authorized": self.artifact_or_output_writes_authorized,
            "evidence_validation_authorized": self.evidence_validation_authorized,
            "promotion_authorized": self.promotion_authorized,
            "longevity_authorizes_promotion": self.longevity_authorizes_promotion,
            "scale_exact_attribution_policy_selected": (
                self.scale_exact_attribution_policy_selected
            ),
            "scale_randomized_coalition_budget_selected": (
                self.scale_randomized_coalition_budget_selected
            ),
            "omitted_higher_order_interactions_recovered": (
                self.omitted_higher_order_interactions_recovered
            ),
            "limitations": list(HCCL_CONTINUAL_LADDER_LIMITATIONS),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLContinualLadderConfig:
        """Reconstruct only the exact, complete versioned declaration."""

        if type(payload) is not dict:
            raise TypeError("ladder config must be a plain dictionary")
        rung_value = payload.get("rung")
        if type(rung_value) is not str:
            raise ValueError("ladder config rung must be a string enum value")
        try:
            rung = HCCLContinualLadderRung(rung_value)
        except ValueError as error:
            raise ValueError("ladder config rung is unsupported") from error
        candidate = cls(rung=rung)
        if _canonical_json_bytes(payload) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("ladder config must exactly match the frozen declaration")
        return candidate

    def to_json(self) -> str:
        """Serialize the complete declaration using canonical strict JSON."""

        return _canonical_json_bytes(self.to_config()).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str) -> HCCLContinualLadderConfig:
        """Parse JSON while rejecting duplicate keys and nonfinite constants."""

        if type(payload) is not str:
            raise TypeError("ladder JSON must be a string")
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("ladder JSON is invalid or non-strict") from error
        if type(decoded) is not dict:
            raise ValueError("ladder JSON must encode one object")
        return cls.from_config(decoded)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLScheduleOccurrence:
    """One evaluator-only absolute half-open regime occurrence."""

    evaluator_regime_name: str
    start: int
    end: int
    cycle_index: int
    segment_index: int
    occurrence_index: int
    regime_occurrence_index: int

    def __post_init__(self) -> None:
        if type(self.evaluator_regime_name) is not str or self.evaluator_regime_name not in (
            _REGIME_NAMES
        ):
            raise ValueError("evaluator_regime_name must be one fixed HCCL regime")
        for name, value, minimum in (
            ("start", self.start, 0),
            ("end", self.end, 1),
            ("cycle_index", self.cycle_index, 0),
            ("segment_index", self.segment_index, 0),
            ("occurrence_index", self.occurrence_index, 0),
            ("regime_occurrence_index", self.regime_occurrence_index, 0),
        ):
            _require_integer(value, name=name, minimum=minimum)
        if self.end <= self.start:
            raise ValueError("schedule occurrence must have positive half-open length")
        if self.segment_index >= len(HCCL_CAUSAL_CORE_SCHEDULE):
            raise ValueError("segment_index is outside the fixed ten-segment geometry")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualSchedule:
    """Frozen evaluator schedule; labels are never a learner input."""

    rung: HCCLContinualLadderRung
    event_count: int
    cycle_count: int
    occurrences: tuple[HCCLScheduleOccurrence, ...]
    evaluator_only: bool = True
    learner_visible_regime_labels: bool = False

    def __post_init__(self) -> None:
        if type(self.rung) is not HCCLContinualLadderRung:
            raise TypeError("schedule rung must be an HCCLContinualLadderRung")
        _require_integer(self.event_count, name="event_count", minimum=1)
        _require_integer(self.cycle_count, name="cycle_count", minimum=1)
        if type(self.occurrences) is not tuple or not self.occurrences:
            raise TypeError("occurrences must be a nonempty tuple")
        if any(type(item) is not HCCLScheduleOccurrence for item in self.occurrences):
            raise TypeError("occurrences must contain only HCCLScheduleOccurrence values")
        if type(self.evaluator_only) is not bool or self.evaluator_only is not True:
            raise ValueError("the ladder schedule is evaluator-only")
        if (
            type(self.learner_visible_regime_labels) is not bool
            or self.learner_visible_regime_labels is not False
        ):
            raise ValueError("regime labels must not be learner-visible")
        expected_events, expected_cycles, _, _ = _rung_geometry(self.rung)
        if self.event_count != expected_events or self.cycle_count != expected_cycles:
            raise ValueError("schedule counts must match the frozen rung")
        source = _source_schedule(self.rung)
        expected_segments = len(source)
        if len(self.occurrences) != expected_segments:
            raise ValueError("schedule must contain every fixed segment in every cycle")
        previous_end = 0
        per_regime: dict[str, int] = {name: 0 for name in _REGIME_NAMES}
        for index, occurrence in enumerate(self.occurrences):
            cycle_index, segment_index = divmod(index, len(HCCL_CAUSAL_CORE_SCHEDULE))
            source_name, source_start, source_end = source[index]
            if occurrence.start != source_start or occurrence.end != source_end:
                raise ValueError("schedule occurrences must give exact contiguous coverage")
            if (
                occurrence.cycle_index != cycle_index
                or occurrence.segment_index != segment_index
                or occurrence.occurrence_index != index
                or occurrence.evaluator_regime_name != source_name
                or occurrence.regime_occurrence_index != per_regime[source_name]
            ):
                raise ValueError("schedule occurrence metadata is not canonical")
            if occurrence.start != previous_end:
                raise ValueError("schedule occurrences must give exact contiguous coverage")
            per_regime[source_name] += 1
            previous_end = occurrence.end
        if previous_end != self.event_count:
            raise ValueError("schedule coverage must end exactly at event_count")

    def regime_occurrence_counts(self) -> dict[str, int]:
        return {
            name: sum(item.evaluator_regime_name == name for item in self.occurrences)
            for name in _REGIME_NAMES
        }

    def regime_event_counts(self) -> dict[str, int]:
        return {
            name: sum(
                item.length for item in self.occurrences if item.evaluator_regime_name == name
            )
            for name in _REGIME_NAMES
        }


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLResourceGeometry:
    """Exact finite calculator output, never an executable policy selection."""

    n_agents: int
    action_count: int
    learned_neighbor_degree: int
    adjacent_action_layers: int
    base_feature_width: int
    generated_feature_count: int
    memory_rows: int
    exact_all_agent_attribution_calls: int
    distinct_action_receipt_vertex_upper_bound: int
    effective_joint_action_bound: int
    unilateral_calls_per_layer: int
    pairwise_coalition_audit_calls_per_layer: int
    factorized_planner_cells_per_agent: int
    factorized_planner_cells_total: int
    raw_pair_candidates: int
    pair_memory_reencode_products_per_agent: int
    calculator_only: bool = True
    selected_executable_attribution_policy: bool = False

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("n_agents", self.n_agents, 1),
            ("action_count", self.action_count, 2),
            ("learned_neighbor_degree", self.learned_neighbor_degree, 0),
            ("adjacent_action_layers", self.adjacent_action_layers, 1),
            ("base_feature_width", self.base_feature_width, 2),
            ("generated_feature_count", self.generated_feature_count, 1),
            ("memory_rows", self.memory_rows, 1),
        ):
            _require_integer(value, name=name, minimum=minimum)
        if self.learned_neighbor_degree >= self.n_agents:
            raise ValueError("learned_neighbor_degree must be smaller than n_agents")
        if type(self.calculator_only) is not bool or self.calculator_only is not True:
            raise ValueError("resource geometry is calculator-only")
        if (
            type(self.selected_executable_attribution_policy) is not bool
            or self.selected_executable_attribution_policy is not False
        ):
            raise ValueError("resource geometry cannot select an executable policy")
        expected = _calculate_resource_values(
            n_agents=self.n_agents,
            action_count=self.action_count,
            learned_neighbor_degree=self.learned_neighbor_degree,
            adjacent_action_layers=self.adjacent_action_layers,
            base_feature_width=self.base_feature_width,
            generated_feature_count=self.generated_feature_count,
            memory_rows=self.memory_rows,
        )
        actual = (
            self.exact_all_agent_attribution_calls,
            self.distinct_action_receipt_vertex_upper_bound,
            self.effective_joint_action_bound,
            self.unilateral_calls_per_layer,
            self.pairwise_coalition_audit_calls_per_layer,
            self.factorized_planner_cells_per_agent,
            self.factorized_planner_cells_total,
            self.raw_pair_candidates,
            self.pair_memory_reencode_products_per_agent,
        )
        if actual != expected:
            raise ValueError("resource geometry values must equal the exact integer formulas")


def calculate_hccl_resource_geometry(
    *,
    n_agents: int,
    action_count: int,
    learned_neighbor_degree: int,
    adjacent_action_layers: int,
    base_feature_width: int,
    generated_feature_count: int,
    memory_rows: int,
) -> HCCLResourceGeometry:
    """Calculate the exact finite bounds without selecting any run policy."""

    values = _calculate_resource_values(
        n_agents=n_agents,
        action_count=action_count,
        learned_neighbor_degree=learned_neighbor_degree,
        adjacent_action_layers=adjacent_action_layers,
        base_feature_width=base_feature_width,
        generated_feature_count=generated_feature_count,
        memory_rows=memory_rows,
    )
    return HCCLResourceGeometry(
        n_agents=n_agents,
        action_count=action_count,
        learned_neighbor_degree=learned_neighbor_degree,
        adjacent_action_layers=adjacent_action_layers,
        base_feature_width=base_feature_width,
        generated_feature_count=generated_feature_count,
        memory_rows=memory_rows,
        exact_all_agent_attribution_calls=values[0],
        distinct_action_receipt_vertex_upper_bound=values[1],
        effective_joint_action_bound=values[2],
        unilateral_calls_per_layer=values[3],
        pairwise_coalition_audit_calls_per_layer=values[4],
        factorized_planner_cells_per_agent=values[5],
        factorized_planner_cells_total=values[6],
        raw_pair_candidates=values[7],
        pair_memory_reencode_products_per_agent=values[8],
    )


def calculate_randomized_coalition_audit_calls(
    *, adjacent_action_layers: int, calls_per_layer_budget: int
) -> int:
    """Total a randomized audit only when its positive budget is explicit."""

    layers = _require_integer(
        adjacent_action_layers, name="adjacent_action_layers", minimum=1
    )
    budget = _require_integer(calls_per_layer_budget, name="calls_per_layer_budget", minimum=1)
    return _checked_multiply(layers, budget, name="randomized coalition audit calls")


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLExactOpportunityCount:
    """One exact named per-event and per-life opportunity count."""

    per_event: int
    per_life: int

    def __post_init__(self) -> None:
        _require_integer(self.per_event, name="per_event", minimum=0)
        _require_integer(self.per_life, name="per_life", minimum=0)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLNamedOpportunityCounts:
    """Named accounting opportunities; these are not measured executions."""

    events: int
    environment_proposal_calls: HCCLExactOpportunityCount | None
    context_updates: HCCLExactOpportunityCount
    fast_state_updates: HCCLExactOpportunityCount
    sequential_lineage_sidecar_proposals: HCCLExactOpportunityCount
    generated_feature_lifecycle_routes: HCCLExactOpportunityCount
    memory_query_write_transactions: HCCLExactOpportunityCount
    behavior_model_candidates: HCCLExactOpportunityCount
    grounded_world_model_candidates: HCCLExactOpportunityCount
    planner_decisions: HCCLExactOpportunityCount
    one_step_joint_planner_cells: HCCLExactOpportunityCount
    real_multistep_backups: HCCLExactOpportunityCount
    dyad_eight_call_cube_selected_for_accounting: bool
    scale_dyad_cube_generalized_or_selected: bool = False

    def __post_init__(self) -> None:
        _require_integer(self.events, name="events", minimum=1)
        if type(self.dyad_eight_call_cube_selected_for_accounting) is not bool:
            raise TypeError("dyad cube accounting declaration must be boolean")
        if (
            type(self.scale_dyad_cube_generalized_or_selected) is not bool
            or self.scale_dyad_cube_generalized_or_selected is not False
        ):
            raise ValueError("the Scale-L4 plan does not generalize or select the dyad cube")
        named = (
            self.context_updates,
            self.fast_state_updates,
            self.sequential_lineage_sidecar_proposals,
            self.generated_feature_lifecycle_routes,
            self.memory_query_write_transactions,
            self.behavior_model_candidates,
            self.grounded_world_model_candidates,
            self.planner_decisions,
            self.one_step_joint_planner_cells,
            self.real_multistep_backups,
        )
        if any(type(item) is not HCCLExactOpportunityCount for item in named):
            raise TypeError("all named opportunity values must be exact counts")
        if self.environment_proposal_calls is not None and type(
            self.environment_proposal_calls
        ) is not HCCLExactOpportunityCount:
            raise TypeError("environment proposal opportunities must be an exact count or None")
        environment = (
            (self.environment_proposal_calls,)
            if self.environment_proposal_calls is not None
            else ()
        )
        for item in (*named, *environment):
            if item.per_life != _checked_multiply(
                item.per_event, self.events, name="per-life opportunity count"
            ):
                raise ValueError("per-life opportunity count must equal per_event * events")
        if self.dyad_eight_call_cube_selected_for_accounting:
            if self.environment_proposal_calls != HCCLExactOpportunityCount(
                per_event=8,
                per_life=_checked_multiply(8, self.events, name="dyad proposal calls"),
            ):
                raise ValueError("dyad accounting must declare exactly eight proposals per event")
        elif self.environment_proposal_calls is not None:
            raise ValueError("non-dyad accounting must not invent an environment proposal policy")
        if self.real_multistep_backups != HCCLExactOpportunityCount(0, 0):
            raise ValueError("the causal-core ladder has zero real multistep backups")


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualLadderPlan:
    """One immutable nonexecuting evaluator plan."""

    config: HCCLContinualLadderConfig
    schedule: HCCLContinualSchedule
    resources: HCCLResourceGeometry
    opportunities: HCCLNamedOpportunityCounts

    def __post_init__(self) -> None:
        if type(self.config) is not HCCLContinualLadderConfig:
            raise TypeError("config must be an HCCLContinualLadderConfig")
        if type(self.schedule) is not HCCLContinualSchedule:
            raise TypeError("schedule must be an HCCLContinualSchedule")
        if type(self.resources) is not HCCLResourceGeometry:
            raise TypeError("resources must be an HCCLResourceGeometry")
        if type(self.opportunities) is not HCCLNamedOpportunityCounts:
            raise TypeError("opportunities must be HCCLNamedOpportunityCounts")
        if self.schedule.rung is not self.config.rung:
            raise ValueError("plan schedule and config rungs must match")
        if self.opportunities.events != self.config.event_count:
            raise ValueError("plan opportunities must cover the exact rung event count")
        expected_resource = (
            HCCL_SCALE_L4_RING_RESOURCE_GEOMETRY
            if self.config.rung is HCCLContinualLadderRung.SCALE_L4
            else HCCL_DYAD_RESOURCE_GEOMETRY
        )
        if self.resources != expected_resource:
            raise ValueError("plan resources must match the frozen rung geometry")


def build_hccl_continual_ladder(
    config: HCCLContinualLadderConfig,
) -> HCCLContinualLadderPlan:
    """Build schedule and arithmetic records without executing any mechanism."""

    if type(config) is not HCCLContinualLadderConfig:
        raise TypeError("config must be an HCCLContinualLadderConfig")
    schedule = _build_schedule(config)
    scale = config.rung is HCCLContinualLadderRung.SCALE_L4
    resources = HCCL_SCALE_L4_RING_RESOURCE_GEOMETRY if scale else HCCL_DYAD_RESOURCE_GEOMETRY
    per_agent = config.learning_agent_count

    def count(per_event: int) -> HCCLExactOpportunityCount:
        return HCCLExactOpportunityCount(
            per_event=per_event,
            per_life=_checked_multiply(
                per_event, config.event_count, name="named opportunity count"
            ),
        )

    opportunities = HCCLNamedOpportunityCounts(
        events=config.event_count,
        environment_proposal_calls=None if scale else count(8),
        context_updates=count(per_agent),
        fast_state_updates=count(per_agent),
        sequential_lineage_sidecar_proposals=count(per_agent),
        generated_feature_lifecycle_routes=count(per_agent),
        memory_query_write_transactions=count(per_agent),
        behavior_model_candidates=count(per_agent),
        grounded_world_model_candidates=count(per_agent),
        planner_decisions=count(per_agent),
        one_step_joint_planner_cells=count(resources.factorized_planner_cells_total),
        real_multistep_backups=count(0),
        dyad_eight_call_cube_selected_for_accounting=not scale,
    )
    return HCCLContinualLadderPlan(
        config=config,
        schedule=schedule,
        resources=resources,
        opportunities=opportunities,
    )


def _build_schedule(config: HCCLContinualLadderConfig) -> HCCLContinualSchedule:
    source = _source_schedule(config.rung)
    occurrences: list[HCCLScheduleOccurrence] = []
    per_regime: dict[str, int] = {name: 0 for name in _REGIME_NAMES}
    for index, (name, start, end) in enumerate(source):
        cycle_index, segment_index = divmod(index, len(HCCL_CAUSAL_CORE_SCHEDULE))
        occurrences.append(
            HCCLScheduleOccurrence(
                evaluator_regime_name=name,
                start=start,
                end=end,
                cycle_index=cycle_index,
                segment_index=segment_index,
                occurrence_index=index,
                regime_occurrence_index=per_regime[name],
            )
        )
        per_regime[name] += 1
    return HCCLContinualSchedule(
        rung=config.rung,
        event_count=config.event_count,
        cycle_count=config.cycle_count,
        occurrences=tuple(occurrences),
    )


def _rung_geometry(rung: HCCLContinualLadderRung) -> tuple[int, int, int, int]:
    if rung is HCCLContinualLadderRung.SMOKE_420:
        return _SMOKE_EVENTS, 1, _DYAD_AGENTS, _DYAD_NEIGHBOR_DEGREE
    if rung is HCCLContinualLadderRung.CORE_L1:
        return _CANONICAL_EVENTS, 1, _DYAD_AGENTS, _DYAD_NEIGHBOR_DEGREE
    if rung is HCCLContinualLadderRung.CORE_L2:
        return (
            hccl_causal_core_lifetime_for_profile(HCCL_CAUSAL_CORE_L2_PROFILE),
            _CORE_L2_CYCLES,
            _DYAD_AGENTS,
            _DYAD_NEIGHBOR_DEGREE,
        )
    if rung is HCCLContinualLadderRung.CORE_L3:
        return (
            hccl_causal_core_lifetime_for_profile(HCCL_CAUSAL_CORE_L3_PROFILE),
            _CORE_L3_CYCLES,
            _DYAD_AGENTS,
            _DYAD_NEIGHBOR_DEGREE,
        )
    if rung is HCCLContinualLadderRung.SCALE_L4:
        return _CANONICAL_EVENTS, 1, _SCALE_L4_AGENTS, _SCALE_L4_RING_DEGREE
    raise AssertionError("unreachable HCCL ladder rung")


def _source_schedule(
    rung: HCCLContinualLadderRung,
) -> tuple[tuple[str, int, int], ...]:
    if rung is HCCLContinualLadderRung.SMOKE_420:
        profile = HCCL_CAUSAL_CORE_SMOKE_PROFILE
    elif rung is HCCLContinualLadderRung.CORE_L2:
        profile = HCCL_CAUSAL_CORE_L2_PROFILE
    elif rung is HCCLContinualLadderRung.CORE_L3:
        profile = HCCL_CAUSAL_CORE_L3_PROFILE
    else:
        profile = HCCL_CAUSAL_CORE_CANONICAL_PROFILE
    return hccl_causal_core_schedule_for_profile(profile)


def _calculate_resource_values(
    *,
    n_agents: int,
    action_count: int,
    learned_neighbor_degree: int,
    adjacent_action_layers: int,
    base_feature_width: int,
    generated_feature_count: int,
    memory_rows: int,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    n_agents = _require_integer(n_agents, name="n_agents", minimum=1)
    action_count = _require_integer(action_count, name="action_count", minimum=2)
    learned_neighbor_degree = _require_integer(
        learned_neighbor_degree, name="learned_neighbor_degree", minimum=0
    )
    adjacent_action_layers = _require_integer(
        adjacent_action_layers, name="adjacent_action_layers", minimum=1
    )
    base_feature_width = _require_integer(
        base_feature_width, name="base_feature_width", minimum=2
    )
    generated_feature_count = _require_integer(
        generated_feature_count, name="generated_feature_count", minimum=1
    )
    memory_rows = _require_integer(memory_rows, name="memory_rows", minimum=1)
    if learned_neighbor_degree >= n_agents:
        raise ValueError("learned_neighbor_degree must be smaller than n_agents")

    coalition_vertices = _checked_power(2, n_agents, name="all-agent coalition vertices")
    exact_calls = _checked_multiply(
        adjacent_action_layers,
        coalition_vertices,
        name="exact all-agent attribution calls",
    )
    distinct_vertices = _checked_add(
        1,
        _checked_multiply(
            adjacent_action_layers,
            coalition_vertices - 1,
            name="distinct action-receipt vertex increments",
        ),
        name="distinct action-receipt vertex upper bound",
    )
    effective_joint_actions = _checked_power(
        action_count, n_agents, name="effective joint-action bound"
    )
    unilateral_calls = _checked_add(n_agents, 1, name="unilateral calls per layer")
    pairs = math.comb(n_agents, 2)
    _check_exact_integer(pairs, name="pairwise coalitions")
    pairwise_calls = _checked_add(
        unilateral_calls, pairs, name="pairwise-coalition audit calls per layer"
    )
    planner_cells_per_agent = _checked_power(
        action_count,
        _checked_add(learned_neighbor_degree, 1, name="planner action exponent"),
        name="factorized planner cells per agent",
    )
    planner_cells_total = _checked_multiply(
        n_agents, planner_cells_per_agent, name="factorized planner cells total"
    )
    raw_pairs = base_feature_width * (base_feature_width - 1) // 2
    _check_exact_integer(raw_pairs, name="raw-pair candidates")
    reencode_products = _checked_multiply(
        2,
        generated_feature_count,
        memory_rows,
        name="pair-memory reencode products per agent",
    )
    return (
        exact_calls,
        distinct_vertices,
        effective_joint_actions,
        unilateral_calls,
        pairwise_calls,
        planner_cells_per_agent,
        planner_cells_total,
        raw_pairs,
        reencode_products,
    )


def _require_integer(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a non-boolean integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    _check_exact_integer(value, name=name)
    return value


def _check_exact_integer(value: int, *, name: str) -> int:
    if value < 0 or value > HCCL_CONTINUAL_LADDER_MAX_EXACT_INTEGER:
        raise OverflowError(f"{name} exceeds the signed 64-bit exact-count limit")
    return value


def _checked_add(*values: int, name: str) -> int:
    return _check_exact_integer(sum(values), name=name)


def _checked_multiply(*values: int, name: str) -> int:
    product = 1
    for value in values:
        if value != 0 and product > HCCL_CONTINUAL_LADDER_MAX_EXACT_INTEGER // value:
            raise OverflowError(f"{name} exceeds the signed 64-bit exact-count limit")
        product *= value
    return _check_exact_integer(product, name=name)


def _checked_power(base: int, exponent: int, *, name: str) -> int:
    if exponent > 62:
        raise OverflowError(f"{name} exceeds the signed 64-bit exact-count limit")
    value = base**exponent
    return _check_exact_integer(value, name=name)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant {value!r} is forbidden")


HCCL_DYAD_RESOURCE_GEOMETRY: Final = calculate_hccl_resource_geometry(
    n_agents=_DYAD_AGENTS,
    action_count=_ACTION_COUNT,
    learned_neighbor_degree=_DYAD_NEIGHBOR_DEGREE,
    adjacent_action_layers=_ADJACENT_ACTION_LAYERS,
    base_feature_width=_BASE_FEATURE_WIDTH,
    generated_feature_count=_GENERATED_FEATURE_COUNT,
    memory_rows=_MEMORY_ROWS,
)
HCCL_SCALE_L4_RING_RESOURCE_GEOMETRY: Final = calculate_hccl_resource_geometry(
    n_agents=_SCALE_L4_AGENTS,
    action_count=_ACTION_COUNT,
    learned_neighbor_degree=_SCALE_L4_RING_DEGREE,
    adjacent_action_layers=_ADJACENT_ACTION_LAYERS,
    base_feature_width=_BASE_FEATURE_WIDTH,
    generated_feature_count=_GENERATED_FEATURE_COUNT,
    memory_rows=_MEMORY_ROWS,
)
