"""Contracts for the pure, nonexecuting HCCL continual ladder geometry."""

from __future__ import annotations

import dataclasses
import json

import pytest

from alberta_framework.core.hccl_continual_ladder import (
    HCCL_CONTINUAL_LADDER_CONFIG_SCHEMA,
    HCCL_CONTINUAL_LADDER_LIMITATIONS,
    HCCL_CONTINUAL_LADDER_MAX_EXACT_INTEGER,
    HCCL_CONTINUAL_LADDER_STATUS,
    HCCL_DYAD_RESOURCE_GEOMETRY,
    HCCL_SCALE_L4_RING_RESOURCE_GEOMETRY,
    HCCLContinualLadderConfig,
    HCCLContinualLadderRung,
    HCCLScheduleOccurrence,
    build_hccl_continual_ladder,
    calculate_hccl_resource_geometry,
    calculate_randomized_coalition_audit_calls,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L2_SCHEDULE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_L3_SCHEDULE,
    HCCL_CAUSAL_CORE_SCHEDULE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    HCCL_CAUSAL_CORE_SMOKE_SCHEDULE,
)

pytestmark = pytest.mark.unit


def _plan(rung: HCCLContinualLadderRung):  # type: ignore[no-untyped-def]
    return build_hccl_continual_ladder(HCCLContinualLadderConfig(rung=rung))


def _schedule_triples(rung: HCCLContinualLadderRung) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (item.evaluator_regime_name, item.start, item.end)
        for item in _plan(rung).schedule.occurrences
    )


@pytest.mark.parametrize("rung", tuple(HCCLContinualLadderRung))
def test_config_is_frozen_strict_and_json_roundtrips(
    rung: HCCLContinualLadderRung,
) -> None:
    config = HCCLContinualLadderConfig(rung=rung)
    encoded = config.to_json()
    payload = json.loads(encoded)

    assert HCCLContinualLadderConfig.from_json(encoded) == config
    assert HCCLContinualLadderConfig.from_config(payload) == config
    assert config.to_json() == json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert payload["schema"] == HCCL_CONTINUAL_LADDER_CONFIG_SCHEMA
    assert payload["status"] == HCCL_CONTINUAL_LADDER_STATUS
    assert payload["limitations"] == list(HCCL_CONTINUAL_LADDER_LIMITATIONS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, "rung", HCCLContinualLadderRung.SMOKE_420)


def test_config_rejects_string_rung_and_every_payload_mutation() -> None:
    with pytest.raises(TypeError, match="HCCLContinualLadderRung"):
        HCCLContinualLadderConfig(rung="core-l1-8998-v1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonauthority"):
        HCCLContinualLadderConfig(execution_authorized=True)
    with pytest.raises(ValueError, match="nonauthority"):
        HCCLContinualLadderConfig(evaluator_only=1)  # type: ignore[arg-type]

    payload = HCCLContinualLadderConfig().to_config()
    for key, value in (
        ("event_count", 8_997),
        ("cycle_count", True),
        ("property_bearing_development_geometry", False),
        ("promotion_authorized", True),
        ("schema", "alberta.hccl-continual-ladder.config.v0"),
    ):
        malformed = dict(payload)
        malformed[key] = value
        with pytest.raises(ValueError, match="frozen declaration"):
            HCCLContinualLadderConfig.from_config(malformed)

    missing = dict(payload)
    del missing["limitations"]
    with pytest.raises(ValueError, match="frozen declaration"):
        HCCLContinualLadderConfig.from_config(missing)
    extra = dict(payload)
    extra["seed"] = 0
    with pytest.raises(ValueError, match="frozen declaration"):
        HCCLContinualLadderConfig.from_config(extra)

    with pytest.raises(TypeError, match="plain dictionary"):
        HCCLContinualLadderConfig.from_config(payload.items())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate|invalid|strict"):
        HCCLContinualLadderConfig.from_json('{"rung":"a","rung":"b"}')
    with pytest.raises(ValueError, match="invalid|strict"):
        HCCLContinualLadderConfig.from_json('{"rung":NaN}')
    with pytest.raises(ValueError, match="one object"):
        HCCLContinualLadderConfig.from_json("[]")


def test_exact_frozen_rung_geometry_and_authority_classification() -> None:
    expected = {
        HCCLContinualLadderRung.SMOKE_420: (420, 1, 2, 1, True, False, False),
        HCCLContinualLadderRung.CORE_L1: (8_998, 1, 2, 1, False, True, False),
        HCCLContinualLadderRung.CORE_L2: (71_984, 8, 2, 1, False, True, True),
        HCCLContinualLadderRung.CORE_L3: (1_007_776, 112, 2, 1, False, True, True),
        HCCLContinualLadderRung.SCALE_L4: (8_998, 1, 4, 2, False, True, False),
    }
    for rung, values in expected.items():
        config = HCCLContinualLadderConfig(rung=rung)
        assert (
            config.event_count,
            config.cycle_count,
            config.learning_agent_count,
            config.fixed_neighbor_degree,
            config.smoke_mechanics_only,
            config.property_bearing_development_geometry,
            config.longevity_geometry,
        ) == values
    assert (
        HCCLContinualLadderConfig(rung=HCCLContinualLadderRung.SMOKE_420).source_schedule_profile
        == HCCL_CAUSAL_CORE_SMOKE_PROFILE
    )
    assert (
        HCCLContinualLadderConfig(rung=HCCLContinualLadderRung.CORE_L3).source_schedule_profile
        == HCCL_CAUSAL_CORE_L3_PROFILE
    )
    scale = HCCLContinualLadderConfig(rung=HCCLContinualLadderRung.SCALE_L4)
    assert scale.agent_topology == "fixed-undirected-four-agent-ring-v1"
    assert scale.fixed_neighbor_indices == ((1, 3), (0, 2), (1, 3), (0, 2))
    assert all(len(neighbors) == 2 for neighbors in scale.fixed_neighbor_indices)
    dyad = HCCLContinualLadderConfig(rung=HCCLContinualLadderRung.CORE_L1)
    assert dyad.agent_topology == "two-agent-dyad-v1"
    assert dyad.fixed_neighbor_indices == ((1,), (0,))


def test_smoke_l1_and_scale_l4_reuse_the_exact_source_schedules() -> None:
    assert _schedule_triples(HCCLContinualLadderRung.SMOKE_420) == (
        HCCL_CAUSAL_CORE_SMOKE_SCHEDULE
    )
    assert _schedule_triples(HCCLContinualLadderRung.CORE_L1) == HCCL_CAUSAL_CORE_SCHEDULE
    assert _schedule_triples(HCCLContinualLadderRung.SCALE_L4) == HCCL_CAUSAL_CORE_SCHEDULE

    smoke = _plan(HCCLContinualLadderRung.SMOKE_420).schedule
    l1 = _plan(HCCLContinualLadderRung.CORE_L1).schedule
    assert smoke.regime_occurrence_counts() == {"A": 5, "B": 2, "C": 2, "D": 1}
    assert smoke.regime_event_counts() == {"A": 207, "B": 82, "C": 92, "D": 39}
    assert l1.regime_occurrence_counts() == {"A": 5, "B": 2, "C": 2, "D": 1}
    assert l1.regime_event_counts() == {"A": 4_453, "B": 1_768, "C": 1_920, "D": 857}
    assert smoke.evaluator_only and not smoke.learner_visible_regime_labels
    assert l1.evaluator_only and not l1.learner_visible_regime_labels


def test_longevity_rungs_reuse_the_exact_executable_world_schedules() -> None:
    l2 = HCCLContinualLadderConfig(rung=HCCLContinualLadderRung.CORE_L2)
    l3 = HCCLContinualLadderConfig(rung=HCCLContinualLadderRung.CORE_L3)

    assert l2.source_schedule_profile == HCCL_CAUSAL_CORE_L2_PROFILE
    assert l3.source_schedule_profile == HCCL_CAUSAL_CORE_L3_PROFILE
    assert _schedule_triples(HCCLContinualLadderRung.CORE_L2) == HCCL_CAUSAL_CORE_L2_SCHEDULE
    assert _schedule_triples(HCCLContinualLadderRung.CORE_L3) == HCCL_CAUSAL_CORE_L3_SCHEDULE


@pytest.mark.parametrize(
    ("rung", "cycles", "events", "occurrences", "event_counts"),
    (
        (
            HCCLContinualLadderRung.CORE_L2,
            8,
            71_984,
            {"A": 47, "B": 16, "C": 16, "D": 1},
            {"A": 41_623, "B": 14_144, "C": 15_360, "D": 857},
        ),
        (
            HCCLContinualLadderRung.CORE_L3,
            112,
            1_007_776,
            {"A": 671, "B": 224, "C": 224, "D": 1},
            {"A": 593_863, "B": 198_016, "C": 215_040, "D": 857},
        ),
    ),
)
def test_longevity_schedules_have_exact_coverage_and_only_one_d(
    rung: HCCLContinualLadderRung,
    cycles: int,
    events: int,
    occurrences: dict[str, int],
    event_counts: dict[str, int],
) -> None:
    schedule = _plan(rung).schedule

    assert schedule.event_count == events == 8_998 * cycles
    assert len(schedule.occurrences) == 10 * cycles
    assert schedule.regime_occurrence_counts() == occurrences
    assert schedule.regime_event_counts() == event_counts
    assert sum(event_counts.values()) == events
    assert schedule.occurrences[0].start == 0
    assert schedule.occurrences[-1].end == events
    assert all(
        left.end == right.start
        for left, right in zip(schedule.occurrences, schedule.occurrences[1:])
    )

    d_occurrences = [
        item for item in schedule.occurrences if item.evaluator_regime_name == "D"
    ]
    assert len(d_occurrences) == 1
    assert (d_occurrences[0].cycle_index, d_occurrences[0].segment_index) == (0, 3)
    for cycle_index in range(1, cycles):
        replacement = schedule.occurrences[cycle_index * 10 + 3]
        assert replacement.evaluator_regime_name == "A"
        assert replacement.length == 857
        assert replacement.cycle_index == cycle_index
        assert replacement.segment_index == 3
    for index, item in enumerate(schedule.occurrences):
        assert item.occurrence_index == index


def test_schedule_metadata_is_frozen_and_fail_closed() -> None:
    schedule = _plan(HCCLContinualLadderRung.CORE_L2).schedule
    first = schedule.occurrences[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(first, "evaluator_regime_name", "D")
    with pytest.raises(ValueError, match="positive half-open"):
        HCCLScheduleOccurrence("A", 1, 1, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="segment_index"):
        HCCLScheduleOccurrence("A", 0, 1, 0, 10, 0, 0)
    with pytest.raises(ValueError, match="exact contiguous coverage"):
        dataclasses.replace(
            schedule,
            occurrences=(dataclasses.replace(first, end=first.end - 1), *schedule.occurrences[1:]),
        )
    with pytest.raises(ValueError, match="learner-visible"):
        dataclasses.replace(schedule, learner_visible_regime_labels=True)


def test_exact_resource_formulas_on_an_independent_geometry() -> None:
    geometry = calculate_hccl_resource_geometry(
        n_agents=3,
        action_count=3,
        learned_neighbor_degree=1,
        adjacent_action_layers=3,
        base_feature_width=5,
        generated_feature_count=7,
        memory_rows=11,
    )
    assert geometry.exact_all_agent_attribution_calls == 3 * 2**3 == 24
    assert geometry.distinct_action_receipt_vertex_upper_bound == 1 + 3 * (2**3 - 1) == 22
    assert geometry.effective_joint_action_bound == 3**3 == 27
    assert geometry.unilateral_calls_per_layer == 3 + 1 == 4
    assert geometry.pairwise_coalition_audit_calls_per_layer == 1 + 3 + 3 == 7
    assert geometry.factorized_planner_cells_per_agent == 3 ** (1 + 1) == 9
    assert geometry.factorized_planner_cells_total == 3 * 9 == 27
    assert geometry.raw_pair_candidates == 5 * 4 // 2 == 10
    assert geometry.pair_memory_reencode_products_per_agent == 2 * 7 * 11 == 154
    assert geometry.calculator_only
    assert not geometry.selected_executable_attribution_policy
    assert calculate_randomized_coalition_audit_calls(
        adjacent_action_layers=3, calls_per_layer_budget=17
    ) == 51


def test_documented_dyad_and_scale_l4_ring_instantiations_are_exact() -> None:
    dyad = HCCL_DYAD_RESOURCE_GEOMETRY
    assert (
        dyad.exact_all_agent_attribution_calls,
        dyad.distinct_action_receipt_vertex_upper_bound,
        dyad.effective_joint_action_bound,
        dyad.unilateral_calls_per_layer,
        dyad.pairwise_coalition_audit_calls_per_layer,
        dyad.factorized_planner_cells_per_agent,
        dyad.factorized_planner_cells_total,
        dyad.raw_pair_candidates,
        dyad.pair_memory_reencode_products_per_agent,
    ) == (8, 7, 4, 3, 4, 4, 8, 120, 1_536)

    scale = HCCL_SCALE_L4_RING_RESOURCE_GEOMETRY
    assert (scale.n_agents, scale.learned_neighbor_degree) == (4, 2)
    assert (
        scale.exact_all_agent_attribution_calls,
        scale.distinct_action_receipt_vertex_upper_bound,
        scale.effective_joint_action_bound,
        scale.unilateral_calls_per_layer,
        scale.pairwise_coalition_audit_calls_per_layer,
        scale.factorized_planner_cells_per_agent,
        scale.factorized_planner_cells_total,
        scale.raw_pair_candidates,
        scale.pair_memory_reencode_products_per_agent,
    ) == (32, 31, 16, 5, 11, 8, 32, 120, 1_536)
    assert scale.calculator_only
    assert not scale.selected_executable_attribution_policy


@pytest.mark.parametrize(
    "kwargs",
    (
        {"n_agents": True},
        {"n_agents": 0},
        {"n_agents": 2, "learned_neighbor_degree": 2},
        {"action_count": 1},
        {"adjacent_action_layers": 0},
        {"base_feature_width": 1},
        {"generated_feature_count": 0},
        {"memory_rows": False},
    ),
)
def test_resource_calculator_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    valid: dict[str, object] = {
        "n_agents": 2,
        "action_count": 2,
        "learned_neighbor_degree": 1,
        "adjacent_action_layers": 2,
        "base_feature_width": 16,
        "generated_feature_count": 12,
        "memory_rows": 64,
    }
    valid.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        calculate_hccl_resource_geometry(**valid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"n_agents": 63, "learned_neighbor_degree": 1},
        {"n_agents": 62, "learned_neighbor_degree": 1, "adjacent_action_layers": 2},
        {"action_count": HCCL_CONTINUAL_LADDER_MAX_EXACT_INTEGER},
        {"base_feature_width": HCCL_CONTINUAL_LADDER_MAX_EXACT_INTEGER},
        {"generated_feature_count": HCCL_CONTINUAL_LADDER_MAX_EXACT_INTEGER},
    ),
)
def test_resource_calculator_rejects_every_overflow(kwargs: dict[str, int]) -> None:
    valid = {
        "n_agents": 2,
        "action_count": 2,
        "learned_neighbor_degree": 1,
        "adjacent_action_layers": 2,
        "base_feature_width": 16,
        "generated_feature_count": 12,
        "memory_rows": 64,
    }
    valid.update(kwargs)
    with pytest.raises(OverflowError, match="exact-count limit"):
        calculate_hccl_resource_geometry(**valid)

    with pytest.raises(OverflowError, match="exact-count limit"):
        calculate_randomized_coalition_audit_calls(
            adjacent_action_layers=2,
            calls_per_layer_budget=HCCL_CONTINUAL_LADDER_MAX_EXACT_INTEGER,
        )


def test_randomized_coalition_calculator_requires_an_explicit_positive_budget() -> None:
    with pytest.raises(TypeError, match="required keyword-only argument"):
        calculate_randomized_coalition_audit_calls(adjacent_action_layers=2)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="at least 1"):
        calculate_randomized_coalition_audit_calls(
            adjacent_action_layers=2, calls_per_layer_budget=0
        )
    with pytest.raises(TypeError, match="non-boolean integer"):
        calculate_randomized_coalition_audit_calls(
            adjacent_action_layers=2, calls_per_layer_budget=True
        )


@pytest.mark.parametrize(
    "rung",
    (
        HCCLContinualLadderRung.SMOKE_420,
        HCCLContinualLadderRung.CORE_L1,
        HCCLContinualLadderRung.CORE_L2,
        HCCLContinualLadderRung.CORE_L3,
    ),
)
def test_every_dyad_rung_has_exact_named_opportunity_counts(
    rung: HCCLContinualLadderRung,
) -> None:
    plan = _plan(rung)
    counts = plan.opportunities
    events = plan.config.event_count

    assert counts.events == events
    assert counts.environment_proposal_calls is not None
    assert (
        counts.environment_proposal_calls.per_event,
        counts.environment_proposal_calls.per_life,
    ) == (8, 8 * events)
    assert counts.dyad_eight_call_cube_selected_for_accounting
    for named in (
        counts.context_updates,
        counts.fast_state_updates,
        counts.sequential_lineage_sidecar_proposals,
        counts.generated_feature_lifecycle_routes,
        counts.memory_query_write_transactions,
        counts.behavior_model_candidates,
        counts.grounded_world_model_candidates,
        counts.planner_decisions,
    ):
        assert (named.per_event, named.per_life) == (2, 2 * events)
    assert (
        counts.one_step_joint_planner_cells.per_event,
        counts.one_step_joint_planner_cells.per_life,
    ) == (8, 8 * events)
    assert (counts.real_multistep_backups.per_event, counts.real_multistep_backups.per_life) == (
        0,
        0,
    )


def test_scale_l4_counts_agent_local_work_but_selects_no_attribution_cube() -> None:
    plan = _plan(HCCLContinualLadderRung.SCALE_L4)
    counts = plan.opportunities

    assert counts.events == 8_998
    assert counts.environment_proposal_calls is None
    assert not counts.dyad_eight_call_cube_selected_for_accounting
    assert not counts.scale_dyad_cube_generalized_or_selected
    for named in (
        counts.context_updates,
        counts.fast_state_updates,
        counts.sequential_lineage_sidecar_proposals,
        counts.generated_feature_lifecycle_routes,
        counts.memory_query_write_transactions,
        counts.behavior_model_candidates,
        counts.grounded_world_model_candidates,
        counts.planner_decisions,
    ):
        assert (named.per_event, named.per_life) == (4, 4 * 8_998)
    assert (
        counts.one_step_joint_planner_cells.per_event,
        counts.one_step_joint_planner_cells.per_life,
    ) == (32, 32 * 8_998)
    assert counts.real_multistep_backups.per_life == 0
    assert plan.resources.exact_all_agent_attribution_calls == 32
    assert plan.resources.calculator_only
    assert not plan.resources.selected_executable_attribution_policy


@pytest.mark.parametrize("rung", tuple(HCCLContinualLadderRung))
def test_no_rung_has_execution_seed_threshold_artifact_evidence_or_promotion_authority(
    rung: HCCLContinualLadderRung,
) -> None:
    plan = _plan(rung)
    payload = plan.config.to_config()

    for field in (
        "learner_visible_regime_labels",
        "execution_authorized",
        "seed_reservation_or_consumption_authorized",
        "threshold_selection_authorized",
        "artifact_or_output_writes_authorized",
        "evidence_validation_authorized",
        "promotion_authorized",
        "longevity_authorizes_promotion",
        "scale_exact_attribution_policy_selected",
        "scale_randomized_coalition_budget_selected",
        "omitted_higher_order_interactions_recovered",
    ):
        assert payload[field] is False
    assert payload["evaluator_only"] is True
    assert not hasattr(plan, "run")
    assert not hasattr(plan, "execute")

    if rung is HCCLContinualLadderRung.SMOKE_420:
        assert payload["smoke_mechanics_only"] is True
        assert payload["property_bearing_development_geometry"] is False
    if rung in (HCCLContinualLadderRung.CORE_L2, HCCLContinualLadderRung.CORE_L3):
        assert payload["longevity_geometry"] is True
        assert payload["promotion_authorized"] is False
