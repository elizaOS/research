"""Contracts for the identity-free future-utility calibration engine."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from alberta_framework.evaluation import (
    _compositional_future_utility_calibration_engine as engine,
)
from alberta_framework.evaluation import compositional_control_life_development as control
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v2_development as consumed_v2,
)

pytestmark = pytest.mark.unit

ENGINE_PATH = (
    Path(__file__).parents[1]
    / "alberta_framework/evaluation/_compositional_future_utility_calibration_engine.py"
)


def _arms() -> tuple[engine.FutureUtilityArmSpec, ...]:
    return (
        engine.FutureUtilityArmSpec(
            "current_mix0_decay095_none",
            "current-utility reference with contribution traces retained",
            0.0,
            0.95,
            "none",
        ),
        engine.FutureUtilityArmSpec(
            "future_mix1_decay095_none",
            "unscaled future-utility endpoint",
            1.0,
            0.95,
            "none",
        ),
        engine.FutureUtilityArmSpec(
            "calibrated_mix05_decay095_none",
            "equal current/future mixture calibration",
            0.5,
            0.95,
            "none",
        ),
        engine.FutureUtilityArmSpec(
            "normalized_mix1_decay095_uncertainty_age",
            "causally age-and-uncertainty-normalized future utility",
            1.0,
            0.95,
            "uncertainty_age",
        ),
        engine.FutureUtilityArmSpec(
            "horizon_mix1_decay883_uncertainty_age",
            "long-horizon normalized future utility (about 883-step half-life)",
            1.0,
            0.999215304851532,
            "uncertainty_age",
        ),
    )


def _geometry() -> engine.FutureUtilityWorkGeometry:
    return engine.FutureUtilityWorkGeometry(
        steps=8_998,
        curation_interval=32,
        active_slots=11,
        candidate_slots=8,
        action_heads=2,
    )


def test_engine_projects_the_consumed_v2_arm_configs_without_importing_it() -> None:
    historical = control.learner_config_for_arm(
        "dovetail_coverage_ancestor_headroom_leftpack"
    )
    configs = {
        arm.name: engine.build_future_utility_learner_config(historical, arm)
        for arm in _arms()
    }

    assert configs == {
        name: consumed_v2._arm_learner_config(name)
        for name in consumed_v2.ARM_NAMES
    }
    varying, departures = engine.validate_future_utility_arm_contrasts(
        historical,
        _arms(),
        configs,
    )
    assert tuple(varying) == engine.INTERVENTION_FIELDS
    assert tuple(departures) == engine.COMMON_DEPARTURE_FIELDS


def test_engine_work_projection_matches_the_consumed_v2_mechanism() -> None:
    geometry = _geometry()
    shared = engine.logical_work_per_arm(geometry)
    assert shared == consumed_v2.logical_work_per_arm(
        consumed_v2.CompositionalFutureUtilityCalibrationV2Protocol()
    )
    for arm in _arms():
        assert engine.intervention_work_per_arm(geometry, arm) == (
            consumed_v2._intervention_work_for_arm(
                consumed_v2.CompositionalFutureUtilityCalibrationV2Protocol(),
                arm.name,
            )
        )
    contract = engine.work_resource_contract(geometry, _arms())
    assert contract["selected_arm_count"] == 5
    assert contract["shared_base_logical_work_matched"] is True
    assert contract["intervention_specific_logical_work_matched"] is False


def test_engine_arm_definitions_are_float32_bound_and_identity_free() -> None:
    definitions = [engine.arm_definition(arm) for arm in _arms()]
    assert [record["name"] for record in definitions] == [arm.name for arm in _arms()]
    assert definitions[-1]["future_utility_trace_decay_f32_bits"] == "3f7fcc93"

    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    source = ENGINE_PATH.read_text(encoding="utf-8")
    assert "calibration_v2" not in source
    assert "calibration_v3" not in source
    assert "development_root" not in source
    assert "_FULL_REPORT_ATTEMPT" not in source
    assert not ({"run_panel", "write_artifact", "select_winner"} & names)


def test_engine_rank_projection_matches_the_consumed_mechanism() -> None:
    active_mask = np.zeros((11,), dtype=np.bool_)
    active_mask[[6, 8]] = True
    active_scores = np.asarray(
        [99, 99, 99, 99, 99, 99, 0.75, 0.5, 0.75, 0.75, 0.25],
        dtype=np.float32,
    )
    candidate_mask = np.zeros((8,), dtype=np.bool_)
    candidate_mask[[1, 3]] = True
    candidate_scores = np.asarray(
        (0.25, 0.75, 0.75, 0.75, 0.5, 0.0, 0.0, 0.0),
        dtype=np.float32,
    )

    assert engine.active_bank_descending_rank(active_mask, active_scores) == (
        consumed_v2._descending_rank(active_mask, active_scores)
    )
    assert engine.candidate_bank_descending_rank(
        candidate_mask,
        candidate_scores,
    ) == consumed_v2._candidate_descending_rank(candidate_mask, candidate_scores)


def _synthetic_recurrence_inputs() -> tuple[
    engine.FutureUtilityEndpointGeometry,
    SimpleNamespace,
]:
    geometry = engine.FutureUtilityEndpointGeometry(
        phase_order=("A", "B", "A", "D", "A", "C", "A", "B", "C", "A"),
        phase_lengths=(1, 2, 3, 4, 5, 6, 7, 8, 9, 20),
        target_names=("A", "B", "C"),
        curation_interval=32,
    )
    steps = geometry.total_steps
    active_slots = np.zeros((steps, 11, 6), dtype=np.bool_)
    candidate_slots = np.zeros((steps, 8, 6), dtype=np.bool_)
    active_slots[:, 6, 0] = True
    candidate_slots[:, 0, 0] = True
    scores = np.zeros((steps, 11), dtype=np.float32)
    candidate_scores = np.zeros((steps, 8), dtype=np.float32)
    events = SimpleNamespace(
        post_active_signature_slots=active_slots,
        post_candidate_signature_slots=candidate_slots,
        direct_active_scores=scores,
        backed_active_scores=scores,
        direct_candidate_scores=candidate_scores,
        augmented_candidate_scores=candidate_scores,
    )
    return geometry, events


def test_engine_pre_recurrence_projection_is_geometry_driven() -> None:
    geometry, events = _synthetic_recurrence_inputs()

    records = engine.pre_recurrence_records(geometry, events)

    assert [(record["target"], record["pre_recurrence_post_step"]) for record in records] == [
        ("A", 3),
        ("A", 10),
        ("A", 21),
        ("B", 28),
        ("C", 36),
        ("A", 45),
    ]


@pytest.mark.parametrize(
    "field",
    (
        "post_active_signature_slots",
        "post_candidate_signature_slots",
        "direct_active_scores",
        "backed_active_scores",
        "direct_candidate_scores",
        "augmented_candidate_scores",
    ),
)
def test_pre_recurrence_rejects_mask_and_score_dtype_drift(field: str) -> None:
    geometry, events = _synthetic_recurrence_inputs()
    value = np.asarray(getattr(events, field))
    replacement = (
        value.astype(np.int8)
        if field.startswith("post_")
        else value.astype(np.float64)
    )
    setattr(events, field, replacement)

    with pytest.raises(TypeError, match="boolean|binary32"):
        engine.pre_recurrence_records(geometry, events)


@pytest.mark.parametrize(
    "broken",
    ("reserved_raw_signature", "active_signature_alias", "candidate_signature_alias"),
)
def test_pre_recurrence_rejects_impossible_signature_slots(broken: str) -> None:
    geometry, events = _synthetic_recurrence_inputs()
    if broken == "reserved_raw_signature":
        events.post_active_signature_slots[:, 0, 0] = True
    elif broken == "active_signature_alias":
        events.post_active_signature_slots[:, control.RAW_DIM, 1] = True
    else:
        events.post_candidate_signature_slots[:, 0, 1] = True

    with pytest.raises(RuntimeError, match="reserved raw|multiple distinct"):
        engine.pre_recurrence_records(geometry, events)


def test_pre_recurrence_rejects_rank_presence_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, events = _synthetic_recurrence_inputs()
    original = engine.active_bank_descending_rank

    def contradictory_rank(mask: object, scores: object) -> dict[str, object]:
        record = original(mask, scores)
        if record["present"] is True:
            return {**record, "present": False}
        return record

    monkeypatch.setattr(engine, "active_bank_descending_rank", contradictory_rank)

    with pytest.raises(RuntimeError, match="rank presence"):
        engine.pre_recurrence_records(geometry, events)


def test_engine_cadence_projection_binds_production_trace_fields_and_counts() -> None:
    endpoint_geometry = engine.FutureUtilityEndpointGeometry(
        phase_order=("A", "B", "A", "D", "A", "C", "A", "B", "C", "A"),
        phase_lengths=(1, 2, 3, 4, 5, 6, 7, 8, 9, 20),
        target_names=("A", "B", "C"),
        curation_interval=32,
    )
    steps = endpoint_geometry.total_steps
    post_step = np.arange(1, steps + 1, dtype=np.int32)
    due = post_step % 32 == 0
    zeros = np.zeros((steps,), dtype=np.bool_)
    margin = zeros.copy()
    margin[[0, 31, 32, 63]] = True
    candidate_margin = np.zeros((steps, 8, 11), dtype=np.bool_)
    candidate_margin[[0, 31, 32, 63], 0, 6] = True
    mutation_masks = {
        name: (
            np.zeros((steps, 11), dtype=np.bool_)
            if name in {"root_change_mask", "cascade_refill_mask", "active_change_mask"}
            else np.zeros((steps, 8), dtype=np.bool_)
            if name.endswith("_mask")
            else zeros.copy()
        )
        for name in engine.REQUIRED_CADENCE_MUTATION_MASK_NAMES
    }
    for name in (
        "decision_should_promote",
        "proposal_formed",
        "has_event",
        "promotion_applied",
        "root_change_applied",
    ):
        mutation_masks[name][31] = True
    mutation_masks["root_change_mask"][31, 6] = True
    mutation_masks["active_change_mask"][31, 6] = True
    mutation_masks["post_promotion_candidate_refresh_mask"][31, 0] = True
    mutation_masks["candidate_refresh_mask"][31, 0] = True
    for name in (
        "decision_should_refresh",
        "proposal_formed",
        "has_event",
    ):
        mutation_masks[name][63] = True
    for name in (
        "ordinary_candidate_refresh_mask",
        "candidate_refresh_mask",
    ):
        mutation_masks[name][63, 0] = True
    trace = SimpleNamespace(
        post_step=post_step,
        decision_update_available=np.ones((steps,), dtype=np.bool_),
        pre_replacement_phase=(post_step - 1) % 32,
        post_replacement_phase=post_step % 32,
        should_try_replace=due,
        decision_margin_passed=margin,
        decision_candidate_margin_eligible=candidate_margin,
        **mutation_masks,
    )
    events = SimpleNamespace(curation_trace=trace)
    totals = {
        "curation_due": 2,
        "proposal": 2,
        "root_change": 1,
        "promotion": 1,
        "cascade_refill": 0,
        "ordinary_candidate_refresh": 1,
        "post_promotion_candidate_refresh": 1,
        "candidate_refresh": 2,
        "candidate_rebound": 0,
        "candidate_overdepth_regeneration": 0,
        "logical_event": 3,
    }

    audit = engine.future_utility_cadence_audit_from_events(
        endpoint_geometry,
        events,
        pinned_due_mask=due,
    )
    closure = engine.validate_future_utility_curation_count_closure(audit, totals)

    assert audit.diagnostic_partitions["decision_margin_passed"] == (
        engine.OpportunityPartition(4, 2, 2)
    )
    assert audit.diagnostic_partitions[
        "decision_candidate_margin_eligible"
    ] == engine.OpportunityPartition(4, 2, 2)
    assert closure["all_checked_counts_close"] is True


def _synthetic_primary_endpoint_inputs() -> tuple[
    engine.FutureUtilityEndpointGeometry,
    SimpleNamespace,
    dict[str, dict[str, object]],
    dict[str, int],
    dict[str, object],
    np.ndarray,
]:
    geometry = engine.FutureUtilityEndpointGeometry(
        phase_order=("A", "B", "A", "D", "A", "C", "A", "B", "C", "A"),
        phase_lengths=(1, 2, 3, 4, 5, 6, 7, 8, 9, 20),
        target_names=("A", "B", "C"),
        curation_interval=32,
    )
    steps = geometry.total_steps
    post_step = np.arange(1, steps + 1, dtype=np.int32)
    due = post_step % geometry.curation_interval == 0
    zeros = np.zeros((steps,), dtype=np.bool_)
    margin = zeros.copy()
    margin[[0, 31, 32, 63]] = True
    candidate_margin = np.zeros((steps, 8, 11), dtype=np.bool_)
    candidate_margin[[0, 31, 32, 63], 0, 6] = True
    mutation_masks = {
        name: (
            np.zeros((steps, 11), dtype=np.bool_)
            if name in {"root_change_mask", "cascade_refill_mask", "active_change_mask"}
            else np.zeros((steps, 8), dtype=np.bool_)
            if name.endswith("_mask")
            else zeros.copy()
        )
        for name in engine.REQUIRED_CADENCE_MUTATION_MASK_NAMES
    }
    for name in (
        "decision_should_promote",
        "proposal_formed",
        "has_event",
        "promotion_applied",
        "root_change_applied",
    ):
        mutation_masks[name][31] = True
    mutation_masks["root_change_mask"][31, 6] = True
    mutation_masks["active_change_mask"][31, 6] = True
    mutation_masks["post_promotion_candidate_refresh_mask"][31, 0] = True
    mutation_masks["candidate_refresh_mask"][31, 0] = True
    mutation_masks["decision_should_refresh"][63] = True
    mutation_masks["proposal_formed"][63] = True
    mutation_masks["has_event"][63] = True
    mutation_masks["ordinary_candidate_refresh_mask"][63, 0] = True
    mutation_masks["candidate_refresh_mask"][63, 0] = True

    active_counts = np.zeros((steps, len(control.SIGNATURE_NAMES)), dtype=np.int32)
    candidate_counts = np.zeros_like(active_counts)
    active_counts[:, control.SIGNATURE_NAMES.index("A")] = 1
    active_counts[10:, control.SIGNATURE_NAMES.index("B")] = 1
    active_counts[20:40, control.SIGNATURE_NAMES.index("C")] = 1
    candidate_counts[:, control.SIGNATURE_NAMES.index("A")] = 1
    candidate_counts[5:, control.SIGNATURE_NAMES.index("B")] = 1

    active_slots = np.zeros(
        (steps, control.ACTIVE_SLOTS, len(control.SIGNATURE_NAMES)),
        dtype=np.bool_,
    )
    candidate_slots = np.zeros(
        (steps, control.CANDIDATE_SLOTS, len(control.SIGNATURE_NAMES)),
        dtype=np.bool_,
    )
    for target_slot, name in enumerate(geometry.target_names):
        index = control.SIGNATURE_NAMES.index(name)
        active_slots[:, control.RAW_DIM + target_slot, index] = (
            active_counts[:, index] > 0
        )
        candidate_slots[:, target_slot, index] = candidate_counts[:, index] > 0
    active_scores = np.zeros((steps, control.ACTIVE_SLOTS), dtype=np.float32)
    candidate_scores = np.zeros((steps, control.CANDIDATE_SLOTS), dtype=np.float32)
    selected_candidate = np.full((steps,), -1, dtype=np.int32)
    selected_destination = np.full((steps,), -1, dtype=np.int32)
    selected_candidate[31] = 0
    selected_destination[31] = 6
    trace = SimpleNamespace(
        post_step=post_step,
        decision_update_available=np.ones((steps,), dtype=np.bool_),
        pre_replacement_phase=(post_step - 1) % geometry.curation_interval,
        post_replacement_phase=post_step % geometry.curation_interval,
        should_try_replace=due,
        decision_margin_passed=margin,
        decision_candidate_margin_eligible=candidate_margin,
        decision_selected_candidate=selected_candidate,
        decision_selected_destination=selected_destination,
        promotion_source_candidate=selected_candidate,
        promotion_destination_active=selected_destination,
        **mutation_masks,
    )
    curation_counts = np.zeros(
        (steps, len(control.CURATION_COUNT_NAMES)),
        dtype=np.int32,
    )
    curation_counts[31] = (1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 2)
    curation_counts[63] = (1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1)
    trace.proposal_count = curation_counts[:, 1]
    trace.root_change_count = curation_counts[:, 2]
    trace.promotion_count = curation_counts[:, 3]
    trace.cascade_refill_count = curation_counts[:, 4]
    trace.ordinary_candidate_refresh_count = curation_counts[:, 5]
    trace.post_promotion_candidate_refresh_count = curation_counts[:, 6]
    trace.candidate_refresh_count = curation_counts[:, 7]
    trace.candidate_rebound_count = curation_counts[:, 8]
    trace.candidate_overdepth_regeneration_count = curation_counts[:, 9]
    trace.logical_event_count = curation_counts[:, 10]
    events = SimpleNamespace(
        curation_trace=trace,
        curation_counts=curation_counts,
        active_signature_counts=active_counts,
        candidate_signature_counts=candidate_counts,
        post_active_signature_slots=active_slots,
        post_candidate_signature_slots=candidate_slots,
        direct_active_scores=active_scores,
        backed_active_scores=active_scores,
        direct_candidate_scores=candidate_scores,
        augmented_candidate_scores=candidate_scores,
    )
    trajectories = {
        name: control._structural_trajectory(
            0,
            active_counts[:, control.SIGNATURE_NAMES.index(name)],
        )
        for name in geometry.target_names
    }
    transitions = {
        name: {
            "acquisition_episode_count": trajectories[name][
                "acquisition_episode_count"
            ],
            "loss_episode_count": trajectories[name]["loss_episode_count"],
            "loss_slot_cause_counts": {
                "promotion_root_replacement": 0,
                "cascade_dependency_refill": 0,
                "unmarked_signature_dependency_change": 0,
            },
            "all_changed_slots_accounted": True,
        }
        for name in geometry.target_names
    }
    curation_audit: dict[str, object] = {
        "due_curation_event_count": 2,
        "all_target_due_events_accounted": True,
        "active_signature_transition_causes": transitions,
        "target_outcome_counts": {
            name: {"admitted": 1, "not_admitted": 1}
            for name in geometry.target_names
        },
    }
    totals = {
        "curation_due": 2,
        "proposal": 2,
        "root_change": 1,
        "promotion": 1,
        "cascade_refill": 0,
        "ordinary_candidate_refresh": 1,
        "post_promotion_candidate_refresh": 1,
        "candidate_refresh": 2,
        "candidate_rebound": 0,
        "candidate_overdepth_regeneration": 0,
        "logical_event": 3,
    }
    return geometry, events, trajectories, totals, curation_audit, due


def test_primary_endpoint_projection_separates_raw_diagnostics_from_mutations() -> None:
    geometry, events, trajectories, totals, curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )

    endpoints = engine.build_future_utility_primary_endpoints(
        geometry,
        events,
        active_trajectories=trajectories,
        curation_totals=totals,
        curation_audit=curation_audit,
        pinned_due_mask=due,
    )

    assert endpoints["endpoint_order"] == list(engine.PRIMARY_ENDPOINT_NAMES)
    margin_passes = cast(dict[str, object], endpoints["margin_passes"])
    candidate_refreshes = cast(dict[str, object], endpoints["candidate_refreshes"])
    cadence_integrity = cast(dict[str, object], endpoints["cadence_integrity"])
    pre_recurrence_presence = cast(list[object], endpoints["pre_recurrence_presence"])
    assert margin_passes == {
        "selected_strict_margin_pass_count": 2,
        "selected_strict_margin_all_step_diagnostic_count": 4,
        "selected_strict_margin_off_opportunity_diagnostic_count": 2,
        "candidate_destination_strict_margin_pair_count": 2,
        "candidate_destination_strict_margin_all_step_diagnostic_count": 4,
        "candidate_destination_strict_margin_off_opportunity_diagnostic_count": 2,
        "due_curation_event_count": 2,
    }
    assert endpoints["promotions"] == {"event_count": 1}
    assert candidate_refreshes["total_refreshed_slot_count"] == 2
    assert cadence_integrity["all_mutations_off_opportunity_count"] == 0
    assert cadence_integrity["curation_counts_close"] is True
    assert len(pre_recurrence_presence) == 6
    assert endpoints["identity_reacquisition_claimed"] is False


def test_primary_endpoint_projection_rejects_one_off_cadence_mutation() -> None:
    geometry, events, trajectories, totals, curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    events.curation_trace.promotion_applied[0] = True

    with pytest.raises(ValueError, match="off-opportunity"):
        engine.build_future_utility_primary_endpoints(
            geometry,
            events,
            active_trajectories=trajectories,
            curation_totals=totals,
            curation_audit=curation_audit,
            pinned_due_mask=due,
        )


@pytest.mark.parametrize(
    ("field", "shape"),
    (
        ("decision_candidate_margin_eligible", (65, 8, 10)),
        ("root_change_mask", (65, 12)),
        ("candidate_refresh_mask", (65, 9)),
    ),
)
def test_cadence_projection_rejects_wrong_trailing_trace_shapes(
    field: str,
    shape: tuple[int, ...],
) -> None:
    geometry, events, _trajectories, _totals, _curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    setattr(events.curation_trace, field, np.zeros(shape, dtype=np.bool_))

    with pytest.raises(RuntimeError, match="shape"):
        engine.future_utility_cadence_audit_from_events(
            geometry,
            events,
            pinned_due_mask=due,
        )


@pytest.mark.parametrize("broken", ("proposal", "post_promotion_refresh"))
def test_curation_count_closure_rejects_production_algebra_breaks(broken: str) -> None:
    geometry, events, _trajectories, totals, _curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    if broken == "proposal":
        events.curation_trace.proposal_formed[63] = False
        totals["proposal"] = 1
    else:
        events.curation_trace.post_promotion_candidate_refresh_mask[31, 0] = False
        events.curation_trace.candidate_refresh_mask[31, 0] = False
        totals["post_promotion_candidate_refresh"] = 0
        totals["candidate_refresh"] = 1
        totals["logical_event"] = 2

    audit = engine.future_utility_cadence_audit_from_events(
        geometry,
        events,
        pinned_due_mask=due,
    )
    with pytest.raises(RuntimeError, match="proposal|post-promotion"):
        engine.validate_future_utility_curation_count_closure(audit, totals)


def test_primary_endpoint_projection_rejects_impossible_structural_lifecycle() -> None:
    geometry, events, trajectories, totals, curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    trajectories["A"]["acquisition_episode_count"] = 2
    trajectories["A"]["structural_reacquisition_count"] = 1

    with pytest.raises(RuntimeError, match="lifecycle"):
        engine.build_future_utility_primary_endpoints(
            geometry,
            events,
            active_trajectories=trajectories,
            curation_totals=totals,
            curation_audit=curation_audit,
            pinned_due_mask=due,
        )


def test_primary_endpoint_projection_rejects_promotion_without_due_margin() -> None:
    geometry, events, trajectories, totals, curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    events.curation_trace.decision_margin_passed[due] = False
    events.curation_trace.decision_candidate_margin_eligible[due] = False

    with pytest.raises(RuntimeError, match="promotion.*margin"):
        engine.build_future_utility_primary_endpoints(
            geometry,
            events,
            active_trajectories=trajectories,
            curation_totals=totals,
            curation_audit=curation_audit,
            pinned_due_mask=due,
        )


def test_primary_endpoint_projection_binds_margin_to_the_same_promotion_step() -> None:
    geometry, events, trajectories, totals, curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    events.curation_trace.decision_margin_passed[31] = False
    events.curation_trace.decision_candidate_margin_eligible[31] = False

    with pytest.raises(RuntimeError, match="same-step|promotion.*margin"):
        engine.build_future_utility_primary_endpoints(
            geometry,
            events,
            active_trajectories=trajectories,
            curation_totals=totals,
            curation_audit=curation_audit,
            pinned_due_mask=due,
        )


@pytest.mark.parametrize(
    "broken",
    (
        "selected_margin_cell",
        "root_destination",
        "refresh_source",
        "promotion_source_audit",
        "promotion_destination_audit",
    ),
)
def test_promotion_binds_selected_candidate_and_destination_cell(broken: str) -> None:
    geometry, events, trajectories, totals, curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    trace = events.curation_trace
    if broken == "selected_margin_cell":
        trace.decision_candidate_margin_eligible[31, 0, 6] = False
        trace.decision_candidate_margin_eligible[31, 1, 7] = True
    elif broken == "root_destination":
        trace.root_change_mask[31, 6] = False
        trace.root_change_mask[31, 7] = True
        trace.active_change_mask[31, 6] = False
        trace.active_change_mask[31, 7] = True
    elif broken == "refresh_source":
        trace.post_promotion_candidate_refresh_mask[31, 0] = False
        trace.post_promotion_candidate_refresh_mask[31, 1] = True
        trace.candidate_refresh_mask[31, 0] = False
        trace.candidate_refresh_mask[31, 1] = True
    elif broken == "promotion_source_audit":
        trace.promotion_source_candidate = trace.promotion_source_candidate.copy()
        trace.promotion_source_candidate[31] = 1
    else:
        trace.promotion_destination_active = trace.promotion_destination_active.copy()
        trace.promotion_destination_active[31] = 7

    with pytest.raises(RuntimeError, match="selected candidate/destination"):
        engine.build_future_utility_primary_endpoints(
            geometry,
            events,
            active_trajectories=trajectories,
            curation_totals=totals,
            curation_audit=curation_audit,
            pinned_due_mask=due,
        )


@pytest.mark.parametrize(
    "broken",
    ("trace_count_value", "trace_count_dtype", "events_count_value", "events_count_dtype"),
)
def test_per_step_production_counts_are_bound_to_mutation_masks(broken: str) -> None:
    geometry, events, trajectories, totals, curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    if broken == "trace_count_value":
        events.curation_trace.proposal_count = events.curation_trace.proposal_count.copy()
        events.curation_trace.proposal_count[31] = 0
    elif broken == "trace_count_dtype":
        events.curation_trace.proposal_count = (
            events.curation_trace.proposal_count.astype(np.int64)
        )
    elif broken == "events_count_value":
        events.curation_counts[31, 1] = 0
    else:
        events.curation_counts = events.curation_counts.astype(np.int64)

    with pytest.raises(RuntimeError, match="count|curation"):
        engine.build_future_utility_primary_endpoints(
            geometry,
            events,
            active_trajectories=trajectories,
            curation_totals=totals,
            curation_audit=curation_audit,
            pinned_due_mask=due,
        )


@pytest.mark.parametrize(
    "broken",
    (
        "transition_acquisition_count",
        "transition_loss_count",
        "unaccounted_transition",
        "unmarked_loss",
        "unaccounted_due_events",
        "outcome_histogram",
    ),
)
def test_curation_audit_lifecycle_reconstructs_from_signature_counts(broken: str) -> None:
    geometry, events, trajectories, totals, curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    transitions = cast(
        dict[str, dict[str, object]],
        curation_audit["active_signature_transition_causes"],
    )
    outcomes = cast(
        dict[str, dict[str, int]],
        curation_audit["target_outcome_counts"],
    )
    if broken == "transition_acquisition_count":
        transitions["A"]["acquisition_episode_count"] = 2
    elif broken == "transition_loss_count":
        transitions["C"]["loss_episode_count"] = 0
    elif broken == "unaccounted_transition":
        transitions["A"]["all_changed_slots_accounted"] = False
    elif broken == "unmarked_loss":
        cast(dict[str, int], transitions["A"]["loss_slot_cause_counts"])[
            "unmarked_signature_dependency_change"
        ] = 1
    elif broken == "unaccounted_due_events":
        curation_audit["all_target_due_events_accounted"] = False
    else:
        outcomes["A"] = {"admitted": 999}

    with pytest.raises(RuntimeError, match="transition|lifecycle|account|outcome"):
        engine.build_future_utility_primary_endpoints(
            geometry,
            events,
            active_trajectories=trajectories,
            curation_totals=totals,
            curation_audit=curation_audit,
            pinned_due_mask=due,
        )


@pytest.mark.parametrize(
    "broken",
    ("slot_count", "fractional_count", "wide_integer_count", "nan_rank"),
)
def test_primary_endpoint_projection_rejects_telemetry_inconsistency(broken: str) -> None:
    geometry, events, trajectories, totals, curation_audit, due = (
        _synthetic_primary_endpoint_inputs()
    )
    if broken == "slot_count":
        events.active_signature_counts[0, control.SIGNATURE_NAMES.index("A")] = 0
    elif broken == "fractional_count":
        counts = events.active_signature_counts.astype(np.float32)
        counts[0, control.SIGNATURE_NAMES.index("A")] = 0.5
        events.active_signature_counts = counts
    elif broken == "wide_integer_count":
        events.active_signature_counts = events.active_signature_counts.astype(np.int64)
    else:
        events.direct_active_scores[2, control.RAW_DIM] = np.nan

    with pytest.raises((TypeError, RuntimeError), match="count|slot|finite|rank"):
        engine.build_future_utility_primary_endpoints(
            geometry,
            events,
            active_trajectories=trajectories,
            curation_totals=totals,
            curation_audit=curation_audit,
            pinned_due_mask=due,
        )


def test_engine_has_no_execution_output_or_scientific_authority() -> None:
    assert engine.DEVELOPMENT_ONLY
    assert not engine.PANEL_EXECUTION_AUTHORIZED
    assert not engine.ROOT_ISSUANCE_AUTHORIZED
    assert not engine.OUTPUT_WRITES_ALLOWED
    assert not engine.EVIDENCE_AUTHORIZED
    assert not engine.SCIENTIFIC_PROMOTION_ALLOWED


@pytest.mark.parametrize(
    "kwargs",
    (
        {"name": "", "role": "r", "mix": 0.0, "trace_decay": 0.9, "normalization": "none"},
        {"name": "a", "role": "", "mix": 0.0, "trace_decay": 0.9, "normalization": "none"},
        {"name": "a", "role": "r", "mix": 1.1, "trace_decay": 0.9, "normalization": "none"},
        {"name": "a", "role": "r", "mix": 0.0, "trace_decay": 1.1, "normalization": "none"},
        {"name": "a", "role": "r", "mix": 0.0, "trace_decay": 0.9, "normalization": "bad"},
    ),
)
def test_arm_spec_rejects_malformed_mechanisms(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        engine.FutureUtilityArmSpec(**kwargs)  # type: ignore[arg-type]


def test_geometry_rejects_bool_zero_and_nondivisible_shapes() -> None:
    with pytest.raises((TypeError, ValueError)):
        engine.FutureUtilityWorkGeometry(True, 32, 11, 8, 2)
    with pytest.raises((TypeError, ValueError)):
        engine.FutureUtilityWorkGeometry(8_998, 0, 11, 8, 2)
