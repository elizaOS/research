"""Bounded declaration and two-update tests for future-utility calibration v2.

These tests never call the production full-report latch or an arm scan.  The
dynamic probes execute exactly two direct learner updates per selected arm.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v2_development as calibration,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    persistent_compositional_state_nbytes,
)

pytestmark = pytest.mark.unit


def _two_update_probe(
    arm_name: str,
) -> tuple[CompositionalFeatureLearner, CompositionalFeatureState]:
    learner = calibration._build_arm_learner(arm_name)
    learner_key = jr.wrap_key_data(
        jnp.asarray(calibration.KEY_MANIFEST["learner_genesis"], dtype=jnp.uint32),
        impl="threefry2x32",
    )
    state = cast(
        CompositionalFeatureState,
        learner.init(6, learner_key).replace(  # type: ignore[attr-defined]
            birth_timestamp=0.0,
            uptime_s=0.0,
        ),
    )
    observation = jnp.asarray(
        (0.2, -0.35, 0.5, -0.65, 0.8, -0.95), dtype=jnp.float32
    )
    for target in (1.0, 0.75):
        state = learner.update(
            state,
            observation,
            jnp.asarray((target, jnp.nan), dtype=jnp.float32),
        ).state
    return learner, state


def test_namespace_root_schedule_and_cadence_are_exact() -> None:
    protocol = calibration.CompositionalFutureUtilityCalibrationV2Protocol()

    assert hashlib.sha256(protocol.namespace.encode("ascii")).hexdigest() == (
        calibration.PROTOCOL_NAMESPACE_SHA256
    )
    assert int(calibration.PROTOCOL_NAMESPACE_SHA256[:8], 16) == 1_924_178_934
    assert protocol.development_root == 1_924_178_934
    assert calibration.DEVELOPMENT_ROOT_HEX == "0x72B0A3F6"
    assert protocol.phase_order == (
        "A",
        "B",
        "A",
        "D",
        "A",
        "C",
        "A",
        "B",
        "C",
        "A",
    )
    assert protocol.phase_lengths == (797, 829, 857, 883, 911, 941, 971, 1009, 1031, 769)
    assert protocol.total_steps == 8_998
    assert protocol.curation_interval == 32
    assert np.cumsum((0, *protocol.phase_lengths)).tolist() == [
        0,
        797,
        1626,
        2483,
        3366,
        4277,
        5218,
        6189,
        7198,
        8229,
        8998,
    ]
    boundaries = np.cumsum((0, *protocol.phase_lengths))
    due = np.arange(32, protocol.total_steps + 1, 32)
    assert [
        int(np.count_nonzero((due > start) & (due <= stop)))
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ] == [24, 26, 27, 28, 28, 30, 30, 31, 33, 24]
    assert due.size == 281
    assert calibration.CompositionalFutureUtilityCalibrationV2Protocol.from_config(
        protocol.to_config()
    ) == protocol


def test_key_manifest_and_stream_digest_are_frozen_before_any_arm_scan() -> None:
    protocol = calibration.CompositionalFutureUtilityCalibrationV2Protocol()
    source = calibration._source_arrays_bound(protocol)

    assert source.key_manifest == {
        "root": [0, 1_924_178_934],
        "observations": [1_189_056_302, 2_383_774_845],
        "exploration": [3_352_410_003, 3_947_271_724],
        "random_actions": [3_382_640_669, 4_117_898_437],
        "learner_genesis": [2_592_838_183, 3_227_537_730],
    }
    assert source.stream_sha256 == (
        "bb741db073a13026425d2cc98cce93a1af1d1b65f2abf24ebc97e43b61abd39c"
    )
    assert source.observations.shape == (8_998, 6)
    assert source.phase_indices.shape == (8_998,)
    assert source.exploration_mask.shape == (8_998,)
    assert source.random_actions.shape == (8_998,)
    assert str(source.observations.dtype) == "float32"
    assert str(source.phase_indices.dtype) == "int32"
    assert str(source.exploration_mask.dtype) == "bool"
    assert str(source.random_actions.dtype) == "int32"


def test_five_arm_matrix_and_config_contrasts_are_isolated() -> None:
    assert calibration.ARM_NAMES == (
        "current_mix0_decay095_none",
        "future_mix1_decay095_none",
        "calibrated_mix05_decay095_none",
        "normalized_mix1_decay095_uncertainty_age",
        "horizon_mix1_decay883_uncertainty_age",
    )
    configs = {name: calibration._arm_learner_config(name) for name in calibration.ARM_NAMES}
    expected = (
        (0.0, 0.95, "none"),
        (1.0, 0.95, "none"),
        (0.5, 0.95, "none"),
        (1.0, 0.95, "uncertainty_age"),
        (1.0, calibration.LONG_TRACE_DECAY, "uncertainty_age"),
    )
    for name, parameters in zip(calibration.ARM_NAMES, expected, strict=True):
        config = configs[name]
        assert (
            config["future_utility_mix"],
            config["future_utility_trace_decay"],
            config["future_utility_normalization"],
        ) == parameters
        assert config["candidate_scoring_mode"] == "legacy"
        assert config["candidate_novelty_admission_bonus"] == 0.0
        assert config["future_utility_trace_mode"] == "contribution"
        assert config["future_utility_normalization_decay"] == 0.99
        assert config["future_utility_rare_task_power"] == 0.0

    varying, _ = calibration._arm_configuration_audit(configs)
    assert tuple(varying) == (
        "future_utility_mix",
        "future_utility_trace_decay",
        "future_utility_normalization",
    )
    contrasts = (
        calibration._differing_fields(
            configs[calibration.ARM_NAMES[0]], configs[calibration.ARM_NAMES[1]]
        ),
        calibration._differing_fields(
            configs[calibration.ARM_NAMES[2]], configs[calibration.ARM_NAMES[1]]
        ),
        calibration._differing_fields(
            configs[calibration.ARM_NAMES[3]], configs[calibration.ARM_NAMES[1]]
        ),
        calibration._differing_fields(
            configs[calibration.ARM_NAMES[4]], configs[calibration.ARM_NAMES[3]]
        ),
    )
    assert contrasts == (
        ("future_utility_mix",),
        ("future_utility_mix",),
        ("future_utility_normalization",),
        ("future_utility_trace_decay",),
    )


def test_long_decay_bits_formula_degeneracy_and_causal_reachability() -> None:
    assert struct.pack(">f", calibration.LONG_TRACE_DECAY).hex() == "3f7fcc93"

    for error in (-2.0, -1.0, 0.5, 1.0, 2.0):
        per_active_reductions = []
        mean_slot_signals = []
        for feature in (-1.0, 1.0):
            delta_weight = 0.01 * error * feature
            contribution = error * feature
            feature_energy = feature * feature
            per_active = max(
                0.0,
                delta_weight * contribution
                - 0.5 * delta_weight * delta_weight * feature_energy,
            )
            per_active_reductions.append(per_active)
            mean_slot_signals.append(per_active / 2.0)
        assert per_active_reductions == [0.00995 * error**2] * 2
        assert mean_slot_signals == [0.004975 * error**2] * 2

    preflight = calibration._static_preflight()
    assert preflight["panel_executed_during_preflight"] is False
    assert preflight["mix_intervention_reaches_ranking"] is True
    assert preflight["mix_intervention_reaches_active_ranking"] is True
    assert preflight["mix_intervention_reaches_candidate_ranking"] is True
    assert preflight["normalization_intervention_reaches_ranking"] is True
    assert preflight["normalization_reaches_active_second_moment_and_ranking"] is True
    assert preflight["normalization_reaches_candidate_second_moment_and_ranking"] is True
    assert preflight["horizon_intervention_reaches_contribution_traces"] is True
    assert preflight["horizon_reaches_active_contribution_trace"] is True
    assert preflight["horizon_reaches_candidate_contribution_trace"] is True
    assert preflight["mixed_utility_overwrite_disabled"] is True
    witness = cast(dict[str, Any], preflight["decay_zero_formula_witness"])
    assert witness["production_core_bound"] is True
    assert witness["production_core_functions"] == [
        "contribution_trace_output_loss_reduction",
        "one_step_output_loss_reduction",
    ]
    for row in cast(list[dict[str, Any]], witness["rows"]):
        assert row["per_active_head_feature_minus_one"] == (
            row["per_active_head_feature_plus_one"]
        )
        assert row["mean_slot_feature_minus_one"] == (
            row["mean_slot_feature_plus_one"]
        )


def test_two_updates_reach_mixed_active_and_candidate_ranking_signals() -> None:
    mixture_learner, mixture_state = _two_update_probe(calibration.ARM_NAMES[2])
    future_learner, future_state = _two_update_probe(calibration.ARM_NAMES[1])

    assert not bool(jnp.array_equal(mixture_state.utilities, future_state.utilities))
    assert not bool(
        jnp.array_equal(
            mixture_state.candidate_utilities,
            future_state.candidate_utilities,
        )
    )
    mixture_ranking = mixture_learner.ranking_diagnostics(mixture_state, 6)
    future_ranking = future_learner.ranking_diagnostics(future_state, 6)
    assert not bool(
        jnp.array_equal(
            mixture_ranking.direct_active_scores,
            future_ranking.direct_active_scores,
        )
    )
    assert not bool(
        jnp.array_equal(
            mixture_ranking.direct_candidate_scores,
            future_ranking.direct_candidate_scores,
        )
    )


def test_two_updates_reach_active_and_candidate_normalization_state() -> None:
    plain_learner, plain_state = _two_update_probe(calibration.ARM_NAMES[1])
    normalized_learner, normalized_state = _two_update_probe(calibration.ARM_NAMES[3])

    assert not bool(jnp.any(plain_state.utility_signal_second_moment))
    assert not bool(jnp.any(plain_state.candidate_utility_signal_second_moment))
    assert bool(jnp.any(normalized_state.utility_signal_second_moment))
    assert bool(jnp.any(normalized_state.candidate_utility_signal_second_moment))
    assert not bool(jnp.array_equal(plain_state.utilities, normalized_state.utilities))
    assert not bool(
        jnp.array_equal(
            plain_state.candidate_utilities,
            normalized_state.candidate_utilities,
        )
    )
    plain_ranking = plain_learner.ranking_diagnostics(plain_state, 6)
    normalized_ranking = normalized_learner.ranking_diagnostics(normalized_state, 6)
    assert not bool(
        jnp.array_equal(
            plain_ranking.direct_active_scores,
            normalized_ranking.direct_active_scores,
        )
    )
    assert not bool(
        jnp.array_equal(
            plain_ranking.direct_candidate_scores,
            normalized_ranking.direct_candidate_scores,
        )
    )


def test_two_updates_reach_active_and_candidate_horizon_traces() -> None:
    short_learner, short_state = _two_update_probe(calibration.ARM_NAMES[3])
    long_learner, long_state = _two_update_probe(calibration.ARM_NAMES[4])

    assert not bool(
        jnp.array_equal(
            short_state.utility_contribution_trace,
            long_state.utility_contribution_trace,
        )
    )
    assert not bool(
        jnp.array_equal(
            short_state.candidate_utility_contribution_trace,
            long_state.candidate_utility_contribution_trace,
        )
    )
    assert not bool(jnp.array_equal(short_state.utilities, long_state.utilities))
    assert not bool(
        jnp.array_equal(short_state.candidate_utilities, long_state.candidate_utilities)
    )
    short_ranking = short_learner.ranking_diagnostics(short_state, 6)
    long_ranking = long_learner.ranking_diagnostics(long_state, 6)
    assert not bool(
        jnp.array_equal(
            short_ranking.direct_active_scores,
            long_ranking.direct_active_scores,
        )
    )
    assert not bool(
        jnp.array_equal(
            short_ranking.direct_candidate_scores,
            long_ranking.direct_candidate_scores,
        )
    )


def test_all_five_geneses_are_bit_identical_with_exact_state_bytes() -> None:
    learner_key = jr.wrap_key_data(
        jnp.asarray(calibration.KEY_MANIFEST["learner_genesis"], dtype=jnp.uint32),
        impl="threefry2x32",
    )
    states = [
        cast(
            CompositionalFeatureState,
            calibration._build_arm_learner(name)
            .init(6, learner_key)
            .replace(  # type: ignore[attr-defined]
                birth_timestamp=0.0,
                uptime_s=0.0,
            ),
        )
        for name in calibration.ARM_NAMES
    ]

    for state in states[1:]:
        chex.assert_trees_all_equal(states[0], state)
    shape_trees = [
        tuple(
            tuple(getattr(leaf, "shape", ()))
            for leaf in jax.tree_util.tree_leaves(state)
        )
        for state in states
    ]
    assert shape_trees == [shape_trees[0]] * 5
    assert {
        persistent_compositional_state_nbytes(state) for state in states
    } == {2_072}


def test_synthetic_primary_endpoint_extraction_is_causal_and_exact() -> None:
    steps = calibration.TOTAL_STEPS
    active_counts = np.ones((steps, 6), dtype=np.int32)
    candidate_counts = np.ones((steps, 6), dtype=np.int32)
    active_slots = np.zeros((steps, 11, 6), dtype=np.bool_)
    candidate_slots = np.zeros((steps, 8, 6), dtype=np.bool_)
    for signature_index, active_slot in enumerate((6, 7, 8)):
        active_slots[:, active_slot, signature_index] = True
        candidate_slots[:, signature_index, signature_index] = True
    active_scores = np.zeros((steps, 11), dtype=np.float32)
    active_scores[:, 6:9] = np.asarray((3.0, 2.0, 1.0), dtype=np.float32)
    candidate_direct_scores = np.broadcast_to(
        np.asarray((0.75, 0.5, 0.75, 0.75, 0.25, 0.0, 0.0, 0.0), dtype=np.float32),
        (steps, 8),
    )
    candidate_augmented_scores = np.broadcast_to(
        np.asarray((0.875, 0.5, 0.75, 0.875, 0.25, 0.0, 0.0, 0.0), dtype=np.float32),
        (steps, 8),
    )

    margin_passed = np.zeros((steps,), dtype=np.bool_)
    margin_passed[0] = True
    margin_passed[31] = True
    margin_pairs = np.zeros((steps, 8, 11), dtype=np.bool_)
    margin_pairs[31, 0, 6] = True
    promotion = np.zeros((steps,), dtype=np.bool_)
    promotion[31] = True
    ordinary_refresh = np.zeros((steps, 8), dtype=np.bool_)
    ordinary_refresh[31, 0] = True
    post_promotion_refresh = np.zeros((steps, 8), dtype=np.bool_)
    candidate_refresh = ordinary_refresh.copy()
    should_refresh = np.zeros((steps,), dtype=np.bool_)
    should_refresh[31] = True
    trace = SimpleNamespace(
        decision_margin_passed=margin_passed,
        decision_candidate_margin_eligible=margin_pairs,
        promotion_applied=promotion,
        ordinary_candidate_refresh_mask=ordinary_refresh,
        post_promotion_candidate_refresh_mask=post_promotion_refresh,
        candidate_refresh_mask=candidate_refresh,
        decision_should_refresh=should_refresh,
    )
    events = SimpleNamespace(
        curation_trace=trace,
        active_signature_counts=active_counts,
        candidate_signature_counts=candidate_counts,
        post_active_signature_slots=active_slots,
        post_candidate_signature_slots=candidate_slots,
        direct_active_scores=active_scores,
        backed_active_scores=active_scores,
        direct_candidate_scores=candidate_direct_scores,
        augmented_candidate_scores=candidate_augmented_scores,
    )
    trajectories = {
        name: {
            "acquisition_episode_count": 1,
            "loss_episode_count": 0,
            "present_at_end": True,
            "structural_reacquisition_count": 0,
        }
        for name in ("A", "B", "C")
    }
    transitions = {
        name: {
            "loss_episode_count": 0,
            "loss_slot_cause_counts": {
                "promotion_root_replacement": 0,
                "cascade_dependency_refill": 0,
                "unmarked_signature_dependency_change": 0,
            },
            "all_changed_slots_accounted": True,
        }
        for name in ("A", "B", "C")
    }
    audit = {
        "due_curation_event_count": 281,
        "active_signature_transition_causes": transitions,
        "target_outcome_counts": {
            "A": {"admitted": 1},
            "B": {},
            "C": {},
        },
    }
    totals = {
        "promotion": 1,
        "ordinary_candidate_refresh": 1,
        "post_promotion_candidate_refresh": 0,
        "candidate_refresh": 1,
        "cascade_refill": 2,
    }

    endpoints = cast(
        dict[str, Any],
        calibration._primary_endpoint_record(
            calibration.CompositionalFutureUtilityCalibrationV2Protocol(),
            events,
            trajectories,
            totals,
            audit,
        ),
    )

    assert endpoints["margin_passes"] == {
        "selected_strict_margin_pass_count": 1,
        "selected_strict_margin_all_step_diagnostic_count": 2,
        "selected_strict_margin_off_opportunity_diagnostic_count": 1,
        "candidate_destination_strict_margin_pair_count": 1,
        "due_curation_event_count": 281,
    }
    assert endpoints["promotions"] == {"event_count": 1}
    assert endpoints["cascade_refill_slot_count"] == 2
    assert endpoints["target_admission_loss_end"]["A"] == {
        "direct_candidate_admission_count": 1,
        "admission_episode_count": 1,
        "loss_episode_count": 0,
        "present_at_end": True,
        "structural_reacquisition_count": 0,
    }
    assert endpoints["a_retention"] == {
        "pre_recurrence_phase_indices": [2, 4, 6, 9],
        "pre_recurrence_presence": [True, True, True, True],
        "present_at_end": True,
    }
    occupancy = endpoints["target_occupancy"]
    assert occupancy["maximum_distinct_active_target_count"] == 3
    assert occupancy["final_active_targets"] == ["A", "B", "C"]
    recurrence = endpoints["pre_recurrence_ranks"]["records"]
    assert [(record["target"], record["pre_recurrence_post_step"]) for record in recurrence] == [
        ("A", 1626),
        ("A", 3366),
        ("A", 5218),
        ("B", 6189),
        ("C", 7198),
        ("A", 8229),
    ]
    first_a = recurrence[0]
    assert first_a["matching_candidate_slots"] == [0]
    assert first_a["candidate_direct_rank"] == {
        "present": True,
        "matching_candidate_slots": [0],
        "matching_score_f32_bits": [0x3F400000],
        "best_score_f32_bits": 0x3F400000,
        "descending_rank_interval": [1, 3],
    }
    assert first_a["candidate_augmented_rank"] == {
        "present": True,
        "matching_candidate_slots": [0],
        "matching_score_f32_bits": [0x3F600000],
        "best_score_f32_bits": 0x3F600000,
        "descending_rank_interval": [1, 2],
    }


def test_work_and_selected_source_snapshot_are_exact_without_panel() -> None:
    protocol = calibration.CompositionalFutureUtilityCalibrationV2Protocol()
    work = calibration.logical_work_per_arm(protocol)

    assert work["learner_updates"] == 8_998
    assert work["curation_update_opportunities"] == 281
    assert work["total_active_feature_value_cells"] == 197_956
    assert work["learner_update_candidate_feature_value_cells"] == 71_984
    assert work["total_q_dot_products"] == 26_994
    assert work["total_q_head_scalar_outputs"] == 53_988
    assert work["candidate_active_correlation_reset_mask_cells"] == 791_824
    assert work["ranking_candidate_active_correlation_cells"] == 791_912
    assert work["persistent_state_nbytes"] == 2_072
    assert work["behavioral_experience_matching_claimed"] is False
    assert calibration._intervention_work_for_arm(
        protocol, calibration.ARM_NAMES[0]
    ) == {
        "utility_mixture_cells": 0,
        "active_second_moment_cells": 0,
        "candidate_second_moment_cells": 0,
        "active_age_debias_cells": 0,
        "candidate_age_debias_cells": 0,
        "active_uncertainty_normalization_cells": 0,
        "candidate_uncertainty_normalization_cells": 0,
    }
    normalized_work = calibration._intervention_work_for_arm(
        protocol, calibration.ARM_NAMES[3]
    )
    assert normalized_work == {
        "utility_mixture_cells": 170_962,
        "active_second_moment_cells": 98_978,
        "candidate_second_moment_cells": 71_984,
        "active_age_debias_cells": 98_978,
        "candidate_age_debias_cells": 71_984,
        "active_uncertainty_normalization_cells": 98_978,
        "candidate_uncertainty_normalization_cells": 71_984,
    }
    contract = calibration._work_resource_contract(protocol)
    assert contract["per_arm_shared_base"] == work
    assert contract["shared_base_logical_work_matched"] is True
    assert contract["stream_shapes_and_update_opportunities_matched"] is True
    assert contract["intervention_specific_logical_work_matched"] is False
    assert contract["total_named_logical_work_equivalence_claimed"] is False
    assert contract["behavior_dependent_branch_work_equivalence_claimed"] is False
    assert "logical_work_matched" not in contract
    assert calibration._selected_source_manifest() == dict(
        calibration._IMPORT_TIME_SELECTED_SOURCE_HASHES
    )


def test_arm_comparison_only_claims_shared_base_work_equality() -> None:
    protocol = calibration.CompositionalFutureUtilityCalibrationV2Protocol()
    shared_base = calibration.logical_work_per_arm(protocol)
    runs: list[dict[str, object]] = []
    for name in calibration.ARM_NAMES:
        runs.append(
            {
                "arm": name,
                "learner_config": calibration._arm_learner_config(name),
                "initial_state_sha256": "shared-genesis",
                "expected_persistent_state_nbytes": 2_072,
                "shared_base_work": shared_base,
                "intervention_specific_work": calibration._intervention_work_for_arm(
                    protocol, name
                ),
                "primary_endpoints": {},
                "secondary_reward_endpoints": {},
            }
        )

    comparison = calibration._arm_comparison(runs)

    assert comparison["shared_base_logical_work_equal"] is True
    assert comparison["stream_shapes_and_update_opportunities_equal"] is True
    assert comparison["intervention_specific_logical_work_equal"] is False
    assert comparison["total_named_logical_work_equivalence_claimed"] is False
    assert comparison["behavior_dependent_branch_work_equivalence_claimed"] is False
    assert "logical_work_equal" not in comparison


def test_composed_bank_rank_is_tie_aware_and_bit_exact() -> None:
    mask = np.zeros((11,), dtype=np.bool_)
    mask[[6, 8]] = True
    scores = np.asarray([99, 99, 99, 99, 99, 99, 0.75, 0.5, 0.75, 0.75, 0.25])

    rank = calibration._descending_rank(mask, scores)

    assert rank == {
        "present": True,
        "matching_composed_slots": [6, 8],
        "matching_score_f32_bits": [0x3F400000, 0x3F400000],
        "best_score_f32_bits": 0x3F400000,
        "descending_rank_interval": [1, 3],
    }


def test_candidate_bank_rank_is_tie_aware_and_bit_exact() -> None:
    mask = np.zeros((8,), dtype=np.bool_)
    mask[[1, 3]] = True
    scores = np.asarray((0.25, 0.75, 0.75, 0.75, 0.5, 0.0, 0.0, 0.0))

    rank = calibration._candidate_descending_rank(mask, scores)

    assert rank == {
        "present": True,
        "matching_candidate_slots": [1, 3],
        "matching_score_f32_bits": [0x3F400000, 0x3F400000],
        "best_score_f32_bits": 0x3F400000,
        "descending_rank_interval": [1, 3],
    }


def test_public_validation_never_starts_or_waits_for_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body: dict[str, object] = {"schema": "bounded-validator-probe"}
    report = {**body, "report_sha256": calibration._json_sha256(body)}
    report_json = calibration._canonical_json(report)
    calls = 0
    entered = threading.Event()
    release = threading.Event()

    def builder(_capability: object) -> str:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5.0)
        return report_json

    latch = calibration._ProcessAttemptLatch(builder)
    monkeypatch.setattr(calibration, "_FULL_REPORT_ATTEMPT", latch)

    before = calibration.validate_compositional_future_utility_calibration_v2_report(
        report
    )
    assert before.valid is False
    assert before.errors == ("the one-shot panel has not completed successfully",)
    with pytest.raises(ValueError, match="has not completed successfully"):
        calibration.compositional_future_utility_calibration_v2_report_json(report)
    assert calls == 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(latch.get)
        assert entered.wait(timeout=5.0)
        in_progress = (
            calibration.validate_compositional_future_utility_calibration_v2_report(
                report
            )
        )
        assert in_progress.valid is False
        with pytest.raises(ValueError, match="has not completed successfully"):
            calibration.compositional_future_utility_calibration_v2_report_json(report)
        assert calls == 1
        release.set()
        assert future.result(timeout=5.0) == report_json

    after = calibration.validate_compositional_future_utility_calibration_v2_report(
        report
    )
    assert after.valid is True
    assert after.errors == ()
    assert (
        calibration.compositional_future_utility_calibration_v2_report_json(report)
        == report_json
    )
    assert calls == 1


def test_process_latch_serializes_one_success_without_full_singleton() -> None:
    calls = 0
    entered = threading.Event()
    release = threading.Event()

    def builder(_capability: object) -> str:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5.0)
        return "sole-value"

    latch = calibration._ProcessAttemptLatch(builder)
    assert latch.completed_value() is None
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(latch.get) for _ in range(8)]
        assert entered.wait(timeout=5.0)
        assert latch.completed_value() is None
        release.set()
        assert [future.result(timeout=5.0) for future in futures] == [
            "sole-value"
        ] * 8
    assert calls == 1
    assert latch.completed_value() == "sole-value"


def test_process_latch_seals_baseexception_failure_without_full_singleton() -> None:
    class FatalProbe(BaseException):
        pass

    calls = 0

    def builder(_capability: object) -> str:
        nonlocal calls
        calls += 1
        raise FatalProbe("sealed")

    latch = calibration._ProcessAttemptLatch(builder)
    with pytest.raises(FatalProbe, match="sealed"):
        latch.get()
    with pytest.raises(RuntimeError, match="sealed after failure") as captured:
        latch.get()
    assert isinstance(captured.value.__cause__, FatalProbe)
    assert calls == 1
    assert latch.completed_value() is None


def test_full_panel_helpers_require_the_live_latch_capability() -> None:
    with pytest.raises(RuntimeError, match="active one-shot capability"):
        calibration._build_report(object())
    with pytest.raises(RuntimeError, match="active one-shot panel capability"):
        calibration._run_arm(
            cast(Any, None),
            cast(Any, None),
            calibration.ARM_NAMES[0],
            _execution_capability=object(),
        )


def test_consumed_root_cannot_reenter_builder_or_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_calls = 0

    def forbidden_source_arrays(
        _protocol: calibration.CompositionalFutureUtilityCalibrationV2Protocol,
    ) -> calibration.BoundSourceArrays:
        nonlocal source_calls
        source_calls += 1
        raise AssertionError("consumed root reached source-array construction")

    latch = calibration._ProcessAttemptLatch(
        lambda capability: calibration._canonical_json(
            calibration._build_report(capability)
        )
    )
    monkeypatch.setattr(calibration, "_FULL_REPORT_ATTEMPT", latch)
    monkeypatch.setattr(calibration, "_source_arrays_bound", forbidden_source_arrays)

    with pytest.raises(RuntimeError, match="consumed by its failed attempt"):
        latch.get()
    assert source_calls == 0
    assert calibration.EXECUTION_ATTEMPTS_AUTHORIZED == 1
    assert calibration.EXECUTION_ATTEMPTS_CONSUMED == 1
    assert calibration.EXECUTION_ATTEMPTS_REMAINING == 0
    assert calibration.EXECUTION_OUTCOME == (
        "failed-invalid-evaluator-cadence-invariant"
    )
