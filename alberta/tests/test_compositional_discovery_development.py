"""Cheap contract tests for the nonpromoting compositional discovery lane."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alberta_framework.core.compositional_features import OP_PRODUCT, OP_RAW, OP_SUM
from alberta_framework.evaluation.compositional_discovery_development import (
    COMPOSITIONAL_DISCOVERY_DEVELOPMENT_STATUS,
    DEFAULT_DEVELOPMENT_SEEDS,
    PREWIRED_DEPTH2_ORACLE,
    PRODUCT_SIGNATURE_CLIPPING_CAVEAT,
    RAW_ONLY,
    STOCHASTIC_DISCOVERY,
    STRUCTURAL_RECURRENCE_DEFINITION,
    CompositionalDiscoveryTrialOutcome,
    ProductChainTrajectory,
    build_development_plan,
    derive_trial_key_manifest,
    detect_product_chain,
    summarize_arm_outcomes,
    summarize_presence_history,
    validate_development_plan,
)

pytestmark = pytest.mark.unit


def _empty_candidates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = np.zeros((0,), dtype=np.int32)
    return empty, empty.copy(), empty.copy()


def _trajectory(
    *,
    ever: bool,
    final: bool,
    first: int | None,
    recurrence: int = 0,
) -> ProductChainTrajectory:
    return ProductChainTrajectory(
        present_at_final_step=final,
        ever_present=ever,
        first_present_step=first,
        presence_snapshot_count=int(ever) + recurrence,
        acquisition_episode_count=int(ever) + recurrence,
        recurrence_count=recurrence,
    )


def _outcome(
    *,
    seed: int,
    arm: str,
    initial: float,
    final: float,
    curation_opportunities: int,
    active_replacements: int,
    promotions: int,
    active: ProductChainTrajectory,
    candidate: ProductChainTrajectory,
) -> CompositionalDiscoveryTrialOutcome:
    return CompositionalDiscoveryTrialOutcome(
        status=COMPOSITIONAL_DISCOVERY_DEVELOPMENT_STATUS,
        development_only=True,
        root_seed_uint32=seed,
        arm_name=arm,
        num_steps=10,
        window=2,
        initial_window_mse=initial,
        final_window_mse=final,
        final_to_initial_mse_ratio=final / initial,
        scheduled_curation_opportunity_count=curation_opportunities,
        active_root_replacement_count=active_replacements,
        promotion_count=promotions,
        active_product_chain=active,
        candidate_product_chain=candidate,
    )


def test_default_plan_is_paired_inert_and_explicitly_nonpromoting() -> None:
    plan = build_development_plan()
    resources = plan.resource_accounting

    assert plan.status == COMPOSITIONAL_DISCOVERY_DEVELOPMENT_STATUS
    assert plan.development_only
    assert not plan.scientific_evidence_authorized
    assert not plan.promotion_authorized
    assert tuple(seed.root_seed_uint32 for seed in plan.seeds) == (
        DEFAULT_DEVELOPMENT_SEEDS
    )
    assert len(plan.seeds) == 8
    assert all(seed.role == "development_consumed_nonpromoting" for seed in plan.seeds)
    assert [arm.name for arm in plan.arms] == [
        RAW_ONLY,
        PREWIRED_DEPTH2_ORACLE,
        STOCHASTIC_DISCOVERY,
    ]
    assert [(arm.active_slots, arm.candidate_slots) for arm in plan.arms] == [
        (4, 0),
        (6, 0),
        (20, 20),
    ]
    assert [arm.frozen_structure for arm in plan.arms] == [True, True, False]
    assert [arm.curation_interval for arm in plan.arms] == [0, 0, 20]

    assert resources.paired_stream_count == 8
    assert resources.trial_count == 24
    assert resources.total_learner_updates == 8 * 3 * 5_000
    assert resources.total_active_slot_update_exposures == 8 * 5_000 * (4 + 6 + 20)
    assert resources.total_candidate_slot_update_exposures == 8 * 5_000 * 20
    assert resources.logical_descriptor_snapshots_per_trial == 5_001
    expected_descriptor_values = 8 * 5_000 * 3 * (4 + 6 + 20 + 20)
    expected_initial_descriptor_values = 8 * 3 * (4 + 6 + 20 + 20)
    assert resources.total_initial_host_descriptor_int32_values == (
        expected_initial_descriptor_values
    )
    assert resources.total_scan_return_descriptor_int32_values == (
        expected_descriptor_values
    )
    assert resources.total_scan_return_descriptor_int32_bytes == (
        expected_descriptor_values * 4
    )
    assert resources.total_logical_descriptor_int32_values == (
        expected_initial_descriptor_values + expected_descriptor_values
    )
    assert resources.total_metric_scalars_recorded == 24 * 5_000
    assert resources.paired_stream_float32_bytes_per_seed == 5_000 * 5 * 4
    assert resources.artifact_bytes_written_by_runner == 0
    assert not resources.compiled_executable_or_peak_ram_bound_claimed
    assert "peak" in resources.accounting_scope

    fields = {field.name for field in dataclasses.fields(type(plan))}
    assert not {"threshold", "accepted", "passed", "evidence"} & fields


def test_threefry_fold_domains_pair_streams_but_separate_arm_learners() -> None:
    seed = DEFAULT_DEVELOPMENT_SEEDS[0]
    manifests = [
        derive_trial_key_manifest(seed, arm)
        for arm in (RAW_ONLY, PREWIRED_DEPTH2_ORACLE, STOCHASTIC_DISCOVERY)
    ]

    assert {manifest.implementation for manifest in manifests} == {"threefry2x32"}
    assert len({manifest.root_key_words_uint32 for manifest in manifests}) == 1
    assert len({manifest.observation_key_words_uint32 for manifest in manifests}) == 1
    assert len({manifest.noise_key_words_uint32 for manifest in manifests}) == 1
    assert len({manifest.learner_key_words_uint32 for manifest in manifests}) == 3
    assert (
        derive_trial_key_manifest(seed, STOCHASTIC_DISCOVERY)
        == manifests[-1]
    )


@pytest.mark.parametrize(
    ("parent_a", "parent_b"),
    [
        # (x0 * x1) * x2
        ([0, 1, 2, 3, 0, 4], [-1, -1, -1, -1, 1, 2]),
        # x0 * (x1 * x2)
        ([0, 1, 2, 3, 1, 0], [-1, -1, -1, -1, 2, 4]),
    ],
)
def test_exact_exponent_detector_counts_both_associative_groupings(
    parent_a: list[int],
    parent_b: list[int],
) -> None:
    empty_ops, empty_a, empty_b = _empty_candidates()
    snapshot = detect_product_chain(
        active_ops=np.asarray(
            [OP_RAW, OP_RAW, OP_RAW, OP_RAW, OP_PRODUCT, OP_PRODUCT],
            dtype=np.int32,
        ),
        active_parent_a=np.asarray(parent_a, dtype=np.int32),
        active_parent_b=np.asarray(parent_b, dtype=np.int32),
        candidate_ops=empty_ops,
        candidate_parent_a=empty_a,
        candidate_parent_b=empty_b,
        feature_dim=4,
    )

    assert snapshot.active_slots == (5,)
    assert snapshot.active_present
    assert not snapshot.candidate_present


def test_detector_resolves_candidate_parents_against_active_bank_only() -> None:
    snapshot = detect_product_chain(
        active_ops=np.asarray(
            [OP_RAW, OP_RAW, OP_RAW, OP_RAW, OP_PRODUCT, OP_SUM], dtype=np.int32
        ),
        active_parent_a=np.asarray([0, 1, 2, 3, 0, 4], dtype=np.int32),
        active_parent_b=np.asarray([-1, -1, -1, -1, 1, 2], dtype=np.int32),
        candidate_ops=np.asarray([OP_PRODUCT, OP_PRODUCT], dtype=np.int32),
        candidate_parent_a=np.asarray([4, 0], dtype=np.int32),
        candidate_parent_b=np.asarray([2, 3], dtype=np.int32),
        feature_dim=4,
    )

    assert not snapshot.active_present
    assert snapshot.candidate_slots == (0,)
    assert "clipped" in PRODUCT_SIGNATURE_CLIPPING_CAVEAT
    assert "identity" in STRUCTURAL_RECURRENCE_DEFINITION


def test_presence_summary_counts_reacquisition_episodes_not_identity() -> None:
    summary = summarize_presence_history([False, True, True, False, True, False])

    assert summary.ever_present
    assert not summary.present_at_final_step
    assert summary.first_present_step == 1
    assert summary.presence_snapshot_count == 3
    assert summary.acquisition_episode_count == 2
    assert summary.recurrence_count == 1

    absent = summarize_presence_history([False, False])
    assert not absent.ever_present
    assert absent.first_present_step is None
    assert absent.recurrence_count == 0


def test_arm_summary_is_descriptive_and_has_no_acceptance_field() -> None:
    never = _trajectory(ever=False, final=False, first=None)
    trials = (
        _outcome(
            seed=1,
            arm=STOCHASTIC_DISCOVERY,
            initial=2.0,
            final=1.0,
            curation_opportunities=5,
            active_replacements=3,
            promotions=1,
            active=_trajectory(ever=True, final=True, first=4, recurrence=1),
            candidate=_trajectory(ever=True, final=False, first=2),
        ),
        _outcome(
            seed=2,
            arm=STOCHASTIC_DISCOVERY,
            initial=4.0,
            final=1.0,
            curation_opportunities=5,
            active_replacements=5,
            promotions=2,
            active=_trajectory(ever=True, final=False, first=8),
            candidate=never,
        ),
        _outcome(
            seed=1,
            arm=RAW_ONLY,
            initial=2.0,
            final=2.0,
            curation_opportunities=0,
            active_replacements=0,
            promotions=0,
            active=never,
            candidate=never,
        ),
    )

    summary = summarize_arm_outcomes(STOCHASTIC_DISCOVERY, trials)
    assert summary.trial_count == 2
    assert summary.mean_initial_window_mse == 3.0
    assert summary.mean_final_window_mse == 1.0
    assert summary.mean_final_to_initial_mse_ratio == 0.375
    assert summary.median_final_to_initial_mse_ratio == 0.375
    assert summary.total_scheduled_curation_opportunities == 10
    assert summary.total_active_root_replacements == 8
    assert summary.total_promotions == 3
    assert summary.active_ever_present_trials == 2
    assert summary.active_final_present_trials == 1
    assert summary.active_first_present_step_median == 6.0
    assert summary.active_total_recurrences == 1
    assert summary.candidate_ever_present_trials == 1
    assert summary.candidate_first_present_step_median == 2.0

    fields = {field.name for field in dataclasses.fields(type(summary))}
    assert not {"threshold", "accepted", "passed", "evidence"} & fields


def test_custom_plan_remains_development_and_validates_budget() -> None:
    plan = build_development_plan(root_seeds=(7,), num_steps=12, window=3)
    assert validate_development_plan(plan) is plan
    assert plan.development_only
    assert plan.resource_accounting.trial_count == 3
    assert plan.resource_accounting.total_learner_updates == 36

    with pytest.raises(ValueError, match="unique"):
        build_development_plan(root_seeds=(7, 7), num_steps=12, window=3)
    with pytest.raises(ValueError, match="window"):
        build_development_plan(root_seeds=(7,), num_steps=12, window=13)
    with pytest.raises(ValueError, match="disjoint"):
        build_development_plan(root_seeds=(7,), num_steps=12, window=7)
    with pytest.raises(ValueError, match="20-slot"):
        build_development_plan(root_seeds=(7,), feature_dim=19)

    forged = dataclasses.replace(
        plan,
        resource_accounting=dataclasses.replace(
            plan.resource_accounting,
            total_learner_updates=37,
        ),
    )
    with pytest.raises(ValueError, match="canonical reconstruction"):
        validate_development_plan(forged)
    with pytest.raises(TypeError, match="exact CompositionalDiscoveryDevelopmentPlan"):
        validate_development_plan(object())  # type: ignore[arg-type]
