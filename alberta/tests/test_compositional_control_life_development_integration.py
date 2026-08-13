"""One consumed-seed compiled smoke for the compositional control life."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import pytest

import alberta_framework.evaluation.compositional_control_life_development as control_life
from alberta_framework.core.compositional_features import CompositionalFeatureLearner
from alberta_framework.evaluation.compositional_control_life_development import (
    DEFAULT_CONSUMED_SEED,
    PHASE_ORDER,
    RAW_PAIR_NAMES,
    SIGNATURE_NAMES,
    build_bound_compositional_control_life_source,
    build_default_protocol,
    build_short_test_protocol,
    execute_compositional_control_life_arm,
    learner_config_for_arm,
    run_compositional_control_life_development,
    validate_compositional_control_life_report,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def compiled_report() -> dict[str, object]:
    return run_compositional_control_life_development(
        build_short_test_protocol(),
        seed=DEFAULT_CONSUMED_SEED,
    )


@pytest.fixture(scope="module")
def canonical_leftpack_report() -> dict[str, object]:
    """Run only the canonical topology-placement counterexample arm."""

    return run_compositional_control_life_development(
        build_default_protocol(),
        seed=DEFAULT_CONSUMED_SEED,
        arm_names=("dovetail_coverage_ancestor_headroom_leftpack",),
    )


def test_authority_free_arm_kernel_reconstructs_the_public_run(
    compiled_report: dict[str, object],
) -> None:
    protocol = build_short_test_protocol()
    (
        key_manifest,
        observations,
        phase_indices,
        exploration_mask,
        random_actions,
        _stream_sha256,
    ) = control_life._stream_arrays(protocol, DEFAULT_CONSUMED_SEED)
    learner_key = jr.wrap_key_data(
        jnp.asarray(key_manifest["learner_genesis"], dtype=jnp.uint32),
        impl="threefry2x32",
    )
    learner = CompositionalFeatureLearner.from_config(
        learner_config_for_arm("myopic_full")
    )

    execution = execute_compositional_control_life_arm(
        protocol,
        learner,
        learner_key,
        observations,
        phase_indices,
        exploration_mask,
        random_actions,
        composed_readout_enabled=True,
    )
    public_run = cast(list[dict[str, Any]], compiled_report["runs"])[0]

    assert execution.scientific_promotion_allowed is False
    assert execution.evidence_authorized is False
    assert execution.output_writes_allowed is False
    assert execution.initial_state_sha256 == public_run["initial_state_sha256"]
    assert execution.final_state_sha256 == public_run["final_state_sha256"]
    assert execution.trace_sha256 == public_run["trace_sha256"]
    assert execution.initial_persistent_state_nbytes == 2_072
    assert execution.final_persistent_state_nbytes == 2_072
    assert tuple(int(value) for value in execution.final_state.step_words) == (
        0,
        protocol.total_steps,
    )


def test_bound_source_builder_reconstructs_the_existing_short_stream() -> None:
    protocol = build_short_test_protocol()
    root = jr.key(DEFAULT_CONSUMED_SEED, impl="threefry2x32")
    source = build_bound_compositional_control_life_source(
        protocol,
        observation_key=jr.fold_in(root, jnp.uint32(control_life.OBSERVATION_DOMAIN)),
        exploration_key=jr.fold_in(root, jnp.uint32(control_life.EXPLORATION_DOMAIN)),
        random_action_key=jr.fold_in(
            root, jnp.uint32(control_life.RANDOM_ACTION_DOMAIN)
        ),
        learner_key=jr.fold_in(root, jnp.uint32(control_life.LEARNER_DOMAIN)),
    )
    (
        old_manifest,
        old_observations,
        old_phase_indices,
        old_exploration_mask,
        old_random_actions,
        old_stream_sha256,
    ) = control_life._stream_arrays(protocol, DEFAULT_CONSUMED_SEED)

    assert dict(source.key_manifest) == {
        name: tuple(words)
        for name, words in old_manifest.items()
        if name != "root"
    }
    assert jnp.array_equal(source.observations, old_observations)
    assert jnp.array_equal(source.phase_indices, old_phase_indices)
    assert jnp.array_equal(source.exploration_mask, old_exploration_mask)
    assert jnp.array_equal(source.random_actions, old_random_actions)
    assert source.stream_sha256 == old_stream_sha256
    assert source.cadence_bound_stream_sha256 != source.stream_sha256
    assert int(jnp.count_nonzero(source.curation_due_mask)) == (
        protocol.total_steps // control_life.CURATION_INTERVAL
    )
    assert source.scientific_promotion_allowed is False
    assert source.evidence_authorized is False
    assert source.output_writes_allowed is False


def test_consumed_seed_short_life_is_finite_persistent_and_descriptive_only(
    compiled_report: dict[str, object],
) -> None:
    protocol = build_short_test_protocol()
    validate_compositional_control_life_report(compiled_report, protocol)

    assert compiled_report["acceptance_status"] == "not-assessed"
    assert compiled_report["evidence_authorized"] is False
    assert compiled_report["scientific_promotion_allowed"] is False
    assert compiled_report["artifact_bytes_written"] == 0
    assert compiled_report["seed"] == DEFAULT_CONSUMED_SEED
    assert compiled_report["arm_order"] == [
        "myopic_full",
        "explore_ancestor",
        "dovetail_coverage_ancestor",
        "dovetail_coverage_ancestor_headroom",
        "dovetail_coverage_ancestor_headroom_leftpack",
        "explore_ancestor_readout_blocked",
        "explore_ancestor_no_slow",
        "depth1_ceiling",
    ]
    identity = cast(Mapping[str, object], compiled_report["identity_tracking"])
    assert identity["v4_birth_ledger_integrated"] is False
    assert identity["reported_reacquisition_kind"] == (
        "bank_level_algebraic_structural_only"
    )

    runs = cast(list[dict[str, Any]], compiled_report["runs"])
    for run in runs:
        assert run["final_step_words_uint32"] == [0, protocol.total_steps]
        assert run["initial_persistent_state_nbytes"] == 2_072
        assert run["final_persistent_state_nbytes"] == 2_072
        assert run["curation_totals"]["curation_due"] == protocol.total_steps // 32
        assert run["initial_state_finite"] is True
        assert run["final_state_finite"] is True
        assert run["all_lifetime_counters_valid"] is True
        assert run["all_lifetime_capacity_available"] is True
        assert run["all_ranking_contracts_valid"] is True
        assert run["all_core_predictions_match_full_q"] is True
        assert len(run["initial_state_sha256"]) == 64
        assert len(run["final_state_sha256"]) == 64
        assert len(run["trace_sha256"]) == 64
        coverage = run["raw_pair_coverage"]
        assert coverage["pair_order"] == list(RAW_PAIR_NAMES)
        assert coverage["pair_count"] == 15
        assert coverage["initial_active_bitset"] == 31
        assert coverage["initial_candidate_bitset"] == 0
        assert coverage["ever_either_count"] == coverage[
            "ever_either_bitset"
        ].bit_count()

    runs_by_name = {run["arm"]: run for run in runs}
    for arm_name in (
        "myopic_full",
        "explore_ancestor",
        "explore_ancestor_readout_blocked",
        "explore_ancestor_no_slow",
    ):
        reachability = runs_by_name[arm_name]["raw_pair_reachability"]
        assert reachability[
            "conditional_theorem_applies_for_entire_observed_life"
        ] is True
        assert reachability[
            "ordinary_fresh_raw_pair_support_for_entire_observed_life"
        ] is False
        assert reachability["cascade_loophole_exercised"] is (
            runs_by_name[arm_name]["curation_totals"]["cascade_refill"] > 0
        )
    depth1_reachability = runs_by_name["depth1_ceiling"]["raw_pair_reachability"]
    assert depth1_reachability[
        "conditional_theorem_applies_for_entire_observed_life"
    ] is False
    assert depth1_reachability[
        "ordinary_fresh_raw_pair_support_for_entire_observed_life"
    ] is True
    coverage_reachability = runs_by_name["dovetail_coverage_ancestor"][
        "raw_pair_reachability"
    ]
    assert coverage_reachability["dovetail_product_coverage_enabled"] is True
    assert coverage_reachability[
        "conditional_theorem_applies_for_entire_observed_life"
    ] is False
    assert coverage_reachability[
        "ordinary_fresh_raw_pair_support_for_entire_observed_life"
    ] is True
    assert runs_by_name["dovetail_coverage_ancestor_headroom"]["learner_config"][
        "topology_headroom_reserve"
    ] is True
    leftpack = runs_by_name["dovetail_coverage_ancestor_headroom_leftpack"]
    assert leftpack["learner_config"]["topology_headroom_reserve"] is True
    assert leftpack["learner_config"]["topology_left_pack_destinations"] is True
    coexistence = leftpack["active_target_coexistence"]
    assert coexistence["target_order"] == ["A", "B", "C"]
    assert sum(coexistence["steps_by_active_target_count"]) == protocol.total_steps
    assert coexistence["active_targets_at_end"] == [
        name
        for name in ("A", "B", "C")
        if leftpack["active_structural_trajectories"][name]["present_at_end"]
    ]
    for phase in leftpack["phase_metrics"]:
        phase_coexistence = phase["active_target_coexistence"]
        assert sum(phase_coexistence["steps_by_active_target_count"]) == phase["steps"]

    assert runs_by_name["explore_ancestor"]["learner_config_sha256"] == (
        runs_by_name["explore_ancestor_readout_blocked"]["learner_config_sha256"]
    )
    assert runs_by_name["explore_ancestor"]["initial_state_sha256"] == (
        runs_by_name["explore_ancestor_readout_blocked"]["initial_state_sha256"]
    )

    run = runs_by_name["explore_ancestor"]
    phases = run["phase_metrics"]
    assert [phase["phase_name"] for phase in phases] == list(PHASE_ORDER)
    assert [phase["steps"] for phase in phases] == list(protocol.phase_lengths)
    for phase in phases:
        assert set(phase["entry_ranking"]) >= {
            "raw_active_utilities",
            "direct_active_scores",
            "backed_active_scores",
            "direct_candidate_scores",
            "candidate_novelty_scores",
            "augmented_candidate_scores",
        }
    for trajectory_group in (
        run["active_structural_trajectories"],
        run["candidate_structural_trajectories"],
    ):
        assert set(trajectory_group) == set(SIGNATURE_NAMES)
        assert all(
            trajectory["identity_reacquisition_claimed"] is False
            for trajectory in trajectory_group.values()
        )


def test_validator_rejects_mutated_authority_and_run_contracts(
    compiled_report: dict[str, object],
) -> None:
    protocol = build_short_test_protocol()
    unauthorized = copy.deepcopy(compiled_report)
    unauthorized["evidence_authorized"] = True
    with pytest.raises(ValueError, match="authority/status"):
        validate_compositional_control_life_report(unauthorized, protocol)

    wrong_clock = copy.deepcopy(compiled_report)
    runs = cast(list[dict[str, object]], wrong_clock["runs"])
    runs[0]["final_step_words_uint32"] = [0, protocol.total_steps - 1]
    with pytest.raises(ValueError, match="lifetime clock"):
        validate_compositional_control_life_report(wrong_clock, protocol)

    wrong_pair_bits = copy.deepcopy(compiled_report)
    pair_runs = cast(list[dict[str, Any]], wrong_pair_bits["runs"])
    pair_runs[0]["raw_pair_coverage"]["ever_active_bitset"] ^= 1
    with pytest.raises(ValueError, match="bitset"):
        validate_compositional_control_life_report(wrong_pair_bits, protocol)


def test_due_curation_audit_binds_decisions_and_accounts_for_losses(
    compiled_report: dict[str, object],
) -> None:
    protocol = build_short_test_protocol()
    runs = cast(list[dict[str, Any]], compiled_report["runs"])
    coverage = next(
        run for run in runs if run["arm"] == "dovetail_coverage_ancestor"
    )
    audit = cast(dict[str, Any], coverage["curation_decision_audit"])
    records = cast(list[dict[str, Any]], audit["due_curation_records"])

    assert audit["due_curation_event_count"] == protocol.total_steps // 32
    assert len(records) == audit["due_curation_event_count"]
    assert audit["all_target_due_events_accounted"] is True
    assert audit["all_shared_p45_bank_losses_accounted"] is True
    assert len(audit["records_sha256"]) == 64
    assert audit["records_canonical_json_bytes"] > 0
    for record in records:
        assert record["decision_due"] is True
        assert record["pre_active_descriptors"]["ops"]
        assert len(record["pre_candidate_descriptors"]["ops"]) == 8
        assert len(record["destination_masks"]["compatible_active_bitsets"]) == 8
        assert set(record["target_admission_outcomes"]) == {
            "A",
            "B",
            "C",
            "shared_p45",
            "obsolete_p12",
        }
        selection = record["selection"]
        assert set(selection) >= {
            "selected_candidate",
            "selected_destination",
            "effective_promotion_margin_f32_bits",
            "margin_rhs_f32_bits",
            "margin_passed",
        }

    work = cast(dict[str, Any], coverage["work"])
    assert work["curation_decision_audit_events"] == len(records)
    assert work["curation_decision_audit_report_json_bytes"] == audit[
        "records_canonical_json_bytes"
    ]
    assert work["curation_decision_audit_ephemeral_bytes"] > 0

    mutated = copy.deepcopy(compiled_report)
    mutated_runs = cast(list[dict[str, Any]], mutated["runs"])
    mutated_coverage = next(
        run for run in mutated_runs if run["arm"] == "dovetail_coverage_ancestor"
    )
    mutated_audit = cast(dict[str, Any], mutated_coverage["curation_decision_audit"])
    mutated_records = cast(list[dict[str, Any]], mutated_audit["due_curation_records"])
    mutated_records[0]["selection"]["margin_passed"] = not mutated_records[0][
        "selection"
    ]["margin_passed"]
    with pytest.raises(ValueError, match="curation decision audit"):
        validate_compositional_control_life_report(mutated, protocol)


def test_headroom_audit_mask_is_bound_into_every_destination_decision(
    compiled_report: dict[str, object],
) -> None:
    runs = cast(list[dict[str, Any]], compiled_report["runs"])
    run = next(
        item for item in runs if item["arm"] == "dovetail_coverage_ancestor_headroom"
    )
    records = cast(
        list[dict[str, Any]],
        run["curation_decision_audit"]["due_curation_records"],
    )
    for record in records:
        masks = record["destination_masks"]
        active = masks["active_eligible_bitset"]
        mature = masks["candidate_mature_bitset"]
        for candidate_slot, compatible in enumerate(
            masks["compatible_active_bitsets"]
        ):
            expected = (
                active
                & masks["topology_compatible_active_bitsets"][candidate_slot]
                & masks["depth_compatible_active_bitsets"][candidate_slot]
                & masks["headroom_compatible_active_bitsets"][candidate_slot]
            )
            if not ((mature >> candidate_slot) & 1):
                expected = 0
            assert compatible == expected

    mutated = copy.deepcopy(compiled_report)
    mutated_runs = cast(list[dict[str, Any]], mutated["runs"])
    mutated_run = next(
        item
        for item in mutated_runs
        if item["arm"] == "dovetail_coverage_ancestor_headroom"
    )
    mutated_audit = mutated_run["curation_decision_audit"]
    mutated_records = mutated_audit["due_curation_records"]
    mutated_records[0]["destination_masks"]["compatible_active_bitsets"][0] ^= 1
    mutated_audit["records_sha256"] = control_life._json_sha256(mutated_records)
    mutated_audit["records_canonical_json_bytes"] = len(
        control_life._canonical_json_bytes(mutated_records)
    )
    mutated_run["work"]["curation_decision_audit_report_json_bytes"] = mutated_audit[
        "records_canonical_json_bytes"
    ]
    with pytest.raises(ValueError, match="final mask relation"):
        validate_compositional_control_life_report(mutated, build_short_test_protocol())


def test_leftpack_audit_binds_margin_mask_lowest_destination_and_loss_causes(
    compiled_report: dict[str, object],
) -> None:
    runs = cast(list[dict[str, Any]], compiled_report["runs"])
    run = next(
        item
        for item in runs
        if item["arm"] == "dovetail_coverage_ancestor_headroom_leftpack"
    )
    audit = cast(dict[str, Any], run["curation_decision_audit"])
    records = cast(list[dict[str, Any]], audit["due_curation_records"])

    for record in records:
        selection = record["selection"]
        assert selection["left_pack_destinations_enabled"] is True
        selected_candidate = selection["selected_candidate"]
        if selected_candidate < 0:
            continue
        mask = record["destination_masks"]["margin_eligible_active_bitsets"][
            selected_candidate
        ]
        expected_destination = -1 if mask == 0 else (mask & -mask).bit_length() - 1
        assert selection["selected_destination"] == expected_destination
        assert selection["left_pack_destination_available"] is (mask != 0)

    causes = audit["active_signature_transition_causes"]
    trajectories = run["active_structural_trajectories"]
    for name in ("A", "B", "C", "shared_p45", "obsolete_p12"):
        assert causes[name]["loss_episode_count"] == trajectories[name][
            "loss_episode_count"
        ]
        assert causes[name]["acquisition_episode_count"] == trajectories[name][
            "acquisition_episode_count"
        ]
        assert causes[name]["all_changed_slots_accounted"] is True


def test_canonical_leftpack_life_pins_the_topology_only_negative_result(
    canonical_leftpack_report: dict[str, object],
) -> None:
    """Keep the consumed-root placement counterexample reproducible.

    This is a development regression, not a promotion threshold.  The exact
    hashes deliberately force any future lifecycle change to re-examine the
    negative interpretation instead of silently drifting the documented run.
    """

    protocol = build_default_protocol()
    validate_compositional_control_life_report(canonical_leftpack_report, protocol)
    assert canonical_leftpack_report["arm_order"] == [
        "dovetail_coverage_ancestor_headroom_leftpack"
    ]
    run = cast(list[dict[str, Any]], canonical_leftpack_report["runs"])[0]

    assert run["lifetime_metrics"]["executed_reward"] == 0.041787063791953766
    assert run["lifetime_metrics"]["greedy_reward"] == 0.03978661924872194
    assert run["curation_totals"] == {
        "logical_event": 671,
        "curation_due": 281,
        "proposal": 281,
        "promotion": 225,
        "root_change": 225,
        "cascade_refill": 87,
        "candidate_refresh": 281,
        "ordinary_candidate_refresh": 56,
        "post_promotion_candidate_refresh": 225,
        "candidate_rebound": 77,
        "candidate_overdepth_regeneration": 1,
    }
    assert run["active_target_coexistence"] == {
        "target_order": ["A", "B", "C"],
        "steps": 8_998,
        "steps_by_active_target_count": [8_166, 832, 0, 0],
        "maximum_active_target_count": 1,
        "all_three_present_steps": 0,
        "all_three_presence_fraction": 0.0,
        "first_all_three_post_step": None,
        "last_all_three_post_step": None,
        "active_targets_at_end": [],
    }
    trajectories = run["active_structural_trajectories"]
    assert {
        name: (
            trajectories[name]["acquisition_episode_count"],
            trajectories[name]["loss_episode_count"],
            trajectories[name]["presence_fraction"],
            trajectories[name]["present_at_end"],
        )
        for name in ("A", "B", "C", "shared_p45", "obsolete_p12")
    } == {
        "A": (4, 4, 0.014223802644738305, False),
        "B": (9, 9, 0.05689521057895322, False),
        "C": (6, 6, 0.021335703967107458, False),
        "shared_p45": (12, 12, 0.1351261251250139, False),
        "obsolete_p12": (13, 13, 0.11023447049672186, False),
    }
    assert run["initial_state_sha256"] == (
        "d3bff4b31a250a7f480c7a9ab05cbfe6d26c4ebd380c391d25143cf780d75b66"
    )
    assert run["final_state_sha256"] == (
        "42b5966da9a859baee1def49d3e9ecd4779e5cab6c15d3050449305856f0a34a"
    )
    assert run["trace_sha256"] == (
        "602fed99878c4ca500ce7ca6882ab42aa2c13f4862abe18dea6eb3d4d4c3b6b4"
    )
    audit = run["curation_decision_audit"]
    assert audit["records_sha256"] == (
        "6c872b8adbf857475875e05994d358aa0529edd9e2666c1f06f5c126e0b201d0"
    )
    assert audit["records_canonical_json_bytes"] == 2_176_971
    assert run["work"]["curation_decision_audit_array_elements"] == 11_310_486
    assert run["work"]["curation_decision_audit_ephemeral_bytes"] == 23_916_684
