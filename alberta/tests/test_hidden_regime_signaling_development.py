"""Development-only audits for the hidden-regime Lewis evaluator."""

import dataclasses
import inspect
from copy import deepcopy

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.slot_signaling_agent import (
    DURABLE_WRITE_SELECTIVE,
    DURABLE_WRITE_WRITABLE,
    REPLACEMENT_TARGET_EVIDENCE,
    REPLACEMENT_TARGET_LRU,
    SCRATCH_SLOT,
    SLOT_DURABLE,
    SLOT_SCRATCH,
    SLOT_VACANT,
    SlotSignalingAgent,
    SlotSignalingConfig,
    slot_signaling_keys,
    slot_signaling_resource_budget,
)
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    BENEFICIARY_FROZEN,
    CONSTANT_CHANNEL,
    CONSTANT_CHANNEL_SYMBOL,
    DEFAULT_DEVELOPMENT_SEGMENT_LENGTHS,
    DEVELOPMENT_CALIBRATION_LIMITATIONS,
    DEVELOPMENT_CANDIDATE_PROVENANCE,
    HELPER_FROZEN,
    HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
    HIDDEN_REGIME_TRACE_SCHEMA,
    MATCHED_CONDITIONS,
    REPLAY_PORTABILITY_SCOPE,
    RESERVED_DEVELOPMENT_SEED_NAMESPACE,
    SELECTIVE_EVIDENCE,
    SELECTIVE_FULL,
    SELECTIVE_LRU,
    SHUFFLED_CHANNEL,
    WRITABLE_EVIDENCE,
    WRITABLE_LRU,
    CommitGenerationLineage,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeSeedPair,
    condition_spec,
    derive_hidden_regime_seed_pairs,
    hidden_regime_lineage_recurrence_segments,
    reconstruct_commit_generation_lineages,
    reconstruct_hidden_regime_retention,
    reconstruct_hidden_regime_summary,
    run_hidden_regime_condition,
    run_hidden_regime_development,
    validate_hidden_regime_development_payload,
    validate_hidden_regime_run_result,
)
from alberta_framework.streams.hidden_regime_signaling import (
    DEFAULT_REGIME_PERMUTATIONS,
    DEFAULT_SEGMENT_LENGTHS,
    DEFAULT_SEGMENT_REGIMES,
    HIDDEN_REGIME_CALIBRATION_MANIFESTS,
    HIDDEN_REGIME_STRUCTURAL_MANIFESTS,
    HiddenRegimeSignalingWorld,
    HiddenRegimeWorldConfig,
    hidden_regime_calibration_world_config,
    hidden_regime_world_config_for_manifest,
    hidden_regime_world_keys,
)

pytestmark = pytest.mark.development


@pytest.fixture(scope="module")
def audit_config() -> HiddenRegimeDevelopmentConfig:
    """A manual CI protocol, much shorter than the unexecuted default."""

    segment_lengths = tuple(4 if regime == 4 else 24 for regime in DEFAULT_SEGMENT_REGIMES)
    return HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=segment_lengths,
            segment_regimes=DEFAULT_SEGMENT_REGIMES,
            regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=SlotSignalingConfig(
            learning_rate=0.5,
            epsilon=0.1,
            relevance_rate=0.5,
            lease_length=4,
            confirmation_steps=1,
            durable_retrieval_threshold=0.1,
            candidate_confirmation_threshold=0.2,
            candidate_confirmation_leases=1,
        ),
        metric_window=4,
    )


@pytest.fixture(scope="module")
def manual_seed_pair() -> HiddenRegimeSeedPair:
    # Explicitly supplied and permanently development-consumed by this test.
    return HiddenRegimeSeedPair(
        namespace="hidden-regime-manual-ci-c-replacement-v1",
        index=33,
        world_seed=1033,
        learner_seed=2033,
    )


@pytest.fixture(scope="module")
def development_report(
    audit_config: HiddenRegimeDevelopmentConfig,
    manual_seed_pair: HiddenRegimeSeedPair,
):
    return run_hidden_regime_development(
        seed_pair=manual_seed_pair,
        config=audit_config,
    )


def test_default_protocol_is_explicitly_calibrated_and_nonpromoting() -> None:
    config = HiddenRegimeDevelopmentConfig()
    expected_lengths = tuple(
        original if regime == 4 else original * 16
        for original, regime in zip(
            DEFAULT_SEGMENT_LENGTHS,
            DEFAULT_SEGMENT_REGIMES,
            strict=True,
        )
    )
    assert DEFAULT_DEVELOPMENT_SEGMENT_LENGTHS == expected_lengths
    assert config.world.segment_lengths == expected_lengths
    assert config.world.segment_lengths[13] == 16
    assert config.learner.learning_rate == 0.25
    assert config.learner.epsilon == 0.1
    assert config.learner.relevance_rate == 0.1
    assert config.learner.lease_length == 16
    assert config.learner.confirmation_steps == 8
    assert config.learner.durable_retrieval_threshold == 0.5
    assert config.learner.candidate_confirmation_threshold == 0.75
    assert config.learner.candidate_confirmation_leases == 3
    assert config.learner.scratch_training_leases_before_retest == 16
    assert config.learner.writable_lru_ablation is False
    payload = config.to_dict()
    assert HIDDEN_REGIME_DEVELOPMENT_SCHEMA.endswith(".v5")
    assert HIDDEN_REGIME_TRACE_SCHEMA.endswith(".v3")
    assert payload["schema_version"] == HIDDEN_REGIME_DEVELOPMENT_SCHEMA
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["acceptance_status"] == "descriptive_only_no_acceptance_gate"
    assert payload["claim_thresholds_frozen"] is False
    learner_payload = payload["learner"]
    assert isinstance(learner_payload, dict)
    assert learner_payload["scratch_training_leases_before_retest"] == 16
    assert payload["development_candidate_provenance"] == DEVELOPMENT_CANDIDATE_PROVENANCE
    assert payload["development_calibration_limitations"] == (DEVELOPMENT_CALIBRATION_LIMITATIONS)
    assert "consumed manual development pairs" in DEVELOPMENT_CANDIDATE_PROVENANCE
    assert "mean 0.845515" in DEVELOPMENT_CALIBRATION_LIMITATIONS
    assert "mean 0.835431" in DEVELOPMENT_CALIBRATION_LIMITATIONS
    assert "0.332--0.336 mean-reward range with zero commits" in (
        DEVELOPMENT_CALIBRATION_LIMITATIONS
    )
    assert "not a held-out comparison" in DEVELOPMENT_CALIBRATION_LIMITATIONS
    assert "same JAX/XLA backend and runtime" in REPLAY_PORTABILITY_SCOPE
    assert payload["replay_portability_scope"] == REPLAY_PORTABILITY_SCOPE
    assert "not aligned to learner lease boundaries" in payload["retention_window_semantics"]
    assert "not direct-retention evidence" in payload["legacy_metric_window_semantics"]


def test_config_and_seed_pair_contracts_are_strict() -> None:
    world = HiddenRegimeWorldConfig(
        segment_lengths=(2,),
        segment_regimes=(0,),
        regime_permutations=((0, 1, 2),),
        repeat_schedule=False,
    )
    for kwargs in (
        {"metric_window": True},
        {"metric_window": 0},
        {"world": dataclasses.replace(world, repeat_schedule=True)},
        {"world": object()},
        {"learner": object()},
        {"learner": SlotSignalingConfig(writable_lru_ablation=True)},
    ):
        with pytest.raises((TypeError, ValueError)):
            HiddenRegimeDevelopmentConfig(**kwargs)  # type: ignore[arg-type]
    for args in (
        ("", 0, 1, 2),
        ("manual", -1, 1, 2),
        ("manual", 0, -1, 2),
        ("manual", 0, 1, 2**32),
    ):
        with pytest.raises(ValueError):
            HiddenRegimeSeedPair(*args)


def test_named_seed_derivation_is_stable_and_owner_separated_without_running() -> None:
    first = derive_hidden_regime_seed_pairs("hidden-regime-manual-derivation-test-v1", 3)
    again = derive_hidden_regime_seed_pairs("hidden-regime-manual-derivation-test-v1", 3)
    other = derive_hidden_regime_seed_pairs("hidden-regime-manual-derivation-test-v2", 3)
    assert first == again
    assert first != other
    assert all(pair.world_seed != pair.learner_seed for pair in first)
    assert len({pair.world_seed for pair in first}) == len(first)
    assert len({pair.learner_seed for pair in first}) == len(first)
    assert all(0 <= pair.world_seed <= 0xFFFFFFFF for pair in first)
    for namespace, count in (("", 1), ("manual", True), ("manual", 0)):
        with pytest.raises(ValueError):
            derive_hidden_regime_seed_pairs(namespace, count)  # type: ignore[arg-type]


def test_manifest_recurrence_episodes_exclude_adjacent_splits_and_use_coalesced_indices() -> None:
    calibration_expected = {
        "hidden-regime-calibration-a-v1": (
            (3, 3206, 2, 1),
            (4, 4105, 1, 1),
            (5, 5244, 0, 1),
            (7, 7143, 1, 2),
            (9, 9192, 0, 2),
            (10, 10353, 1, 3),
            (11, 11499, 3, 1),
            (12, 12534, 0, 3),
            (14, 13452, 3, 2),
            (15, 14461, 1, 4),
            (16, 15365, 0, 4),
        ),
        "hidden-regime-calibration-b-v1": (
            (4, 4124, 1, 1),
            (5, 5266, 2, 1),
            (6, 6156, 1, 2),
            (7, 7182, 0, 1),
            (9, 9236, 1, 3),
            (10, 10389, 0, 2),
            (11, 11526, 3, 1),
            (13, 12563, 0, 3),
            (14, 13471, 1, 4),
            (15, 14363, 3, 2),
            (16, 15378, 0, 4),
        ),
        "hidden-regime-calibration-c-v1": (
            (3, 3187, 1, 1),
            (4, 4354, 2, 1),
            (5, 5251, 0, 1),
            (6, 6149, 1, 2),
            (7, 7176, 0, 2),
            (9, 9236, 0, 3),
            (10, 10387, 3, 1),
            (11, 11414, 1, 3),
            (12, 12559, 0, 4),
            (14, 13483, 0, 5),
            (15, 14631, 3, 2),
            (16, 15646, 1, 4),
        ),
    }
    assert set(calibration_expected) == set(HIDDEN_REGIME_CALIBRATION_MANIFESTS)
    for name, expected in calibration_expected.items():
        world = hidden_regime_calibration_world_config(name)
        episodes = hidden_regime_lineage_recurrence_segments(world)
        actual = tuple(
            (
                segment,
                sum(world.segment_lengths[:segment]),
                regime,
                occurrence,
            )
            for segment, regime, occurrence in episodes
        )
        assert actual == expected
        assert all(
            world.segment_regimes[segment - 1] != regime
            for segment, regime, _ in episodes
        )

    structural_expected = {
        "hidden-regime-structural-a-v1": (
            (3, 0, 1),
            (4, 1, 1),
            (5, 2, 1),
            (6, 0, 2),
            (7, 1, 2),
            (9, 0, 3),
            (10, 3, 1),
            (11, 1, 3),
            (12, 0, 4),
            (14, 1, 4),
            (15, 3, 2),
            (16, 0, 5),
        ),
        "hidden-regime-structural-b-v1": (
            (3, 1, 1),
            (4, 0, 1),
            (5, 2, 1),
            (6, 1, 2),
            (7, 0, 2),
            (9, 1, 3),
            (10, 3, 1),
            (11, 0, 3),
            (13, 3, 2),
            (14, 0, 4),
            (15, 1, 4),
            (16, 0, 5),
        ),
        "hidden-regime-structural-c-v1": (
            (3, 0, 1),
            (4, 2, 1),
            (5, 1, 1),
            (6, 0, 2),
            (8, 1, 2),
            (9, 0, 3),
            (10, 3, 1),
            (11, 1, 3),
            (13, 0, 4),
            (14, 1, 4),
            (15, 3, 2),
            (16, 0, 5),
        ),
    }
    assert set(structural_expected) == set(HIDDEN_REGIME_STRUCTURAL_MANIFESTS)
    for name, expected in structural_expected.items():
        world = hidden_regime_world_config_for_manifest(name)
        assert hidden_regime_lineage_recurrence_segments(world) == expected


def test_retention_records_disclose_coalesced_and_raw_occurrence_indices(
    development_report,
) -> None:
    run = development_report.runs[0]
    regimes = list(run.config.world.segment_regimes)
    regimes[:4] = [0, 0, 1, 0]
    config = dataclasses.replace(
        run.config,
        world=dataclasses.replace(
            run.config.world,
            segment_regimes=tuple(regimes),
        ),
    )
    records, aggregate = reconstruct_hidden_regime_retention(
        _without_commits(run.trace),
        config,
    )
    assert not any(record.segment_index == 1 for record in records)
    first_genuine_a_recurrence = next(
        record for record in records if record.segment_index == 3
    )
    assert first_genuine_a_recurrence.occurrence_index == 1
    assert first_genuine_a_recurrence.raw_segment_occurrence_index == 2
    assert aggregate.recurrence_count == len(records)


def test_reserved_namespace_is_fail_closed_and_never_used_by_tests(
    audit_config: HiddenRegimeDevelopmentConfig,
) -> None:
    reserved_pair = HiddenRegimeSeedPair(
        RESERVED_DEVELOPMENT_SEED_NAMESPACE,
        0,
        1,
        2,
    )
    with pytest.raises(ValueError, match="intentionally unexecuted"):
        run_hidden_regime_condition(
            SELECTIVE_FULL,
            seed_pair=reserved_pair,
            config=audit_config,
        )


def test_condition_interventions_are_exact_and_shape_neutral() -> None:
    specs = {condition: condition_spec(condition) for condition in MATCHED_CONDITIONS}
    assert tuple(specs) == MATCHED_CONDITIONS
    assert specs[SELECTIVE_FULL].channel == "direct"
    assert specs[SELECTIVE_FULL].helper_write
    assert specs[SELECTIVE_FULL].beneficiary_write
    assert not specs[SELECTIVE_FULL].writable_lru_ablation
    assert SELECTIVE_EVIDENCE == SELECTIVE_FULL
    assert specs[SELECTIVE_FULL].durable_write_policy == DURABLE_WRITE_SELECTIVE
    assert specs[SELECTIVE_FULL].replacement_target_policy == REPLACEMENT_TARGET_EVIDENCE
    assert specs[WRITABLE_EVIDENCE].durable_write_policy == DURABLE_WRITE_WRITABLE
    assert specs[WRITABLE_EVIDENCE].replacement_target_policy == REPLACEMENT_TARGET_EVIDENCE
    assert specs[SELECTIVE_LRU].durable_write_policy == DURABLE_WRITE_SELECTIVE
    assert specs[SELECTIVE_LRU].replacement_target_policy == REPLACEMENT_TARGET_LRU
    assert specs[WRITABLE_LRU].durable_write_policy == DURABLE_WRITE_WRITABLE
    assert specs[WRITABLE_LRU].replacement_target_policy == REPLACEMENT_TARGET_LRU
    assert specs[WRITABLE_LRU].writable_lru_ablation
    assert not specs[HELPER_FROZEN].helper_write
    assert specs[HELPER_FROZEN].beneficiary_write
    assert specs[BENEFICIARY_FROZEN].helper_write
    assert not specs[BENEFICIARY_FROZEN].beneficiary_write
    assert specs[CONSTANT_CHANNEL].channel == "constant_0"
    assert specs[SHUFFLED_CHANNEL].channel == "shuffled"
    with pytest.raises(ValueError):
        condition_spec("oracle")  # type: ignore[arg-type]


def test_condition_order_is_one_canonical_subsequence_and_alias_duplicates_fail_closed(
    audit_config: HiddenRegimeDevelopmentConfig,
    manual_seed_pair: HiddenRegimeSeedPair,
) -> None:
    malformed_orders = (
        (WRITABLE_LRU, SELECTIVE_FULL),
        (SELECTIVE_FULL, WRITABLE_LRU, SELECTIVE_LRU),
        (SELECTIVE_FULL, SELECTIVE_EVIDENCE),
    )
    for conditions in malformed_orders:
        with pytest.raises(ValueError, match="canonical|unique"):
            run_hidden_regime_development(
                seed_pair=manual_seed_pair,
                config=audit_config,
                conditions=conditions,
            )


def test_factorial_conditions_have_identical_state_tree_dtypes_and_exact_budget(
    development_report,
) -> None:
    factorial = (SELECTIVE_FULL, WRITABLE_EVIDENCE, SELECTIVE_LRU, WRITABLE_LRU)
    runs = {run.condition: run for run in development_report.runs}
    flattened = [jax.tree_util.tree_flatten(runs[name].final_state) for name in factorial]
    reference_leaves, reference_tree = flattened[0]
    reference_shapes = tuple(leaf.shape for leaf in reference_leaves)
    reference_dtypes = tuple(leaf.dtype for leaf in reference_leaves)
    for name, (leaves, tree) in zip(factorial, flattened, strict=True):
        assert tree == reference_tree
        assert tuple(leaf.shape for leaf in leaves) == reference_shapes
        assert tuple(leaf.dtype for leaf in leaves) == reference_dtypes
        budget = slot_signaling_resource_budget(runs[name].final_state)
        assert budget.helper.state_scalars == 69
        assert budget.helper.state_bytes == 276
        assert budget.beneficiary.state_scalars == 69
        assert budget.beneficiary.state_bytes == 276
        assert budget.state_scalars == 138
        assert budget.state_bytes == 552
    assert runs[SELECTIVE_LRU].summary.selective_immutability_applicable
    assert not runs[WRITABLE_EVIDENCE].summary.selective_immutability_applicable


def test_learner_oracle_boundary_is_structural_and_pre_reward_causal() -> None:
    helper_parameters = inspect.signature(SlotSignalingAgent.select_helper).parameters
    beneficiary_parameters = inspect.signature(SlotSignalingAgent.select_beneficiary).parameters
    update_parameters = inspect.signature(SlotSignalingAgent.update).parameters
    assert set(helper_parameters) == {"self", "state", "private_cue"}
    assert set(beneficiary_parameters) == {"self", "state", "delivered_message"}
    assert "target" not in update_parameters
    assert "regime" not in update_parameters

    identity = HiddenRegimeSignalingWorld(
        HiddenRegimeWorldConfig(
            segment_lengths=(2,),
            segment_regimes=(0,),
            regime_permutations=((0, 1, 2),),
        )
    )
    shifted = HiddenRegimeSignalingWorld(
        HiddenRegimeWorldConfig(
            segment_lengths=(2,),
            segment_regimes=(0,),
            regime_permutations=((1, 2, 0),),
        )
    )
    world_keys = hidden_regime_world_keys(jr.key(91))
    identity_state = identity.init(world_keys)
    shifted_state = shifted.init(world_keys)
    learner = SlotSignalingAgent()
    learner_state = learner.init(slot_signaling_keys(jr.key(92)))
    identity_observation = identity.observe(identity_state)
    shifted_observation = shifted.observe(shifted_state)
    assert int(identity_observation.helper_cue) == int(shifted_observation.helper_cue)
    helper_a = learner.select_helper(learner_state.helper, identity_observation.helper_cue)
    helper_b = learner.select_helper(learner_state.helper, shifted_observation.helper_cue)
    beneficiary_a = learner.select_beneficiary(learner_state.beneficiary, helper_a.action)
    beneficiary_b = learner.select_beneficiary(learner_state.beneficiary, helper_b.action)
    assert int(helper_a.action) == int(helper_b.action)
    assert int(beneficiary_a.action) == int(beneficiary_b.action)
    transition_a, _ = identity.step(
        identity_state,
        helper_a.action,
        beneficiary_a.action,
    )
    transition_b, _ = shifted.step(
        shifted_state,
        helper_b.action,
        beneficiary_b.action,
    )
    assert int(transition_a.oracle.target) != int(transition_b.oracle.target)


def test_all_matched_runs_reconstruct_and_hold_552_bytes(development_report) -> None:
    assert tuple(run.condition for run in development_report.runs) == MATCHED_CONDITIONS
    for run in development_report.runs:
        assert validate_hidden_regime_run_result(run) == ()
        assert run.resource.initial_state_bytes == 552
        assert run.resource.final_state_bytes == 552
        assert run.resource.resource_constant
        assert run.resource.resource_matched
        assert run.trace.reward.shape == (run.config.num_steps,)
        assert run.trace.world_terminated.dtype == jnp.bool_
        assert run.trace.world_discount.dtype == jnp.float32
        assert not np.any(run.trace.world_terminated)
        np.testing.assert_array_equal(
            run.trace.world_discount,
            np.ones((run.config.num_steps,), dtype=np.float32),
        )
        assert run.trace.helper_value_bits_pre.shape == (run.config.num_steps, 4, 3, 3)
        assert run.trace.helper_value_bits_pre.dtype == jnp.uint32
        assert run.trace.world_cue_key_data_pre.shape == (run.config.num_steps, 2)
        assert run.trace.world_channel_key_data_post.dtype == jnp.uint32
        assert run.trace.helper_policy_key_data_pre.shape == (run.config.num_steps, 2)
        assert run.trace.helper_relevance_mean_pre.shape == (run.config.num_steps, 4)
        assert run.trace.helper_relevance_mass_post.dtype == jnp.float32
        assert run.trace.helper_failed_leases_pre.dtype == jnp.int32
        assert run.trace.helper_idle_leases_post.shape == (run.config.num_steps, 4)
        assert run.trace.helper_status_pre.dtype == jnp.int8
        assert run.trace.helper_generation_pre.dtype == jnp.int32
        assert run.trace.helper_scratch_failed_leases_pre.dtype == jnp.int32
        assert run.trace.helper_scratch_retest_started.dtype == jnp.bool_
        np.testing.assert_array_equal(
            run.trace.helper_scratch_failed_leases_post,
            run.trace.beneficiary_scratch_failed_leases_post,
        )
        np.testing.assert_array_equal(
            run.trace.helper_scratch_retest_started,
            run.trace.beneficiary_scratch_retest_started,
        )
        np.testing.assert_array_equal(
            run.trace.world_cue_pre[1:],
            run.trace.world_cue_post[:-1],
        )
        np.testing.assert_array_equal(
            run.trace.helper_relevance_mean_pre[1:],
            run.trace.helper_relevance_mean_post[:-1],
        )
        np.testing.assert_array_equal(
            run.trace.helper_policy_key_data_pre[1:],
            run.trace.helper_policy_key_data_post[:-1],
        )
        assert run.summary.legacy_metric_window == run.config.metric_window
        assert run.summary.legacy_recurrence_entry_reward_mean == (
            run.summary.recurrence_entry_reward_mean
        )


def test_selective_summary_exposes_recurrence_replacement_and_transient_contract(
    development_report,
) -> None:
    selective = development_report.runs[0]
    summary = selective.summary
    labels = {item.regime_label for item in summary.recurrence_by_regime}
    assert {"A", "B", "C-old", "C-new", "D-short"} <= labels
    for label in ("A", "B", "C-old", "C-new"):
        item = next(item for item in summary.recurrence_by_regime if item.regime_label == label)
        assert item.recurrence_count >= 1
        assert item.recurrence_entry_reward_mean is not None
        assert item.recurrence_recovery_mean is not None
    assert summary.recurrence_entry_reward_mean is not None
    assert summary.recurrence_recovery_mean is not None
    assert summary.helper_effective_learning_update_count > 0
    assert summary.beneficiary_effective_learning_update_count > 0
    assert summary.both_roles_learned
    assert summary.c_old_to_c_new_replacement_count == 1
    assert summary.c_old_to_c_new_target_slots == (2,)
    assert len(summary.c_old_to_c_new_generation_pairs) == 1
    assert summary.c_old_to_c_new_exactly_one_target
    assert summary.d_short_checked
    assert summary.d_short_non_displacement
    assert summary.selective_durable_bit_immutable_until_atomic_replacement
    assert summary.helper_selective_mutation_violations == 0
    assert summary.beneficiary_selective_mutation_violations == 0


def test_retention_summary_uses_exact_world_entry_windows_and_exposes_missingness(
    development_report,
) -> None:
    run = development_report.runs[0]
    summary = run.summary
    records = summary.recurrence_retention
    aggregate = summary.retention
    expected_recurrences = sum(
        max(0, DEFAULT_SEGMENT_REGIMES.count(regime) - 1)
        for regime in set(DEFAULT_SEGMENT_REGIMES)
    )
    assert len(records) == expected_recurrences == aggregate.recurrence_count
    assert aggregate.complete_first_world_window_count == expected_recurrences
    assert aggregate.missing_first_world_window_count == 0
    assert aggregate.dormant_probe_available_count + aggregate.dormant_probe_missing_count == (
        expected_recurrences
    )
    assert aggregate.exact_generation_relock_count + (
        aggregate.exact_generation_relock_missing_count
    ) == expected_recurrences
    assert aggregate.legacy_first_exposure_comparison_available_count + (
        aggregate.legacy_first_exposure_comparison_missing_count
    ) == expected_recurrences
    assert aggregate.legacy_recurrence_to_first_exposure_error_rate_ratio_defined_count + (
        aggregate.legacy_recurrence_to_first_exposure_error_rate_ratio_undefined_count
    ) == aggregate.legacy_first_exposure_comparison_available_count
    assert aggregate.lineage_retention_applicable_count + (
        aggregate.acquisition_coverage_failure_count
    ) == expected_recurrences
    assert aggregate.qualification_coverage_denominator == expected_recurrences
    assert aggregate.latest_qualified_acquisition_comparison_denominator + (
        aggregate.latest_qualified_acquisition_comparison_not_applicable_count
    ) == expected_recurrences
    assert aggregate.latest_qualified_acquisition_comparison_available_count + (
        aggregate.latest_qualified_acquisition_comparison_missing_count
    ) == aggregate.latest_qualified_acquisition_comparison_denominator
    assert aggregate.qualified_lineage_survival_denominator == (
        aggregate.prior_qualified_lineage_count
    )
    assert aggregate.selected_lineage_probe_denominator == (
        aggregate.lineage_retention_applicable_count
    )
    assert aggregate.selected_lineage_probe_available_count + (
        aggregate.selected_lineage_survival_failure_count
    ) == aggregate.selected_lineage_probe_denominator
    assert aggregate.selected_lineage_not_applicable_count == (
        aggregate.acquisition_coverage_failure_count
    )
    assert aggregate.selected_entry_metric_denominator == (
        aggregate.selected_lineage_probe_available_count
    )
    assert aggregate.selected_exact_generation_relock_conditional_denominator == (
        aggregate.selected_lineage_probe_available_count
    )
    assert aggregate.selected_exact_generation_relock_all_qualified_denominator == (
        aggregate.lineage_retention_applicable_count
    )
    for record in records:
        assert record.first_world_window_length == run.config.learner.lease_length
        assert record.world_entry_steps_to_first_learner_boundary == (
            run.config.learner.lease_length - record.world_entry_learner_lease_offset
        )
        assert record.first_world_window_complete
        assert record.first_world_window_errors is not None
        assert record.first_world_window_error_rate == pytest.approx(
            record.first_world_window_errors / record.first_world_window_length
        )
        slots = [probe.slot for probe in record.eligible_dormant_generations]
        assert slots == sorted(slots)
        assert len(slots) == len(set(slots))
        if record.dormant_probe_available:
            assert record.best_dormant_slot is not None
            assert record.best_dormant_generation is not None
            expected_best = min(
                record.eligible_dormant_generations,
                key=lambda item: (-item.composed_greedy_accuracy, item.slot),
            )
            assert record.best_dormant_slot == expected_best.slot
            assert record.best_dormant_generation == expected_best.generation
            assert record.best_dormant_composed_greedy_accuracy == (
                expected_best.composed_greedy_accuracy
            )
        else:
            assert record.eligible_dormant_generations == ()
        if record.exact_generation_relock_observed:
            assert record.observed_learner_boundaries_until_relock is not None
            assert record.scratch_entered_before_relock is not None
            assert record.durable_retrieval_before_scratch == (
                not record.scratch_entered_before_relock
            )


def _hand_construct_intact_dormant_relock(run):
    """Install a perfect dormant generation and an exact pre-scratch relock."""

    trace = run.trace
    segment_index = 2  # first A recurrence in the manual default-order schedule
    start = sum(run.config.world.segment_lengths[:segment_index])
    end = start + run.config.world.segment_lengths[segment_index]
    relock_step = int(np.flatnonzero(np.asarray(trace.helper_lease_boundary)[start:end])[0]) + start
    identity_bits = np.eye(3, dtype=np.float32).view(np.uint32)
    replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        active_pre = jnp.asarray(getattr(trace, f"{role}_active_slot_pre"))
        active_pre = active_pre.at[start : relock_step + 1].set(2)
        active_pre = active_pre.at[relock_step].set(1)
        active_post = jnp.asarray(getattr(trace, f"{role}_active_slot_post"))
        active_post = active_post.at[relock_step].set(1)
        status_pre = jnp.asarray(getattr(trace, f"{role}_status_pre"))
        status_post = jnp.asarray(getattr(trace, f"{role}_status_post"))
        generation_pre = jnp.asarray(getattr(trace, f"{role}_generation_pre"))
        generation_post = jnp.asarray(getattr(trace, f"{role}_generation_post"))
        status_pre = status_pre.at[start, 1:4].set(SLOT_DURABLE)
        generation_pre = (
            generation_pre.at[start, 1].set(7).at[start, 2].set(8).at[start, 3].set(9)
        )
        status_pre = status_pre.at[relock_step, 1].set(SLOT_DURABLE)
        status_post = status_post.at[relock_step, 1].set(SLOT_DURABLE)
        generation_pre = generation_pre.at[relock_step, 1].set(7)
        generation_post = generation_post.at[relock_step, 1].set(7)
        value_bits_pre = jnp.asarray(getattr(trace, f"{role}_value_bits_pre"))
        value_bits_pre = value_bits_pre.at[start, 1].set(jnp.asarray(identity_bits))
        value_bits_pre = value_bits_pre.at[start, 3].set(jnp.asarray(identity_bits))
        durable_relevant = jnp.asarray(getattr(trace, f"{role}_durable_relevant"))
        durable_relevant = durable_relevant.at[relock_step].set(True)
        replacements.update(
            {
                f"{role}_active_slot_pre": active_pre,
                f"{role}_active_slot_post": active_post,
                f"{role}_status_pre": status_pre,
                f"{role}_status_post": status_post,
                f"{role}_generation_pre": generation_pre,
                f"{role}_generation_post": generation_post,
                f"{role}_value_bits_pre": value_bits_pre,
                f"{role}_durable_relevant": durable_relevant,
            }
        )
    return dataclasses.replace(trace, **replacements), segment_index, relock_step


_IDENTITY_TABLE_BITS = np.eye(3, dtype=np.float32).view(np.uint32)
_ZERO_TABLE_BITS = np.zeros((3, 3), dtype=np.float32).view(np.uint32)


def _without_commits(trace: HiddenRegimePrimitiveTrace) -> HiddenRegimePrimitiveTrace:
    replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        for suffix in (
            "committed_slot",
            "committed_generation",
            "retired_slot",
            "retired_generation",
        ):
            value = jnp.asarray(getattr(trace, f"{role}_{suffix}"))
            replacements[f"{role}_{suffix}"] = jnp.full_like(value, -1)
    return dataclasses.replace(trace, **replacements)


def _with_commit(
    trace: HiddenRegimePrimitiveTrace,
    *,
    step: int,
    slot: int,
    generation: int,
    helper_bits: np.ndarray = _IDENTITY_TABLE_BITS,
    beneficiary_bits: np.ndarray = _IDENTITY_TABLE_BITS,
    commit_segment_index: int | None = None,
    commit_segment_step: int = 0,
    regime_id: int = 0,
) -> HiddenRegimePrimitiveTrace:
    replacements: dict[str, object] = {}
    for role, table in (
        ("helper", helper_bits),
        ("beneficiary", beneficiary_bits),
    ):
        committed_slot = jnp.asarray(getattr(trace, f"{role}_committed_slot"))
        committed_generation = jnp.asarray(
            getattr(trace, f"{role}_committed_generation")
        )
        value_bits_post = jnp.asarray(getattr(trace, f"{role}_value_bits_post"))
        replacements[f"{role}_committed_slot"] = committed_slot.at[step].set(slot)
        replacements[f"{role}_committed_generation"] = (
            committed_generation.at[step].set(generation)
        )
        replacements[f"{role}_value_bits_post"] = value_bits_post.at[step, slot].set(
            jnp.asarray(table, dtype=jnp.uint32)
        )
    if commit_segment_index is not None:
        replacements["segment_index"] = jnp.asarray(trace.segment_index).at[step].set(
            commit_segment_index
        )
        replacements["segment_step"] = jnp.asarray(trace.segment_step).at[step].set(
            commit_segment_step
        )
        replacements["regime_id"] = jnp.asarray(trace.regime_id).at[step].set(
            regime_id
        )
    return dataclasses.replace(trace, **replacements)


def _with_entry_generation(
    trace: HiddenRegimePrimitiveTrace,
    *,
    start: int,
    slot: int,
    generation: int,
    helper_bits: np.ndarray = _IDENTITY_TABLE_BITS,
    beneficiary_bits: np.ndarray = _IDENTITY_TABLE_BITS,
    helper_active: int = 2,
    beneficiary_active: int = 2,
) -> HiddenRegimePrimitiveTrace:
    replacements: dict[str, object] = {}
    for role, table, active in (
        ("helper", helper_bits, helper_active),
        ("beneficiary", beneficiary_bits, beneficiary_active),
    ):
        status_pre = jnp.asarray(getattr(trace, f"{role}_status_pre"))
        generation_pre = jnp.asarray(getattr(trace, f"{role}_generation_pre"))
        value_bits_pre = jnp.asarray(getattr(trace, f"{role}_value_bits_pre"))
        active_pre = jnp.asarray(getattr(trace, f"{role}_active_slot_pre"))
        replacements[f"{role}_status_pre"] = status_pre.at[start, slot].set(
            SLOT_DURABLE
        )
        replacements[f"{role}_generation_pre"] = generation_pre.at[
            start,
            slot,
        ].set(generation)
        replacements[f"{role}_value_bits_pre"] = value_bits_pre.at[start, slot].set(
            jnp.asarray(table, dtype=jnp.uint32)
        )
        replacements[f"{role}_active_slot_pre"] = active_pre.at[start].set(active)
    return dataclasses.replace(trace, **replacements)


def _first_a_recurrence_bounds(run) -> tuple[int, int, int]:
    segment_index = 2
    start = sum(run.config.world.segment_lengths[:segment_index])
    end = start + run.config.world.segment_lengths[segment_index]
    return segment_index, start, end


def _qualified_a_lineage_trace(
    run,
    *,
    slot: int = 1,
    generation: int = 7,
) -> tuple[HiddenRegimePrimitiveTrace, int, int, int]:
    segment_index, start, end = _first_a_recurrence_bounds(run)
    trace = _with_commit(
        _without_commits(run.trace),
        step=0,
        slot=slot,
        generation=generation,
    )
    trace = _with_entry_generation(
        trace,
        start=start,
        slot=slot,
        generation=generation,
    )
    return trace, segment_index, start, end


def test_hand_constructed_fast_relearning_does_not_pass_direct_retention_probe(
    development_report,
) -> None:
    run = development_report.runs[0]
    trace = run.trace
    segment_index = 2
    start = sum(run.config.world.segment_lengths[:segment_index])
    end = start + run.config.world.segment_lengths[segment_index]
    rewards = jnp.asarray(trace.reward).at[start:end].set(jnp.float32(1.0))
    rewards = rewards.at[start : start + 2].set(jnp.float32(0.0))
    acquisition_start = 0
    rewards = rewards.at[acquisition_start : acquisition_start + 4].set(jnp.float32(1.0))
    rewards = rewards.at[acquisition_start].set(jnp.float32(0.0))
    empty_status = jnp.asarray(
        [SLOT_SCRATCH, SLOT_VACANT, SLOT_VACANT, SLOT_VACANT],
        dtype=jnp.int8,
    )
    empty_generation = jnp.zeros((4,), dtype=jnp.int32)
    fast_trace = dataclasses.replace(
        trace,
        reward=rewards,
        helper_active_slot_pre=jnp.asarray(trace.helper_active_slot_pre).at[start].set(0),
        beneficiary_active_slot_pre=jnp.asarray(trace.beneficiary_active_slot_pre)
        .at[start]
        .set(0),
        helper_status_pre=jnp.asarray(trace.helper_status_pre)
        .at[start]
        .set(empty_status),
        beneficiary_status_pre=jnp.asarray(trace.beneficiary_status_pre)
        .at[start]
        .set(empty_status),
        helper_generation_pre=jnp.asarray(trace.helper_generation_pre)
        .at[start]
        .set(empty_generation),
        beneficiary_generation_pre=jnp.asarray(trace.beneficiary_generation_pre)
        .at[start]
        .set(empty_generation),
    )
    summary = reconstruct_hidden_regime_summary(fast_trace, run.config, run.condition)
    legacy_segment = next(
        item for item in summary.segment_rewards if item.segment_index == segment_index
    )
    direct = next(
        item for item in summary.recurrence_retention if item.segment_index == segment_index
    )
    assert legacy_segment.late_reward == 1.0
    assert direct.first_world_window_reward == 0.5
    assert direct.legacy_recurrence_minus_first_exposure_error_rate_delta == 0.25
    assert direct.legacy_recurrence_to_first_exposure_error_rate_ratio == 2.0
    assert direct.legacy_recurrence_to_first_exposure_error_rate_ratio_defined
    assert not direct.dormant_probe_available
    assert direct.eligible_dormant_generations == ()
    assert not direct.exact_generation_relock_observed
    assert not direct.durable_retrieval_before_scratch


def test_lucky_unrelated_dormant_table_does_not_create_acquisition_lineage(
    development_report,
) -> None:
    run = development_report.runs[0]
    segment_index, start, _ = _first_a_recurrence_bounds(run)
    trace = _with_entry_generation(
        _without_commits(run.trace),
        start=start,
        slot=1,
        generation=91,
    )
    records, aggregate = reconstruct_hidden_regime_retention(trace, run.config)
    record = next(item for item in records if item.segment_index == segment_index)
    assert record.dormant_probe_available
    assert record.best_dormant_composed_greedy_accuracy == 1.0
    assert record.prior_same_regime_lineages == ()
    assert record.prior_qualified_lineage_count == 0
    assert not record.lineage_retention_applicable
    assert record.acquisition_coverage_failure
    assert record.latest_prior_qualified_survived is None
    assert record.latest_prior_qualified_lineage_index is None
    assert record.any_prior_qualified_survived is None
    assert not record.selected_lineage_available
    assert record.selected_lineage_entry_composed_greedy_accuracy is None
    assert record.selected_exact_generation_relock_observed is None
    assert aggregate.acquisition_coverage_failure_count > 0
    assert aggregate.selected_lineage_not_applicable_count == (
        aggregate.acquisition_coverage_failure_count
    )


def test_no_prior_commit_is_acquisition_not_applicable_not_zero_retention(
    development_report,
) -> None:
    run = development_report.runs[0]
    segment_index, _, _ = _first_a_recurrence_bounds(run)
    records, _ = reconstruct_hidden_regime_retention(
        _without_commits(run.trace),
        run.config,
    )
    record = next(item for item in records if item.segment_index == segment_index)
    assert record.prior_same_regime_lineage_count == 0
    assert record.prior_qualified_lineage_count == 0
    assert record.acquisition_coverage_failure
    assert not record.lineage_retention_applicable
    assert record.latest_prior_qualified_survived is None
    assert record.latest_prior_qualified_lineage_index is None
    assert record.any_prior_qualified_survived is None
    assert not record.selected_lineage_available
    assert record.selected_lineage_entry_composed_greedy_accuracy is None
    assert record.selected_lineage_joint_bit_exact_preserved is None


def test_unqualified_prior_commit_is_disclosed_but_excluded_from_selection(
    development_report,
) -> None:
    run = development_report.runs[0]
    segment_index, start, _ = _first_a_recurrence_bounds(run)
    trace = _with_commit(
        _without_commits(run.trace),
        step=0,
        slot=1,
        generation=7,
        helper_bits=_ZERO_TABLE_BITS,
        beneficiary_bits=_ZERO_TABLE_BITS,
    )
    trace = _with_entry_generation(
        trace,
        start=start,
        slot=1,
        generation=7,
        helper_bits=_ZERO_TABLE_BITS,
        beneficiary_bits=_ZERO_TABLE_BITS,
    )
    lineages = reconstruct_commit_generation_lineages(trace, run.config)
    assert len(lineages) == 1
    lineage = lineages[0]
    assert isinstance(lineage, CommitGenerationLineage)
    assert lineage.target_mapping == (0, 1, 2)
    assert lineage.committed_composed_greedy_mapping == (0, 0, 0)
    assert lineage.committed_composed_greedy_accuracy == pytest.approx(1.0 / 3.0)
    assert not lineage.committed_composed_greedy_tie_free
    assert not lineage.acquisition_qualified
    assert len(lineage.helper_table_uint32_bits) == 9
    assert len(lineage.beneficiary_table_uint32_bits) == 9
    records, _ = reconstruct_hidden_regime_retention(trace, run.config)
    record = next(item for item in records if item.segment_index == segment_index)
    assert record.prior_same_regime_lineage_count == 1
    assert record.prior_unqualified_lineage_count == 1
    assert record.prior_qualified_lineage_count == 0
    assert record.latest_prior_qualified_lineage_index is None
    assert len(record.prior_same_regime_lineages) == 1
    assert not record.prior_same_regime_lineages[0].acquisition_qualified
    assert record.prior_same_regime_lineages[0].synchronized_generation_survives
    assert record.acquisition_coverage_failure
    assert not record.selected_lineage_available


def test_correct_first_index_mapping_with_a_tied_argmax_is_not_acquisition(
    development_report,
) -> None:
    run = development_report.runs[0]
    tied_helper = np.eye(3, dtype=np.float32)
    tied_helper[0, 1] = np.float32(1.0)
    trace = _with_commit(
        _without_commits(run.trace),
        step=0,
        slot=1,
        generation=7,
        helper_bits=tied_helper.view(np.uint32),
        beneficiary_bits=_IDENTITY_TABLE_BITS,
    )

    (lineage,) = reconstruct_commit_generation_lineages(trace, run.config)

    assert lineage.committed_composed_greedy_mapping == lineage.target_mapping
    assert lineage.committed_composed_greedy_accuracy == 1.0
    assert not lineage.committed_composed_greedy_tie_free
    assert not lineage.acquisition_qualified


def test_asymmetric_commit_lineage_fails_closed_instead_of_omitting_event(
    development_report,
) -> None:
    run = development_report.runs[0]
    trace = _without_commits(run.trace)
    trace = dataclasses.replace(
        trace,
        helper_committed_slot=jnp.asarray(trace.helper_committed_slot).at[0].set(1),
        helper_committed_generation=jnp.asarray(trace.helper_committed_generation)
        .at[0]
        .set(7),
    )
    with pytest.raises(ValueError, match="synchronized valid slot/generation"):
        reconstruct_commit_generation_lineages(trace, run.config)
    hostile = dataclasses.replace(run, trace=trace)
    errors = validate_hidden_regime_run_result(hostile)
    assert "synchronized lifecycle field committed_slot differs by role" in errors
    assert any("summary lineage reconstruction failed closed" in error for error in errors)


def test_qualified_lineage_preserves_exact_selective_commit_content(
    development_report,
) -> None:
    run = development_report.runs[0]
    trace, segment_index, start, end = _qualified_a_lineage_trace(run)
    lineages = reconstruct_commit_generation_lineages(trace, run.config)
    assert len(lineages) == 1
    lineage = lineages[0]
    assert lineage.acquisition_qualified
    assert lineage.committed_composed_greedy_tie_free
    assert lineage.committed_composed_greedy_mapping == lineage.target_mapping
    records, aggregate = reconstruct_hidden_regime_retention(trace, run.config)
    record = next(item for item in records if item.segment_index == segment_index)
    assert record.lineage_retention_applicable
    assert record.latest_prior_qualified_survived is True
    assert record.any_prior_qualified_survived is True
    assert record.selected_lineage_available
    assert record.selected_lineage_commit_step == 0
    assert record.selected_lineage_entry_activity_status == "dormant"
    assert record.selected_lineage_entry_composed_greedy_mapping == (0, 1, 2)
    assert record.selected_lineage_entry_composed_greedy_accuracy == 1.0
    assert record.selected_lineage_entry_minus_commit_accuracy == 0.0
    assert record.selected_lineage_helper_bit_exact_preserved is True
    assert record.selected_lineage_beneficiary_bit_exact_preserved is True
    assert record.selected_lineage_joint_bit_exact_preserved is True
    assert record.selected_lineage_zero_helper_accuracy == pytest.approx(1.0 / 3.0)
    assert record.selected_lineage_zero_beneficiary_accuracy == pytest.approx(1.0 / 3.0)
    assert record.selected_lineage_role_swapped_accuracy == 1.0
    assert aggregate.selected_entry_metric_denominator > 0
    reward_hostile = dataclasses.replace(
        trace,
        reward=jnp.asarray(trace.reward).at[start:end].set(jnp.float32(0.0)),
    )
    hostile_records, _ = reconstruct_hidden_regime_retention(reward_hostile, run.config)
    reward_hostile_record = next(
        item for item in hostile_records if item.segment_index == segment_index
    )
    assert reward_hostile_record.prior_same_regime_lineages == (
        record.prior_same_regime_lineages
    )
    assert reward_hostile_record.selected_lineage_index == record.selected_lineage_index
    assert reward_hostile_record.selected_lineage_commit_step == (
        record.selected_lineage_commit_step
    )


def test_same_generation_writable_content_corruption_survives_but_fails_preservation(
    development_report,
) -> None:
    run = development_report.runs[0]
    trace, segment_index, start, _ = _qualified_a_lineage_trace(run)
    trace = _with_entry_generation(
        trace,
        start=start,
        slot=1,
        generation=7,
        helper_bits=_ZERO_TABLE_BITS,
        beneficiary_bits=_ZERO_TABLE_BITS,
    )
    records, _ = reconstruct_hidden_regime_retention(trace, run.config)
    record = next(item for item in records if item.segment_index == segment_index)
    probe = record.prior_same_regime_lineages[0]
    assert probe.synchronized_generation_survives
    assert probe.entry_composed_greedy_mapping == (0, 0, 0)
    assert probe.entry_composed_greedy_accuracy == pytest.approx(1.0 / 3.0)
    assert probe.entry_minus_commit_accuracy == pytest.approx(-2.0 / 3.0)
    assert not probe.helper_bit_exact_preserved
    assert not probe.beneficiary_bit_exact_preserved
    assert not probe.joint_bit_exact_preserved
    assert record.latest_prior_qualified_survived is True
    assert record.any_prior_qualified_survived is True
    assert record.selected_lineage_available
    assert record.selected_lineage_entry_composed_greedy_accuracy == pytest.approx(
        1.0 / 3.0
    )
    assert record.selected_lineage_joint_bit_exact_preserved is False


def test_latest_qualified_evicted_while_older_survives_selects_canonical_older_lineage(
    development_report,
) -> None:
    run = development_report.runs[0]
    segment_index, start, _ = _first_a_recurrence_bounds(run)
    trace = _without_commits(run.trace)
    trace = _with_commit(trace, step=0, slot=1, generation=7)
    trace = _with_commit(trace, step=1, slot=2, generation=8)
    trace = _with_entry_generation(
        trace,
        start=start,
        slot=1,
        generation=7,
    )
    replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        status = jnp.asarray(getattr(trace, f"{role}_status_pre")).at[start, 2].set(
            SLOT_VACANT
        )
        generation = jnp.asarray(
            getattr(trace, f"{role}_generation_pre")
        ).at[start, 2].set(0)
        replacements[f"{role}_status_pre"] = status
        replacements[f"{role}_generation_pre"] = generation
    trace = dataclasses.replace(trace, **replacements)
    records, _ = reconstruct_hidden_regime_retention(trace, run.config)
    record = next(item for item in records if item.segment_index == segment_index)
    assert [item.commit_step for item in record.prior_same_regime_lineages] == [0, 1]
    assert all(item.acquisition_qualified for item in record.prior_same_regime_lineages)
    assert record.latest_prior_qualified_commit_step == 1
    assert record.latest_prior_qualified_lineage_index == 1
    assert record.latest_prior_qualified_survived is False
    assert record.any_prior_qualified_survived is True
    assert record.surviving_qualified_lineage_count == 1
    assert record.selected_lineage_commit_step == 0
    assert record.selected_lineage_index == 0
    assert record.selected_lineage_slot == 1
    assert record.selected_lineage_generation == 7
    assert not record.prior_same_regime_lineages[1].synchronized_generation_survives


def test_latest_surviving_qualified_lineage_is_selected_even_when_older_scores_better(
    development_report,
) -> None:
    run = development_report.runs[0]
    segment_index, start, _ = _first_a_recurrence_bounds(run)
    trace = _without_commits(run.trace)
    trace = _with_commit(trace, step=0, slot=1, generation=7)
    trace = _with_commit(trace, step=1, slot=2, generation=8)
    trace = _with_entry_generation(
        trace,
        start=start,
        slot=1,
        generation=7,
        helper_active=3,
        beneficiary_active=3,
    )
    trace = _with_entry_generation(
        trace,
        start=start,
        slot=2,
        generation=8,
        helper_bits=_ZERO_TABLE_BITS,
        beneficiary_bits=_ZERO_TABLE_BITS,
        helper_active=3,
        beneficiary_active=3,
    )
    records, _ = reconstruct_hidden_regime_retention(trace, run.config)
    record = next(item for item in records if item.segment_index == segment_index)
    entry_accuracies = [
        probe.entry_composed_greedy_accuracy
        for probe in record.prior_same_regime_lineages
    ]
    assert entry_accuracies[0] == 1.0
    assert entry_accuracies[1] == pytest.approx(1.0 / 3.0)
    assert record.latest_prior_qualified_survived is True
    assert record.latest_prior_qualified_lineage_index == 1
    assert record.surviving_qualified_lineage_count == 2
    assert record.selected_lineage_commit_step == 1
    assert record.selected_lineage_index == 1
    assert record.selected_lineage_slot == 2
    assert record.selected_lineage_entry_composed_greedy_accuracy == pytest.approx(
        1.0 / 3.0
    )
    assert record.selected_lineage_joint_bit_exact_preserved is False
    assert record.best_dormant_slot == 1
    assert record.best_dormant_composed_greedy_accuracy == 1.0


def test_lineage_aggregate_denominators_separate_coverage_survival_and_selection(
    development_report,
) -> None:
    run = development_report.runs[0]
    config = HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=(4, 4, 4),
            segment_regimes=(0, 1, 0),
            regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=run.config.learner,
        metric_window=4,
    )
    start = 8
    trace = _without_commits(run.trace)
    trace = _with_commit(trace, step=0, slot=1, generation=7)
    trace = _with_commit(trace, step=1, slot=2, generation=8)
    trace = _with_entry_generation(
        trace,
        start=start,
        slot=1,
        generation=7,
    )
    evict_latest: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        evict_latest[f"{role}_status_pre"] = jnp.asarray(
            getattr(trace, f"{role}_status_pre")
        ).at[start, 2].set(SLOT_VACANT)
        evict_latest[f"{role}_generation_pre"] = jnp.asarray(
            getattr(trace, f"{role}_generation_pre")
        ).at[start, 2].set(0)
    trace = dataclasses.replace(trace, **evict_latest)
    records, aggregate = reconstruct_hidden_regime_retention(trace, config)
    assert len(records) == aggregate.recurrence_count == 1
    assert aggregate.qualification_coverage_denominator == 1
    assert aggregate.lineage_retention_applicable_count == 1
    assert aggregate.acquisition_coverage_failure_count == 0
    assert aggregate.qualification_coverage_fraction == 1.0
    assert aggregate.prior_qualified_lineage_count == 2
    assert aggregate.surviving_qualified_lineage_count == 1
    assert aggregate.qualified_lineage_survival_denominator == 2
    assert aggregate.qualified_lineage_survival_fraction == 0.5
    assert aggregate.latest_qualified_version_survival_count == 0
    assert aggregate.latest_qualified_version_survival_denominator == 1
    assert aggregate.latest_qualified_version_survival_fraction == 0.0
    assert aggregate.any_qualified_knowledge_survival_count == 1
    assert aggregate.any_qualified_knowledge_survival_fraction == 1.0
    assert aggregate.selected_lineage_probe_available_count == 1
    assert aggregate.selected_lineage_probe_denominator == 1
    assert aggregate.selected_lineage_survival_failure_count == 0
    assert aggregate.selected_lineage_not_applicable_count == 0
    assert aggregate.selected_lineage_survival_fraction_given_qualified_prior == 1.0
    assert aggregate.selected_entry_metric_denominator == 1
    assert aggregate.selected_bit_exact_preservation_conditional_denominator == 1
    assert aggregate.selected_bit_exact_preservation_all_qualified_denominator == 1

    evict_all: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        evict_all[f"{role}_status_pre"] = jnp.asarray(
            getattr(trace, f"{role}_status_pre")
        ).at[start, 1].set(SLOT_VACANT)
        evict_all[f"{role}_generation_pre"] = jnp.asarray(
            getattr(trace, f"{role}_generation_pre")
        ).at[start, 1].set(0)
    no_survivor_trace = dataclasses.replace(trace, **evict_all)
    _, no_survivor = reconstruct_hidden_regime_retention(no_survivor_trace, config)
    assert no_survivor.lineage_retention_applicable_count == 1
    assert no_survivor.selected_lineage_probe_available_count == 0
    assert no_survivor.selected_lineage_survival_failure_count == 1
    assert no_survivor.selected_lineage_not_applicable_count == 0
    assert no_survivor.selected_lineage_survival_fraction_given_qualified_prior == 0.0
    assert no_survivor.selected_entry_metric_denominator == 0
    assert no_survivor.selected_entry_composed_greedy_accuracy_mean is None
    assert no_survivor.selected_exact_generation_relock_conditional_denominator == 0
    assert no_survivor.selected_exact_generation_relock_all_qualified_denominator == 1
    assert no_survivor.selected_exact_generation_relock_fraction_all_qualified == 0.0
    assert (
        no_survivor.selected_durable_retrieval_before_scratch_fraction_all_qualified
        == 0.0
    )

    _, no_acquisition = reconstruct_hidden_regime_retention(
        _without_commits(run.trace),
        config,
    )
    assert no_acquisition.lineage_retention_applicable_count == 0
    assert no_acquisition.acquisition_coverage_failure_count == 1
    assert no_acquisition.selected_lineage_survival_failure_count == 0
    assert no_acquisition.selected_lineage_not_applicable_count == 1
    assert no_acquisition.selected_lineage_probe_denominator == 0
    assert no_acquisition.selected_lineage_survival_fraction_given_qualified_prior is None


def test_acquisition_window_binds_latest_qualified_episode_not_first_exposure(
    development_report,
) -> None:
    run = development_report.runs[0]
    config = HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=(4, 4, 4, 4, 4),
            segment_regimes=(0, 1, 0, 1, 0),
            regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=run.config.learner,
        metric_window=4,
    )
    trace = _with_commit(
        _without_commits(run.trace),
        step=8,
        slot=1,
        generation=7,
        commit_segment_index=2,
    )
    trace = _with_entry_generation(
        trace,
        start=16,
        slot=1,
        generation=7,
    )
    rewards = jnp.asarray(trace.reward)
    rewards = rewards.at[0:4].set(jnp.float32(0.0))
    rewards = rewards.at[8:12].set(
        jnp.asarray([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    )
    rewards = rewards.at[16:20].set(
        jnp.asarray([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    )
    trace = dataclasses.replace(trace, reward=rewards)
    records, aggregate = reconstruct_hidden_regime_retention(trace, config)
    second_a = next(record for record in records if record.segment_index == 2)
    third_a = next(record for record in records if record.segment_index == 4)
    assert second_a.acquisition_coverage_failure
    assert third_a.latest_prior_qualified_commit_step == 8
    assert third_a.latest_prior_qualified_lineage_index == 0
    assert third_a.latest_qualified_acquisition_segment_index == 2
    assert third_a.latest_qualified_acquisition_episode_length == 4
    assert third_a.latest_qualified_acquisition_world_window_error_rate == 0.25
    assert third_a.first_world_window_error_rate == 0.5
    assert (
        third_a.recurrence_minus_latest_qualified_acquisition_error_rate_delta
        == 0.25
    )
    assert third_a.recurrence_to_latest_qualified_acquisition_error_rate_ratio == 2.0
    assert (
        third_a.recurrence_to_latest_qualified_acquisition_error_rate_ratio_defined
        is True
    )
    assert third_a.legacy_first_exposure_world_window_error_rate == 1.0
    assert third_a.legacy_recurrence_minus_first_exposure_error_rate_delta == -0.5
    assert aggregate.latest_qualified_acquisition_comparison_available_count == 1
    assert aggregate.latest_qualified_acquisition_comparison_denominator == 1
    assert aggregate.latest_qualified_acquisition_comparison_not_applicable_count == 2


def test_qualified_commit_in_adjacent_raw_segment_binds_coalesced_episode_start(
    development_report,
) -> None:
    run = development_report.runs[0]
    config = HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=(4, 4, 4, 4, 4, 4),
            segment_regimes=(0, 1, 0, 0, 1, 0),
            regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=run.config.learner,
        metric_window=4,
    )
    trace = _with_commit(
        _without_commits(run.trace),
        step=12,
        slot=1,
        generation=7,
        commit_segment_index=3,
    )
    trace = _with_entry_generation(
        trace,
        start=20,
        slot=1,
        generation=7,
    )
    rewards = jnp.asarray(trace.reward).at[8:12].set(
        jnp.asarray([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    )
    trace = dataclasses.replace(trace, reward=rewards)
    records, _ = reconstruct_hidden_regime_retention(trace, config)
    assert not any(record.segment_index == 3 for record in records)
    third_a = next(record for record in records if record.segment_index == 5)
    lineage = third_a.prior_same_regime_lineages[-1]
    assert lineage.commit_segment_index == 3
    assert third_a.latest_prior_qualified_lineage_index == 0
    assert third_a.latest_qualified_acquisition_segment_index == 2
    assert third_a.latest_qualified_acquisition_episode_length == 8
    assert third_a.latest_qualified_acquisition_world_window_error_rate == 0.25


def test_latest_evicted_baseline_stays_latest_while_selected_content_uses_older(
    development_report,
) -> None:
    run = development_report.runs[0]
    config = HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=(4, 4, 4, 4, 4),
            segment_regimes=(0, 1, 0, 1, 0),
            regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=run.config.learner,
        metric_window=4,
    )
    trace = _without_commits(run.trace)
    trace = _with_commit(
        trace,
        step=0,
        slot=1,
        generation=7,
        commit_segment_index=0,
    )
    trace = _with_commit(
        trace,
        step=8,
        slot=2,
        generation=8,
        commit_segment_index=2,
    )
    trace = _with_entry_generation(
        trace,
        start=16,
        slot=1,
        generation=7,
    )
    evicted: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        evicted[f"{role}_status_pre"] = jnp.asarray(
            getattr(trace, f"{role}_status_pre")
        ).at[16, 2].set(SLOT_VACANT)
        evicted[f"{role}_generation_pre"] = jnp.asarray(
            getattr(trace, f"{role}_generation_pre")
        ).at[16, 2].set(0)
    rewards = jnp.asarray(trace.reward)
    rewards = rewards.at[0:4].set(jnp.float32(1.0))
    rewards = rewards.at[8:12].set(
        jnp.asarray([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    )
    rewards = rewards.at[16:20].set(
        jnp.asarray([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    )
    trace = dataclasses.replace(trace, reward=rewards, **evicted)
    records, _ = reconstruct_hidden_regime_retention(trace, config)
    third_a = next(record for record in records if record.segment_index == 4)
    assert third_a.latest_prior_qualified_commit_step == 8
    assert third_a.latest_prior_qualified_lineage_index == 1
    assert third_a.latest_prior_qualified_survived is False
    assert third_a.selected_lineage_commit_step == 0
    assert third_a.selected_lineage_index == 0
    assert third_a.latest_qualified_acquisition_segment_index == 2
    assert third_a.latest_qualified_acquisition_world_window_error_rate == 0.25
    assert (
        third_a.recurrence_minus_latest_qualified_acquisition_error_rate_delta
        == 0.25
    )


def test_world_entry_window_names_misalignment_and_short_segments_explicitly(
    development_report,
) -> None:
    run = development_report.runs[0]
    jittered_lengths = list(run.config.world.segment_lengths)
    jittered_lengths[0] += 3
    jittered_lengths[-1] -= 3
    jittered = dataclasses.replace(
        run.config,
        world=dataclasses.replace(
            run.config.world,
            segment_lengths=tuple(jittered_lengths),
        ),
    )
    jittered_records, _ = reconstruct_hidden_regime_retention(run.trace, jittered)
    first_a_recurrence = next(item for item in jittered_records if item.segment_index == 2)
    assert first_a_recurrence.world_entry_learner_lease_offset == 3
    assert first_a_recurrence.world_entry_steps_to_first_learner_boundary == 1
    assert first_a_recurrence.first_world_window_length == run.config.learner.lease_length

    lease_16 = dataclasses.replace(
        run.config,
        learner=dataclasses.replace(run.config.learner, lease_length=16),
    )
    lease_16_records, _ = reconstruct_hidden_regime_retention(run.trace, lease_16)
    aligned_16 = next(item for item in lease_16_records if item.segment_index == 2)
    assert aligned_16.world_entry_learner_lease_offset == 0
    assert aligned_16.world_entry_steps_to_first_learner_boundary == 16
    start = sum(run.config.world.segment_lengths[:2])
    offset_15_trace = dataclasses.replace(
        run.trace,
        helper_lease_offset_pre=jnp.asarray(run.trace.helper_lease_offset_pre)
        .at[start]
        .set(15),
    )
    offset_15_records, _ = reconstruct_hidden_regime_retention(offset_15_trace, lease_16)
    offset_15 = next(item for item in offset_15_records if item.segment_index == 2)
    assert offset_15.world_entry_learner_lease_offset == 15
    assert offset_15.world_entry_steps_to_first_learner_boundary == 1

    short_lengths = list(run.config.world.segment_lengths)
    transferred = short_lengths[2] - (run.config.learner.lease_length - 1)
    short_lengths[2] -= transferred
    short_lengths[-1] += transferred
    short = dataclasses.replace(
        run.config,
        world=dataclasses.replace(run.config.world, segment_lengths=tuple(short_lengths)),
    )
    short_records, short_aggregate = reconstruct_hidden_regime_retention(run.trace, short)
    short_recurrence = next(item for item in short_records if item.segment_index == 2)
    assert not short_recurrence.first_world_window_complete
    assert short_recurrence.first_world_window_reward is None
    assert short_recurrence.first_world_window_errors is None
    assert short_aggregate.missing_first_world_window_count >= 1


def test_hand_constructed_intact_dormant_memory_passes_probe_and_role_zeroing_is_chance(
    development_report,
) -> None:
    run = development_report.runs[0]
    trace, segment_index, relock_step = _hand_construct_intact_dormant_relock(run)
    records, _ = reconstruct_hidden_regime_retention(trace, run.config)
    record = next(item for item in records if item.segment_index == segment_index)
    assert record.dormant_probe_available
    assert [item.slot for item in record.eligible_dormant_generations] == [1, 3]
    probe = record.eligible_dormant_generations[0]
    assert (probe.slot, probe.generation) == (1, 7)
    assert probe.composed_greedy_accuracy == 1.0
    assert probe.zero_helper_accuracy == pytest.approx(1.0 / 3.0)
    assert probe.zero_beneficiary_accuracy == pytest.approx(1.0 / 3.0)
    assert probe.role_swapped_accuracy == 1.0
    assert record.best_dormant_slot == 1
    assert record.best_dormant_generation == 7
    assert record.best_dormant_composed_greedy_accuracy == 1.0
    assert record.exact_generation_relock_observed
    assert record.observed_learner_boundaries_until_relock == 1
    assert record.scratch_entered_before_relock is False
    assert record.durable_retrieval_before_scratch
    for role in ("helper", "beneficiary"):
        assert int(getattr(trace, f"{role}_active_slot_pre")[relock_step]) == 1
        assert int(getattr(trace, f"{role}_active_slot_post")[relock_step]) == 1
        assert int(getattr(trace, f"{role}_status_pre")[relock_step, 1]) == SLOT_DURABLE
        assert int(getattr(trace, f"{role}_status_post")[relock_step, 1]) == SLOT_DURABLE
        assert int(getattr(trace, f"{role}_generation_pre")[relock_step, 1]) == 7
        assert int(getattr(trace, f"{role}_generation_post")[relock_step, 1]) == 7
        assert bool(getattr(trace, f"{role}_lease_boundary")[relock_step])
        assert bool(getattr(trace, f"{role}_durable_relevant")[relock_step])


def test_hand_constructed_generation_tamper_invalidates_exact_relock(
    development_report,
) -> None:
    run = development_report.runs[0]
    trace, segment_index, relock_step = _hand_construct_intact_dormant_relock(run)
    intact, _ = reconstruct_hidden_regime_retention(trace, run.config)
    intact_record = next(item for item in intact if item.segment_index == segment_index)
    assert intact_record.exact_generation_relock_observed
    tampered = dataclasses.replace(
        trace,
        helper_generation_post=jnp.asarray(trace.helper_generation_post)
        .at[relock_step, 1]
        .set(9),
    )
    hostile, _ = reconstruct_hidden_regime_retention(tampered, run.config)
    hostile_record = next(item for item in hostile if item.segment_index == segment_index)
    assert hostile_record.best_dormant_generation == 7
    assert not hostile_record.exact_generation_relock_observed
    assert hostile_record.observed_learner_boundaries_until_relock is None
    assert not hostile_record.durable_retrieval_before_scratch


def _with_primary_dormant_relock(
    trace: HiddenRegimePrimitiveTrace,
    *,
    start: int,
    end: int,
    slot: int = 1,
    generation: int = 7,
) -> tuple[HiddenRegimePrimitiveTrace, int]:
    boundary_mask = np.logical_and(
        np.asarray(trace.helper_lease_boundary)[start:end],
        np.asarray(trace.beneficiary_lease_boundary)[start:end],
    )
    relock_step = start + int(np.flatnonzero(boundary_mask)[0])
    replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        active_pre = jnp.asarray(getattr(trace, f"{role}_active_slot_pre"))
        active_post = jnp.asarray(getattr(trace, f"{role}_active_slot_post"))
        active_pre = active_pre.at[start:relock_step].set(2).at[relock_step].set(slot)
        active_post = (
            active_post.at[start:relock_step].set(2).at[relock_step].set(slot)
        )
        status_pre = jnp.asarray(getattr(trace, f"{role}_status_pre"))
        status_post = jnp.asarray(getattr(trace, f"{role}_status_post"))
        generation_pre = jnp.asarray(getattr(trace, f"{role}_generation_pre"))
        generation_post = jnp.asarray(getattr(trace, f"{role}_generation_post"))
        relevant = jnp.asarray(getattr(trace, f"{role}_durable_relevant"))
        replacements[f"{role}_active_slot_pre"] = active_pre
        replacements[f"{role}_active_slot_post"] = active_post
        replacements[f"{role}_status_pre"] = status_pre.at[relock_step, slot].set(
            SLOT_DURABLE
        )
        replacements[f"{role}_status_post"] = status_post.at[relock_step, slot].set(
            SLOT_DURABLE
        )
        replacements[f"{role}_generation_pre"] = generation_pre.at[
            relock_step,
            slot,
        ].set(generation)
        replacements[f"{role}_generation_post"] = generation_post.at[
            relock_step,
            slot,
        ].set(generation)
        replacements[f"{role}_durable_relevant"] = relevant.at[relock_step].set(True)
    return dataclasses.replace(trace, **replacements), relock_step


def test_selected_dormant_lineage_requires_exact_full_lease_relock_and_generation(
    development_report,
) -> None:
    run = development_report.runs[0]
    trace, segment_index, start, end = _qualified_a_lineage_trace(run)
    trace, relock_step = _with_primary_dormant_relock(trace, start=start, end=end)
    records, _ = reconstruct_hidden_regime_retention(trace, run.config)
    record = next(item for item in records if item.segment_index == segment_index)
    assert record.selected_lineage_entry_activity_status == "dormant"
    assert record.selected_exact_generation_relock_observed is True
    assert record.selected_first_exact_generation_relock_step == relock_step
    assert record.selected_first_exact_generation_relock_segment_step == (
        relock_step - start
    )
    assert record.selected_exact_generation_relock_phase == "post"
    assert record.selected_observed_learner_boundaries_until_relock == 1
    assert record.selected_scratch_entered_before_relock is False
    assert record.selected_durable_retrieval_before_scratch is True

    tampered = dataclasses.replace(
        trace,
        helper_generation_post=jnp.asarray(trace.helper_generation_post)
        .at[relock_step, 1]
        .set(8),
    )
    hostile_records, _ = reconstruct_hidden_regime_retention(tampered, run.config)
    hostile = next(
        item for item in hostile_records if item.segment_index == segment_index
    )
    assert hostile.selected_lineage_generation == 7
    assert hostile.selected_exact_generation_relock_observed is False
    assert hostile.selected_observed_learner_boundaries_until_relock is None
    assert hostile.selected_durable_retrieval_before_scratch is False


def test_selected_lineage_active_at_entry_is_immediate_but_mixed_roles_are_not(
    development_report,
) -> None:
    run = development_report.runs[0]
    trace, segment_index, start, end = _qualified_a_lineage_trace(run)
    active_replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        active_replacements[f"{role}_active_slot_pre"] = jnp.asarray(
            getattr(trace, f"{role}_active_slot_pre")
        ).at[start:end].set(1)
        active_replacements[f"{role}_active_slot_post"] = jnp.asarray(
            getattr(trace, f"{role}_active_slot_post")
        ).at[start:end].set(1)
    active_trace = dataclasses.replace(trace, **active_replacements)
    active_records, _ = reconstruct_hidden_regime_retention(active_trace, run.config)
    active = next(
        item for item in active_records if item.segment_index == segment_index
    )
    assert active.selected_lineage_entry_activity_status == "active"
    assert active.selected_exact_generation_relock_observed is True
    assert active.selected_first_exact_generation_relock_step == start
    assert active.selected_first_exact_generation_relock_segment_step == 0
    assert active.selected_exact_generation_relock_phase == "pre"
    assert active.selected_observed_learner_boundaries_until_relock == 0
    assert active.selected_scratch_entered_before_relock is False
    assert active.selected_durable_retrieval_before_scratch is True

    active_then_scratch_trace = dataclasses.replace(
        active_trace,
        helper_active_slot_post=jnp.asarray(active_trace.helper_active_slot_post)
        .at[start]
        .set(SCRATCH_SLOT),
        beneficiary_active_slot_post=jnp.asarray(
            active_trace.beneficiary_active_slot_post
        )
        .at[start]
        .set(SCRATCH_SLOT),
    )
    active_then_scratch_records, _ = reconstruct_hidden_regime_retention(
        active_then_scratch_trace,
        run.config,
    )
    active_then_scratch = next(
        item
        for item in active_then_scratch_records
        if item.segment_index == segment_index
    )
    assert active_then_scratch.selected_first_exact_generation_relock_step == start
    assert active_then_scratch.selected_exact_generation_relock_phase == "pre"
    assert active_then_scratch.selected_first_scratch_entry_step == start
    assert active_then_scratch.selected_first_scratch_entry_phase == "post"
    assert active_then_scratch.selected_scratch_entered_before_relock is False
    assert active_then_scratch.selected_durable_retrieval_before_scratch is True

    mixed_trace = dataclasses.replace(
        trace,
        helper_active_slot_pre=jnp.asarray(trace.helper_active_slot_pre)
        .at[start:end]
        .set(1),
        helper_active_slot_post=jnp.asarray(trace.helper_active_slot_post)
        .at[start:end]
        .set(1),
        beneficiary_active_slot_pre=jnp.asarray(trace.beneficiary_active_slot_pre)
        .at[start:end]
        .set(2),
        beneficiary_active_slot_post=jnp.asarray(trace.beneficiary_active_slot_post)
        .at[start:end]
        .set(2),
    )
    mixed_records, _ = reconstruct_hidden_regime_retention(mixed_trace, run.config)
    mixed = next(item for item in mixed_records if item.segment_index == segment_index)
    assert mixed.selected_lineage_available
    assert mixed.selected_lineage_entry_activity_status == "mixed"
    assert mixed.selected_exact_generation_relock_observed is False
    assert mixed.selected_durable_retrieval_before_scratch is False


def test_final_transition_switch_into_scratch_is_not_omitted_from_event_scan(
    development_report,
) -> None:
    run = development_report.runs[0]
    trace, segment_index, start, end = _qualified_a_lineage_trace(run)
    replacements: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        active_pre = jnp.asarray(getattr(trace, f"{role}_active_slot_pre")).at[
            start:end
        ].set(2)
        active_post = jnp.asarray(getattr(trace, f"{role}_active_slot_post")).at[
            start:end
        ].set(2)
        replacements[f"{role}_active_slot_pre"] = active_pre
        replacements[f"{role}_active_slot_post"] = active_post.at[end - 1].set(
            SCRATCH_SLOT
        )
    trace = dataclasses.replace(trace, **replacements)
    records, _ = reconstruct_hidden_regime_retention(trace, run.config)
    record = next(item for item in records if item.segment_index == segment_index)
    assert record.selected_lineage_entry_activity_status == "dormant"
    assert record.selected_exact_generation_relock_observed is False
    assert record.selected_first_scratch_entry_step == end - 1
    assert record.selected_first_scratch_entry_segment_step == end - start - 1
    assert record.selected_first_scratch_entry_phase == "post"
    assert record.selected_scratch_entered_before_relock is True
    assert record.selected_scratch_entered_before_relock_or_segment_end is True
    assert record.selected_durable_retrieval_before_scratch is False


def test_manual_retention_reconstruction_and_run_are_deterministic(
    audit_config: HiddenRegimeDevelopmentConfig,
    manual_seed_pair: HiddenRegimeSeedPair,
) -> None:
    first = run_hidden_regime_condition(
        SELECTIVE_FULL,
        seed_pair=manual_seed_pair,
        config=audit_config,
    )
    second = run_hidden_regime_condition(
        SELECTIVE_FULL,
        seed_pair=manual_seed_pair,
        config=audit_config,
    )
    assert first.summary.retention.to_dict() == second.summary.retention.to_dict()
    assert [item.to_dict() for item in first.summary.commit_generation_lineages] == [
        item.to_dict() for item in second.summary.commit_generation_lineages
    ]
    assert [item.to_dict() for item in first.summary.recurrence_retention] == [
        item.to_dict() for item in second.summary.recurrence_retention
    ]
    reconstructed, aggregate = reconstruct_hidden_regime_retention(
        first.trace,
        first.config,
    )
    assert reconstructed == first.summary.recurrence_retention
    assert aggregate == first.summary.retention


def test_atomic_replacement_is_exactly_paired_by_slot_and_generation(
    development_report,
) -> None:
    trace = development_report.runs[0].trace
    replacements = np.flatnonzero(np.asarray(trace.helper_retired_slot) >= 0)
    assert replacements.size == 1
    step = int(replacements[0])
    slot = int(trace.helper_retired_slot[step])
    assert slot == int(trace.helper_committed_slot[step])
    assert slot == int(trace.beneficiary_retired_slot[step])
    assert slot == int(trace.beneficiary_committed_slot[step])
    assert int(trace.helper_retired_generation[step]) == int(
        trace.helper_generation_pre[step, slot]
    )
    assert int(trace.helper_committed_generation[step]) == int(
        trace.helper_generation_post[step, slot]
    )
    assert int(trace.helper_generation_pre[step, slot]) != int(
        trace.helper_generation_post[step, slot]
    )
    assert not bool(trace.helper_selective_mutation_violation[step, slot])
    assert not bool(trace.beneficiary_selective_mutation_violation[step, slot])


def test_freeze_controls_enforce_structural_write_and_joint_commit_gates(
    development_report,
) -> None:
    by_condition = {run.condition: run for run in development_report.runs}
    helper_frozen = by_condition[HELPER_FROZEN]
    beneficiary_frozen = by_condition[BENEFICIARY_FROZEN]
    assert not np.any(helper_frozen.trace.helper_value_write)
    assert np.any(helper_frozen.trace.beneficiary_value_write)
    assert not np.any(np.asarray(helper_frozen.trace.helper_committed_slot) >= 0)
    assert not np.any(np.asarray(helper_frozen.trace.beneficiary_committed_slot) >= 0)
    assert np.any(beneficiary_frozen.trace.helper_value_write)
    assert not np.any(beneficiary_frozen.trace.beneficiary_value_write)
    assert not np.any(np.asarray(beneficiary_frozen.trace.helper_committed_slot) >= 0)
    assert not np.any(np.asarray(beneficiary_frozen.trace.beneficiary_committed_slot) >= 0)
    np.testing.assert_array_equal(
        np.asarray(helper_frozen.final_state.helper.values).view(np.uint32),
        np.zeros((4, 3, 3), dtype=np.float32).view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(beneficiary_frozen.final_state.beneficiary.values).view(np.uint32),
        np.zeros((4, 3, 3), dtype=np.float32).view(np.uint32),
    )


def test_no_channel_and_shuffled_channel_are_distinct_exact_controls(
    development_report,
) -> None:
    by_condition = {run.condition: run for run in development_report.runs}
    constant = by_condition[CONSTANT_CHANNEL]
    shuffled = by_condition[SHUFFLED_CHANNEL]
    np.testing.assert_array_equal(
        constant.trace.delivered_message,
        np.full((constant.config.num_steps,), CONSTANT_CHANNEL_SYMBOL, dtype=np.int8),
    )
    assert len(np.unique(np.asarray(shuffled.trace.delivered_message))) == 3
    assert not np.array_equal(
        shuffled.trace.delivered_message,
        shuffled.trace.helper_message,
    )
    assert validate_hidden_regime_run_result(constant) == ()
    assert validate_hidden_regime_run_result(shuffled) == ()


def test_paired_control_metrics_reconstruct_without_an_acceptance_gate(
    development_report,
) -> None:
    selective = development_report.runs[0]
    assert tuple(item.condition for item in development_report.paired_controls) == tuple(
        run.condition for run in development_report.runs[1:]
    )
    for metric, run in zip(
        development_report.paired_controls,
        development_report.runs[1:],
        strict=True,
    ):
        assert metric.resource_bytes == 552
        assert metric.mean_prequential_reward == run.summary.mean_prequential_reward
        assert metric.delta_vs_selective_full == pytest.approx(
            run.summary.mean_prequential_reward - selective.summary.mean_prequential_reward
        )
        assert metric.recurrence_entry_reward_mean == run.summary.recurrence_entry_reward_mean
    payload = development_report.to_dict(include_traces=False)
    assert payload["oracle_upper_bound_included"] is False
    assert payload["artifact_written"] is False
    assert payload["reserved_namespace_executed"] is False
    assert validate_hidden_regime_development_payload(payload) == ()


def test_hostile_same_generation_durable_bit_change_is_rejected(development_report) -> None:
    run = development_report.runs[0]
    trace = run.trace
    durable = np.asarray(trace.helper_status_pre) == SLOT_DURABLE
    replacement = np.asarray(trace.helper_retired_slot)[:, None] >= 0
    candidates = np.argwhere(np.logical_and(durable, ~replacement))
    assert candidates.size > 0
    step, slot = (int(value) for value in candidates[0])
    post = jnp.asarray(trace.helper_value_bits_post)
    post = post.at[step, slot, 0, 0].set(post[step, slot, 0, 0] ^ jnp.uint32(1))
    forged_audit = jnp.asarray(trace.helper_selective_mutation_violation).at[step, slot].set(True)
    hostile_trace = dataclasses.replace(
        trace,
        helper_value_bits_post=post,
        helper_selective_mutation_violation=forged_audit,
    )
    hostile = dataclasses.replace(run, trace=hostile_trace)
    errors = validate_hidden_regime_run_result(hostile)
    assert "helper selective durable bits changed without atomic replacement" in errors


def test_hostile_domains_and_unpaired_retirement_fail_without_indexing_crash(
    development_report,
) -> None:
    run = development_report.runs[0]
    bad_cue = dataclasses.replace(
        run,
        trace=dataclasses.replace(
            run.trace,
            helper_cue=jnp.asarray(run.trace.helper_cue).at[0].set(-1),
        ),
    )
    assert "trace.helper_cue is outside the ternary domain" in (
        validate_hidden_regime_run_result(bad_cue)
    )

    step = int(np.flatnonzero(np.asarray(run.trace.helper_committed_slot) < 0)[0])
    unpaired = dataclasses.replace(
        run,
        trace=dataclasses.replace(
            run.trace,
            helper_retired_slot=jnp.asarray(run.trace.helper_retired_slot).at[step].set(1),
            helper_retired_generation=jnp.asarray(run.trace.helper_retired_generation)
            .at[step]
            .set(1),
        ),
    )
    assert "helper retirement occurred without an atomic commit" in (
        validate_hidden_regime_run_result(unpaired)
    )


def test_hostile_frozen_scratch_write_cannot_hide_behind_a_false_write_flag(
    development_report,
) -> None:
    run = next(run for run in development_report.runs if run.condition == HELPER_FROZEN)
    post = jnp.asarray(run.trace.helper_value_bits_post)
    post = post.at[0, 0, 0, 0].set(post[0, 0, 0, 0] ^ jnp.uint32(1))
    hostile = dataclasses.replace(
        run,
        trace=dataclasses.replace(run.trace, helper_value_bits_post=post),
    )
    assert "helper value bank does not reconstruct from writes and commits" in (
        validate_hidden_regime_run_result(hostile)
    )


def test_hostile_oracle_summary_and_resource_mutations_fail_closed(development_report) -> None:
    run = development_report.runs[0]
    target = jnp.asarray(run.trace.oracle_target).at[0].set((run.trace.oracle_target[0] + 1) % 3)
    oracle_hostile = dataclasses.replace(
        run,
        trace=dataclasses.replace(run.trace, oracle_target=target),
    )
    assert "oracle_target does not match evaluator-only permutation" in (
        validate_hidden_regime_run_result(oracle_hostile)
    )

    summary_hostile = dataclasses.replace(
        run,
        summary=dataclasses.replace(
            run.summary,
            mean_prequential_reward=run.summary.mean_prequential_reward + 0.01,
        ),
    )
    assert "summary does not reconstruct exactly from primitive trace" in (
        validate_hidden_regime_run_result(summary_hostile)
    )
    resource_hostile = dataclasses.replace(
        run,
        resource=dataclasses.replace(run.resource, final_state_bytes=556),
    )
    errors = validate_hidden_regime_run_result(resource_hostile)
    assert "resource report does not match initial/final persistent state" in errors
    assert "condition is not the exact constant 552-byte dyad" in errors


def test_actual_world_termination_and_discount_leaves_are_strictly_bound(
    development_report,
) -> None:
    run = development_report.runs[0]
    terminated = dataclasses.replace(
        run,
        trace=dataclasses.replace(
            run.trace,
            world_terminated=jnp.asarray(run.trace.world_terminated).at[0].set(True),
        ),
    )
    assert (
        "world termination trace is not the actual continuing-world false leaf"
        in validate_hidden_regime_run_result(terminated)
    )
    discounted = dataclasses.replace(
        run,
        trace=dataclasses.replace(
            run.trace,
            world_discount=jnp.asarray(run.trace.world_discount)
            .at[0]
            .set(jnp.float32(0.0)),
        ),
    )
    assert "world discount trace is not the actual continuing-world one leaf" in (
        validate_hidden_regime_run_result(discounted)
    )


def test_named_rng_action_and_untraced_final_state_mutations_fail_replay(
    development_report,
) -> None:
    run = development_report.runs[0]
    message = jnp.asarray(run.trace.helper_message).at[0].set((run.trace.helper_message[0] + 1) % 3)
    action_hostile = dataclasses.replace(
        run,
        trace=dataclasses.replace(run.trace, helper_message=message),
    )
    assert "trace.helper_message differs from deterministic named-RNG replay" in (
        validate_hidden_regime_run_result(action_hostile)
    )

    helper = dataclasses.replace(
        run.final_state.helper,
        relevance_mean=run.final_state.helper.relevance_mean.at[0].add(jnp.float32(0.125)),
    )
    final_hostile = dataclasses.replace(
        run,
        final_state=dataclasses.replace(run.final_state, helper=helper),
    )
    assert "final_state.helper.relevance_mean differs from replay" in (
        validate_hidden_regime_run_result(final_hostile)
    )

    keyed_helper = dataclasses.replace(
        run.final_state.helper,
        key=jr.fold_in(run.final_state.helper.key, 99),
    )
    key_hostile = dataclasses.replace(
        run,
        final_state=dataclasses.replace(run.final_state, helper=keyed_helper),
    )
    assert "final_state.helper.key differs from replay" in (
        validate_hidden_regime_run_result(key_hostile)
    )


def test_serialized_envelope_rejects_scope_resource_and_schema_attacks(
    development_report,
) -> None:
    payload = development_report.to_dict(include_traces=False)
    attacks: list[tuple[dict[str, object], str]] = []

    extra = deepcopy(payload)
    extra["acceptance_threshold"] = 0.8
    attacks.append((extra, "strict v4 schema"))

    promoted = deepcopy(payload)
    promoted["scientific_promotion_allowed"] = True
    attacks.append((promoted, "promotion"))

    reserved = deepcopy(payload)
    reserved["reserved_namespace_executed"] = True
    attacks.append((reserved, "unexecuted"))

    resource = deepcopy(payload)
    runs = resource["runs"]
    assert isinstance(runs, list)
    first = runs[0]
    assert isinstance(first, dict)
    first_resource = first["resource"]
    assert isinstance(first_resource, dict)
    first_resource["final_state_bytes"] = 548
    attacks.append((resource, "552-byte"))

    nested_schema = deepcopy(payload)
    nested_runs = nested_schema["runs"]
    assert isinstance(nested_runs, list)
    nested_first = nested_runs[0]
    assert isinstance(nested_first, dict)
    nested_first["oracle_policy"] = True
    attacks.append((nested_schema, "run 0 fields"))

    paired_delta = deepcopy(payload)
    paired_controls = paired_delta["paired_controls"]
    assert isinstance(paired_controls, list)
    first_control = paired_controls[0]
    assert isinstance(first_control, dict)
    first_control["delta_vs_selective_full"] = 999.0
    attacks.append((paired_delta, "reward delta"))

    nested_config = deepcopy(payload)
    config = nested_config["config"]
    assert isinstance(config, dict)
    config["portable_artifact"] = True
    attacks.append((nested_config, "config does not reconstruct"))

    nested_summary = deepcopy(payload)
    summary_runs = nested_summary["runs"]
    assert isinstance(summary_runs, list)
    summary_first = summary_runs[0]
    assert isinstance(summary_first, dict)
    summary = summary_first["summary"]
    assert isinstance(summary, dict)
    summary["scientific_claim"] = True
    attacks.append((nested_summary, "summary fields"))

    for hostile, fragment in attacks:
        assert any(
            fragment in error for error in validate_hidden_regime_development_payload(hostile)
        )


def test_serialized_trace_contents_are_bound_to_exact_replay(development_report) -> None:
    payload = development_report.to_dict(include_traces=True)
    assert validate_hidden_regime_development_payload(payload) == ()
    hostile = deepcopy(payload)
    runs = hostile["runs"]
    assert isinstance(runs, list)
    first = runs[0]
    assert isinstance(first, dict)
    trace = first["trace"]
    assert isinstance(trace, dict)
    reward = trace["reward"]
    assert isinstance(reward, list)
    reward[0] = 1.0 - float(reward[0])
    assert "serialized report differs from exact deterministic replay" in (
        validate_hidden_regime_development_payload(hostile)
    )

    hostile_world = deepcopy(payload)
    world_runs = hostile_world["runs"]
    assert isinstance(world_runs, list)
    world_first = world_runs[0]
    assert isinstance(world_first, dict)
    world_trace = world_first["trace"]
    assert isinstance(world_trace, dict)
    cue_keys = world_trace["world_cue_key_data_post"]
    assert isinstance(cue_keys, list)
    assert isinstance(cue_keys[0], list)
    cue_keys[0][0] = int(cue_keys[0][0]) ^ 1
    assert "serialized report differs from exact deterministic replay" in (
        validate_hidden_regime_development_payload(hostile_world)
    )

    hostile_termination = deepcopy(payload)
    termination_runs = hostile_termination["runs"]
    assert isinstance(termination_runs, list)
    termination_trace = termination_runs[0]["trace"]
    assert isinstance(termination_trace, dict)
    termination_trace["world_terminated"][0] = True
    assert "serialized report differs from exact deterministic replay" in (
        validate_hidden_regime_development_payload(hostile_termination)
    )

    hostile_discount = deepcopy(payload)
    discount_runs = hostile_discount["runs"]
    assert isinstance(discount_runs, list)
    discount_trace = discount_runs[0]["trace"]
    assert isinstance(discount_trace, dict)
    discount_trace["world_discount"][0] = 0.0
    assert "serialized report differs from exact deterministic replay" in (
        validate_hidden_regime_development_payload(hostile_discount)
    )


def test_serialized_commit_lineage_is_canonical_and_mutation_bound(
    development_report,
) -> None:
    payload = development_report.to_dict(include_traces=False)
    runs = payload["runs"]
    assert isinstance(runs, list)
    first = runs[0]
    assert isinstance(first, dict)
    summary = first["summary"]
    assert isinstance(summary, dict)
    lineages = summary["commit_generation_lineages"]
    assert isinstance(lineages, list)
    assert lineages
    assert [lineage["lineage_index"] for lineage in lineages] == list(
        range(len(lineages))
    )
    assert [lineage["commit_step"] for lineage in lineages] == sorted(
        lineage["commit_step"] for lineage in lineages
    )
    assert all(len(lineage["target_mapping"]) == 3 for lineage in lineages)
    assert all(len(lineage["helper_table_uint32_bits"]) == 9 for lineage in lineages)
    assert all(
        len(lineage["beneficiary_table_uint32_bits"]) == 9 for lineage in lineages
    )

    malformed = deepcopy(payload)
    malformed_lineages = malformed["runs"][0]["summary"]["commit_generation_lineages"]
    malformed_lineages[0]["helper_table_uint32_bits"].pop()
    assert any(
        "commit lineage 0 fixed arrays are malformed" in error
        for error in validate_hidden_regime_development_payload(malformed)
    )

    mutated = deepcopy(payload)
    mutated_lineages = mutated["runs"][0]["summary"]["commit_generation_lineages"]
    mutated_lineages[0]["helper_table_uint32_bits"][0] ^= 1
    assert "serialized report differs from exact deterministic replay" in (
        validate_hidden_regime_development_payload(mutated)
    )

    recurrence_records = summary["recurrence_retention"]
    lineage_record_index = next(
        index
        for index, record in enumerate(recurrence_records)
        if record["prior_same_regime_lineages"]
    )
    probe_mutation = deepcopy(payload)
    probes = probe_mutation["runs"][0]["summary"]["recurrence_retention"]
    probes[lineage_record_index]["prior_same_regime_lineages"][0]["generation"] += 1
    assert "serialized report differs from exact deterministic replay" in (
        validate_hidden_regime_development_payload(probe_mutation)
    )


def test_serialized_retention_records_bind_all_dormant_generations_and_best_choice(
    development_report,
) -> None:
    payload = development_report.to_dict(include_traces=False)
    runs = payload["runs"]
    assert isinstance(runs, list)
    first = runs[0]
    assert isinstance(first, dict)
    summary = first["summary"]
    assert isinstance(summary, dict)
    records = summary["recurrence_retention"]
    assert isinstance(records, list)
    record_index = next(
        index
        for index, record in enumerate(records)
        if isinstance(record, dict) and record["eligible_dormant_generations"]
    )

    missing = deepcopy(payload)
    missing_runs = missing["runs"]
    assert isinstance(missing_runs, list)
    missing_first = missing_runs[0]
    assert isinstance(missing_first, dict)
    missing_summary = missing_first["summary"]
    assert isinstance(missing_summary, dict)
    missing_records = missing_summary["recurrence_retention"]
    assert isinstance(missing_records, list)
    missing_record = missing_records[record_index]
    assert isinstance(missing_record, dict)
    del missing_record["eligible_dormant_generations"]
    assert any(
        "retention record" in error
        for error in validate_hidden_regime_development_payload(missing)
    )

    forged_best = deepcopy(payload)
    forged_runs = forged_best["runs"]
    assert isinstance(forged_runs, list)
    forged_first = forged_runs[0]
    assert isinstance(forged_first, dict)
    forged_summary = forged_first["summary"]
    assert isinstance(forged_summary, dict)
    forged_records = forged_summary["recurrence_retention"]
    assert isinstance(forged_records, list)
    forged_record = forged_records[record_index]
    assert isinstance(forged_record, dict)
    forged_record["best_dormant_generation"] = 2_000_000_000
    assert "serialized report differs from exact deterministic replay" in (
        validate_hidden_regime_development_payload(forged_best)
    )
