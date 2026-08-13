"""Fixed-seed development contracts for natural generated-D birth feasibility."""

from __future__ import annotations

import dataclasses

import pytest

from alberta_framework.evaluation import (
    generated_natural_d_birth_probe as natural_d_probe,
)
from alberta_framework.evaluation.generated_class_d_mapping_twin import (
    D_MAPPING_NEVER_SEEN_TWIN,
)
from alberta_framework.evaluation.generated_class_recurrence import FULL_LIFECYCLE
from alberta_framework.evaluation.generated_natural_d_birth_probe import (
    GENERATED_NATURAL_D_BIRTH_PROBE_STATUS,
    GENERATED_NATURAL_D_BIRTH_RESULT_STATUS,
    OUTCOME_ACTIVE_CONTRIBUTION,
    PROBE_SEED,
    PROBE_STOP_POST_STEP,
    GeneratedNaturalDBirthProbePermit,
    GeneratedNaturalDBirthProbeResult,
    build_generated_natural_d_birth_probe_permit,
    run_generated_natural_d_birth_probe,
    validate_generated_natural_d_birth_probe_permit,
    validate_generated_natural_d_birth_probe_result,
)

pytestmark = pytest.mark.development


def _historical_permit() -> GeneratedNaturalDBirthProbePermit:
    return natural_d_probe._canonical_permit()  # noqa: SLF001


def test_historical_probe_permit_is_narrow_but_current_sources_fail_closed() -> None:
    permit = _historical_permit()

    pinned = dict(permit.dependency_source_sha256)
    live = dict(natural_d_probe._live_dependency_hashes())  # noqa: SLF001
    assert {name for name in pinned if pinned[name] != live[name]} == {
        "alberta_framework.core.compositional_features",
        "alberta_framework.evaluation.generated_birth_identity_trace_binding",
    }
    with pytest.raises(ValueError, match="dependency source hashes drifted"):
        build_generated_natural_d_birth_probe_permit()

    assert permit.status == GENERATED_NATURAL_D_BIRTH_PROBE_STATUS
    assert permit.development_only
    assert not permit.executes_upstream_d_mapping_v0_contract
    assert not permit.upstream_d_mapping_v0_execution_authorized
    assert not permit.upstream_d_mapping_v0_runner_authorized
    assert permit.canonical_dataset_reused_read_only
    assert permit.fixed_in_memory_execution_authorized
    assert permit.fixed_source_replay_authorized
    assert permit.fixed_seed_uint32 == PROBE_SEED == 101
    assert permit.fixed_arm_order == (FULL_LIFECYCLE, D_MAPPING_NEVER_SEEN_TWIN)
    assert permit.fixed_learner_control_name == FULL_LIFECYCLE
    assert permit.fixed_stop_post_step == PROBE_STOP_POST_STEP == 3_928
    assert permit.stop_is_complete_twin_first_true_d_exposure
    assert permit.learner_visible_fields == ("raw_features", "target")
    assert not any(
        (
            permit.artifact_writes_authorized,
            permit.outputs_authorized,
            permit.search_authorized,
            permit.threshold_authorized,
            permit.evidence_authorized,
            permit.scientific_promotion_allowed,
            permit.feature_or_lineage_injection_authorized,
            permit.evaluator_chosen_curation_slot_authorized,
            permit.learner_config_change_authorized,
        )
    )
    assert permit.expected_transactions_per_arm == 3_928
    assert permit.expected_curation_opportunities_per_arm == 122
    assert permit.wall_clock_threshold is None


def test_probe_permit_seed_arm_and_authority_tampering_fail_closed() -> None:
    permit = _historical_permit()

    with pytest.raises(ValueError, match="dependency source hashes drifted"):
        validate_generated_natural_d_birth_probe_permit(permit)

    with pytest.raises(ValueError, match="exact canonical permit"):
        validate_generated_natural_d_birth_probe_permit(
            dataclasses.replace(permit, fixed_seed_uint32=102)
        )
    with pytest.raises(ValueError, match="exact canonical permit"):
        validate_generated_natural_d_birth_probe_permit(
            dataclasses.replace(
                permit,
                fixed_arm_order=(
                    permit.fixed_arm_order[1],
                    permit.fixed_arm_order[0],
                ),
            )
        )
    with pytest.raises(ValueError, match="exact canonical permit"):
        validate_generated_natural_d_birth_probe_permit(
            dataclasses.replace(permit, search_authorized=True)
        )


@pytest.fixture(scope="module")
def fixed_probe_result() -> GeneratedNaturalDBirthProbeResult:
    try:
        permit = build_generated_natural_d_birth_probe_permit()
    except ValueError as error:
        if "dependency source hashes drifted" not in str(error):
            raise
        pytest.skip(
            "the consumed Natural-D v0 source closure is invalid; a reviewed, "
            "new-version nonpromoting replay is required"
        )
    return run_generated_natural_d_birth_probe(permit)


@pytest.mark.slow
def test_fixed_paired_probe_authenticates_natural_birth_and_first_use(
    fixed_probe_result: GeneratedNaturalDBirthProbeResult,
) -> None:
    result = fixed_probe_result

    assert result.status == GENERATED_NATURAL_D_BIRTH_RESULT_STATUS
    assert result.paired_steps_executed == 3_928
    assert result.total_authenticated_transactions == 7_856
    assert result.first_d_target_bit_mismatch_count == 421
    assert result.non_first_d_target_bit_mismatch_count == 0
    assert result.initial_core_state_bit_exact_across_arms
    assert result.learner_key_bit_exact_across_arms_every_step
    assert result.learner_visible_field_schema_equal_across_arms
    assert result.raw_observation_values_bit_exact_across_arms
    assert result.target_value_differences_confined_to_first_d
    assert result.artifact_bytes_written == 0
    assert result.thresholds_applied == 0
    assert result.searches_executed == 0
    assert not result.evidence_authorized
    assert not result.promotion_authorized

    reference, twin = result.arm_results
    assert reference.arm_name == FULL_LIFECYCLE
    assert twin.arm_name == D_MAPPING_NEVER_SEEN_TWIN
    assert reference.initial_core_state_sha256 == twin.initial_core_state_sha256
    assert reference.final_learner_key_words_uint32 == (
        twin.final_learner_key_words_uint32
    )

    for arm in result.arm_results:
        assert arm.authenticated_transaction_count == 3_928
        assert arm.source_replay_authenticated_transaction_count == 3_928
        assert len(arm.transaction_sha256s) == 3_928
        assert arm.every_transaction_source_replay_authenticated
        assert arm.curation_opportunity_count == 122
        assert arm.curation_event_count == 122
        assert arm.persistent_capacity_unchanged
        assert arm.initial_persistent_core_state_nbytes == (
            arm.final_persistent_core_state_nbytes
        )
        assert arm.initial_persistent_ledger_state_nbytes == (
            arm.final_persistent_ledger_state_nbytes
        )
        assert arm.final_step_words_uint32 == (0, 3_928)
        assert arm.final_ledger_step_words_uint32 == (0, 3_928)
        assert arm.exact_any_birth_post_steps == (800,)
        assert arm.expanded_any_birth_post_steps == (800,)
        assert arm.exact_candidate_count_increase_post_steps[0] == 800
        first_birth = arm.curation_transactions[24]
        assert first_birth.post_step == 800
        assert first_birth.transaction_sha256 == arm.transaction_sha256s[799]
        assert first_birth.snapshot.candidate_exact_root_count == 1
        assert first_birth.snapshot.candidate_expanded_mask_count == 1
        assert len(first_birth.snapshot.candidate_expanded_birth_identities) == 1
        assert arm.final_snapshot.active_exact_root_count == 1
        assert arm.final_snapshot.candidate_exact_root_count == 0
        assert arm.first_prequential_head_use is not None
        assert arm.first_nonzero_prequential_contribution is not None
        assert arm.descriptive_outcome == OUTCOME_ACTIVE_CONTRIBUTION
        assert arm.artifacts_written == 0
        assert not arm.evidence_authorized
        assert not arm.promotion_authorized

    assert reference.exact_active_count_increase_post_steps[0] == 1_728
    assert twin.exact_active_count_increase_post_steps[0] == 3_584
    assert reference.first_prequential_head_use is not None
    assert reference.first_nonzero_prequential_contribution is not None
    assert twin.first_prequential_head_use is not None
    assert twin.first_nonzero_prequential_contribution is not None
    assert reference.first_prequential_head_use.zero_based_step_index == 1_728
    assert reference.first_prequential_head_use.true_d_exposure_ordinal == 112
    assert reference.first_nonzero_prequential_contribution.zero_based_step_index == (
        1_728
    )
    assert twin.first_prequential_head_use.zero_based_step_index == 3_584
    assert twin.first_prequential_head_use.true_d_exposure_ordinal == 46
    assert twin.first_nonzero_prequential_contribution.zero_based_step_index == 3_584
    assert reference.true_d_exposure_count == 810
    assert reference.deranged_d_exposure_count == 0
    assert twin.true_d_exposure_count == 389
    assert twin.deranged_d_exposure_count == 421


@pytest.mark.slow
def test_fixed_probe_result_revalidates_and_tamper_fails_closed(
    fixed_probe_result: GeneratedNaturalDBirthProbeResult,
) -> None:
    assert (
        validate_generated_natural_d_birth_probe_result(fixed_probe_result)
        is fixed_probe_result
    )
    with pytest.raises(ValueError, match="forbidden output"):
        validate_generated_natural_d_birth_probe_result(
            dataclasses.replace(fixed_probe_result, searches_executed=1)
        )
