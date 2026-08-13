"""Strict unit contracts for the matched nonpromoting WP7.4 evaluator."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.calibrated_extended_search_control import (
    SEARCH_MODE_COMBINED,
    SEARCH_MODE_MODEL_FREE_EXTENDED_Q,
    SEARCH_MODE_OPTION_MODEL,
    SEARCH_MODE_PRIMITIVE_MODEL,
)
from alberta_framework.evaluation.calibrated_extended_search_development import (
    ASSESSMENT_STATUS,
    CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_STATUS,
    CANONICAL_ARM_ORDER,
    MODEL_FREE_VS_OPTION_MODEL,
    MODEL_SNAPSHOT_BOUNDARY,
    PRIMITIVE_VS_COMBINED,
    SOURCE_CALIBRATION_BOUNDARY,
    CalibratedExtendedSearchDevelopmentConfig,
    CalibratedExtendedSearchDevelopmentEvaluator,
    CalibratedSearchDevelopmentSuite,
    build_calibrated_search_model_snapshot,
    build_calibrated_search_source_runtime_manifest,
    reconstruct_calibrated_search_evaluator_trace,
    validate_calibrated_search_development_suite,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def completed() -> tuple[
    CalibratedExtendedSearchDevelopmentEvaluator,
    CalibratedSearchDevelopmentSuite,
]:
    evaluator = CalibratedExtendedSearchDevelopmentEvaluator(
        CalibratedExtendedSearchDevelopmentConfig()
    )
    return evaluator, evaluator.run()


def test_config_is_strict_nonpromoting_and_has_finite_protocol_caps() -> None:
    config = CalibratedExtendedSearchDevelopmentConfig()
    payload = config.to_config()
    assert CalibratedExtendedSearchDevelopmentConfig.from_config(payload) == config
    assert payload["assessment_status"] == ASSESSMENT_STATUS
    assert payload["model_snapshot_boundary"] == MODEL_SNAPSHOT_BOUNDARY
    assert payload["source_calibration_boundary"] == SOURCE_CALIBRATION_BOUNDARY
    assert payload["thresholds"] is None
    for name in (
        "held_out_seeds_used",
        "thresholds_frozen",
        "artifact_writes_authorized",
        "evidence_authorized",
        "scientific_promotion_allowed",
        "policy_authority",
    ):
        assert payload[name] is False
    with pytest.raises(ValueError, match="fields"):
        CalibratedExtendedSearchDevelopmentConfig.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="uint32"):
        CalibratedExtendedSearchDevelopmentConfig(seed=2**32)
    assert CalibratedExtendedSearchDevelopmentConfig(seed=2**32 - 1).seed == 2**32 - 1
    with pytest.raises(ValueError, match="whole fixed candidate cycles"):
        CalibratedExtendedSearchDevelopmentConfig(num_steps=7)
    with pytest.raises(ValueError, match=r"\[1, 100000\]"):
        CalibratedExtendedSearchDevelopmentConfig(num_steps=100_002)
    with pytest.raises(ValueError, match="preloaded-counter headroom"):
        CalibratedExtendedSearchDevelopmentConfig(max_observations=7)
    boundary = CalibratedExtendedSearchDevelopmentConfig(max_observations=8)
    assert boundary.maximum_anchor_trial_increments == 6
    assert boundary.maximum_candidate_observation_increments == 1
    assert boundary.required_max_observations == 8
    boundary_suite = CalibratedExtendedSearchDevelopmentEvaluator(boundary).run()
    assert all(
        record.summary.final_anchor_revisit_trials == (8, 8)
        for record in boundary_suite.arm_records
    )
    with pytest.raises(ValueError, match="requires at least 15"):
        CalibratedExtendedSearchDevelopmentConfig(
            calibration_evidence_floor=9,
            max_observations=14,
        )
    assert (
        CalibratedExtendedSearchDevelopmentConfig(
            calibration_evidence_floor=9,
            max_observations=15,
        ).required_max_observations
        == 15
    )
    with pytest.raises(ValueError, match="requires at least 21"):
        CalibratedExtendedSearchDevelopmentConfig(
            model_support_floor=20,
            max_observations=20,
        )


def test_source_runtime_snapshot_and_trace_reconstruct_exactly() -> None:
    config = CalibratedExtendedSearchDevelopmentConfig()
    manifest = build_calibrated_search_source_runtime_manifest()
    assert manifest.schema.endswith("manifest.v1")
    assert len(manifest.source_files) == 2
    assert all(len(item.sha256) == 64 and item.nbytes > 0 for item in manifest.source_files)
    assert len(manifest.manifest_sha256) == 64
    assert manifest.prng_impl == "threefry2x32"
    assert manifest.chex_version
    assert manifest.operating_system
    assert manifest.operating_system_release
    assert manifest.machine
    assert manifest.device_count == len(manifest.device_platforms)
    assert manifest.device_count == len(manifest.device_kinds)
    assert manifest.local_device_count <= manifest.device_count
    assert manifest.backend_platform_version
    assert manifest.default_prng_impl == "threefry2x32"
    assert "unobservable-compiler-and-host-determinants" in (
        manifest.runtime_identity_scope
    )
    snapshot = build_calibrated_search_model_snapshot(config, manifest)
    trace = reconstruct_calibrated_search_evaluator_trace(config, snapshot)
    replayed = reconstruct_calibrated_search_evaluator_trace(config, snapshot)

    assert snapshot.source_manifest_sha256 == manifest.manifest_sha256
    assert snapshot.snapshot_sha256 == bytes(
        np.asarray(trace.snapshot_sha256_bytes, dtype=np.uint8)
    ).hex()
    assert np.array_equal(np.asarray(trace.decision_ids), np.asarray(replayed.decision_ids))
    assert np.array_equal(
        np.asarray(trace.external_returns), np.asarray(replayed.external_returns)
    )
    assert np.array_equal(
        np.asarray(trace.evaluator_key_after[:-1]),
        np.asarray(trace.evaluator_key_before[1:]),
    )
    # One exact cycle exposes all C=M(K+N)=6 executed candidate identities.
    flat = np.where(
        np.asarray(trace.executed_kinds) == 0,
        np.asarray(trace.executed_semantic_indices),
        2 + np.asarray(trace.executed_semantic_indices),
    ) * 2 + np.asarray(trace.decision_anchor_indices)
    np.testing.assert_array_equal(flat, np.arange(6, dtype=np.int32))
    assert bool(jnp.all(trace.natural_completions))
    assert not bool(jnp.any(trace.censored))


def test_suite_has_four_exact_modes_no_assessment_and_required_contrasts(
    completed: tuple[
        CalibratedExtendedSearchDevelopmentEvaluator,
        CalibratedSearchDevelopmentSuite,
    ],
) -> None:
    _, suite = completed
    assert suite.status == CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_STATUS
    assert suite.assessment_status == ASSESSMENT_STATUS
    assert suite.canonical_arm_order == CANONICAL_ARM_ORDER == (
        SEARCH_MODE_MODEL_FREE_EXTENDED_Q,
        SEARCH_MODE_PRIMITIVE_MODEL,
        SEARCH_MODE_OPTION_MODEL,
        SEARCH_MODE_COMBINED,
    )
    assert tuple(record.mode for record in suite.arm_records) == CANONICAL_ARM_ORDER
    assert tuple(contrast.name for contrast in suite.contrasts) == (
        MODEL_FREE_VS_OPTION_MODEL,
        PRIMITIVE_VS_COMBINED,
    )
    assert suite.thresholds is None
    assert suite.aggregate_verdict is None
    assert suite.artifact_output_path is None
    assert suite.authenticated_replay_verified is False
    assert all(
        contrast.assessment_status == ASSESSMENT_STATUS
        and contrast.threshold is None
        and contrast.verdict is None
        for contrast in suite.contrasts
    )
    assert all(
        value is False
        for value in (
            suite.held_out_seeds_used,
            suite.thresholds_frozen,
            suite.artifact_writes_authorized,
            suite.evidence_authorized,
            suite.scientific_promotion_allowed,
            suite.policy_authority,
        )
    )
    assert validate_calibrated_search_development_suite(suite) == ()


def test_common_random_number_budget_and_terminal_resource_accounting_are_exact(
    completed: tuple[
        CalibratedExtendedSearchDevelopmentEvaluator,
        CalibratedSearchDevelopmentSuite,
    ],
) -> None:
    evaluator, suite = completed
    audit = suite.matched_audit
    assert audit.errors == ()
    assert all(
        (
            audit.trace_reconstruction_passed,
            audit.immutable_initial_snapshot_passed,
            audit.common_random_number_trace_passed,
            audit.identical_anchor_bank_passed,
            audit.identical_candidate_opportunities_passed,
            audit.equal_real_experience_passed,
            audit.one_shared_backup_budget_passed,
            audit.source_binding_passed,
        )
    )
    expected_attempts = evaluator.config.num_steps * evaluator.config.backup_budget
    for record in suite.arm_records:
        assert record.accounting.real_experience_records == evaluator.config.num_steps
        assert record.accounting.candidate_capacity == 6
        assert record.accounting.shared_backup_budget == evaluator.config.backup_budget
        assert record.accounting.expected_backup_attempts == expected_attempts
        assert record.accounting.actual_backup_attempts == expected_attempts
        assert record.accounting.actual_learner_updates <= expected_attempts
        assert record.accounting.planner_rng_draws == 0
        assert record.accounting.persistent_state_growth_bytes == 0
        np.testing.assert_array_equal(
            np.asarray(record.trace.backup_attempts),
            np.full(evaluator.config.num_steps, evaluator.config.backup_budget),
        )
    accounting = suite.evaluator_accounting
    assert accounting.unique_real_experience_records == evaluator.config.num_steps
    assert accounting.total_matched_real_experience_deliveries == 4 * evaluator.config.num_steps
    assert accounting.total_backup_attempts == 4 * expected_attempts
    assert accounting.maximum_total_learner_updates == 4 * expected_attempts
    assert accounting.actual_total_learner_updates <= 4 * expected_attempts
    assert accounting.evaluator_scalar_random_draws == 2 * evaluator.config.num_steps
    assert accounting.planner_random_draws == 0
    assert accounting.source_snapshot_array_nbytes > 100
    assert accounting.evaluator_trace_array_nbytes > 100
    assert accounting.resumable_state_array_nbytes > 1_000
    assert accounting.per_observation_state_growth_bytes == 0
    assert accounting.evaluator_num_steps_terminal_cap == 100_000
    assert accounting.controller_signed_int32_counter_terminal_cap == 2**31 - 1
    assert accounting.evaluator_uint32_seed_terminal_cap == 2**32 - 1


def test_raw_support_calibration_reachability_and_mode_isolation_are_visible(
    completed: tuple[
        CalibratedExtendedSearchDevelopmentEvaluator,
        CalibratedSearchDevelopmentSuite,
    ],
) -> None:
    evaluator, suite = completed
    for record in suite.arm_records:
        trace = record.trace
        assert bool(jnp.all(trace.arm_transaction_valid))
        assert bool(jnp.all(trace.observe_transaction_valid))
        assert bool(jnp.all(trace.value_calibration_ready_count == 6))
        assert bool(jnp.all(trace.reachability_ready_count == 6))
        assert bool(jnp.all(trace.support_ready_count == 6))
        assert record.summary.final_support_counts == (3, 3, 3, 3, 3, 3)
        assert record.summary.final_value_change_counts == (3, 3, 3, 3, 3, 3)
        assert record.summary.final_anchor_revisit_trials == (8, 8)
        assert len(record.summary.final_anchor_revisit_successes) == 2
        assert record.summary.eligible_candidate_opportunities > 0
        assert np.isfinite(record.summary.mean_priority)
        assert np.isfinite(record.summary.final_q_l1)
        assert record.final_controller_state.state_revision == evaluator.config.num_steps
        assert record.final_controller_state.source_digest.tolist() == (
            suite.model_snapshot.source_digest.tolist()
        )
        assert np.array_equal(
            np.asarray(record.final_controller_state.anchor_bank),
            np.asarray(suite.model_snapshot.anchor_bank),
        )
    # Static modes expose their intended model-error calibration family only.
    assert suite.arm_records[1].summary.final_model_error_counts[-2:] == (2, 2)
    assert suite.arm_records[2].summary.final_model_error_counts[:4] == (2, 2, 2, 2)
    assert suite.arm_records[0].summary.final_model_error_counts == (3, 3, 3, 3, 3, 3)
    assert suite.arm_records[3].summary.final_model_error_counts == (3, 3, 3, 3, 3, 3)
    assert suite.contrasts[0].final_q_l1_distance > 0.0
    assert suite.contrasts[1].final_q_l1_distance > 0.0


@pytest.mark.parametrize("tamper", ("status", "trace", "budget", "manifest", "threshold"))
def test_structural_validator_rejects_tamper(
    completed: tuple[
        CalibratedExtendedSearchDevelopmentEvaluator,
        CalibratedSearchDevelopmentSuite,
    ],
    tamper: str,
) -> None:
    _, suite = completed
    if tamper == "status":
        changed = dataclasses.replace(suite, assessment_status="accepted")
    elif tamper == "trace":
        trace = suite.evaluator_trace.replace(  # type: ignore[attr-defined]
            external_returns=suite.evaluator_trace.external_returns.at[0].add(1.0)
        )
        changed = dataclasses.replace(suite, evaluator_trace=trace)
    elif tamper == "budget":
        record = suite.arm_records[0]
        accounting = dataclasses.replace(record.accounting, shared_backup_budget=999)
        records = (dataclasses.replace(record, accounting=accounting), *suite.arm_records[1:])
        changed = dataclasses.replace(suite, arm_records=records)
    elif tamper == "manifest":
        manifest = dataclasses.replace(
            suite.source_runtime_manifest, manifest_sha256="0" * 64
        )
        changed = dataclasses.replace(suite, source_runtime_manifest=manifest)
    else:
        changed = dataclasses.replace(suite, thresholds={"forbidden": 1})  # type: ignore[arg-type]
    errors = validate_calibrated_search_development_suite(changed)
    assert errors


def test_partial_state_has_zero_tail_and_refuses_finalization() -> None:
    evaluator = CalibratedExtendedSearchDevelopmentEvaluator(
        CalibratedExtendedSearchDevelopmentConfig(seed=77)
    )
    partial = evaluator.advance(evaluator.init(), steps=2)
    assert evaluator.validate_state(partial) == ()
    assert bool(jnp.all(partial.backup_attempts[:, :2] == 2))
    assert not bool(jnp.any(partial.backup_attempts[:, 2:]))
    assert bool(jnp.all(partial.resolved_candidate_index[:, 2:] == -1))
    with pytest.raises(RuntimeError, match="before all fixed real transitions"):
        evaluator.finalize(partial)
