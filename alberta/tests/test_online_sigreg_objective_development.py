"""Mechanism checks for the development-only online SIGReg objective probe."""

from __future__ import annotations

import dataclasses
import inspect
import math

import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework.evaluation.online_sigreg_objective_development as lane
from alberta_framework.evaluation.online_sigreg_objective_development import (
    ARM_ORDER,
    ASSESSMENT_STATUS,
    DEVELOPMENT_KEYS_FROZEN,
    DEVELOPMENT_ONLY,
    EVIDENCE_CLAIMED,
    EVIDENCE_LEVEL,
    OUTPUT_WRITES_ALLOWED,
    PHASE_IDENTIFIERS_EXPOSED,
    PREDICTION_ONLY,
    SCIENTIFIC_PROMOTION_ALLOWED,
    SIGREG_INERT,
    SIGREG_ROUTED,
    THRESHOLDS_FROZEN,
    TRANSITION_REPLAYED,
    OnlineSIGRegDevelopmentConfig,
    OnlineSIGRegDevelopmentReport,
    build_online_sigreg_source,
    run_online_sigreg_objective_development,
    validate_online_sigreg_development_report,
    validate_online_sigreg_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module")
def small_config() -> OnlineSIGRegDevelopmentConfig:
    return OnlineSIGRegDevelopmentConfig(
        phase_steps=10,
        summary_window=2,
        context_size=4,
        latent_dim=2,
        sigreg_projections=3,
        development_key=19,
    )


@pytest.fixture(scope="module")
def small_report(
    small_config: OnlineSIGRegDevelopmentConfig,
) -> OnlineSIGRegDevelopmentReport:
    return run_online_sigreg_objective_development(small_config)


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("phase_steps", True, TypeError),
        ("phase_steps", 3, ValueError),
        ("summary_window", 0, ValueError),
        ("context_size", 2, ValueError),
        ("context_size", 25, ValueError),
        ("latent_dim", 0, ValueError),
        ("sigreg_projections", 0, ValueError),
        ("prediction_step_size", True, TypeError),
        ("prediction_step_size", 1, TypeError),
        ("prediction_step_size", np.float32(0.1), TypeError),
        ("prediction_step_size", 1.0e300, ValueError),
        ("prediction_step_size", 1.0e-300, ValueError),
        ("sigreg_step_size", float("nan"), ValueError),
        ("probe_ridge", float(np.finfo(np.float32).eps), ValueError),
        ("probe_ridge", 0.0, ValueError),
        ("min_latent_std", 0, TypeError),
        ("min_latent_std", -0.1, ValueError),
        ("development_key", -1, ValueError),
    ],
)
def test_config_fails_closed(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        OnlineSIGRegDevelopmentConfig(**{field: value})  # type: ignore[arg-type]


def test_config_requires_causal_probe_before_late_a_window() -> None:
    with pytest.raises(
        ValueError,
        match="late initial-A window must be disjoint from the probe-fit prefix",
    ):
        OnlineSIGRegDevelopmentConfig(
            phase_steps=8,
            summary_window=4,
            context_size=6,
        )

    # The fit uses observations 0..context_size-1.  A window beginning at
    # context_size-1 would reuse the final fit sample as an evaluation sample.
    with pytest.raises(
        ValueError,
        match="late initial-A window must be disjoint from the probe-fit prefix",
    ):
        OnlineSIGRegDevelopmentConfig(
            phase_steps=8,
            summary_window=4,
            context_size=5,
        )


def test_source_is_continuous_reconstructable_and_has_no_phase_array(
    small_config: OnlineSIGRegDevelopmentConfig,
) -> None:
    source = build_online_sigreg_source(small_config)
    reconstructed = build_online_sigreg_source(small_config)

    assert validate_online_sigreg_source(source) == ()
    assert source.input_sha256 == reconstructed.input_sha256
    assert source.generator_contract_sha256 == reconstructed.generator_contract_sha256
    assert source.observations.shape == (small_config.total_steps, 4)
    assert source.actions.shape == (small_config.total_steps,)
    assert source.next_observations.shape == (small_config.total_steps, 4)
    assert not hasattr(source, "phase_ids")
    assert not hasattr(source, "regimes")
    np.testing.assert_array_equal(
        np.asarray(source.observations[1:]),
        np.asarray(source.next_observations[:-1]),
    )

    physical = np.asarray(source.observations[:, :2])
    next_physical = np.asarray(source.next_observations[:, :2])
    np.testing.assert_allclose(
        np.linalg.norm(physical, axis=1),
        np.ones((small_config.total_steps,), dtype=np.float32),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    signed_rotation = (
        physical[:, 0] * next_physical[:, 1]
        - physical[:, 1] * next_physical[:, 0]
    )
    steps = small_config.phase_steps
    assert np.all(signed_rotation[:steps] > 0.0)
    assert np.all(signed_rotation[steps : 2 * steps] < 0.0)
    assert np.all(signed_rotation[2 * steps :] > 0.0)

    tampered = dataclasses.replace(
        source,
        next_observations=source.next_observations.at[0, 0].add(
            jnp.asarray(0.25, dtype=jnp.float32)
        ),
    )
    errors = validate_online_sigreg_source(tampered)
    assert "source digest does not match its arrays" in errors
    assert "next_observations does not reconstruct bit-exactly" in errors


def test_source_validator_fails_closed_before_malformed_values_are_dereferenced(
    small_config: OnlineSIGRegDevelopmentConfig,
) -> None:
    source = build_online_sigreg_source(small_config)
    assert validate_online_sigreg_source(object()) == (
        "source must be an exact OnlineSIGRegSource",
    )

    wrong_config = dataclasses.replace(source, config=object())  # type: ignore[arg-type]
    assert validate_online_sigreg_source(wrong_config) == (
        "source config must be an exact OnlineSIGRegDevelopmentConfig",
    )

    wrong_array = dataclasses.replace(
        source,
        observations=np.asarray(source.observations),  # type: ignore[arg-type]
    )
    assert "observations must be a JAX Array" in validate_online_sigreg_source(
        wrong_array
    )

    wrong_digest = dataclasses.replace(
        source,
        input_sha256=7,  # type: ignore[arg-type]
    )
    assert "source input digest must be lowercase sha256" in (
        validate_online_sigreg_source(wrong_digest)
    )


def test_report_is_explicitly_l0_nonpromoting_and_valid(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    assert validate_online_sigreg_development_report(small_report) == ()
    assert EVIDENCE_LEVEL == "L0"
    assert DEVELOPMENT_ONLY is True
    assert ASSESSMENT_STATUS == "not_assessed"
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert OUTPUT_WRITES_ALLOWED is False
    assert EVIDENCE_CLAIMED is False
    assert THRESHOLDS_FROZEN is False
    assert DEVELOPMENT_KEYS_FROZEN is False
    assert PHASE_IDENTIFIERS_EXPOSED is False
    assert TRANSITION_REPLAYED is False
    assert small_report.status == "development_only_descriptive_not_assessed"
    assert small_report.descriptive_claims_only is True
    assert small_report.scientific_promotion_allowed is False
    assert small_report.output_writes_allowed is False
    assert len(small_report.limitations) >= 6


def test_prediction_api_cannot_accept_an_outcome_or_phase_identifier() -> None:
    parameters = tuple(inspect.signature(lane._predict_next_latent).parameters)
    assert parameters == ("state", "observation", "action")
    step_parameters = tuple(inspect.signature(lane._online_step).parameters)
    assert step_parameters == (
        "state",
        "inputs",
        "learner_config",
        "objective_enabled",
        "route_sigreg_gradient",
    )
    assert "phase" not in " ".join(step_parameters)
    assert "future" not in " ".join(step_parameters)
    learner_config = lane._learner_config(OnlineSIGRegDevelopmentConfig())
    assert not hasattr(learner_config, "phase_steps")
    assert not hasattr(learner_config, "summary_window")


def test_timing_contract_assigns_pre_and_post_outcome_ownership(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    timing = small_report.timing
    assert timing.pre_outcome_prediction_inputs == (
        "observation_t",
        "action_t",
        "pre_update_state_t",
    )
    assert "next_observation_t" in timing.outcome_revealed_after_prediction
    assert "post-update state" in timing.regularizer_update_effective
    assert "event t+1" in timing.regularizer_update_effective
    assert "before the first routed SIGReg update" in timing.frozen_probe_fit
    assert "never reach the learner" in timing.frozen_probe_measurement
    assert "after the life" in timing.evaluator_only_segmentation
    assert any("in-sample intervention sanity" in item for item in small_report.limitations)
    assert any("no action selection" in item for item in small_report.limitations)


def test_arms_share_initial_state_persistent_resources_and_first_prediction(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    assert tuple(arm.name for arm in small_report.arms) == ARM_ORDER
    assert small_report.common_initial_state is True
    assert small_report.persistent_resources_equal is True
    first = small_report.arms[0]
    for arm in small_report.arms:
        assert arm.source_input_sha256 == small_report.source.input_sha256
        assert arm.initial_state_sha256 == first.initial_state_sha256
        assert arm.resource == first.resource
        assert arm.resource.initial_total_nbytes == arm.resource.final_total_nbytes
        assert arm.resource.exact_component_sum_matches_total is True
        assert bool(
            arm.trajectory.latent_prediction_mse[0]
            == first.trajectory.latent_prediction_mse[0]
        )


def test_inert_control_computes_then_discards_without_changing_the_learner(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    arms = {arm.name: arm for arm in small_report.arms}
    prediction = arms[PREDICTION_ONLY]
    inert = arms[SIGREG_INERT]
    routed = arms[SIGREG_ROUTED]
    eligible = small_report.config.eligible_sigreg_steps

    assert small_report.routed_and_inert_objective_compute_matched is True
    assert small_report.inert_and_prediction_only_learners_identical is True
    assert prediction.sigreg_objective_enabled is False
    assert prediction.work.sigreg_objective_gradient_evaluations == 0
    assert prediction.work.sigreg_gradients_discarded == 0
    assert prediction.work.sigreg_gradients_rejected_nonfinite == 0
    assert inert.sigreg_objective_enabled is True
    assert inert.sigreg_gradient_routed_by_design is False
    assert inert.work.sigreg_objective_gradient_evaluations == eligible
    assert inert.work.sigreg_gradients_discarded == eligible
    assert inert.work.sigreg_gradients_routed == 0
    assert inert.work.sigreg_gradients_rejected_nonfinite == 0
    assert routed.sigreg_objective_enabled is True
    assert routed.sigreg_gradient_routed_by_design is True
    assert routed.work.sigreg_objective_gradient_evaluations == eligible
    assert routed.work.sigreg_gradients_routed == eligible
    assert routed.work.sigreg_gradients_discarded == 0
    assert routed.work.sigreg_gradients_rejected_nonfinite == 0
    assert inert.final_learner_sha256 == prediction.final_learner_sha256
    assert routed.final_learner_sha256 != inert.final_learner_sha256

    for field in (
        "latent_prediction_mse",
        "physical_probe_mse",
        "nuisance_probe_mse",
        "post_update_sigreg_loss",
        "post_update_latent_std_min",
        "post_update_collapsed_fraction",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(prediction.trajectory, field)),
            np.asarray(getattr(inert.trajectory, field)),
        )

    np.testing.assert_array_equal(
        np.asarray(prediction.trajectory.sigreg_objective_loss),
        np.zeros((small_report.config.total_steps,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(prediction.trajectory.sigreg_gradient_norm),
        np.zeros((small_report.config.total_steps,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(prediction.trajectory.sigreg_candidate_update_norm),
        np.zeros((small_report.config.total_steps,), dtype=np.float32),
    )
    inert_evaluated = np.asarray(inert.trajectory.sigreg_gradient_evaluated)
    routed_evaluated = np.asarray(routed.trajectory.sigreg_gradient_evaluated)
    assert np.array_equal(inert_evaluated, routed_evaluated)
    assert np.any(np.asarray(inert.trajectory.sigreg_gradient_norm)[inert_evaluated] > 0.0)
    assert np.any(
        np.asarray(inert.trajectory.sigreg_candidate_update_norm)[inert_evaluated] > 0.0
    )
    assert not np.any(
        np.asarray(inert.trajectory.sigreg_gradient_rejected_nonfinite)
    )
    assert not np.any(
        np.asarray(routed.trajectory.sigreg_gradient_rejected_nonfinite)
    )


def test_overflowing_diagnostic_norm_is_zeroed_and_marked_nonfinite() -> None:
    norm, finite = lane._finite_l2_norm(
        jnp.asarray((2.0e38, 2.0e38), dtype=jnp.float32)
    )
    assert not bool(finite)
    assert float(norm) == 0.0


def test_sigreg_first_affects_only_the_next_event_and_probe_is_prefix_frozen(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    arms = {arm.name: arm for arm in small_report.arms}
    inert = arms[SIGREG_INERT]
    routed = arms[SIGREG_ROUTED]
    first_route = small_report.config.first_sigreg_transition

    np.testing.assert_array_equal(
        np.asarray(routed.trajectory.latent_prediction_mse[: first_route + 1]),
        np.asarray(inert.trajectory.latent_prediction_mse[: first_route + 1]),
    )
    assert bool(routed.trajectory.sigreg_gradient_routed[first_route])
    assert not bool(routed.trajectory.sigreg_gradient_routed[first_route - 1])
    assert not bool(routed.trajectory.probe_available[first_route])
    assert bool(routed.trajectory.probe_available[first_route + 1])
    assert bool(
        routed.trajectory.sigreg_objective_loss[first_route]
        == inert.trajectory.sigreg_objective_loss[first_route]
    )
    assert bool(
        routed.trajectory.sigreg_gradient_norm[first_route]
        == inert.trajectory.sigreg_gradient_norm[first_route]
    )
    assert bool(
        routed.trajectory.sigreg_candidate_update_norm[first_route]
        == inert.trajectory.sigreg_candidate_update_norm[first_route]
    )
    assert bool(
        routed.trajectory.post_update_sigreg_loss[first_route]
        != inert.trajectory.post_update_sigreg_loss[first_route]
    )

    # The readout is fit before the first routed regularizer update, so all
    # arms freeze identical diagnostic weights even though one encoder then
    # follows a different trajectory.
    assert len({arm.frozen_probe_sha256 for arm in small_report.arms}) == 1


def test_raw_feature_collapse_and_recurrence_metrics_are_descriptive(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    for arm in small_report.arms:
        trajectory = arm.trajectory
        total = small_report.config.total_steps
        for field in (
            "latent_prediction_mse",
            "physical_probe_mse",
            "nuisance_probe_mse",
            "sigreg_objective_loss",
            "sigreg_gradient_norm",
            "sigreg_candidate_update_norm",
            "post_update_sigreg_loss",
            "post_update_latent_std_min",
            "post_update_collapsed_fraction",
        ):
            values = np.asarray(getattr(trajectory, field))
            assert values.shape == (total,)
            assert np.all(np.isfinite(values))
        assert np.all(np.asarray(trajectory.post_update_collapsed_fraction) >= 0.0)
        assert np.all(np.asarray(trajectory.post_update_collapsed_fraction) <= 1.0)
        assert not np.any(
            np.asarray(trajectory.sigreg_gradient_rejected_nonfinite)
        )

        for recurrence in dataclasses.astuple(arm.recurrence):
            initial, entry, late, entry_initial, entry_late, late_initial = recurrence
            assert entry_initial == pytest.approx(entry - initial)
            assert entry_late == pytest.approx(entry - late)
            assert late_initial == pytest.approx(late - initial)
            assert all(math.isfinite(value) for value in recurrence)
        assert tuple(phase.name for phase in arm.phase_metrics) == (
            "A_initial",
            "B_interference",
            "A_recurrence",
        )
        assert all(phase.probe_measurements > 0 for phase in arm.phase_metrics)
        assert all(phase.sigreg_measurements > 0 for phase in arm.phase_metrics)


def test_clocks_and_single_pass_work_are_exact(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    total = small_report.config.total_steps
    for arm in small_report.arms:
        np.testing.assert_array_equal(
            np.asarray(arm.trajectory.pre_step_count),
            np.arange(total, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(arm.trajectory.post_step_count),
            np.arange(1, total + 1, dtype=np.int32),
        )
        assert arm.work.transitions_consumed_once == total
        assert arm.work.transition_replays == 0
        assert arm.work.prediction_gradient_evaluations == total
        assert arm.work.frozen_probe_fits == 1
        assert arm.work.allocator_peak_assessed is False
        assert arm.work.wall_clock_matched is False


def test_persistent_and_named_transient_bytes_are_exact(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    config = small_report.config
    resource = small_report.arms[0].resource
    expected_learner_float_elements = (
        4 * config.latent_dim
        + config.latent_dim
        + (config.latent_dim + 2) * config.latent_dim
        + config.latent_dim
    )
    assert resource.learner_parameter_nbytes == 4 * expected_learner_float_elements
    assert resource.observation_context_nbytes == 4 * config.context_size * 4
    assert resource.sigreg_direction_nbytes == (
        4 * config.sigreg_projections * config.latent_dim
    )
    assert resource.frozen_probe_nbytes == 4 * (config.latent_dim + 1) * 4
    assert resource.scalar_control_nbytes == 13  # three int32 scalars plus one bool
    assert resource.initial_total_nbytes == (
        resource.learner_parameter_nbytes
        + resource.observation_context_nbytes
        + resource.sigreg_direction_nbytes
        + resource.frozen_probe_nbytes
        + resource.scalar_control_nbytes
    )
    expected_trajectory_bytes = sum(
        np.asarray(getattr(small_report.arms[0].trajectory, field.name)).nbytes
        for field in dataclasses.fields(type(small_report.arms[0].trajectory))
    )
    expected_source_bytes = sum(
        np.asarray(value).nbytes
        for value in (
            small_report.source.observations,
            small_report.source.actions,
            small_report.source.next_observations,
        )
    )
    assert resource.trajectory_output_nbytes == expected_trajectory_bytes
    assert resource.shared_source_input_nbytes == expected_source_bytes
    assert resource.retained_report_array_nbytes_including_shared_source == (
        expected_trajectory_bytes + expected_source_bytes
    )
    assert "excludes Python objects" in resource.accounting_scope

    arms = {arm.name: arm for arm in small_report.arms}
    for arm in small_report.arms:
        tensors = arm.work.logical_transient_tensors
        assert arm.work.logical_transient_payload_nbytes == sum(
            tensor.total_named_payload_nbytes for tensor in tensors
        )
        for tensor in tensors:
            assert tensor.dtype == "float32"
            assert tensor.nbytes_per_evaluation == 4 * math.prod(tensor.shape)
            assert tensor.total_named_payload_nbytes == (
                tensor.nbytes_per_evaluation * tensor.evaluations
            )
    inert_manifest = tuple(
        (tensor.name, tensor.shape, tensor.evaluations)
        for tensor in arms[SIGREG_INERT].work.logical_transient_tensors
    )
    routed_manifest = tuple(
        (tensor.name, tensor.shape, tensor.evaluations)
        for tensor in arms[SIGREG_ROUTED].work.logical_transient_tensors
    )
    assert routed_manifest == inert_manifest
    assert any(
        tensor.name == "sigreg_encoder_candidate"
        for tensor in arms[SIGREG_INERT].work.logical_transient_tensors
    )
    assert not any(
        tensor.name == "sigreg_encoder_gradient"
        for tensor in arms[PREDICTION_ONLY].work.logical_transient_tensors
    )


def test_validator_rejects_promotion_work_and_clock_relabeling(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    promoted = dataclasses.replace(
        small_report,
        status="accepted",
        assessment_status="accepted",
        scientific_promotion_allowed=True,
    )
    assert "L0 development nonauthority contract changed" in (
        validate_online_sigreg_development_report(promoted)
    )

    inert = small_report.arms[1]
    forged_work = dataclasses.replace(
        inert.work,
        sigreg_gradients_discarded=inert.work.sigreg_gradients_discarded - 1,
    )
    forged = dataclasses.replace(
        small_report,
        arms=(
            small_report.arms[0],
            dataclasses.replace(inert, work=forged_work),
            *small_report.arms[2:],
        ),
    )
    assert "sigreg_inert work accounting does not reconstruct" in (
        validate_online_sigreg_development_report(forged)
    )

    first = small_report.arms[0]
    forged_clock = dataclasses.replace(
        first.trajectory,
        post_step_count=first.trajectory.post_step_count.at[0].set(
            jnp.asarray(7, dtype=jnp.int32)
        ),
    )
    forged = dataclasses.replace(
        small_report,
        arms=(dataclasses.replace(first, trajectory=forged_clock), *small_report.arms[1:]),
    )
    assert "prediction_only raw trajectory does not reconstruct" in (
        validate_online_sigreg_development_report(forged)
    )


def test_validator_guards_report_types_and_canonical_config_binding(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    assert validate_online_sigreg_development_report(object()) == (
        "report must be an exact OnlineSIGRegDevelopmentReport",
    )
    malformed_config = dataclasses.replace(
        small_report,
        config=object(),  # type: ignore[arg-type]
    )
    assert validate_online_sigreg_development_report(malformed_config) == (
        "report config must be an exact OnlineSIGRegDevelopmentConfig",
    )

    different_config = dataclasses.replace(
        small_report.config,
        development_key=small_report.config.development_key + 1,
    )
    changed_source = dataclasses.replace(
        small_report.source,
        config=different_config,
    )
    forged = dataclasses.replace(small_report, source=changed_source)
    errors = validate_online_sigreg_development_report(forged)
    assert "report config is not canonically bound to source config" in errors
    assert "report source does not exactly reconstruct" in errors


def test_validator_binds_timing_limitations_hashes_resources_and_all_metrics(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    first = small_report.arms[0]
    tampered_trajectory = dataclasses.replace(
        first.trajectory,
        latent_prediction_mse=first.trajectory.latent_prediction_mse.at[0].add(
            jnp.asarray(0.001, dtype=jnp.float32)
        ),
    )
    first_phase = dataclasses.replace(first.phase_metrics[0], start=1)
    tampered_phases = (first_phase, *first.phase_metrics[1:])
    recurrence = first.recurrence
    tampered_recurrence = dataclasses.replace(
        recurrence,
        physical_probe_mse=dataclasses.replace(
            recurrence.physical_probe_mse,
            initial_a_late=recurrence.physical_probe_mse.initial_a_late + 0.001,
        ),
    )
    tampered_resource = dataclasses.replace(
        first.resource,
        initial_total_nbytes=first.resource.initial_total_nbytes + 4,
    )
    tampered_arm = dataclasses.replace(
        first,
        final_learner_sha256="0" * 64,
        resource=tampered_resource,
        trajectory=tampered_trajectory,
        phase_metrics=tampered_phases,
        recurrence=tampered_recurrence,
    )
    tampered_timing = dataclasses.replace(
        small_report.timing,
        evaluator_only_segmentation="before the life",
    )
    forged = dataclasses.replace(
        small_report,
        timing=tampered_timing,
        limitations=(*small_report.limitations[:-1], "forged limitation"),
        arms=(tampered_arm, *small_report.arms[1:]),
        inert_and_prediction_only_learners_identical=False,
    )
    errors = validate_online_sigreg_development_report(forged)
    assert "timing ownership does not exactly reconstruct" in errors
    assert "limitations do not exactly reconstruct" in errors
    assert "prediction_only identity or hash binding changed" in errors
    assert "prediction_only resource accounting does not reconstruct" in errors
    assert "prediction_only raw trajectory does not reconstruct" in errors
    assert "prediction_only phase metrics do not reconstruct" in errors
    assert "prediction_only recurrence metrics do not reconstruct" in errors
    assert "inert_and_prediction_only_learners_identical does not reconstruct" in errors
    assert "inert and prediction-only identity does not hold" in errors


def test_validator_uses_exact_array_bytes_not_numeric_equality(
    small_report: OnlineSIGRegDevelopmentReport,
) -> None:
    first = small_report.arms[0]
    # Both values compare numerically equal, but the sign bit is part of the
    # deterministic report contract.
    assert float(first.trajectory.sigreg_objective_loss[0]) == 0.0
    signed_zero = first.trajectory.sigreg_objective_loss.at[0].set(
        jnp.asarray(-0.0, dtype=jnp.float32)
    )
    assert np.array_equal(
        np.asarray(first.trajectory.sigreg_objective_loss),
        np.asarray(signed_zero),
    )
    forged_trajectory = dataclasses.replace(
        first.trajectory,
        sigreg_objective_loss=signed_zero,
    )
    forged = dataclasses.replace(
        small_report,
        arms=(dataclasses.replace(first, trajectory=forged_trajectory), *small_report.arms[1:]),
    )
    assert "prediction_only raw trajectory does not reconstruct" in (
        validate_online_sigreg_development_report(forged)
    )
