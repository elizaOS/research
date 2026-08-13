"""Focused checks for the development-only latent recurrence probe."""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.sigreg import (
    SIGRegConfig,
    sample_sigreg_directions,
    sigreg_diagnostics,
)
from alberta_framework.evaluation.latent_world_model_recurrence_development import (
    ARM_ORDER,
    ASSESSMENT_STATUS,
    DEVELOPMENT_ONLY,
    EVIDENCE_CLAIMED,
    FIXED_ENCODER,
    OUTPUT_WRITES_ALLOWED,
    RESETS_EXPOSED,
    SCIENTIFIC_PROMOTION_ALLOWED,
    TASK_IDENTIFIERS_EXPOSED,
    THRESHOLDS_FROZEN,
    TRAINABLE_COLLAPSE_GATED,
    TRAINABLE_PREDICTION_ONLY,
    LatentWorldModelRecurrenceDevelopmentReport,
    RecurringVectorDynamicsProbeConfig,
    build_recurring_vector_dynamics_source,
    run_latent_world_model_recurrence_development,
    validate_latent_world_model_recurrence_development_report,
    validate_recurring_vector_dynamics_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module")
def small_report() -> LatentWorldModelRecurrenceDevelopmentReport:
    return run_latent_world_model_recurrence_development(
        RecurringVectorDynamicsProbeConfig(
            phase_steps=8,
            summary_window=3,
            latent_dim=3,
            hidden_sizes=(),
            sigreg_projections=5,
            development_key=17,
        )
    )


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("phase_steps", 1, ValueError),
        ("summary_window", 0, ValueError),
        ("summary_window", 65, ValueError),
        ("latent_dim", True, TypeError),
        ("hidden_sizes", (4, 0), ValueError),
        ("predictor_step_size", float("nan"), ValueError),
        ("collapse_gate_threshold", 1.0, ValueError),
        ("development_key", -1, ValueError),
    ],
)
def test_probe_config_fails_closed(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        RecurringVectorDynamicsProbeConfig(**{field: value})  # type: ignore[arg-type]


def test_source_is_uninterrupted_matched_and_bit_reconstructable() -> None:
    config = RecurringVectorDynamicsProbeConfig(
        phase_steps=5,
        summary_window=2,
        latent_dim=3,
        hidden_sizes=(),
        sigreg_projections=4,
    )
    source = build_recurring_vector_dynamics_source(config)
    reconstructed = build_recurring_vector_dynamics_source(config)

    assert validate_recurring_vector_dynamics_source(source) == ()
    assert source.input_sha256 == reconstructed.input_sha256
    assert source.generator_contract_sha256 == reconstructed.generator_contract_sha256
    assert source.observations.shape == (15, 4)
    assert source.actions.shape == (15,)
    np.testing.assert_array_equal(
        np.asarray(source.observations[1:]),
        np.asarray(source.next_observations[:-1]),
    )

    nominal = np.asarray(source.next_observations)
    physical_delta = np.asarray(source.physical_violation_next_observations) - nominal
    nuisance_delta = np.asarray(source.nuisance_perturbation_next_observations) - nominal
    np.testing.assert_array_equal(physical_delta[:, 2:], np.zeros((15, 2), np.float32))
    np.testing.assert_array_equal(nuisance_delta[:, :2], np.zeros((15, 2), np.float32))
    np.testing.assert_allclose(
        np.linalg.norm(physical_delta[:, :2], axis=1),
        np.linalg.norm(nuisance_delta[:, 2:], axis=1),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    # The source keeps persistent excitation on one shared state manifold:
    # unlike a damped trajectory, it cannot make recurrence artificially easy
    # by collapsing the physical signal toward zero.  A and B rotate that
    # shared manifold in opposite directions without exposing the phase.
    physical = np.asarray(source.observations[:, :2])
    next_physical = nominal[:, :2]
    np.testing.assert_allclose(
        np.linalg.norm(physical, axis=1),
        np.ones((15,), dtype=np.float32),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    signed_rotation = (
        physical[:, 0] * next_physical[:, 1]
        - physical[:, 1] * next_physical[:, 0]
    )
    assert np.all(signed_rotation[:5] > 0.0)
    assert np.all(signed_rotation[5:10] < 0.0)
    assert np.all(signed_rotation[10:] > 0.0)

    tampered = dataclasses.replace(
        source,
        rewards=source.rewards.at[0].add(jnp.asarray(0.5, dtype=jnp.float32)),
    )
    errors = validate_recurring_vector_dynamics_source(tampered)
    assert "source input digest does not match its arrays" in errors
    assert "rewards does not reconstruct bit-exactly" in errors


def test_report_is_explicitly_nonpromoting_and_locally_valid(
    small_report: LatentWorldModelRecurrenceDevelopmentReport,
) -> None:
    report = small_report
    assert validate_latent_world_model_recurrence_development_report(report) == ()
    assert DEVELOPMENT_ONLY is True
    assert ASSESSMENT_STATUS == "not_assessed"
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert OUTPUT_WRITES_ALLOWED is False
    assert THRESHOLDS_FROZEN is False
    assert EVIDENCE_CLAIMED is False
    assert TASK_IDENTIFIERS_EXPOSED is False
    assert RESETS_EXPOSED is False
    assert report.development_only is True
    assert report.assessment_status == "not_assessed"
    assert report.descriptive_claims_only is True
    assert report.scientific_promotion_allowed is False
    assert report.output_writes_allowed is False
    assert report.evidence_claimed is False
    assert len(report.limitations) >= 4


def test_three_arms_have_exact_common_inputs_randomness_and_resources(
    small_report: LatentWorldModelRecurrenceDevelopmentReport,
) -> None:
    report = small_report
    assert tuple(arm.name for arm in report.arms) == ARM_ORDER
    assert report.fixed_persistent_resources_equal is True
    assert report.matched_nonintervention_config is True
    assert report.binding.source_reconstruction_valid is True
    assert report.binding.common_randomness_preserved is True

    first = report.arms[0]
    for arm in report.arms:
        assert arm.source_input_sha256 == report.source.input_sha256
        assert arm.initial_state_sha256 == first.initial_state_sha256
        assert arm.sigreg_directions_sha256 == first.sigreg_directions_sha256
        assert arm.nonintervention_config_sha256 == first.nonintervention_config_sha256
        assert arm.resource == first.resource
        assert arm.resource.initial_state_nbytes > 0
        assert arm.resource.initial_state_nbytes == first.resource.initial_state_nbytes
        assert arm.resource.final_state_nbytes == first.resource.final_state_nbytes
        assert arm.resource.initial_array_elements == first.resource.initial_array_elements
        assert arm.resource.final_array_elements == first.resource.final_array_elements

    # All interventions occur after the first prediction, so its prequential
    # errors must be exactly common across arms.
    for field in ("latent_errors", "reward_errors", "discount_errors"):
        expected = getattr(first.trajectory, field)[0]
        for arm in report.arms[1:]:
            assert bool(getattr(arm.trajectory, field)[0] == expected)


def test_encoder_interventions_and_raw_prequential_metrics_are_recorded(
    small_report: LatentWorldModelRecurrenceDevelopmentReport,
) -> None:
    arms = {arm.name: arm for arm in small_report.arms}
    fixed = arms[FIXED_ENCODER]
    prediction_only = arms[TRAINABLE_PREDICTION_ONLY]
    gated = arms[TRAINABLE_COLLAPSE_GATED]

    assert fixed.encoder_learning is False
    assert fixed.encoder_update_rate == 0.0
    assert fixed.encoder_gate_rate == 0.0
    assert prediction_only.encoder_learning is True
    assert prediction_only.collapse_gate_threshold == 1.0
    assert prediction_only.encoder_update_rate > 0.0
    assert prediction_only.encoder_gate_rate == 0.0
    assert gated.encoder_learning is True
    assert gated.collapse_gate_threshold < 1.0
    assert gated.encoder_gate_rate > 0.0
    assert gated.encoder_update_rate < prediction_only.encoder_update_rate

    total = small_report.config.total_steps
    for arm in small_report.arms:
        trajectory = arm.trajectory
        assert trajectory.latent_errors.shape == (total,)
        assert trajectory.reward_errors.shape == (total,)
        assert trajectory.discount_errors.shape == (total,)
        assert trajectory.target_next_embeddings.shape == (
            total,
            small_report.config.latent_dim,
        )
        np.testing.assert_allclose(
            np.asarray(trajectory.joint_errors),
            np.asarray(
                trajectory.latent_errors + trajectory.reward_errors + trajectory.discount_errors
            ),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        assert bool(jnp.all(trajectory.world_updates))
        assert arm.world_update_rate == 1.0


def test_recurrence_sigreg_and_surprise_are_descriptive_recomputations(
    small_report: LatentWorldModelRecurrenceDevelopmentReport,
) -> None:
    report = small_report
    _, direction_key = jr.split(jr.key(report.config.development_key))
    sigreg_config = SIGRegConfig(n_projections=report.config.sigreg_projections)
    directions = sample_sigreg_directions(
        direction_key,
        report.config.latent_dim,
        sigreg_config,
    )

    for arm in report.arms:
        recurrence = arm.recurrence
        assert recurrence.entry_forgetting.joint == pytest.approx(
            recurrence.recurrence_entry.joint - recurrence.initial_a_reference.joint
        )
        assert recurrence.within_recurrence_recovery.joint == pytest.approx(
            recurrence.recurrence_entry.joint - recurrence.recurrence_late.joint
        )
        assert recurrence.residual_forgetting.joint == pytest.approx(
            recurrence.recurrence_late.joint - recurrence.initial_a_reference.joint
        )

        for phase in arm.phase_summaries:
            diagnostics = sigreg_diagnostics(
                arm.trajectory.target_next_embeddings[phase.start : phase.stop],
                directions,
                sigreg_config,
            )
            assert phase.sigreg.loss == pytest.approx(float(diagnostics.loss))
            assert phase.sigreg.latent_std_min == pytest.approx(float(diagnostics.latent_std_min))
            assert all(math.isfinite(value) for value in dataclasses.astuple(phase.sigreg))

        separation = arm.overall_surprise
        assert separation.physical_excess == pytest.approx(
            separation.physical_violation - separation.nominal
        )
        assert separation.nuisance_excess == pytest.approx(
            separation.nuisance_perturbation - separation.nominal
        )
        assert separation.physical_minus_nuisance == pytest.approx(
            separation.physical_violation - separation.nuisance_perturbation
        )
        # Separation is deliberately recorded with either sign; there is no
        # hidden pass threshold in this development evaluator.
        assert math.isfinite(separation.physical_minus_nuisance)


def test_report_validator_rejects_promotion_or_resource_relabeling(
    small_report: LatentWorldModelRecurrenceDevelopmentReport,
) -> None:
    promoted = dataclasses.replace(
        small_report,
        assessment_status="accepted",
        scientific_promotion_allowed=True,
    )
    assert "development-only nonpromotion contract changed" in (
        validate_latent_world_model_recurrence_development_report(promoted)
    )

    mislabeled = dataclasses.replace(
        small_report,
        fixed_persistent_resources_equal=False,
    )
    assert "persistent resource equality is false or inconsistent" in (
        validate_latent_world_model_recurrence_development_report(mislabeled)
    )

    status_relabel = dataclasses.replace(small_report, status="accepted")
    assert "development-only nonpromotion contract changed" in (
        validate_latent_world_model_recurrence_development_report(status_relabel)
    )


def test_report_validator_reconstructs_common_randomness_not_only_equality(
    small_report: LatentWorldModelRecurrenceDevelopmentReport,
) -> None:
    forged_digest = "0" * 64
    forged = dataclasses.replace(
        small_report,
        binding=dataclasses.replace(
            small_report.binding,
            initial_state_sha256=forged_digest,
        ),
        arms=tuple(
            dataclasses.replace(arm, initial_state_sha256=forged_digest)
            for arm in small_report.arms
        ),
    )
    errors = validate_latent_world_model_recurrence_development_report(forged)
    assert "initial-state binding does not reconstruct" in errors
    assert any("initial state does not reconstruct" in error for error in errors)


def test_report_validator_recomputes_update_rates(
    small_report: LatentWorldModelRecurrenceDevelopmentReport,
) -> None:
    first = small_report.arms[0]
    forged_trajectory = dataclasses.replace(
        first.trajectory,
        world_updates=first.trajectory.world_updates.at[0].set(False),
    )
    forged = dataclasses.replace(
        small_report,
        arms=(dataclasses.replace(first, trajectory=forged_trajectory), *small_report.arms[1:]),
    )
    assert "fixed_encoder world update rate does not reconstruct" in (
        validate_latent_world_model_recurrence_development_report(forged)
    )
