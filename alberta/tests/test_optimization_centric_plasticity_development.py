"""Development contracts for optimization-centric plasticity diagnostics."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alberta_framework.evaluation.optimization_centric_plasticity_development import (
    ASSESSMENT_STATUS,
    DEVELOPMENT_STATUS,
    L2_CONSTRAINED,
    UNCONSTRAINED,
    OptimizationCentricPlasticityDevelopmentConfig,
    diagnose_frozen_learner_switch,
    freeze_learner_snapshot,
    reconstruct_optimization_centric_plasticity_development,
    run_optimization_centric_plasticity_development,
    thaw_learner_snapshot,
    validate_optimization_centric_plasticity_development,
    verify_learner_snapshot,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def report():  # type: ignore[no-untyped-def]
    return run_optimization_centric_plasticity_development()


def test_recurring_protocol_retains_reconstructable_raw_diagnostics(report) -> None:  # type: ignore[no-untyped-def]
    config = OptimizationCentricPlasticityDevelopmentConfig()

    assert report.status == DEVELOPMENT_STATUS
    assert report.assessment_status == ASSESSMENT_STATUS == "not_assessed"
    assert report.development_only
    assert not report.output_writes_authorized
    assert not report.thresholds_authorized
    assert not report.evidence_authorized
    assert not report.promotion_authorized
    assert not report.scientific_promotion_allowed
    assert report.config == config
    assert tuple(step.evaluator_regime_id for step in report.protocol) == (
        "A",
        "A",
        "A",
        "A",
        "B",
        "B",
        "B",
        "B",
        "A",
        "A",
        "A",
        "A",
    )
    assert report.learner_visible_fields == ("observation", "target")
    assert report.evaluator_only_fields == (
        "evaluator_regime_id",
        "phase_index",
        "phase_boundary",
    )

    assert tuple(condition.name for condition in report.conditions) == (
        UNCONSTRAINED,
        L2_CONSTRAINED,
    )
    assert len({condition.training_stream_sha256 for condition in report.conditions}) == 1
    assert len({condition.initial_snapshot.sha256 for condition in report.conditions}) == 1
    assert report.matched_intervention.same_initial_snapshot
    assert report.matched_intervention.same_ordered_training_stream
    assert report.matched_intervention.equal_update_opportunities
    assert report.matched_intervention.only_declared_difference == "l2_parameter_constraint"
    assert report.matched_intervention.constrained_projection_applications > 0

    for condition in report.conditions:
        assert len(condition.phases) == 3
        assert len(condition.switch_diagnostics) == 2
        assert tuple(phase.evaluator_regime_id for phase in condition.phases) == (
            "A",
            "B",
            "A",
        )
        for phase in condition.phases:
            assert len(phase.updates) == config.updates_per_phase
            assert len(phase.parameter_change.delta_float32_bits) == config.parameter_count
            assert phase.parameter_change.changed_coordinate_count > 0
        for diagnostic in condition.switch_diagnostics:
            assert len(diagnostic.old_task_gradient.gradient_float32_bits) == (
                config.parameter_count
            )
            assert len(diagnostic.incoming_task_gradient.gradient_float32_bits) == (
                config.parameter_count
            )
            assert len(diagnostic.local_perturbations) == config.local_direction_count
            assert diagnostic.incoming_zero_gradient == (
                diagnostic.incoming_task_gradient.l2_norm
                <= config.zero_gradient_descriptive_floor
            )
            assert diagnostic.local_optimum_trapped == (
                diagnostic.incoming_zero_gradient
                and all(
                    not probe.improving_direction_found
                    for probe in diagnostic.local_perturbations
                )
            )
            assert not diagnostic.dormancy.used_as_zero_gradient_proxy
            assert not diagnostic.dormancy.used_in_local_optimum_rule
            assert len(diagnostic.dormancy.mean_absolute_activation_float32_bits) == (
                config.hidden_dim
            )

    validate_optimization_centric_plasticity_development(report)
    assert reconstruct_optimization_centric_plasticity_development(report) == report


def test_snapshots_are_bit_exact_immutable_and_tampering_fails_closed(report) -> None:  # type: ignore[no-untyped-def]
    snapshot = report.initial_snapshot
    before = snapshot.parameter_float32_bits
    parameters = np.array(thaw_learner_snapshot(snapshot), copy=True)
    parameters[0] = np.float32(parameters[0] + 100.0)

    assert snapshot.parameter_float32_bits == before
    verify_learner_snapshot(snapshot, report.config)

    tampered_words = (before[0] ^ 1, *before[1:])
    tampered = dataclasses.replace(
        snapshot,
        parameter_float32_bits=tampered_words,
    )
    with pytest.raises(ValueError, match="snapshot"):
        verify_learner_snapshot(tampered, report.config)
    with pytest.raises(ValueError, match="snapshot"):
        thaw_learner_snapshot(tampered)

    condition = report.conditions[0]
    diagnostic = condition.switch_diagnostics[0]
    proxy_dormancy = dataclasses.replace(
        diagnostic.dormancy,
        used_as_zero_gradient_proxy=True,
    )
    proxy_diagnostic = dataclasses.replace(diagnostic, dormancy=proxy_dormancy)
    proxy_condition = dataclasses.replace(
        condition,
        switch_diagnostics=(proxy_diagnostic, *condition.switch_diagnostics[1:]),
    )
    proxy_report = dataclasses.replace(
        report,
        conditions=(proxy_condition, *report.conditions[1:]),
    )
    with pytest.raises(ValueError, match="dormancy"):
        validate_optimization_centric_plasticity_development(
            proxy_report,
            reconstruct=False,
        )


def test_raw_gradient_and_parameter_metrics_recompute_exactly(report) -> None:  # type: ignore[no-untyped-def]
    config = report.config
    for condition in report.conditions:
        for diagnostic in condition.switch_diagnostics:
            old = np.asarray(
                diagnostic.old_task_gradient.to_float32(),
                dtype=np.float64,
            )
            incoming = np.asarray(
                diagnostic.incoming_task_gradient.to_float32(),
                dtype=np.float64,
            )
            old_norm = float(np.linalg.norm(old))
            incoming_norm = float(np.linalg.norm(incoming))
            dot = float(np.dot(old, incoming))
            assert diagnostic.old_task_gradient.l2_norm == old_norm
            assert diagnostic.incoming_task_gradient.l2_norm == incoming_norm
            assert diagnostic.gradient_dot_product == dot
            if old_norm == 0.0 or incoming_norm == 0.0:
                assert diagnostic.gradient_cosine is None
                assert diagnostic.gradient_alignment == "undefined"
            else:
                assert diagnostic.gradient_cosine == dot / (old_norm * incoming_norm)
                expected = (
                    "aligned"
                    if dot > config.alignment_descriptive_floor
                    else "conflicting"
                    if dot < -config.alignment_descriptive_floor
                    else "orthogonal"
                )
                assert diagnostic.gradient_alignment == expected

        for phase in condition.phases:
            before = np.asarray(thaw_learner_snapshot(phase.start_snapshot))
            after = np.asarray(thaw_learner_snapshot(phase.end_snapshot))
            delta = np.asarray(phase.parameter_change.to_float32())
            np.testing.assert_array_equal(delta, np.asarray(after - before, dtype=np.float32))
            assert phase.parameter_change.l2_displacement == float(
                np.linalg.norm(delta.astype(np.float64))
            )

    resources = report.resources
    assert resources.condition_count == 2
    assert resources.phase_count == 6
    assert resources.switch_diagnostic_count == 4
    assert resources.training_update_opportunities == 24
    assert resources.realized_training_updates == 24
    assert resources.training_gradient_evaluations == 24
    assert resources.diagnostic_gradient_evaluations == 8
    assert resources.local_perturbation_loss_evaluations == 16
    assert resources.dormant_activation_batch_evaluations == 4
    assert resources.parameter_projection_attempts == 12
    assert resources.learner_snapshot_freezes == 25
    assert resources.config_bound_snapshot_verification_calls == 117
    assert resources.artifact_bytes_written == 0
    assert resources.output_write_calls == 0


def test_zero_gradient_local_probe_rule_is_independent_of_dormancy() -> None:
    config = OptimizationCentricPlasticityDevelopmentConfig()
    zero_snapshot = freeze_learner_snapshot(
        np.zeros((config.parameter_count,), dtype=np.float32),
        revision=0,
        config=config,
    )

    first_switch = diagnose_frozen_learner_switch(
        zero_snapshot,
        switch_index=0,
        config=config,
    )
    recurring_switch = diagnose_frozen_learner_switch(
        zero_snapshot,
        switch_index=1,
        config=config,
    )

    assert first_switch.incoming_task_gradient.l2_norm == 0.0
    assert recurring_switch.incoming_task_gradient.l2_norm == 0.0
    assert first_switch.incoming_zero_gradient
    assert recurring_switch.incoming_zero_gradient
    assert first_switch.dormancy.dormant_fraction == 1.0
    assert recurring_switch.dormancy.dormant_fraction == 1.0
    assert not first_switch.local_optimum_trapped
    assert recurring_switch.local_optimum_trapped
    assert any(
        probe.improving_direction_found
        for probe in first_switch.local_perturbations
    )
    assert all(
        not probe.improving_direction_found
        for probe in recurring_switch.local_perturbations
    )
    assert not first_switch.dormancy.used_in_local_optimum_rule
    assert not recurring_switch.dormancy.used_in_local_optimum_rule


def test_identity_config_and_raw_measurement_tampering_are_rejected(report) -> None:  # type: ignore[no-untyped-def]
    bad_config = dataclasses.replace(
        report,
        config_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="config"):
        validate_optimization_centric_plasticity_development(
            bad_config,
            reconstruct=False,
        )

    bad_runtime = dataclasses.replace(
        report,
        runtime_identity_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="runtime"):
        validate_optimization_centric_plasticity_development(
            bad_runtime,
            reconstruct=False,
        )

    bad_source_item = dataclasses.replace(
        report.source_manifest[0],
        sha256="0" * 64,
    )
    bad_source = dataclasses.replace(
        report,
        source_manifest=(bad_source_item, *report.source_manifest[1:]),
    )
    with pytest.raises(ValueError, match="source"):
        validate_optimization_centric_plasticity_development(
            bad_source,
            reconstruct=False,
        )

    condition = report.conditions[0]
    diagnostic = condition.switch_diagnostics[0]
    bad_gradient = dataclasses.replace(
        diagnostic.incoming_task_gradient,
        gradient_float32_bits=(
            diagnostic.incoming_task_gradient.gradient_float32_bits[0] ^ 1,
            *diagnostic.incoming_task_gradient.gradient_float32_bits[1:],
        ),
    )
    bad_diagnostic = dataclasses.replace(
        diagnostic,
        incoming_task_gradient=bad_gradient,
    )
    bad_condition = dataclasses.replace(
        condition,
        switch_diagnostics=(bad_diagnostic, *condition.switch_diagnostics[1:]),
    )
    bad_report = dataclasses.replace(
        report,
        conditions=(bad_condition, *report.conditions[1:]),
    )
    with pytest.raises(ValueError, match="report|gradient"):
        validate_optimization_centric_plasticity_development(
            bad_report,
            reconstruct=False,
        )
