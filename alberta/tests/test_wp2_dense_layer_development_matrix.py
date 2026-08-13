"""Contracts for the frozen, nonwriting WP2 dense-layer development matrix."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import alberta_framework
import alberta_framework.evaluation as evaluation_api
from alberta_framework.evaluation.wp2_dense_layer_development_matrix import (
    ADAM_ARM,
    ADAMO_ARM,
    CPR_ARM,
    SGD_ARM,
    SPECTRAL_ARM,
    WP2_DENSE_LAYER_ASSESSMENT_STATUS,
    WP2_DENSE_LAYER_RESOURCE_COMPARABILITY,
    WP2DenseLayerDevelopmentConfig,
    WP2DenseLayerDevelopmentReport,
    reconstruct_wp2_dense_layer_development_matrix,
    restore_wp2_dense_layer_checkpoint_parameters,
    run_wp2_dense_layer_development_matrix,
    validate_wp2_dense_layer_development_matrix,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def report() -> WP2DenseLayerDevelopmentReport:
    return run_wp2_dense_layer_development_matrix()


def test_protocol_and_authority_are_frozen_before_execution(
    report: WP2DenseLayerDevelopmentReport,
) -> None:
    config = WP2DenseLayerDevelopmentConfig()
    assert report.config == config
    assert report.assessment_status == WP2_DENSE_LAYER_ASSESSMENT_STATUS == "not_assessed"
    assert report.resource_comparability == WP2_DENSE_LAYER_RESOURCE_COMPARABILITY
    assert report.resource_comparability == "not_assessed"
    assert report.development_only
    assert not report.output_writes_authorized
    assert not report.tuning_authorized
    assert not report.winner_selection_authorized
    assert not report.default_selection_authorized
    assert not report.efficacy_claim_authorized
    assert not report.evidence_authorized
    assert not report.promotion_authorized
    assert not report.scientific_promotion_allowed
    assert tuple(arm.name for arm in report.arms) == (
        SGD_ARM,
        ADAM_ARM,
        SPECTRAL_ARM,
        ADAMO_ARM,
        CPR_ARM,
    )
    assert config.phase_order == ("A", "B", "A")
    assert config.input_dim == 2
    assert config.hidden_dim == 4
    assert config.updates_per_phase == 8
    assert report.initialization_key_words_uint32 != report.data_key_words_uint32
    assert report.protocol_sha256
    assert alberta_framework.WP2DenseLayerDevelopmentConfig is WP2DenseLayerDevelopmentConfig
    assert evaluation_api.WP2DenseLayerDevelopmentReport is WP2DenseLayerDevelopmentReport
    assert (
        alberta_framework.run_wp2_dense_layer_development_matrix
        is run_wp2_dense_layer_development_matrix
    )
    with pytest.raises(ValueError, match="frozen"):
        dataclasses.replace(config, adam_learning_rate=0.123)


def test_raw_prequential_probe_switch_and_parameter_traces_are_retained(
    report: WP2DenseLayerDevelopmentReport,
) -> None:
    expected_updates = report.config.updates_per_phase * len(report.config.phase_order)
    initial_checkpoints = {arm.initial_checkpoint.parameter_float32_bits for arm in report.arms}
    stream_digests = {arm.training_stream_sha256 for arm in report.arms}
    assert len(initial_checkpoints) == 1
    assert len(stream_digests) == 1

    for arm in report.arms:
        assert len(arm.prequential_trace) == expected_updates
        assert len(arm.probe_trace) == expected_updates + 1
        assert len(arm.phase_metrics) == 3
        assert len(arm.switch_metrics) == 2
        assert len(arm.representation_trace) == expected_updates + 1
        assert tuple(phase.regime_id for phase in arm.phase_metrics) == ("A", "B", "A")
        assert all(np.isfinite(row.squared_error) for row in arm.prequential_trace)
        assert all(np.isfinite(row.a_mse) and np.isfinite(row.b_mse) for row in arm.probe_trace)
        assert all(
            0.0 <= row.dormant_fraction <= 1.0 and 0.0 <= row.effective_rank <= 4.0
            for row in arm.representation_trace
        )
        assert all(
            metric.parameter_change.changed_coordinate_count > 0 for metric in arm.phase_metrics
        )
        assert all(
            metric.parameter_change.bitwise_churn_fraction > 0.0 for metric in arm.phase_metrics
        )
        assert all(metric.half_gap_recovery_steps >= 0 for metric in arm.switch_metrics)
        assert arm.phase_metrics[1].old_task_forgetting is not None
        assert arm.phase_metrics[2].old_task_forgetting is not None


def test_mechanism_work_and_state_bytes_are_declared_but_never_called_matched(
    report: WP2DenseLayerDevelopmentReport,
) -> None:
    arms = {arm.name: arm for arm in report.arms}
    updates = report.config.updates_per_phase * 3

    assert arms[SGD_ARM].work.task_gradient_evaluations == updates
    assert arms[ADAM_ARM].work.adam_moment_updates == updates
    assert arms[SPECTRAL_ARM].work.spectral_evaluations == updates
    assert arms[SPECTRAL_ARM].work.spectral_power_matvecs == 2 * updates
    assert arms[ADAMO_ARM].work.adamo_gram_gradient_evaluations == updates
    assert arms[ADAMO_ARM].work.adamo_gram_matrix_elements > 0
    assert arms[CPR_ARM].work.cpr_per_example_gradient_evaluations == updates
    assert arms[CPR_ARM].work.cpr_reset_events > 0
    assert arms[CPR_ARM].work.cpr_initialization_draws > 0
    assert all(arm.work.output_write_calls == 0 for arm in report.arms)
    assert all(arm.work.artifact_bytes_written == 0 for arm in report.arms)
    assert all(arm.state_bytes.total_persistent_bytes > 0 for arm in report.arms)
    assert len({arm.state_bytes.total_persistent_bytes for arm in report.arms}) > 1
    assert not report.resources_matched
    assert report.resource_comparability == "not_assessed"
    assert "Gram" in report.resource_noncomparability_reason
    assert "reset" in report.resource_noncomparability_reason


def test_in_memory_checkpoint_and_exact_replay_fail_closed(
    report: WP2DenseLayerDevelopmentReport,
) -> None:
    validate_wp2_dense_layer_development_matrix(report)
    assert reconstruct_wp2_dense_layer_development_matrix(report) == report
    assert report.replay_verified
    assert report.resources.output_write_calls == 0
    assert report.resources.artifact_bytes_written == 0

    for arm in report.arms:
        restored = restore_wp2_dense_layer_checkpoint_parameters(
            arm.final_checkpoint,
            report.config,
        )
        np.testing.assert_array_equal(
            restored.view(np.uint32),
            np.asarray(arm.final_checkpoint.parameter_float32_bits, dtype=np.uint32),
        )

    first = report.arms[0]
    tampered_checkpoint = dataclasses.replace(
        first.final_checkpoint,
        parameter_float32_bits=(
            first.final_checkpoint.parameter_float32_bits[0] ^ 1,
            *first.final_checkpoint.parameter_float32_bits[1:],
        ),
    )
    tampered_arm = dataclasses.replace(first, final_checkpoint=tampered_checkpoint)
    tampered = dataclasses.replace(report, arms=(tampered_arm, *report.arms[1:]))
    with pytest.raises(ValueError, match="checkpoint|replay|report"):
        validate_wp2_dense_layer_development_matrix(tampered, reconstruct=False)


def test_repeated_in_memory_execution_is_exact(report: WP2DenseLayerDevelopmentReport) -> None:
    assert run_wp2_dense_layer_development_matrix() == report
