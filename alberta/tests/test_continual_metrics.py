"""Tests for continual-learning retention, transfer, and recovery metrics."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.utils.metrics import (
    compute_backward_transfer,
    compute_forward_transfer,
    compute_per_task_forgetting,
    compute_prequential_performance,
    compute_recovery_lengths,
    compute_stability_gap,
    summarize_continual_learning,
)


def test_accuracy_matrix_forgetting_and_backward_transfer() -> None:
    performance = np.array(
        [
            [0.80, 0.55, np.nan],
            [0.60, 0.70, 0.45],
            [0.75, 0.80, 0.90],
        ]
    )
    first_exposure = [0, 1, 2]

    forgetting = compute_per_task_forgetting(performance, first_exposure)
    transfer = compute_backward_transfer(performance, first_exposure)
    forward = compute_forward_transfer(
        performance,
        first_exposure,
        baseline_performance=[0.50, 0.50, 0.50],
    )

    np.testing.assert_allclose(forgetting, [0.05, 0.0, 0.0])
    np.testing.assert_allclose(transfer, [-0.05, 0.10, 0.0])
    assert np.isnan(forward[0])
    np.testing.assert_allclose(forward[1:], [0.05, -0.05])


def test_loss_matrix_normalizes_metric_directions() -> None:
    losses = np.array(
        [
            [0.20, np.nan],
            [0.50, 0.40],
            [0.25, 0.30],
        ]
    )

    forgetting = compute_per_task_forgetting(
        losses,
        [0, 1],
        higher_is_better=False,
    )
    transfer = compute_backward_transfer(
        losses,
        [0, 1],
        higher_is_better=False,
    )

    np.testing.assert_allclose(forgetting, [0.05, 0.0])
    np.testing.assert_allclose(transfer, [-0.05, 0.10])


def test_stability_gap_and_prequential_performance_ignore_nan_probes() -> None:
    online = np.array([0.9, 0.4, np.nan, 0.8, 1.0])
    gap = compute_stability_gap(online, 0.8)

    assert compute_prequential_performance(online) == pytest.approx(0.775)
    assert gap.mean == pytest.approx(0.1)
    assert gap.maximum == pytest.approx(0.4)
    np.testing.assert_allclose(
        gap.per_step,
        [0.0, 0.4, np.nan, 0.0, 0.0],
        equal_nan=True,
    )


def test_recovery_lengths_are_bounded_by_next_change() -> None:
    online = [0.9, 0.2, 0.4, 0.8, 0.9, 0.1, 0.7, 0.85, 0.86]

    recovery = compute_recovery_lengths(
        online,
        change_points=[1, 5],
        threshold=0.8,
        window_size=2,
    )

    np.testing.assert_array_equal(recovery, [4, 4])


def test_recovery_counts_the_full_sustained_window() -> None:
    recovery = compute_recovery_lengths(
        [0.9, 0.9, 0.9],
        change_points=[0],
        threshold=0.8,
        window_size=3,
    )

    np.testing.assert_array_equal(recovery, [3])


def test_recovery_reports_minus_one_when_threshold_is_not_reached() -> None:
    recovery = compute_recovery_lengths(
        [0.2, 0.3, 0.4],
        change_points=[0],
        threshold=0.8,
        window_size=2,
    )

    np.testing.assert_array_equal(recovery, [-1])


def test_summary_preserves_direct_evidence_arrays() -> None:
    performance = np.array(
        [
            [0.80, np.nan],
            [0.60, 0.70],
            [0.75, 0.80],
        ]
    )
    online = [0.8, 0.6, 0.9]

    summary = summarize_continual_learning(
        performance,
        first_exposure=[0, 1],
        online_performance=online,
        reference_performance=0.8,
    )

    assert summary.final_performance == pytest.approx(0.775)
    assert summary.prequential_performance == pytest.approx(0.7666666667)
    assert summary.mean_forgetting == pytest.approx(0.025)
    assert summary.max_forgetting == pytest.approx(0.05)
    assert summary.backward_transfer == pytest.approx(0.025)
    assert summary.stability_gap_mean == pytest.approx(0.0666666667)
    assert summary.stability_gap_max == pytest.approx(0.2)
    np.testing.assert_allclose(summary.per_task_final_performance, [0.75, 0.80])


@pytest.mark.parametrize(
    ("matrix", "first_exposure", "message"),
    [
        ([0.1, 0.2], [0], "shape"),
        ([[0.1, 0.2]], [0], "one row index per task"),
        ([[0.1, 0.2]], [0, 1], "must index"),
        ([[0.1, np.nan]], [0, 0], "no finite evaluation"),
    ],
)
def test_task_matrix_validation(
    matrix: list[float] | list[list[float]],
    first_exposure: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_per_task_forgetting(matrix, first_exposure)
