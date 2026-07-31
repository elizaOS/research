"""Smoke tests for utils.timing and utils.visualization.

Visualization tests run on the headless Agg backend, operate purely on
in-memory figure objects, and perform no file I/O.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alberta_framework.utils.experiments import (  # noqa: E402
    AggregatedResults,
    MetricSummary,
)
from alberta_framework.utils.timing import Timer, format_duration  # noqa: E402
from alberta_framework.utils.visualization import (  # noqa: E402
    create_comparison_figure,
    plot_final_performance_bars,
    plot_hyperparameter_heatmap,
    plot_learning_curves,
    plot_step_size_evolution,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def test_format_duration_branches() -> None:
    assert format_duration(0.5) == "0.50s"
    assert format_duration(90.5) == "1m 30.50s"
    assert format_duration(3665) == "1h 1m 5.00s"


def test_timer_monotonic_accumulation() -> None:
    with Timer("smoke", verbose=False) as timer:
        first = timer.elapsed()
        time.sleep(0.01)
        second = timer.elapsed()

    assert 0.0 <= first <= second
    assert timer.end_time >= timer.start_time
    assert timer.duration >= second
    assert timer.duration == pytest.approx(timer.end_time - timer.start_time)

    # A second timed block accumulates independently and stays non-negative.
    with Timer("smoke-2", verbose=False) as second_timer:
        time.sleep(0.001)
    assert second_timer.duration > 0.0
    assert second_timer.start_time >= timer.end_time


def test_timer_print_fn_and_repr() -> None:
    messages: list[str] = []
    with Timer("Custom op", print_fn=messages.append):
        pass

    assert len(messages) == 1
    assert messages[0].startswith("Custom op completed in ")

    silent = Timer("silent", verbose=False)
    assert repr(silent) == "Timer(name='silent')"
    with silent:
        time.sleep(0.001)
    assert "duration=" in repr(silent)


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------


def _aggregated(name: str, seed_offset: int, n_steps: int = 120) -> AggregatedResults:
    rng = np.random.default_rng(seed_offset)
    error = rng.uniform(0.1, 1.0, size=(3, n_steps))
    step_size = rng.uniform(0.01, 0.1, size=(3, n_steps))
    finals = error[:, -1]
    summary = {
        "squared_error": MetricSummary(
            mean=float(finals.mean()),
            std=float(finals.std()),
            min=float(finals.min()),
            max=float(finals.max()),
            n_seeds=3,
            values=finals,
        )
    }
    return AggregatedResults(
        config_name=name,
        seeds=[0, 1, 2],
        metric_arrays={"squared_error": error, "mean_step_size": step_size},
        summary=summary,
    )


@pytest.fixture
def results() -> dict[str, AggregatedResults]:
    return {"lms": _aggregated("lms", 0), "idbd": _aggregated("idbd", 1)}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_learning_curves_returns_figure_and_axes(results) -> None:
    fig, ax = plot_learning_curves(results, window_size=5)
    assert isinstance(fig, plt.Figure)
    assert ax.figure is fig
    assert len(ax.lines) == 2
    assert {line.get_label() for line in ax.lines} == {"lms", "idbd"}


def test_plot_final_performance_bars(results) -> None:
    fig, ax = plot_final_performance_bars(results)
    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) == 2
    assert [label.get_text() for label in ax.get_xticklabels()] == ["lms", "idbd"]


def test_plot_step_size_evolution(results) -> None:
    fig, ax = plot_step_size_evolution(results)
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) == 2
    assert ax.get_yscale() == "log"


def test_plot_hyperparameter_heatmap_handles_missing_cells(results) -> None:
    fig, ax = plot_hyperparameter_heatmap(
        results,
        param1_name="optimizer",
        param1_values=["lms", "idbd"],
        param2_name="suffix",
        param2_values=["", "_missing"],
        name_pattern="{p1}{p2}",
    )
    assert isinstance(fig, plt.Figure)
    assert len(ax.images) == 1
    # Only the first column resolves to a known config; the second is NaN.
    data = np.ma.filled(np.asarray(ax.images[0].get_array(), dtype=float), np.nan)
    assert np.isfinite(data[:, 0]).all()
    assert np.isnan(data[:, 1]).all()


def test_create_comparison_figure_has_four_panels(results) -> None:
    fig = create_comparison_figure(results)
    assert isinstance(fig, plt.Figure)
    # 2x2 panels plus the heatmap-free layout; colorbars would add axes.
    assert len(fig.axes) == 4
    titles = {ax.get_title() for ax in fig.axes}
    assert "Learning Curves" in titles
    assert "Final Performance" in titles
