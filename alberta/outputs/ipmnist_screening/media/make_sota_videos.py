"""Render the SOTA-comparison videos from stored 200-task development curves.

Two videos (development-grade visualizations of nonpromoting artifacts):

1. ``sota_race.mp4`` — progressive reveal of per-task online accuracy over
   the 200-task protocol: published-config UPGD-W reproduction vs the best
   protocol-pure arm (``adamw_cbp_r3e4``), the no-backprop tracking control
   (``rff_rls``), and the champion (``sigma0_ndecay099``), with running-mean
   counters and the published-SOTA band highlighted.
2. ``mechanism_cascade.mp4`` — animated buildup of the mechanism
   decomposition: 0.7791 -> +conditioning -> +gate -> +fast tracking ->
   0.86245, one rung at a time.

Every number is read from ``confirm_full/`` shard artifacts; nothing is
hard-coded except labels. Rendering requires matplotlib + ffmpeg.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"

ARMS = {
    "upgd_w_control": ("Published SOTA (UPGD-W, ICLR 2024 config)", "#888888"),
    "adamw_cbp_r3e4": ("AdamW + CBP (ours, protocol-pure)", "#1f77b4"),
    "rff_rls": ("Random features + RLS (ours, no backprop)", "#2ca02c"),
    "sigma0_ndecay099": ("Fast-conditioned gated SGD (ours, champion)", "#d62728"),
}


def load_mean_curve(arm: str) -> np.ndarray:
    curves = []
    for p in sorted(glob.glob(str(ROOT / "confirm_full" / f"{arm}_seed*.json"))):
        curves.append(np.asarray(json.load(open(p))["per_task_accuracy"], dtype=float))
    if not curves:
        raise FileNotFoundError(f"no confirm_full shards for {arm}")
    n = min(len(c) for c in curves)
    return np.mean([c[:n] for c in curves], axis=0)


def render_race() -> None:
    data = {arm: load_mean_curve(arm) for arm in ARMS}
    n_tasks = min(len(c) for c in data.values())
    smooth = {a: np.convolve(c[:n_tasks], np.ones(5) / 5, mode="valid") for a, c in data.items()}

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.tick_params(colors="#bbbbbb")
    ax.set_xlim(0, n_tasks)
    ax.set_ylim(0.55, 0.92)
    ax.set_xlabel("Task (input permutation #)", color="#dddddd", fontsize=12)
    ax.set_ylabel("Online accuracy (per task, 5-task smoothed)", color="#dddddd", fontsize=12)
    ax.set_title(
        "Input-permuted MNIST, ICLR-2024 protocol — 1M examples, one per step, 200 tasks\n"
        "development-grade stored curves, nonpromoting",
        color="#ffffff",
        fontsize=13,
    )
    ax.axhspan(0.75, 0.78, color="#666666", alpha=0.25)
    ax.text(
        n_tasks * 0.99, 0.765, "published SOTA band", color="#aaaaaa",
        ha="right", va="center", fontsize=10,
    )

    lines, counters = {}, {}
    for i, (arm, (label, color)) in enumerate(ARMS.items()):
        (lines[arm],) = ax.plot([], [], color=color, lw=2.2, label=label)
        counters[arm] = ax.text(
            0.985, 0.30 - 0.055 * i, "", transform=ax.transAxes, color=color,
            ha="right", va="center", fontsize=13, family="monospace", fontweight="bold",
        )
    ax.legend(loc="lower left", facecolor="#1a1d24", edgecolor="#444444",
              labelcolor="#eeeeee", fontsize=11)

    frames = min(len(c) for c in smooth.values())

    def update(f: int):
        artists = []
        for arm in ARMS:
            c = smooth[arm]
            lines[arm].set_data(np.arange(f + 1), c[: f + 1])
            running = float(np.mean(data[arm][: f + 1]))
            counters[arm].set_text(f"{running:0.4f}")
            artists.extend([lines[arm], counters[arm]])
        return artists

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(str(MEDIA / "sota_race.mp4"), writer=animation.FFMpegWriter(fps=12, bitrate=4000))
    plt.close(fig)
    print("wrote", MEDIA / "sota_race.mp4")


def render_cascade() -> None:
    rungs = [
        ("Published SOTA\n(UPGD-W as published)", load_mean_curve("upgd_w_control").mean(), "#888888"),
        ("+ EMA input\nconditioning (SGD only)", load_mean_curve("sgd_ema_norm").mean(), "#1f77b4"),
        ("+ utility gate\n(σ=0 gated SGD)", load_mean_curve("upgd_ema_norm_sigma0").mean(), "#9467bd"),
        ("+ fast tracking\n(decay 0.99) — champion", load_mean_curve("sigma0_ndecay099").mean(), "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.tick_params(colors="#bbbbbb")
    ax.set_ylim(0.74, 0.888)
    ax.set_xlim(-0.6, len(rungs) - 0.4)
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([r[0] for r in rungs], color="#dddddd", fontsize=11)
    ax.set_ylabel("Average online accuracy (200 tasks)", color="#dddddd", fontsize=12)
    ax.set_title(
        "The mechanism cascade — every rung beats published SOTA\n"
        "development-grade stored means, nonpromoting",
        color="#ffffff", fontsize=13,
    )
    ax.axhline(rungs[0][1], color="#888888", ls="--", lw=1, alpha=0.7)

    bars = ax.bar(
        range(len(rungs)), [0] * len(rungs),
        color=[r[2] for r in rungs], width=0.62, bottom=0.74,
    )
    values = [
        ax.text(i, 0.745, "", ha="center", color="#ffffff", fontsize=14,
                fontweight="bold", family="monospace")
        for i in range(len(rungs))
    ]
    deltas = [
        ax.text(i, 0.86, "", ha="center", color=rungs[i][2], fontsize=12,
                family="monospace")
        for i in range(len(rungs))
    ]

    per_bar = 30
    frames = per_bar * len(rungs) + 24

    def update(f: int):
        artists = []
        for i, (_, val, _c) in enumerate(rungs):
            start = i * per_bar
            t = np.clip((f - start) / per_bar, 0.0, 1.0)
            eased = 1 - (1 - t) ** 3
            h = (val - 0.74) * eased
            bars[i].set_height(h)
            if t > 0:
                values[i].set_text(f"{0.74 + h:0.4f}" if t < 1 else f"{val:0.4f}")
                values[i].set_y(0.74 + h + 0.004)
            if t >= 1 and i > 0:
                deltas[i].set_text(f"+{val - rungs[i - 1][1]:0.3f}")
                deltas[i].set_y(val + 0.016)
            artists.extend([bars[i], values[i], deltas[i]])
        return artists

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(str(MEDIA / "mechanism_cascade.mp4"),
              writer=animation.FFMpegWriter(fps=24, bitrate=4000))
    plt.close(fig)
    print("wrote", MEDIA / "mechanism_cascade.mp4")


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    render_race()
    render_cascade()
