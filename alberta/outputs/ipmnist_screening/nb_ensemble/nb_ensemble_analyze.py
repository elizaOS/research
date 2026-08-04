"""Curve analysis for the nb_ensemble transient attack.

Development-only diagnostic (never promotable evidence). Reads the per-step
traces written by nb_ensemble_runs.py for the ensemble arm(s), the shiftnorm
champion, and the naive_bayes tracker (same seed, same protocol schedule =
paired streams), and reports:

- mean within-task accuracy by step bucket (tasks >= 1: the post-shift
  transient; task 0 warmup separately);
- first-500-step accuracy: ensemble vs champion vs NB (the attack's target
  window);
- the ORACLE SWITCH bound: mean over (task >= 1, step) of
  max(champion, naive_bayes) per-step accuracy — the ceiling any
  accuracy-weighted vote of these two members can reach on this stream;
- switching dynamics: mean NB vote weight per within-task step bucket
  (switch-in lag, switch-back lag, mid-task residual weight).

Usage:
  .venv/bin/python outputs/ipmnist_screening/nb_ensemble/nb_ensemble_analyze.py \
      --ensemble nb_ensemble_champion --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

BUCKETS = (
    (0, 50), (50, 100), (100, 250), (250, 500), (500, 1000),
    (1000, 2000), (2000, 3500), (3500, 5000),
)


def _load(tag: str, seed: int) -> np.ndarray:
    return np.load(OUT / f"{tag}_seed{seed}_per_step.npy").astype(np.float64)


def _bucket_means(curve_2d: np.ndarray) -> dict[str, float]:
    return {
        f"{a}-{b}": round(float(curve_2d[:, a:b].mean()), 6) for a, b in BUCKETS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", default="nb_ensemble_champion")
    parser.add_argument("--champion", default="sigma0_shiftnorm_d099")
    parser.add_argument("--nb", default="naive_bayes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ens = _load(args.ensemble, args.seed)
    champ = _load(args.champion, args.seed)
    nb = _load(args.nb, args.seed)
    n_tasks = min(ens.shape[0], champ.shape[0], nb.shape[0])
    ens, champ, nb = ens[:n_tasks], champ[:n_tasks], nb[:n_tasks]

    # Post-shift tasks only (task 0 is from-scratch warmup, not a shift).
    shifted = slice(1, n_tasks)
    oracle = np.maximum(champ[shifted], nb[shifted])

    report: dict[str, object] = {
        "schema": "alberta.ipmnist_nb_ensemble.analysis.v1",
        "evidence_class": "development_screening_diagnostic",
        "development_only": True,
        "seed": args.seed,
        "n_tasks": n_tasks,
        "arms": {
            "ensemble": args.ensemble,
            "champion": args.champion,
            "nb": args.nb,
        },
        "mean_accuracy": {
            "ensemble": round(float(ens.mean()), 6),
            "champion": round(float(champ.mean()), 6),
            "nb": round(float(nb.mean()), 6),
            "oracle_switch_bound_shifted_tasks": round(float(oracle.mean()), 6),
        },
        "task0_warmup_mean": {
            "ensemble": round(float(ens[0].mean()), 6),
            "champion": round(float(champ[0].mean()), 6),
            "nb": round(float(nb[0].mean()), 6),
        },
        "shifted_task_bucket_means": {
            "ensemble": _bucket_means(ens[shifted]),
            "champion": _bucket_means(champ[shifted]),
            "nb": _bucket_means(nb[shifted]),
            "oracle_switch": _bucket_means(oracle),
        },
        "first_500_shifted": {
            "ensemble": round(float(ens[shifted, :500].mean()), 6),
            "champion": round(float(champ[shifted, :500].mean()), 6),
            "nb": round(float(nb[shifted, :500].mean()), 6),
            "oracle_switch": round(float(oracle[:, :500].mean()), 6),
        },
        "plateau_3500_5000": {
            "ensemble": round(float(ens[shifted, 3500:].mean()), 6),
            "champion": round(float(champ[shifted, 3500:].mean()), 6),
            "nb": round(float(nb[shifted, 3500:].mean()), 6),
        },
    }

    weight_path = OUT / f"{args.ensemble}_seed{args.seed}_nb_weight.npy"
    if weight_path.exists():
        w = np.load(weight_path).astype(np.float64)[:n_tasks]
        report["nb_vote_weight_bucket_means_shifted"] = {
            f"{a}-{b}": round(float(w[shifted, a:b].mean()), 6) for a, b in BUCKETS
        }
        # Switch-in speed: first within-task step (mean over shifted tasks)
        # where the NB weight crosses 0.5 from below, if it does.
        crossed = (w[shifted] > 0.5).argmax(axis=1)
        never = ~(w[shifted] > 0.5).any(axis=1)
        report["nb_weight_crosses_half"] = {
            "fraction_of_tasks": round(float(1.0 - never.mean()), 4),
            "mean_step_when_crossed": (
                round(float(crossed[~never].mean()), 1) if (~never).any() else None
            ),
        }

    out_path = OUT / f"analysis_{args.ensemble}_seed{args.seed}.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
