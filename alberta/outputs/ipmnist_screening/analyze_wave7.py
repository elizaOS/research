"""Wave-7 analysis: merge sigma0-family shards paired against the sigma0 champion.

Writes summary_sigma0.json (control = upgd_ema_norm_sigma0) and prints the
ranked paired table plus the >0.002 confirmation picks.
"""

import json
from pathlib import Path

import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import merge_shards

DIR = Path("outputs/ipmnist_screening")
ARMS = [
    "sigma0_ndecay099",
    "sigma0_ndecay09999",
    "sigma0_eps1e6",
    "sigma0_eps1e4",
    "sigma0_hidden_norm",
    "sigma0_gate_beta05",
    "sigma0_gate_beta2",
    "sigma0_localgate",
]
FAMILY = ARMS + ["upgd_ema_norm_sigma0", "upgd_ema_norm", "sgd_ema_norm"]

paths = []
for name in FAMILY:
    for seed in (0, 1, 2):
        p = DIR / "shards" / f"{name}_seed{seed}.json"
        if p.exists():
            paths.append(p)
        else:
            print(f"MISSING {p}")

summary = merge_shards(paths, control_name="upgd_ema_norm_sigma0")
out = DIR / "summary_sigma0.json"
out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {out} ({summary['n_shards']} shards)\n")

print(f"{'arm':26s} {'acc60':>8s} {'paired_vs_sigma0':>17s} {'all_improve':>12s}")
picks = []
for e in summary["results"]:
    pv = e.get("paired_vs_control") or {}
    md = pv.get("mean_diff")
    print(
        f"{e['config_name']:26s} {e['average_online_accuracy_mean']:8.5f} "
        f"{(f'{md:+.5f}' if md is not None else '   (control)'):>17s} "
        f"{str(pv.get('all_seeds_improve', '')):>12s}"
    )
    if e["config_name"] in ARMS and md is not None and md > 0.002:
        picks.append((md, e["config_name"]))

picks.sort(reverse=True)
print("\nconfirmation picks (paired > +0.002 vs upgd_ema_norm_sigma0):", [n for _, n in picks])
jobs = []
for _, name in picks:
    for seed in (0, 1, 2):
        if not (DIR / "confirm_full" / f"{name}_seed{seed}.json").exists():
            jobs.append(f"{name} {seed}\n")
(DIR / "confirm_full" / "jobs_wave7.txt").write_text("".join(jobs))
print(f"wrote {len(jobs)} confirm jobs -> confirm_full/jobs_wave7.txt")

# quick per-seed dump for the report
for name in ARMS:
    vals = []
    for seed in (0, 1, 2):
        p = DIR / "shards" / f"{name}_seed{seed}.json"
        if p.exists():
            d = json.loads(p.read_text())
            vals.append(round(float(np.mean(d["per_task_accuracy"])), 5))
    print(name, vals)
