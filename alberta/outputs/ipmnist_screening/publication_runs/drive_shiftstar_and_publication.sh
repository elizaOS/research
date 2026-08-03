#!/usr/bin/env bash
# Autonomous remainder of the shiftd099 thread:
#   1. wait for the 18 mini-star screen shards (jobs_shiftstar.txt)
#   2. rank vs the incumbent sigma0_shiftnorm_d099; auto-confirm bar +0.002
#      paired, all seeds positive
#   3. 200-task confirm any winners (seeds 0-2, exact step mode)
#   4. pick final-best by 200-task confirmed mean
#   5. 20-seed publication runs (seeds 0-19, 200 tasks, exact step mode,
#      P=12) for final-best + sigma0_ndecay099
#   6. write publication_runs/RESULTS.md + shiftstar_results.json and append
#      the mini-star verdict to FINAL_REPORT.md
# All outputs are development-grade, permanently nonpromoting.
set -u
cd /home/shaw/milady/research/alberta
SCR=outputs/ipmnist_screening
PUB=$SCR/publication_runs

# --- 1. wait for the screen (max ~40 min) -----------------------------------
for _ in $(seq 1 160); do
  n=$(ls $SCR/shards/ 2>/dev/null | grep -cE 'd098_seed|d099_(k|f|r)')
  [ "$n" -ge 18 ] && break
  sleep 15
done
echo "screen shards present: $(ls $SCR/shards/ | grep -cE 'd098_seed|d099_(k|f|r)')"

# --- 2. rank and emit confirm jobs ------------------------------------------
.venv/bin/python - <<'EOF'
import glob
import json

import numpy as np

SCR = "outputs/ipmnist_screening"
ARMS = [
    "sigma0_shiftnorm_d099_k05", "sigma0_shiftnorm_d099_k2",
    "sigma0_shiftnorm_d098", "sigma0_shiftnorm_d099_f08",
    "sigma0_shiftnorm_d099_f095", "sigma0_shiftnorm_d099_r200",
]
BAR = 0.002


def seed_means(cfg, base=f"{SCR}/shards"):
    out = {}
    for p in sorted(glob.glob(f"{base}/{cfg}_seed*.json")):
        d = json.load(open(p))
        s = int(p.split("seed")[-1].split(".")[0])
        out[s] = float(np.asarray(d["per_task_accuracy"], dtype=float).mean())
    return out


inc = seed_means("sigma0_shiftnorm_d099")
rows, candidates = [], []
for arm in ARMS:
    sm = seed_means(arm)
    common = sorted(set(sm) & set(inc))
    if not common:
        rows.append({"arm": arm, "status": "missing"})
        continue
    mean = float(np.mean(list(sm.values())))
    diffs = [sm[s] - inc[s] for s in common]
    entry = {
        "arm": arm, "screen_mean": round(mean, 6),
        "per_seed": {s: round(sm[s], 6) for s in common},
        "paired_delta_vs_shiftnorm_d099": round(float(np.mean(diffs)), 6),
        "all_seeds_positive": bool(all(d > 0 for d in diffs)),
    }
    entry["confirm_candidate"] = bool(
        np.mean(diffs) > BAR and entry["all_seeds_positive"]
    )
    if entry["confirm_candidate"]:
        candidates.append(arm)
    rows.append(entry)
out = {
    "incumbent": "sigma0_shiftnorm_d099",
    "incumbent_screen_mean": round(float(np.mean(list(inc.values()))), 6),
    "bar": BAR, "results": rows, "confirm_candidates": candidates,
}
json.dump(out, open(f"{SCR}/shiftstar_results.json", "w"), indent=1, sort_keys=True)
with open(f"{SCR}/confirm_full/jobs_shiftstar_confirm.txt", "w") as f:
    for arm in candidates:
        for s in (0, 1, 2):
            f.write(f"{arm} {s}\n")
print("candidates:", candidates)
EOF

# --- 3. confirm winners at 200 tasks ----------------------------------------
if [ -s $SCR/confirm_full/jobs_shiftstar_confirm.txt ]; then
  xargs -P 6 -n 2 bash $SCR/confirm_full/worker_confirm.sh \
    < $SCR/confirm_full/jobs_shiftstar_confirm.txt
fi

# --- 4. pick final best by confirmed 200-task mean --------------------------
FINAL_BEST=$(.venv/bin/python - <<'EOF'
import glob
import json

import numpy as np

SCR = "outputs/ipmnist_screening"
star = json.load(open(f"{SCR}/shiftstar_results.json"))
arms = ["sigma0_shiftnorm_d099"] + star["confirm_candidates"]
best, best_mean, table = None, -1.0, {}
for arm in arms:
    vals = [
        float(np.asarray(json.load(open(p))["per_task_accuracy"], dtype=float).mean())
        for p in sorted(glob.glob(f"{SCR}/confirm_full/{arm}_seed*.json"))
    ]
    if len(vals) < 3:
        continue
    m = float(np.mean(vals))
    table[arm] = {"mean": round(m, 6), "per_seed": [round(v, 6) for v in vals]}
    if m > best_mean:
        best, best_mean = arm, m
star["confirm_200task"] = table
star["final_best"] = best
json.dump(star, open(f"{SCR}/shiftstar_results.json", "w"), indent=1, sort_keys=True)
print(best)
EOF
)
echo "final best arm: $FINAL_BEST"

# --- 5. publication runs: 20 seeds x 200 tasks, final-best + champion -------
: > $PUB/jobs_publication.txt
for cfg in "$FINAL_BEST" sigma0_ndecay099; do
  for s in $(seq 0 19); do echo "$cfg $s" >> $PUB/jobs_publication.txt; done
done
sort -u $PUB/jobs_publication.txt -o $PUB/jobs_publication.txt
xargs -P 12 -n 2 bash $PUB/worker_pub.sh < $PUB/jobs_publication.txt
echo "publication shards: $(ls $PUB/*_seed*.json 2>/dev/null | wc -l)"

# --- 6. RESULTS.md + FINAL_REPORT.md appendix -------------------------------
FINAL_BEST="$FINAL_BEST" .venv/bin/python - <<'EOF'
import glob
import json
import os

import numpy as np

SCR = "outputs/ipmnist_screening"
PUB = f"{SCR}/publication_runs"
best = os.environ["FINAL_BEST"]


def stats(pattern):
    vals = {}
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p))
        s = int(p.split("seed")[-1].split(".")[0])
        vals[s] = float(np.asarray(d["per_task_accuracy"], dtype=float).mean())
    if not vals:
        return None
    v = np.array([vals[s] for s in sorted(vals)])
    return {
        "n": len(v), "mean": float(v.mean()),
        "stderr": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
        "per_seed": {s: round(vals[s], 6) for s in sorted(vals)},
    }


pub_best = stats(f"{PUB}/{best}_seed*.json")
pub_champ = stats(f"{PUB}/sigma0_ndecay099_seed*.json")
star = json.load(open(f"{SCR}/shiftstar_results.json"))


def held_out(st):
    ho = {s: m for s, m in st["per_seed"].items() if s >= 3}
    v = np.array(list(ho.values()))
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def fmt(st):
    return f"{st['mean']:.5f} +/- {st['stderr']:.5f} (n={st['n']})"


hb = held_out(pub_best)
hc = held_out(pub_champ)
best_hp = json.load(open(sorted(glob.glob(f"{PUB}/{best}_seed*.json"))[0]))[
    "hyperparameters"
]
lines = f"""# IPMNIST 20-seed publication runs (development-grade, nonpromoting)

Scope: `development_screening_diagnostic` — these numbers are the durable
descriptive claim artifact for the screening campaign at the published seed
count (n=20), and are **permanently nonpromoting** under the repository's
evidence rules (no preregistered frozen protocol; the screening source is not
registry-bound). Protocol: ICLR-2024 input-permuted MNIST — 200 tasks x 5000
steps, one example per step, 300x150 ReLU MLP, average online accuracy,
exact per-step noise mode (both arms are noise-free, sigma=0). Runner:
`worker_pub.sh` -> `alberta_framework.benchmarks.ipmnist_screening run`.

## Headline (seeds 0-19)

| Arm | n | Mean online acc +/- stderr |
|---|---:|---|
| `{best}` (final best) | {pub_best['n']} | **{pub_best['mean']:.5f} +/- {pub_best['stderr']:.5f}** |
| `sigma0_ndecay099` (prior champion) | {pub_champ['n']} | {pub_champ['mean']:.5f} +/- {pub_champ['stderr']:.5f} |

Cited references (previously stored artifacts, not rerun here):

| Reference | n | Mean online acc +/- stderr | Source |
|---|---:|---|---|
| `upgd_ema_norm` (UPGD-W + EMA input norm) | 10 | 0.85362 +/- 0.00007 | `confirm_full/upgd_ema_norm_seed*.json` |
| `upgd_w` published-config reproduction (baseline) | 10 | 0.77915 +/- 0.00006 | `outputs/upgd_ipmnist/partials/upgd_w_seed*.json` |
| `adamw` published-config reproduction | 10 | 0.71900 +/- 0.00059 | `outputs/upgd_ipmnist/partials/adamw_seed*.json` |

## Selection-bias caveat

Seeds 0-2 of both headline arms were consumed by screening/selection; seeds
3-19 are selection-untouched. Held-out-only means (seeds 3-19, n=17):

- `{best}`: {hb[0]:.5f} +/- {hb[1]:.5f}
- `sigma0_ndecay099`: {hc[0]:.5f} +/- {hc[1]:.5f}

## Final-best hyperparameters

`{best}`: {json.dumps(best_hp, sort_keys=True)}

## Per-seed means

`{best}`: {json.dumps(pub_best['per_seed'])}

`sigma0_ndecay099`: {json.dumps(pub_champ['per_seed'])}

## Mini-star context

Screen-and-confirm chain for the shift-detector mini-star (60-task screen,
paired vs `sigma0_shiftnorm_d099`, +0.002 auto-confirm bar; 200-task
confirmation for candidates): `../shiftstar_results.json`. The 200-task
3-seed confirmation that made `sigma0_shiftnorm_d099` the record holder:
mean 0.86459 seeds [0.864213, 0.864415, 0.865129]
(`../confirm_full/sigma0_shiftnorm_d099_seed*.json`).
"""
open(f"{PUB}/RESULTS.md", "w").write(lines)

conf = star.get("confirm_200task", {})
rows = "\n".join(
    f"- `{r['arm']}` screen {r.get('screen_mean')} "
    f"(paired {r.get('paired_delta_vs_shiftnorm_d099'):+} vs incumbent, "
    f"all-seeds-positive={r.get('all_seeds_positive')}"
    + (
        f"; 200-task confirm {conf[r['arm']]['mean']}"
        if r["arm"] in conf else ""
    )
    + ")"
    for r in star["results"] if "screen_mean" in r
)
appendix = f"""

## Shift-detector mini-star verdicts (auto-appended)

60-task screen, seeds 0-2, paired vs `sigma0_shiftnorm_d099`
(incumbent screen {star['incumbent_screen_mean']}), bar +{star['bar']}:

{rows}

Final best arm by 200-task confirmed mean: **`{star['final_best']}`**.
20-seed publication runs (final best + `sigma0_ndecay099`, seeds 0-19,
200 tasks, exact step mode): `publication_runs/RESULTS.md`.
"""
open(f"{SCR}/FINAL_REPORT.md", "a").write(appendix)
print("RESULTS.md written; final_best:", best)
EOF
echo "DRIVER-COMPLETE"
