#!/usr/bin/env bash
# Wave-7 endgame: wait for the 24 sigma0_* screening shards (jobs7.txt), pair
# them against upgd_ema_norm_sigma0 (analyze_wave7.py -> summary_sigma0.json +
# confirm_full/jobs_wave7.txt with picks > +0.002 paired), run the 200-task
# confirmations, re-merge the master summary, and write wave7_results.json.
# Every step is idempotent; safe to re-run. Run under setsid; survives exit.
set -u
cd /home/shaw/milady/research/alberta
PY=.venv/bin/python
DIR=outputs/ipmnist_screening
exec >> "$DIR/endgame7.log" 2>&1
echo "=== endgame7 start $(date -Is)"

# ---- 1. Wait for all 24 wave-7 screening shards (restart workers if dead).
deadline=$(( $(date +%s) + 3600 ))
while :; do
  n=$(ls "$DIR"/shards/sigma0_*.json 2>/dev/null | wc -l)
  echo "$(date -Is) sigma0 shards=$n/24"
  [ "$n" -ge 24 ] && break
  [ "$(date +%s)" -gt "$deadline" ] && { echo "TIMEOUT-1 shards=$n"; break; }
  if ! pgrep -f "ipmnist_screening run" > /dev/null; then
    echo "$(date -Is) workers died; relaunching jobs7 (idempotent)"
    ( cd "$DIR" && xargs -P 12 -n 2 bash worker.sh < jobs7.txt ) &
  fi
  sleep 60
done

# ---- 2. Paired merge vs the sigma0 champion + confirmation picks (>0.002).
$PY "$DIR/analyze_wave7.py" || { echo "analyze FAILED"; exit 1; }

# ---- 3. 200-task confirmations for the picks (idempotent per shard).
if [ -s "$DIR/confirm_full/jobs_wave7.txt" ]; then
  echo "$(date -Is) launching wave-7 confirmations"
  ( cd "$DIR" && xargs -P 12 -n 2 bash confirm_full/worker_confirm.sh \
      < confirm_full/jobs_wave7.txt )
else
  echo "$(date -Is) no confirmation picks above +0.002 paired"
fi

# ---- 4. Re-merge the master 60-task summary (all arms, published control).
$PY -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards "$DIR"/shards/*.json \
  --control-name upgd_w_control \
  --output "$DIR/summary.json" || { echo "master merge FAILED"; exit 1; }

# ---- 5. Wave-7 verdicts: confirmed 200-task means paired vs the sigma0
# champion's confirm shards and vs the upgd_ema_norm 10-seed champion.
$PY - <<'PYEOF'
import json
from pathlib import Path

import numpy as np

DIR = Path("outputs/ipmnist_screening")


def confirm_mean(name: str, seeds=(0, 1, 2)):
    vals = {}
    for s in seeds:
        p = DIR / "confirm_full" / f"{name}_seed{s}.json"
        if p.exists():
            d = json.loads(p.read_text())
            vals[s] = float(np.mean(d["per_task_accuracy"]))
    return vals


base = confirm_mean("upgd_ema_norm_sigma0")
champion = confirm_mean("upgd_ema_norm", seeds=range(10))
picks = []
jobs = DIR / "confirm_full" / "jobs_wave7.txt"
if jobs.exists():
    picks = sorted({line.split()[0] for line in jobs.read_text().split("\n") if line.strip()})
summary60 = json.loads((DIR / "summary_sigma0.json").read_text())
rows = []
for name in picks:
    vals = confirm_mean(name)
    common = [s for s in vals if s in base]
    diffs = [vals[s] - base[s] for s in common]
    rows.append(
        {
            "config_name": name,
            "confirm_seed_means": {str(s): round(v, 6) for s, v in vals.items()},
            "confirm_mean": round(float(np.mean(list(vals.values()))), 6) if vals else None,
            "paired_vs_sigma0_mean_diff": round(float(np.mean(diffs)), 6) if diffs else None,
            "all_seeds_improve": bool(diffs and all(d > 0 for d in diffs)),
        }
    )
out = {
    "schema": "alberta.ipmnist_screening.wave7_results.v1",
    "evidence_class": "development_screening_diagnostic",
    "scientific_promotion_allowed": False,
    "base_confirm_upgd_ema_norm_sigma0": {str(s): round(v, 6) for s, v in base.items()},
    "champion_confirm_upgd_ema_norm_mean_n10": round(float(np.mean(list(champion.values()))), 6),
    "confirmed": rows,
    "screening_summary_sigma0": "summary_sigma0.json",
}
(DIR / "wave7_results.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print("WAVE7 VERDICTS:", json.dumps(rows))
PYEOF

echo "=== endgame7 done $(date -Is)"
