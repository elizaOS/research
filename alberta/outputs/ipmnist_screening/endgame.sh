#!/usr/bin/env bash
# Autonomous endgame for the beat-SOTA screening sweep.
# Waits for all 60 proxy shards, validates the proxy, merges the ranked
# table, launches full-protocol pool64 confirmations for every candidate
# (paired proxy diff > +0.005 vs upgd_w_control, plus the top 2 regardless),
# waits for those, and writes FINAL_REPORT.md with the honest verdict.
# Idempotent; safe to re-run. Survives session exit (run under setsid).
set -u
cd /home/shaw/milady/research/alberta
PY=.venv/bin/python
DIR=outputs/ipmnist_screening
LOG="$DIR/endgame.log"
exec >> "$LOG" 2>&1
echo "=== endgame start $(date -Is)"

# ---- 1. Wait for all 60 proxy shards (45 original + 15 new arms), max 8h.
deadline=$(( $(date +%s) + 8*3600 ))
while :; do
  n=$(ls "$DIR"/shards/*.json 2>/dev/null | wc -l)
  echo "$(date -Is) shards=$n/60"
  [ "$n" -ge 60 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "TIMEOUT waiting for shards; proceeding with $n shards"
    break
  fi
  # Resurrect dead workers if none are running and shards are incomplete.
  if ! pgrep -f "ipmnist_screening run" > /dev/null; then
    echo "$(date -Is) no workers alive; resuming both job files"
    setsid bash -c "cd $PWD/$DIR && xargs -P 5 -n 2 bash worker.sh < jobs.txt" >/dev/null 2>&1 &
    setsid bash -c "cd $PWD/$DIR && xargs -P 2 -n 2 bash worker.sh < jobs2.txt" >/dev/null 2>&1 &
  fi
  sleep 300
done

# ---- 2. Validate the proxy against the pinned full-run prefix values.
$PY -m alberta_framework.benchmarks.ipmnist_screening validate-proxy \
  --shards "$DIR"/shards/upgd_w_control_seed*.json \
           "$DIR"/shards/adamw_control_seed*.json \
  --partials-dir outputs/upgd_ipmnist/partials \
  --output "$DIR/proxy_validation.json" || echo "validate-proxy FAILED"

# ---- 3. Merge the ranked table.
$PY -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards "$DIR"/shards/*.json \
  --control-name upgd_w_control \
  --output "$DIR/summary.json" || { echo "merge FAILED"; exit 1; }

# ---- 4. Pick confirmation candidates and launch pool64 200-task runs.
$PY - <<'PYEOF'
import json
from pathlib import Path

d = json.load(open("outputs/ipmnist_screening/summary.json"))
rows = d.get("results") or d.get("ranked") or []
if isinstance(rows, dict):
    rows = [dict(name=k, **v) for k, v in rows.items()]

def paired(r):
    pv = r.get("paired_vs_control") or {}
    if isinstance(pv.get("mean_diff"), (int, float)):
        return pv["mean_diff"]
    return None

cands, rest = [], []
for r in rows:
    name = r.get("config_name") or r.get("name") or r.get("config") or ""
    if not name or name in ("upgd_w_control", "adamw_control"):
        continue
    p = paired(r)
    if p is None:
        continue
    (cands if p > 0.005 else rest).append((p, name))
cands.sort(reverse=True)
rest.sort(reverse=True)
picks = [n for _, n in cands] + [n for _, n in rest[:2]]
Path("outputs/ipmnist_screening/confirm_full/jobs_candidates.txt").write_text(
    "".join(f"{n} {s}\n" for n in picks for s in (0, 1, 2))
)
print("confirmation candidates:", picks)
PYEOF

if [ -s "$DIR/confirm_full/jobs_candidates.txt" ]; then
  echo "$(date -Is) launching confirmations"
  ( cd "$DIR" && xargs -P 6 -n 2 bash confirm_full/worker_confirm.sh \
      < confirm_full/jobs_candidates.txt )
fi

# ---- 5. Final report with the honest verdict.
$PY - <<'PYEOF'
import glob
import json
from pathlib import Path

import numpy as np

EXACT_CONTROL = {0: 0.77906, 1: 0.77903, 2: 0.77932}
SOTA = 0.7791

summary = json.load(open("outputs/ipmnist_screening/summary.json"))
val = {}
vp = Path("outputs/ipmnist_screening/proxy_validation.json")
if vp.exists():
    val = json.load(open(vp))

confirm = {}
for p in glob.glob("outputs/ipmnist_screening/confirm_full/*_seed*.json"):
    d = json.load(open(p))
    name = d.get("config") or d.get("name") or Path(p).stem.rsplit("_seed", 1)[0]
    seed = d.get("seed")
    if seed is None:
        seed = int(Path(p).stem.rsplit("_seed", 1)[1])
    accs = d.get("per_task_accuracy")
    if accs is None:
        continue
    arr = np.asarray(accs, dtype=float)
    confirm.setdefault(name, {})[int(seed)] = float(arr.mean())

pool_ctrl = confirm.pop("upgd_w_control", {})
lines = ["# Beat-SOTA screening: final report", "",
         f"Proxy validation: {json.dumps(val)[:400]}", "",
         "## Full-protocol pool64 confirmations (200 tasks, seeds 0-2)", "",
         f"Pool64 upgd_w_control: {pool_ctrl} (exact controls {EXACT_CONTROL}, "
         "known pool-vs-exact delta -0.00012)", ""]
verdict = []
for name, seeds in sorted(confirm.items()):
    vals = [seeds[s] for s in sorted(seeds)]
    mean = float(np.mean(vals))
    beats_all = all(
        seeds.get(s) is not None and pool_ctrl.get(s) is not None and seeds[s] > pool_ctrl[s]
        for s in (0, 1, 2)
    ) if len(seeds) == 3 and len(pool_ctrl) == 3 else False
    tag = ("BEATS-SOTA" if (mean > SOTA and beats_all)
           else "TIES" if abs(mean - SOTA) <= 0.002 else "BELOW")
    verdict.append(f"- {name}: mean {mean:.5f} seeds {vals} -> {tag}")
lines += verdict or ["- (no confirmation shards found)"]
lines += ["", "## Screening ranked table (proxy, 60 tasks)", "",
          "```json", json.dumps(summary, indent=1)[:6000], "```"]
Path("outputs/ipmnist_screening/FINAL_REPORT.md").write_text("\n".join(lines))
print("\n".join(verdict) if verdict else "no confirmations")
PYEOF

echo "=== endgame done $(date -Is)"
