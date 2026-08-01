#!/usr/bin/env bash
# Second-stage endgame: run the jobs3 arms (FADE head, SwiftTD-stabilized
# UPGD x IDBD) once the main sweep drains, then re-merge ALL shards and
# refresh candidates/confirmations/FINAL_REPORT. Every step is idempotent
# and safe to run after (or concurrently with) endgame.sh, whose own merge
# it strictly supersedes. Run under setsid; survives session exit.
set -u
cd /home/shaw/milady/research/alberta
PY=.venv/bin/python
DIR=outputs/ipmnist_screening
LOG="$DIR/endgame2.log"
exec >> "$LOG" 2>&1
echo "=== endgame2 start $(date -Is)"

# ---- 1. Wait for the main 60 shards, then run jobs3 (6 shards), max 10h.
deadline=$(( $(date +%s) + 10*3600 ))
while :; do
  n=$(ls "$DIR"/shards/*.json 2>/dev/null | wc -l)
  echo "$(date -Is) shards=$n (waiting for 60 before jobs3)"
  [ "$n" -ge 60 ] && break
  [ "$(date +%s)" -gt "$deadline" ] && { echo "TIMEOUT-1 at $n shards"; break; }
  sleep 300
done

echo "$(date -Is) running jobs3 arms"
( cd "$DIR" && xargs -P 3 -n 2 bash worker.sh < jobs3.txt )

while :; do
  n=$(ls "$DIR"/shards/*.json 2>/dev/null | wc -l)
  echo "$(date -Is) shards=$n (waiting for 66)"
  [ "$n" -ge 66 ] && break
  [ "$(date +%s)" -gt "$deadline" ] && { echo "TIMEOUT-2 at $n shards"; break; }
  # jobs3 workers are idempotent; restart if all died short of 66.
  if ! pgrep -f "ipmnist_screening run" > /dev/null; then
    ( cd "$DIR" && xargs -P 3 -n 2 bash worker.sh < jobs3.txt ) &
  fi
  sleep 300
done

# ---- 2. Re-merge everything (supersedes endgame.sh's 60-shard merge).
$PY -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards "$DIR"/shards/*.json \
  --control-name upgd_w_control \
  --output "$DIR/summary.json" || { echo "merge FAILED"; exit 1; }

# ---- 3. Refresh candidates; launch any confirmations not yet present.
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
print("confirmation candidates (final):", picks)
PYEOF

if [ -s "$DIR/confirm_full/jobs_candidates.txt" ]; then
  echo "$(date -Is) launching confirmations (idempotent per shard)"
  ( cd "$DIR" && xargs -P 6 -n 2 bash confirm_full/worker_confirm.sh \
      < confirm_full/jobs_candidates.txt )
fi

# ---- 4. Final report (same logic as endgame.sh; overwrites with full data).
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
lines = ["# Beat-SOTA screening: final report (all arms incl. FADE + SwiftTD)", "",
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
lines += ["", "## Screening ranked table (proxy, 60 tasks, all 22 arms)", "",
          "```json", json.dumps(summary, indent=1)[:8000], "```"]
Path("outputs/ipmnist_screening/FINAL_REPORT.md").write_text("\n".join(lines))
print("FINAL VERDICTS:")
print("\n".join(verdict) if verdict else "no confirmations")
PYEOF

echo "=== endgame2 done $(date -Is)"
