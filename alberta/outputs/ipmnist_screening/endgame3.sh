#!/usr/bin/env bash
# Third-stage endgame (wave-4 arms): wait for jobs4.txt to be published AND
# for the main sweep + jobs3 arms to drain (shards >= 66), then run jobs4 via
# worker.sh, re-merge ALL shards, refresh candidates/confirmations, and
# regenerate FINAL_REPORT.md. Every step is idempotent and safe to run after
# (or concurrently with) endgame.sh/endgame2.sh, whose merges it strictly
# supersedes. Run under setsid; survives session exit.
set -u
cd /home/shaw/milady/research/alberta
PY=.venv/bin/python
DIR=outputs/ipmnist_screening
LOG="$DIR/endgame3.log"
exec >> "$LOG" 2>&1
echo "=== endgame3 start $(date -Is)"

# ---- 1. Wait for jobs4.txt AND the 66 main+jobs3 shards, max 6h.
deadline=$(( $(date +%s) + 6*3600 ))
while :; do
  n=$(ls "$DIR"/shards/*.json 2>/dev/null | wc -l)
  havejobs=no; [ -s "$DIR/jobs4.txt" ] && havejobs=yes
  echo "$(date -Is) shards=$n jobs4=$havejobs (waiting for jobs4.txt + 66 shards)"
  [ "$havejobs" = yes ] && [ "$n" -ge 66 ] && break
  [ "$(date +%s)" -gt "$deadline" ] && { echo "TIMEOUT-1 shards=$n jobs4=$havejobs"; break; }
  sleep 300
done

# ---- 2. Run jobs4 arms (skipped entirely if the file never appeared).
if [ -s "$DIR/jobs4.txt" ]; then
  echo "$(date -Is) running jobs4 arms"
  ( cd "$DIR" && xargs -P 4 -n 2 bash worker.sh < jobs4.txt )

  # Wait until every (config, seed) named in jobs4.txt has a shard.
  while :; do
    missing=0
    while read -r cfg seed; do
      [ -n "${cfg:-}" ] || continue
      [ -f "$DIR/shards/${cfg}_seed${seed}.json" ] || missing=$((missing+1))
    done < "$DIR/jobs4.txt"
    echo "$(date -Is) jobs4 shards missing=$missing"
    [ "$missing" -eq 0 ] && break
    [ "$(date +%s)" -gt "$deadline" ] && { echo "TIMEOUT-2 jobs4 missing=$missing"; break; }
    # jobs4 workers are idempotent; restart if all died short of complete.
    if ! pgrep -f "ipmnist_screening run" > /dev/null; then
      ( cd "$DIR" && xargs -P 4 -n 2 bash worker.sh < jobs4.txt ) &
    fi
    sleep 300
  done
else
  echo "$(date -Is) jobs4.txt never appeared; merging existing shards only"
fi

# ---- 3. Re-merge everything (supersedes endgame2.sh's merge).
$PY -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards "$DIR"/shards/*.json \
  --control-name upgd_w_control \
  --output "$DIR/summary.json" || { echo "merge FAILED"; exit 1; }

# ---- 4. Refresh candidates; launch only confirmations not yet present.
# Picker schema: results[].config_name + paired_vs_control.{mean_diff,
# confirmation_candidate}. Seeds 0-2 only — extension seeds (3-9) belong to
# the dedicated ext lanes and must not be double-launched from here.
$PY - <<'PYEOF'
import json
from pathlib import Path

d = json.load(open("outputs/ipmnist_screening/summary.json"))
rows = d.get("results") or []

picks = []
for r in rows:
    name = r.get("config_name") or ""
    if not name or name in ("upgd_w_control", "adamw_control"):
        continue
    pv = r.get("paired_vs_control") or {}
    md = pv.get("mean_diff")
    if pv.get("confirmation_candidate") or (
        isinstance(md, (int, float)) and md > 0.005
    ):
        picks.append((md if isinstance(md, (int, float)) else 0.0, name))
picks.sort(reverse=True)

new = []
for _, n in picks:
    for s in (0, 1, 2):
        if not Path(
            f"outputs/ipmnist_screening/confirm_full/{n}_seed{s}.json"
        ).exists():
            new.append(f"{n} {s}\n")
Path("outputs/ipmnist_screening/confirm_full/jobs_cand4.txt").write_text(
    "".join(new)
)
print("confirmation candidates (wave-4 final):", [n for _, n in picks])
print("NEW confirmation jobs to launch:", len(new))
PYEOF

if [ -s "$DIR/confirm_full/jobs_cand4.txt" ]; then
  echo "$(date -Is) launching new confirmations (idempotent per shard)"
  ( cd "$DIR" && xargs -P 4 -n 2 bash confirm_full/worker_confirm.sh \
      < confirm_full/jobs_cand4.txt )
fi

# ---- 5. Final report — verdict logic copied from endgame2.sh (incl. the
# pool-vs-exact caveat), with TWO documented deviations:
#   (a) beats_all is computed on the seed-0-2 paired subset instead of
#       requiring len(seeds)==3, because extension seeds (3-9) now land in
#       the same directory and would otherwise make BEATS-SOTA unreachable
#       for the very arms the extensions support;
#   (b) the arm name is read from "config_name" — in the shard schema
#       "config" is the protocol dict, so endgame2's `d.get("config")`
#       pattern crashes with `unhashable type: dict` (verified on
#       confirm_full/upgd_w_wd0005_seed0.json).
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
    name = d.get("config_name") or Path(p).stem.rsplit("_seed", 1)[0]
    seed = d.get("seed")
    if seed is None:
        seed = int(Path(p).stem.rsplit("_seed", 1)[1])
    accs = d.get("per_task_accuracy")
    if accs is None:
        continue
    arr = np.asarray(accs, dtype=float)
    confirm.setdefault(name, {})[int(seed)] = float(arr.mean())

pool_ctrl = confirm.pop("upgd_w_control", {})
lines = ["# Beat-SOTA screening: final report (all arms incl. wave-4)", "",
         f"Proxy validation: {json.dumps(val)[:400]}", "",
         "## Full-protocol pool64 confirmations (200 tasks, seeds 0-2)", "",
         f"Pool64 upgd_w_control: {pool_ctrl} (exact controls {EXACT_CONTROL}, "
         "known pool-vs-exact delta -0.00012)", ""]
verdict = []
for name, seeds in sorted(confirm.items()):
    vals = [seeds[s] for s in sorted(seeds)]
    mean = float(np.mean(vals))
    core = {s: seeds[s] for s in (0, 1, 2) if s in seeds}
    beats_all = all(
        core.get(s) is not None and pool_ctrl.get(s) is not None and core[s] > pool_ctrl[s]
        for s in (0, 1, 2)
    ) if len(core) == 3 and len(pool_ctrl) == 3 else False
    tag = ("BEATS-SOTA" if (mean > SOTA and beats_all)
           else "TIES" if abs(mean - SOTA) <= 0.002 else "BELOW")
    verdict.append(f"- {name}: mean {mean:.5f} seeds {vals} -> {tag}")
lines += verdict or ["- (no confirmation shards found)"]
lines += ["", "## Screening ranked table (proxy, 60 tasks, all arms)", "",
          "```json", json.dumps(summary, indent=1)[:8000], "```"]
Path("outputs/ipmnist_screening/FINAL_REPORT.md").write_text("\n".join(lines))
print("FINAL VERDICTS:")
print("\n".join(verdict) if verdict else "no confirmations")
PYEOF

echo "=== endgame3 done $(date -Is)"
