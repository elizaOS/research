#!/usr/bin/env bash
# Frontier round-2 driver (development screening lane only): fast-norm-decay
# neighborhood of the confirmed sigma0_ndecay099 winner (0.86245 @ 200 tasks)
# plus the noisy-champion transplant ema_norm_ndecay099.
# 1. Runs 12 60-task screening shards (4 arms x seeds 0-2) via worker.sh.
# 2. Confirms at 200 tasks: any sigma0_ndecay* arm whose paired screen delta
#    vs sigma0_ndecay099 exceeds +0.002, and ema_norm_ndecay099
#    unconditionally (it challenges the overall 0.85362 champion).
# 3. Writes frontier2_results.json (development_screening_diagnostic).
set -u
cd /home/shaw/milady/research/alberta
JOBS=outputs/ipmnist_screening/jobs_frontier2.txt
LOG=outputs/ipmnist_screening/frontier2.log
: > "$JOBS"
for cfg in sigma0_ndecay09 sigma0_ndecay095 sigma0_ndecay098 ema_norm_ndecay099; do
  for s in 0 1 2; do echo "$cfg $s" >> "$JOBS"; done
done

while true; do
  n=$(ls outputs/ipmnist_screening/shards 2>/dev/null \
    | grep -cE '^(sigma0_ndecay09_|sigma0_ndecay095_|sigma0_ndecay098_|ema_norm_ndecay099_)')
  [ "$n" -ge 12 ] && break
  if ! pgrep -f "ipmnist_screening run --config-name (sigma0_ndecay09|ema_norm_ndecay099)" > /dev/null; then
    echo "launching missing round-2 shards ($n/12 done)" >> "$LOG"
    xargs -n2 -P12 bash outputs/ipmnist_screening/worker.sh < "$JOBS" >> "$LOG" 2>&1
  fi
  sleep 60
done

winners=$(.venv/bin/python - <<'PY'
import json
base = {}
for s in (0, 1, 2):
    base[s] = json.load(open(f"outputs/ipmnist_screening/shards/sigma0_ndecay099_seed{s}.json"))[
        "average_online_accuracy"]
for cfg in ("sigma0_ndecay09", "sigma0_ndecay095", "sigma0_ndecay098"):
    deltas = []
    for s in (0, 1, 2):
        d = json.load(open(f"outputs/ipmnist_screening/shards/{cfg}_seed{s}.json"))
        deltas.append(d["average_online_accuracy"] - base[s])
    if sum(deltas) / len(deltas) > 0.002 and all(x > 0 for x in deltas):
        print(cfg)
print("ema_norm_ndecay099")
PY
)
: > outputs/ipmnist_screening/jobs_frontier2_confirm.txt
for cfg in $winners; do
  for s in 0 1 2; do
    echo "$cfg $s" >> outputs/ipmnist_screening/jobs_frontier2_confirm.txt
  done
done
xargs -n2 -P6 bash outputs/ipmnist_screening/confirm_full/worker_confirm.sh \
  < outputs/ipmnist_screening/jobs_frontier2_confirm.txt >> "$LOG" 2>&1

.venv/bin/python - > outputs/ipmnist_screening/frontier2_results.json <<'PY'
import glob
import json
out = {"evidence_class": "development_screening_diagnostic", "arms": {}}
for cfg in ("sigma0_ndecay09", "sigma0_ndecay095", "sigma0_ndecay098",
            "ema_norm_ndecay099", "sigma0_ndecay099"):
    entry = {}
    for kind, pat in (("screen", "shards"), ("confirm", "confirm_full")):
        vals = []
        for p in sorted(glob.glob(f"outputs/ipmnist_screening/{pat}/{cfg}_seed*.json")):
            vals.append(json.load(open(p))["average_online_accuracy"])
        if vals:
            entry[kind] = {"n": len(vals), "mean": sum(vals) / len(vals), "per_seed": vals}
    out["arms"][cfg] = entry
print(json.dumps(out, indent=2))
PY
echo "frontier2 pipeline complete $(date -u +%FT%TZ)" >> "$LOG"
