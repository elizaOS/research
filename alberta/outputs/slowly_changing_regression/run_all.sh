#!/usr/bin/env bash
# SCR v2 full protocol: 300 shards (3 methods x 100 seeds) -> merge -> validate.
# Idempotent per shard (immutable atomic writes refuse overwrite); resumable by
# re-running this script. Runs nice'd so screening confirmations keep priority.
set -u
cd /home/shaw/milady/research/alberta
PY=.venv/bin/python
DIR=outputs/slowly_changing_regression
PLAN="$DIR/plan.v2.json"
LOG="$DIR/run_all.log"
exec >> "$LOG" 2>&1
echo "=== SCR run_all start $(date -Is)"
mkdir -p "$DIR/shards"

# Job list: method seed pairs.
JOBS="$DIR/jobs.txt"
if [ ! -s "$JOBS" ]; then
  for m in publication_bp_relu_sgd alberta_cbp_relu_local_extension alberta_upgd_relu_local_extension; do
    for s in $(seq 0 99); do echo "$m $s"; done
  done > "$JOBS"
fi

worker() {
  m="$1"; s="$2"
  out="outputs/slowly_changing_regression/shards/${m}_seed${s}.json"
  [ -f "$out" ] && { echo "skip $m $s"; return 0; }
  OMP_NUM_THREADS=1 nice -n 15 .venv/bin/python -m \
    alberta_framework.benchmarks.slowly_changing_regression run-shard \
    --plan outputs/slowly_changing_regression/plan.v2.json \
    --method "$m" --seed-id "$s" --output "$out" \
    > "outputs/slowly_changing_regression/shards/${m}_seed${s}.log" 2>&1 \
    || echo "FAILED $m $s"
}
export -f worker

xargs -P 6 -n 2 bash -c 'worker "$@"' _ < "$JOBS"

n=$(ls "$DIR"/shards/*.json 2>/dev/null | wc -l)
echo "$(date -Is) shards done: $n/300"
if [ "$n" -lt 300 ]; then
  echo "INCOMPLETE — rerun this script to resume"
  exit 1
fi

$PY -m alberta_framework.benchmarks.slowly_changing_regression merge \
  --plan "$PLAN" --shards-dir "$DIR/shards" \
  --output "$DIR/replication.v2.json" || { echo "merge FAILED"; exit 1; }

$PY -m alberta_framework.benchmarks.slowly_changing_regression validate \
  --artifact "$DIR/replication.v2.json" && echo "SCR-VALIDATED" || echo "SCR-VALIDATE-FAILED"

$PY - <<'PYEOF'
import json
d = json.load(open("outputs/slowly_changing_regression/replication.v2.json"))
o = d.get("orderings") or d.get("summary", {}).get("orderings") or {}
print("SCR-ORDERINGS:", json.dumps(o)[:600])
PYEOF
echo "=== SCR run_all done $(date -Is)"
