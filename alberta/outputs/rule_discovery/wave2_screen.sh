#!/usr/bin/env bash
# Wave-2 discovered-arm screening/confirmation pool.
# Usage: wave2_screen.sh screen|confirm arm1 [arm2 ...]
# Idempotent per (arm, seed): reuses the campaign worker convention.
set -u
cd /home/shaw/milady/research/alberta
mode="$1"; shift
case "$mode" in
  screen) n_tasks=60; out_dir=outputs/ipmnist_screening/shards ;;
  confirm) n_tasks=200; out_dir=outputs/ipmnist_screening/confirm_full ;;
  *) echo "mode must be screen|confirm" >&2; exit 2 ;;
esac
mkdir -p "$out_dir" outputs/ipmnist_screening/logs
pids=()
for cfg in "$@"; do
  for seed in 0 1 2; do
    out="${out_dir}/${cfg}_seed${seed}.json"
    log="outputs/ipmnist_screening/logs/${cfg}_${mode}_seed${seed}.log"
    if [ -f "$out" ]; then
      echo "skip ${cfg} seed ${seed} (${mode} shard exists)"
      continue
    fi
    OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
      --config-name "$cfg" --seed "$seed" --n-tasks "$n_tasks" --task-length 5000 \
      --out "$out" --progress-every 20 > "$log" 2>&1 &
    pids+=($!)
  done
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
echo "wave2 ${mode} pool done (status ${status})"
exit $status
