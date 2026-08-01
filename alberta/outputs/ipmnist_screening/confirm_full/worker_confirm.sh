#!/usr/bin/env bash
# One (config, seed) full-protocol pool64 confirmation shard (200 tasks).
# Screening-only pool-noise approximation: shards live here and NEVER enter
# outputs/upgd_ipmnist/partials or any promoted artifact lifecycle.
set -u
cd /home/shaw/milady/research/alberta
cfg="$1"
seed="$2"
out="outputs/ipmnist_screening/confirm_full/${cfg}_seed${seed}.json"
log="outputs/ipmnist_screening/confirm_full/${cfg}_seed${seed}.log"
if [ -f "$out" ]; then
  echo "skip ${cfg} seed ${seed} (shard exists)"
  exit 0
fi
OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$cfg" --seed "$seed" --n-tasks 200 --task-length 5000 \
  --noise-mode pool --noise-pool-steps 64 \
  --out "$out" --progress-every 20 > "$log" 2>&1
status=$?
# Noise-free arms (e.g. adamw_cbp) reject pool mode; exact step mode is
# equivalent for them (they consume no perturbation noise) and no slower.
if [ $status -ne 0 ] && grep -q "unsupported for" "$log"; then
  echo "retrying ${cfg} seed ${seed} with --noise-mode step (noise-free arm)"
  OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
    --config-name "$cfg" --seed "$seed" --n-tasks 200 --task-length 5000 \
    --noise-mode step \
    --out "$out" --progress-every 20 > "$log" 2>&1
  status=$?
fi
if [ $status -ne 0 ]; then
  echo "FAILED ${cfg} seed ${seed} (exit ${status}) — see ${log}"
fi
exit $status
