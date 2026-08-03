#!/usr/bin/env bash
# One (config, seed) publication-grade full-protocol shard: 200 tasks x 5000
# steps, exact per-step noise mode (noise-free arms consume no noise; exact
# mode is the protocol-pure path for them).  Idempotent + atomic via the
# screening CLI's occupied-output refusal: a complete shard is never
# overwritten, so concurrent workers are safe.
#
# Scope: development-grade, permanently nonpromoting
# (development_screening_diagnostic).  These shards NEVER enter
# outputs/upgd_ipmnist/partials or any promoted artifact lifecycle.
set -u
cd /home/shaw/milady/research/alberta
cfg="$1"
seed="$2"
out="outputs/ipmnist_screening/publication_runs/${cfg}_seed${seed}.json"
log="outputs/ipmnist_screening/publication_runs/${cfg}_seed${seed}.log"
if [ -f "$out" ]; then
  echo "skip ${cfg} seed ${seed} (shard exists)"
  exit 0
fi
OMP_NUM_THREADS=1 .venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening run \
  --config-name "$cfg" --seed "$seed" --n-tasks 200 --task-length 5000 \
  --noise-mode step \
  --out "$out" --progress-every 20 > "$log" 2>&1
status=$?
if [ $status -ne 0 ]; then
  echo "FAILED ${cfg} seed ${seed} (exit ${status}) — see ${log}"
fi
exit $status
