#!/usr/bin/env bash
# Frontier-extensions pipeline driver (development screening lane only).
# 1. Waits for the 24 sigma0_* 60-task screening shards (launched via worker.sh).
# 2. Ranks each arm vs the upgd_ema_norm_sigma0 base, paired by seed.
# 3. Launches 200-task pool64 confirmations (worker_confirm.sh) for every arm
#    whose paired delta beats the base by > 0.002.
# 4. Writes outputs/ipmnist_screening/frontier_results.json with the ranked
#    table (evidence_class: development_screening_diagnostic — never promotes).
set -u
cd /home/shaw/milady/research/alberta
ARMS="sigma0_hidden_norm sigma0_localgate sigma0_ndecay099 sigma0_ndecay09999 sigma0_eps1e6 sigma0_eps1e4 sigma0_gate_beta05 sigma0_gate_beta2"

# 1. wait for all 24 screening shards; if the original workers died (no
#    screening python processes alive) relaunch the missing jobs idempotently
#    (worker.sh skips shards that already exist).
while true; do
  n=$(ls outputs/ipmnist_screening/shards 2>/dev/null | grep -c '^sigma0_')
  [ "$n" -ge 24 ] && break
  if ! pgrep -f "ipmnist_screening run --config-name sigma0_" > /dev/null; then
    echo "relaunching missing sigma0 screening shards ($n/24 done)" \
      >> outputs/ipmnist_screening/frontier_confirm.log
    xargs -n2 -P12 bash outputs/ipmnist_screening/worker.sh \
      < outputs/ipmnist_screening/jobs_frontier.txt \
      >> outputs/ipmnist_screening/frontier_confirm.log 2>&1
  fi
  sleep 60
done

# 2+3. rank and pick confirmation candidates
.venv/bin/python outputs/ipmnist_screening/frontier_rank.py \
  > outputs/ipmnist_screening/frontier_screen_ranked.txt 2>&1

winners=$(.venv/bin/python outputs/ipmnist_screening/frontier_rank.py --winners-only 2>/dev/null)
if [ -n "$winners" ]; then
  : > outputs/ipmnist_screening/jobs_frontier_confirm.txt
  for cfg in $winners; do
    for s in 0 1 2; do
      echo "$cfg $s" >> outputs/ipmnist_screening/jobs_frontier_confirm.txt
    done
  done
  xargs -n2 -P9 bash outputs/ipmnist_screening/confirm_full/worker_confirm.sh \
    < outputs/ipmnist_screening/jobs_frontier_confirm.txt \
    >> outputs/ipmnist_screening/frontier_confirm.log 2>&1
fi

# 4. final ranked results (screen + any confirms)
.venv/bin/python outputs/ipmnist_screening/frontier_rank.py --json \
  > outputs/ipmnist_screening/frontier_results.json 2>/dev/null
echo "frontier pipeline complete $(date -u +%FT%TZ)" >> outputs/ipmnist_screening/frontier_confirm.log
