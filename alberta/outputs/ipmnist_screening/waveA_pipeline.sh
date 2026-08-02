#!/usr/bin/env bash
# Wave-A update-rule-family screen (development lane only): colnorm_gate,
# muon_gate, lion_gate vs champion sigma0_ndecay099. Auto-confirms at 200
# tasks any arm with paired screen delta > +0.002 vs the champion.
set -u
cd /home/shaw/milady/research/alberta
JOBS=outputs/ipmnist_screening/jobs_waveA.txt
LOG=outputs/ipmnist_screening/waveA.log
: > "$JOBS"
for cfg in colnorm_gate muon_gate lion_gate; do
  for s in 0 1 2; do echo "$cfg $s" >> "$JOBS"; done
done

while true; do
  n=$(ls outputs/ipmnist_screening/shards 2>/dev/null \
    | grep -cE '^(colnorm_gate|muon_gate|lion_gate)_seed[0-2]\.json')
  [ "$n" -ge 9 ] && break
  if ! pgrep -f "ipmnist_screening run --config-name (colnorm_gate|muon_gate|lion_gate)" > /dev/null; then
    echo "launching missing wave-A shards ($n/9 done)" >> "$LOG"
    xargs -n2 -P9 bash outputs/ipmnist_screening/worker.sh < "$JOBS" >> "$LOG" 2>&1
  fi
  sleep 60
done

.venv/bin/python outputs/ipmnist_screening/waveA_rank.py > outputs/ipmnist_screening/waveA_results.json 2>> "$LOG"

winners=$(.venv/bin/python -c "
import json
d = json.load(open('outputs/ipmnist_screening/waveA_results.json'))
print(' '.join(a for a, v in d['arms'].items()
               if v.get('paired_delta_vs_champion', -1) > 0.002
               and all(x > 0 for x in v.get('per_seed_delta', [-1]))))
" 2>> "$LOG")
if [ -n "$winners" ]; then
  : > outputs/ipmnist_screening/jobs_waveA_confirm.txt
  for cfg in $winners; do
    for s in 0 1 2; do echo "$cfg $s" >> outputs/ipmnist_screening/jobs_waveA_confirm.txt; done
  done
  xargs -n2 -P9 bash outputs/ipmnist_screening/confirm_full/worker_confirm.sh \
    < outputs/ipmnist_screening/jobs_waveA_confirm.txt >> "$LOG" 2>&1
  .venv/bin/python outputs/ipmnist_screening/waveA_rank.py > outputs/ipmnist_screening/waveA_results.json 2>> "$LOG"
fi
echo "waveA pipeline complete $(date -u +%FT%TZ)" >> "$LOG"
