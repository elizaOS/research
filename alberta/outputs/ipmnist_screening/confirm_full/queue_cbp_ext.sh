#!/usr/bin/env bash
# Queued launcher for adamw_cbp confirmation seeds 3-9 (jobs_cbp_ext.txt),
# which were written but never launched. Waits for the two currently-running
# confirmation lanes (adamw_cbp 0-2, upgd_w_wd0005 3-9) to drain so total
# load stays flat, then runs the ext jobs at -P 4. Idempotent — worker_confirm
# skips any shard that already exists. Run under setsid.
set -u
cd /home/shaw/milady/research/alberta
DIR=outputs/ipmnist_screening
exec >> "$DIR/confirm_full/cbp_ext_queue.log" 2>&1
echo "=== queue_cbp_ext start $(date -Is)"
deadline=$(( $(date +%s) + 12*3600 ))
while :; do
  ready=yes
  for s in 0 1 2; do
    [ -f "$DIR/confirm_full/adamw_cbp_seed${s}.json" ] || ready=no
  done
  for s in 3 4 5 6 7 8 9; do
    [ -f "$DIR/confirm_full/upgd_w_wd0005_seed${s}.json" ] || ready=no
  done
  [ "$ready" = yes ] && break
  [ "$(date +%s)" -gt "$deadline" ] && { echo "TIMEOUT waiting; launching anyway"; break; }
  echo "$(date -Is) waiting for running confirm lanes to drain"
  sleep 180
done
echo "$(date -Is) launching adamw_cbp seeds 3-9 at -P 4"
( cd "$DIR" && nice -n 5 xargs -P 4 -n 2 bash confirm_full/worker_confirm.sh \
    < confirm_full/jobs_cbp_ext.txt )
echo "=== queue_cbp_ext done $(date -Is)"
