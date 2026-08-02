#!/usr/bin/env bash
# Live confirmation-verdict viewer (read-only, safe to run any time):
#   bash outputs/ipmnist_screening/watch_verdicts.sh          one shot
#   watch -n 60 bash outputs/ipmnist_screening/watch_verdicts.sh
# Same verdict computation as endgame2/endgame3 (pool64 confirmations paired
# against pool64 upgd_w_control on seeds 0-2), plus all-seed means and the
# live progress line of every confirmation still in flight.
set -u
cd /home/shaw/milady/research/alberta
.venv/bin/python - <<'PYEOF'
import glob
import json
import time
from pathlib import Path

import numpy as np

EXACT_CONTROL = {0: 0.77906, 1: 0.77903, 2: 0.77932}
SOTA = 0.7791

confirm = {}
for p in glob.glob("outputs/ipmnist_screening/confirm_full/*_seed*.json"):
    d = json.load(open(p))
    # NB: in the shard schema "config" is the protocol dict; the arm name
    # is "config_name" (fall back to the filename stem).
    name = d.get("config_name") or Path(p).stem.rsplit("_seed", 1)[0]
    seed = d.get("seed")
    if seed is None:
        seed = int(Path(p).stem.rsplit("_seed", 1)[1])
    accs = d.get("per_task_accuracy")
    if accs is None:
        continue
    confirm.setdefault(name, {})[int(seed)] = float(
        np.asarray(accs, dtype=float).mean()
    )

pool_ctrl = confirm.pop("upgd_w_control", {})
print(f"=== confirmation verdicts @ {time.strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"Pool64 upgd_w_control: {pool_ctrl}")
print(f"(exact controls {EXACT_CONTROL}, known pool-vs-exact delta -0.00012; "
      f"SOTA {SOTA})")
print()
if not confirm:
    print("(no confirmation shards yet)")
for name, seeds in sorted(confirm.items()):
    vals = [seeds[s] for s in sorted(seeds)]
    mean = float(np.mean(vals))
    core = {s: seeds[s] for s in (0, 1, 2) if s in seeds}
    beats_all = (
        len(core) == 3
        and len(pool_ctrl) == 3
        and all(core[s] > pool_ctrl[s] for s in (0, 1, 2))
    )
    tag = ("BEATS-SOTA" if (mean > SOTA and beats_all)
           else "TIES" if abs(mean - SOTA) <= 0.002 else "BELOW")
    print(f"- {name}: mean {mean:.5f} over seeds {sorted(seeds)} -> {tag}"
          f"  (paired seeds0-2 all-improve: {beats_all})")

print()
print("In flight (log without finished shard):")
inflight = False
for p in sorted(glob.glob("outputs/ipmnist_screening/confirm_full/*_seed*.log")):
    if Path(p[:-4] + ".json").exists():
        continue
    lines = [ln.strip() for ln in open(p, errors="replace") if " task " in ln]
    if lines:
        inflight = True
        print(" ", Path(p).stem, "->", lines[-1].split("INFO ")[-1])
if not inflight:
    print("  (none)")
PYEOF
