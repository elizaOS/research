#!/usr/bin/env python
"""Translate promoted wave-2 search genomes into screening-arm registry rows.

Reads ``search_gauss_v2.json`` (the gauss-lane search artifact), takes the
``promoted`` rows (holdout survivors over the budget-matched tuned
champion-form baseline), and prints two registry tuples per survivor:

- ``disc2_rK``     — the discovered flags + constants VERBATIM;
- ``disc2_rK_pscale`` — the same flag structure at champion-scale core
  constants (lr/wd/norm_decay/fast_decay/shift_k/utility_decay from
  ``sigma0_shiftnorm_d099``), keeping the discovered constants of the new
  mechanisms — the wave-1 lesson (`disc_r1_pscale_norms`) made this the
  transfer-relevant translation.

Development tooling only; promotes nothing by itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FLAGS = (
    "norm", "shift_reset", "gate", "decay_to_init", "surprise_budget",
    "meta_decay", "utility_shift_reset", "w1_shift_reset", "hidden_rms",
    "rls_head", "rls_reset_p", "nb_member", "lr_anneal", "layer_lr",
    "kalman_norm",
)
PARAMS = (
    "lr", "weight_decay", "norm_decay", "fast_decay", "shift_k",
    "utility_decay", "gate_beta", "surprise_gain", "surprise_fast",
    "surprise_slow", "meta_gain", "rls_lambda", "nb_decay", "vote_decay",
    "anneal_lo", "anneal_hi", "layer_lr_ratio", "kalman_q",
)
PARAM_TO_HP = {"lr": "step_size"}
CHAMPION_SCALE = {
    "step_size": 0.01, "weight_decay": 0.01, "norm_decay": 0.99,
    "fast_decay": 0.9, "shift_k": 1.0, "utility_decay": 0.9999,
    "gate_beta": 1.0,
}


def hp_dict(config: dict[str, float], pscale: bool) -> dict[str, float]:
    hp: dict[str, float] = {}
    for flag in FLAGS:
        hp[f"flag_{flag}"] = float(config[flag])
    for param in PARAMS:
        hp[PARAM_TO_HP.get(param, param)] = float(config[param])
    if pscale:
        hp.update(CHAMPION_SCALE)
    return hp


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/rule_discovery/search_gauss_v2.json")
    payload = json.loads(path.read_text())
    print(f"# promoted rows: {len(payload['promoted'])}")
    print(f"# baseline holdout: {payload['baseline']['holdout_accuracy']:.5f}")
    for rank, row in enumerate(payload["promoted"], start=1):
        config = row["config"]
        print(f"\n# --- disc2_r{rank}: {row['description']}")
        print(f"#     holdout {row['holdout_accuracy']:.5f} per-task {row['holdout_per_task']}")
        for suffix, pscale in (("", False), ("_pscale", True)):
            hp = hp_dict(config, pscale)
            print(f"(\n    \"disc2_r{rank}{suffix}\",")
            print("    {")
            for key, value in hp.items():
                print(f"        \"{key}\": {value!r},")
            print("    },\n),")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
