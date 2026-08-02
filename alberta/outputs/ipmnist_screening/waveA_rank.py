"""Rank wave-A update-rule arms vs the sigma0_ndecay099 champion (dev lane)."""

import glob
import json

import numpy as np

ROOT = "outputs/ipmnist_screening"
ARMS = ("colnorm_gate", "muon_gate", "lion_gate")
CHAMPION = "sigma0_ndecay099"


def acc(path: str) -> float:
    return float(np.asarray(json.load(open(path))["per_task_accuracy"], dtype=float).mean())


def main() -> None:
    base = {s: acc(f"{ROOT}/shards/{CHAMPION}_seed{s}.json") for s in (0, 1, 2)}
    out = {
        "evidence_class": "development_screening_diagnostic",
        "champion": CHAMPION,
        "champion_screen": base,
        "arms": {},
    }
    for cfg in ARMS:
        paths = sorted(glob.glob(f"{ROOT}/shards/{cfg}_seed[0-2].json"))
        if len(paths) < 3:
            continue
        vals = [acc(p) for p in paths]
        deltas = [v - base[s] for s, v in enumerate(vals)]
        entry = {
            "screen_mean": sum(vals) / len(vals),
            "per_seed": vals,
            "paired_delta_vs_champion": sum(deltas) / len(deltas),
            "per_seed_delta": deltas,
        }
        confirms = sorted(glob.glob(f"{ROOT}/confirm_full/{cfg}_seed*.json"))
        if confirms:
            cvals = [acc(p) for p in confirms]
            entry["confirm_200task"] = {"n": len(cvals), "mean": sum(cvals) / len(cvals),
                                        "per_seed": cvals}
        out["arms"][cfg] = entry
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
