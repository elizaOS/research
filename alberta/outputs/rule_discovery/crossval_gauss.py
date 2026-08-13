"""Cross-suite transfer check for discovered rules (development, nonpromoting).

Evaluates the search's promoted/candidate genomes + the tuned champion-form
baseline on the CANONICAL Gaussian micro suite (micro_continual gauss-v1),
including the ``recurrence`` family — a memory axis the digits search suite
does not contain. Fresh seeds (201-203), never used by search or holdout.

Usage: .venv/bin/python outputs/rule_discovery/crossval_gauss.py \
    outputs/rule_discovery/search_v1.json outputs/rule_discovery/crossval_gauss_v1.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import _atomic_write_json
from alberta_framework.benchmarks.micro_continual import (
    MicroStreamConfig,
    generate_stream,
)
from alberta_framework.benchmarks.rule_discovery import (
    NONPROMOTING_POLICY,
    _batched_run,
    describe_genome,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params

FAMILIES = ("input_permutation", "scale_shift", "recurrence")
SEEDS = (201, 202, 203)
GEOMETRY = dict(n_regimes=12, regime_length=500, dim=32, n_classes=10)


def main() -> int:
    search_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    search = json.loads(search_path.read_text())
    rows = [("tuned_baseline", search["baseline"]["genome"])]
    rows += [
        (f"cand_{row['rank_by_search_fitness']}", row["genome"])
        for row in search["candidates"]
    ]
    genomes = jnp.asarray(np.asarray([g for _, g in rows], dtype=np.float32))
    net = IPMNISTConfig(
        n_tasks=GEOMETRY["n_regimes"],
        task_length=GEOMETRY["regime_length"],
        input_dim=GEOMETRY["dim"],
        hidden1=32,
        hidden2=16,
        n_classes=GEOMETRY["n_classes"],
    )
    results: dict[str, dict[str, float]] = {name: {} for name, _ in rows}
    for family in FAMILIES:
        config = MicroStreamConfig(
            family=family,
            recurrence_pool=4 if family == "recurrence" else 5,
            **GEOMETRY,
        )
        total = np.zeros(len(rows))
        for seed in SEEDS:
            stream = generate_stream(config, seed)
            params = init_mlp_params(jr.key(np.uint32(seed)), net)
            mean_accuracy, _ = _batched_run(
                genomes, params, stream.x, stream.y, config.regime_length
            )
            total += np.asarray(mean_accuracy)
        for index, (name, _) in enumerate(rows):
            results[name][family] = float(total[index] / len(SEEDS))
    payload = {
        "schema": "alberta.rule_discovery.crossval_gauss.v1",
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "source_search": str(search_path),
        "families": list(FAMILIES),
        "seeds": list(SEEDS),
        "geometry": GEOMETRY,
        "rows": [
            {
                "name": name,
                "description": describe_genome(np.asarray(genome, dtype=np.float32)),
                "per_family": results[name],
                "mean": float(np.mean(list(results[name].values()))),
            }
            for name, genome in rows
        ],
    }
    _atomic_write_json(out_path, payload)
    for row in payload["rows"]:
        print(
            f"{row['name']:16s} mean={row['mean']:.4f} "
            + " ".join(f"{k}={v:.4f}" for k, v in row["per_family"].items())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
