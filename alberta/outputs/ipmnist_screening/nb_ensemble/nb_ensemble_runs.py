"""Per-step instrumentation for the nb_ensemble transient-attack arms.

Development-only diagnostic (never promotable evidence). Reuses the exact
screening machinery (specs, seed derivation, schedule, per-step RNG chain)
from alberta_framework.benchmarks.ipmnist_screening — protocol permutations,
identical trajectories to the shards — but records PER-STEP online accuracy
(uint8) and, for the ensemble arms, the per-step naive-Bayes vote weight
``softmax(ens_beta * member_acc)[1]`` read from the carried state BEFORE each
update (float16), so within-task switching dynamics can be read directly.

Writes JSON + .npy arrays to outputs/ipmnist_screening/nb_ensemble/
(new, non-pinned directory; the ceiling/ conventions).

Usage:
  .venv/bin/python outputs/ipmnist_screening/nb_ensemble/nb_ensemble_runs.py \
      --arm nb_ensemble_champion --seed 0 --n-tasks 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    NBEnsembleState,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
    load_mnist_train,
)

OUT = Path(__file__).resolve().parent


def run_arm_per_step(
    spec_name: str, seed: int, n_tasks: int, progress_every: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run one registered arm under the protocol, recording per-step traces.

    Returns (per_step_accuracy uint8 [n_tasks, task_length],
             per_step_nb_weight float16 [n_tasks, task_length]  (0 for
             non-ensemble arms),
             per_task_accuracy float64 [n_tasks], wall_seconds).
    """
    config = IPMNISTConfig(n_tasks=n_tasks)
    spec = screening_spec(spec_name)
    ens_beta = float(spec.hyperparameters.get("ens_beta", 0.0))
    is_ensemble = spec_name.startswith("nb_ensemble")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    data_x_np, data_y_np = load_mnist_train(default_openml_data_home())
    data_x = jnp.asarray(data_x_np, dtype=jnp.float32)
    data_y = jnp.asarray(data_y_np, dtype=jnp.int32)
    n_train = int(data_x.shape[0])

    root = jr.key(jnp.uint32(seed))
    key_init, key_schedule, key_noise = jr.split(root, 3)
    params = init_mlp_params(key_init, config)
    schedule = build_schedule(key_schedule, config, n_train)
    state = init_fn(params)

    def nb_weight(step_state: object) -> jnp.ndarray:
        if is_ensemble:
            assert isinstance(step_state, NBEnsembleState)
            return jax.nn.softmax(ens_beta * step_state.member_acc)[1]
        return jnp.float32(0.0)

    def run_task(params, state, key, permutation, examples):
        def one_step(carry, example):
            step_params, step_state, key = carry
            x = data_x[example][permutation]
            y = data_y[example]
            w_nb = nb_weight(step_state)
            key, step_key = jr.split(key)
            new_params, new_state, metrics = step_fn(
                step_params, step_state, x, y, step_key
            )
            accuracy, _, _ = metrics
            return (new_params, new_state, key), (accuracy, w_nb)

        (params, state, key), (accuracies, weights) = jax.lax.scan(
            one_step, (params, state, key), examples
        )
        return params, state, key, accuracies, weights

    run_task_jit = jax.jit(run_task)
    per_step = np.zeros((n_tasks, config.task_length), dtype=np.uint8)
    per_w = np.zeros((n_tasks, config.task_length), dtype=np.float16)
    per_task = np.zeros(n_tasks, dtype=np.float64)
    started = time.monotonic()
    for task in range(n_tasks):
        params, state, key_noise, accuracies, weights = run_task_jit(
            params, state, key_noise, schedule.permutations[task],
            schedule.example_indices[task],
        )
        acc = np.asarray(accuracies)
        per_step[task] = acc.astype(np.uint8)
        per_w[task] = np.asarray(weights).astype(np.float16)
        per_task[task] = float(acc.mean())
        if (task + 1) % progress_every == 0:
            print(
                f"{spec_name} seed={seed} task {task + 1}/{n_tasks} "
                f"acc={per_task[task]:.4f} elapsed={time.monotonic() - started:.0f}s",
                flush=True,
            )
    return per_step, per_w, per_task, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-tasks", type=int, default=60)
    args = parser.parse_args()

    per_step, per_w, per_task, wall = run_arm_per_step(
        args.arm, args.seed, args.n_tasks
    )
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_seed{args.seed}"
    np.save(OUT / f"{tag}_per_step.npy", per_step)
    if args.arm.startswith("nb_ensemble"):
        np.save(OUT / f"{tag}_nb_weight.npy", per_w)
    payload = {
        "schema": "alberta.ipmnist_nb_ensemble.run.v1",
        "evidence_class": "development_screening_diagnostic",
        "development_only": True,
        "arm": args.arm,
        "seed": args.seed,
        "n_tasks": args.n_tasks,
        "task_length": 5000,
        "per_task_accuracy": [round(float(v), 8) for v in per_task],
        "mean_accuracy": round(float(per_task.mean()), 8),
        "wall_clock_seconds": round(wall, 2),
        "jax": jax.__version__,
    }
    (OUT / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    print(f"WROTE {tag} mean={per_task.mean():.5f} wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
