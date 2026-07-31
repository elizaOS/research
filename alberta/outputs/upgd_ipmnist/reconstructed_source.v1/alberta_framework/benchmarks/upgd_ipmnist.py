"""Protocol-exact UPGD Input-permuted MNIST benchmark (Elsayed & Mahmood, ICLR 2024).

This lane reproduces the *online Input-permuted MNIST* protocol of

    Elsayed, M. & Mahmood, A. R. (2024). Addressing Loss of Plasticity and
    Catastrophic Forgetting in Continual Learning. ICLR 2024.
    https://openreview.net/forum?id=sKPzAXoylB

exactly as released in the authors' MIT-licensed repository
(https://github.com/mohmdelsayed/upgd, audited at commit
``b75e90ad4b09c28971ac9dbb902a8fd86709b28c`` -- the same commit profiled by
:mod:`alberta_framework.core.canonical_upgd`):

- 1,000,000 MNIST training examples presented one per step (batch size 1),
  drawn from the canonical 60,000-example train split; each 5,000-step task
  reshuffles the split and takes the first 5,000 examples without replacement
  (``core/task/input_permuted_mnist.py`` recreates its ``DataLoader`` with
  ``shuffle=True`` at every permutation).
- A fresh input-pixel permutation every 5,000 steps => 200 tasks. The first
  task is itself permuted (upstream calls ``permute()`` at step 0).
- Inputs scaled to ``[-1, 1]`` (torchvision ``ToTensor`` then
  ``Normalize((0.5,), (0.5,))``).
- Network: 784 -> 300 -> ReLU -> 150 -> ReLU -> 10 MLP
  (``core/network/fcn_relu.py`` ``FullyConnectedReLU``), PyTorch default
  ``nn.Linear`` init: weights and biases ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
- Loss: 10-class softmax cross-entropy on the single example.
- Metric: online accuracy of the prediction made *before* each update,
  averaged per task; headline number = mean over tasks (= mean over steps
  because tasks have equal length here). Plasticity per step is
  ``clip(1 - loss_after_update / max(loss, 1e-8), 0, 1)`` on the same example
  (``core/run/run_stats.py``).
- Learners (best published hyperparameters, from
  ``experiments/statistics_input_permuted_mnist.py`` at the audited commit):

  * **UPGD-W** = ``FirstOrderGlobalUPGDLearner`` -- utility-gated perturbed
    SGD with decoupled weight decay. Implemented here by
    :class:`~alberta_framework.core.canonical_upgd.CanonicalUPGD` with
    ``profile="official_experiment_global"``, ``mode="protecting"``, which is
    equation-exact against the released ``FirstOrderGlobalUPGD`` optimizer.
    Best config: ``lr=0.01, beta_utility=0.9999, sigma=0.1,
    weight_decay=0.01``.
  * **AdamW** = the released ``core/optim/adam.py`` Adam, which applies
    decoupled weight decay (``p.data.add_(p.data, alpha=-wd*lr)`` before the
    Adam step). Implemented by
    :class:`~alberta_framework.core.baseline_optimizers.Adam` with
    ``weight_decay``. Best config: ``lr=1e-4, beta1=0.0, beta2=0.99,
    eps=1e-8, weight_decay=0.0`` (the paper's grid selected zero decay, so
    the published "Adam" baseline *is* AdamW at wd=0).

Documented deviations from the released code (none change the protocol):

- Upstream draws each task's pixel permutation from an *unseeded*
  ``numpy.random.default_rng()``, so its permutation sequence is not
  reproducible. Here every stream (init, permutations, per-task example
  draws, perturbation noise) derives from the run seed via ``jr.key`` /
  ``jr.fold_in``.
- Upstream's per-task logging buffer flushes at ``i % 5000 == 0`` *after*
  step ``i`` runs, so each logged block covers steps ``[5000t+1, 5000(t+1)]``
  -- shifted one step past the permutation boundary -- and the final 4,999
  steps are dropped. This lane uses exact blocks ``[5000t, 5000(t+1))``
  aligned with the permutation schedule; the difference is 1 step in 5,000.
- Bias corrections are computed in float32 (upstream mixes float64 Python
  scalars into float32 tensors).
- The inner loop uses :func:`lean_upgd_w_update`, a scan-optimized
  restatement of the ``CanonicalUPGD`` ``official_experiment_global``
  protecting profile (the canonical transform's finiteness/mask bookkeeping
  triples CPU step time on this all-finite protocol). A supplied-noise parity
  unit test pins it to the canonical implementation exactly. Perturbations
  are drawn as one flat ``N(0, sigma^2)`` vector per step and sliced per
  parameter -- same distribution as upstream's per-tensor ``randn_like``.

MNIST arrives through the same OpenML plumbing the step2 runners use
(``sklearn.datasets.fetch_openml("mnist_784", version=1)``); the loader
reuses the existing step2 cache when present instead of re-downloading. The
first 60,000 OpenML rows are the canonical torchvision train split.

Run via::

    python -m alberta_framework.benchmarks.upgd_ipmnist --seeds 10 \
        --output outputs/upgd_ipmnist/results.v1.json

Benchmark executions happen through this CLI, never inside pytest.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.baseline_optimizers import Adam
from alberta_framework.core.canonical_upgd import CanonicalUPGD, CanonicalUPGDConfig

logger = logging.getLogger(__name__)

IPMNISTLearner = Literal["upgd_w", "adamw"]

#: Best published hyperparameters for the statistics (headline-figure) runs,
#: from ``experiments/statistics_input_permuted_mnist.py`` at commit
#: ``b75e90ad4b09c28971ac9dbb902a8fd86709b28c``.
UPGD_W_PROTOCOL_HYPERPARAMETERS: dict[str, float] = {
    "step_size": 0.01,
    "utility_decay": 0.9999,
    "noise_std": 0.1,
    "weight_decay": 0.01,
}
ADAMW_PROTOCOL_HYPERPARAMETERS: dict[str, float] = {
    "step_size": 1e-4,
    "beta1": 0.0,
    "beta2": 0.99,
    "eps": 1e-8,
    "weight_decay": 0.0,
}

#: Published reference numbers this lane compares against. The accuracies are
#: approximate read-offs from the paper's Input-permuted MNIST online-accuracy
#: figure (the paper reports curves, not a table); the hyperparameters and
#: protocol are exact from the released code. The paper averages 20 seeds.
PAPER_REFERENCE: dict[str, Any] = {
    "citation": (
        "Elsayed & Mahmood (2024). Addressing Loss of Plasticity and "
        "Catastrophic Forgetting in Continual Learning. ICLR 2024."
    ),
    "openreview": "https://openreview.net/forum?id=sKPzAXoylB",
    "official_repository": "https://github.com/mohmdelsayed/upgd",
    "official_commit": "b75e90ad4b09c28971ac9dbb902a8fd86709b28c",
    "protocol_files": [
        "core/task/input_permuted_mnist.py",
        "core/network/fcn_relu.py",
        "core/run/run_stats.py",
        "experiments/statistics_input_permuted_mnist.py",
        "core/optim/weight_upgd/first_order.py",
        "core/optim/adam.py",
    ],
    "n_seeds": 20,
    "approximate_average_online_accuracy": {
        "upgd_w": 0.78,
        "adamw": 0.68,
        "s_ewc_s_si_s_mas": [0.70, 0.72],
    },
    "qualitative": (
        "UPGD-W holds ~0.78 flat across all 200 tasks; Adam(W) decays toward "
        "~0.68; streaming EWC/SI/MAS land between them."
    ),
    "reference_kind": "figure_readoff",
}

#: Absolute reproduction-gap threshold beyond which a run must be flagged for
#: investigation rather than reported as a clean reproduction.
REPRODUCTION_GAP_THRESHOLD = 0.02

_PLASTICITY_LOSS_FLOOR = 1e-8


@dataclass(frozen=True)
class IPMNISTConfig:
    """Input-permuted MNIST protocol configuration.

    Defaults are the published ICLR-2024 protocol; tests shrink every field.

    Args:
        n_tasks: Number of permutation tasks (published: 200).
        task_length: Online steps per task (published: 5,000).
        input_dim: Flattened input dimensionality (published: 784).
        hidden1: First hidden layer width (published: 300).
        hidden2: Second hidden layer width (published: 150).
        n_classes: Output classes (published: 10).
    """

    n_tasks: int = 200
    task_length: int = 5000
    input_dim: int = 784
    hidden1: int = 300
    hidden2: int = 150
    n_classes: int = 10

    def __post_init__(self) -> None:
        for name in ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def n_steps(self) -> int:
        """Total online steps (examples) in a run."""
        return self.n_tasks * self.task_length

    @property
    def is_protocol_exact(self) -> bool:
        """Whether this configuration is the published ICLR-2024 protocol."""
        return self == IPMNISTConfig()

    def to_config(self) -> dict[str, int]:
        """Return a JSON-serializable configuration."""
        return {
            "n_tasks": self.n_tasks,
            "task_length": self.task_length,
            "input_dim": self.input_dim,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "n_classes": self.n_classes,
        }


@chex.dataclass(frozen=True)
class IPMNISTSchedule:
    """Per-seed permutation and example schedule.

    Attributes:
        permutations: ``(n_tasks, input_dim)`` int32; row ``t`` is the pixel
            permutation active for every step of task ``t``.
        example_indices: ``(n_tasks, task_length)`` int32; row ``t`` holds the
            dataset rows presented during task ``t``, drawn without
            replacement from a fresh per-task shuffle of the train split.
    """

    permutations: Array
    example_indices: Array


def task_index_for_step(step: Array | int, task_length: int) -> Array | int:
    """Return the task index active at ``step`` (permutes every ``task_length``)."""
    return step // task_length


def build_schedule(key: Array, config: IPMNISTConfig, n_train: int) -> IPMNISTSchedule:
    """Build the deterministic permutation/example schedule for one seed.

    Args:
        key: Per-seed schedule key.
        config: Protocol configuration.
        n_train: Number of rows in the train split; must be at least
            ``config.task_length`` so tasks sample without replacement.

    Returns:
        The schedule; task ``t`` is exactly steps
        ``[t * task_length, (t + 1) * task_length)``.
    """
    if n_train < config.task_length:
        raise ValueError(
            f"n_train={n_train} is smaller than task_length={config.task_length}; "
            "per-task sampling is without replacement"
        )
    key_perm, key_sample = jr.split(key)
    tasks = jnp.arange(config.n_tasks)
    permutations = jax.vmap(
        lambda task: jr.permutation(jr.fold_in(key_perm, task), config.input_dim)
    )(tasks)
    example_indices = jax.vmap(
        lambda task: jr.permutation(jr.fold_in(key_sample, task), n_train)[: config.task_length]
    )(tasks)
    return IPMNISTSchedule(  # type: ignore[call-arg]
        permutations=permutations.astype(jnp.int32),
        example_indices=example_indices.astype(jnp.int32),
    )


def init_mlp_params(key: Array, config: IPMNISTConfig) -> dict[str, Array]:
    """Initialize the 300x150 ReLU MLP with PyTorch-default Linear init.

    ``nn.Linear.reset_parameters`` draws weights from
    ``kaiming_uniform_(a=sqrt(5))`` = U(-1/sqrt(fan_in), +1/sqrt(fan_in)) and
    biases from the same interval.
    """
    sizes = [
        (config.input_dim, config.hidden1),
        (config.hidden1, config.hidden2),
        (config.hidden2, config.n_classes),
    ]
    keys = jr.split(key, 2 * len(sizes))
    params: dict[str, Array] = {}
    for index, (fan_in, fan_out) in enumerate(sizes):
        bound = 1.0 / math.sqrt(fan_in)
        params[f"w{index + 1}"] = jr.uniform(
            keys[2 * index], (fan_in, fan_out), jnp.float32, -bound, bound
        )
        params[f"b{index + 1}"] = jr.uniform(
            keys[2 * index + 1], (fan_out,), jnp.float32, -bound, bound
        )
    return params


def mlp_logits(params: dict[str, Array], x: Array) -> Array:
    """Forward pass of the protocol MLP for a single flattened example."""
    hidden = jax.nn.relu(x @ params["w1"] + params["b1"])
    hidden = jax.nn.relu(hidden @ params["w2"] + params["b2"])
    return hidden @ params["w3"] + params["b3"]


def cross_entropy_loss(
    params: dict[str, Array], x: Array, y: Array
) -> tuple[Array, Array]:
    """Softmax cross-entropy of one example; returns ``(loss, logits)``."""
    logits = mlp_logits(params, x)
    return -jax.nn.log_softmax(logits)[y], logits


LearnerInitFn = Callable[[dict[str, Array]], Any]
LearnerStepFn = Callable[
    [dict[str, Array], Any, dict[str, Array], Array],
    tuple[dict[str, Array], Any],
]


@chex.dataclass(frozen=True)
class LeanUPGDState:
    """Carry state of the lean UPGD-W step: raw utility EMA + global clock."""

    utility: dict[str, Array]
    step: Array


def canonical_upgd_w(hp: dict[str, float]) -> CanonicalUPGD:
    """The audited reference optimizer the lean step must match exactly."""
    return CanonicalUPGD(
        CanonicalUPGDConfig(
            step_size=hp["step_size"],
            utility_decay=hp["utility_decay"],
            noise_std=hp["noise_std"],
            weight_decay=hp["weight_decay"],
            mode="protecting",
            profile="official_experiment_global",
            normalization=None,
        )
    )


def lean_upgd_w_update(
    params: dict[str, Array],
    state: LeanUPGDState,
    grads: dict[str, Array],
    noise: dict[str, Array],
    hp: dict[str, float],
) -> tuple[dict[str, Array], LeanUPGDState]:
    """One UPGD-W step, equation-identical to ``FirstOrderGlobalUPGD``.

    This is a scan-optimized restatement of
    ``CanonicalUPGD(profile="official_experiment_global", mode="protecting")``
    for the all-finite, no-mask benchmark inner loop (the canonical transform
    spends most of its time on finiteness/mask bookkeeping this protocol
    never exercises). ``tests/test_upgd_ipmnist.py`` holds a supplied-noise
    parity test asserting exact agreement with the canonical implementation;
    any equation change must keep that test passing.

    ``noise`` is the already-scaled perturbation ``xi ~ N(0, sigma^2)``.
    """
    beta = hp["utility_decay"]
    step_size = hp["step_size"]
    decay = 1.0 - step_size * hp["weight_decay"]
    count = state.step + jnp.array(1, dtype=jnp.int32)
    utility = {
        name: beta * state.utility[name] + (1.0 - beta) * (-grads[name] * params[name])
        for name in params
    }
    global_max = jnp.max(
        jnp.stack([jnp.max(utility[name]) for name in sorted(params)])
    )
    bias_correction = 1.0 - jnp.power(
        jnp.asarray(beta, dtype=jnp.float32), count.astype(jnp.float32)
    )
    new_params = {}
    for name in params:
        gate = jax.nn.sigmoid((utility[name] / bias_correction) / global_max)
        new_params[name] = params[name] * decay - step_size * (
            (grads[name] + noise[name]) * (1.0 - gate)
        )
    return new_params, LeanUPGDState(utility=utility, step=count)  # type: ignore[call-arg]


def _make_upgd_w_learner(hp: dict[str, float]) -> tuple[LearnerInitFn, LearnerStepFn]:
    """UPGD-W (``FirstOrderGlobalUPGD``) via the lean parity-tested step."""
    noise_std = hp["noise_std"]

    def init_fn(params: dict[str, Array]) -> LeanUPGDState:
        return LeanUPGDState(  # type: ignore[call-arg]
            utility={name: jnp.zeros_like(value) for name, value in params.items()},
            step=jnp.array(0, dtype=jnp.int32),
        )

    def step_fn(
        params: dict[str, Array], state: LeanUPGDState, grads: dict[str, Array], key: Array
    ) -> tuple[dict[str, Array], LeanUPGDState]:
        names = sorted(params)
        shapes = [params[name].shape for name in names]
        counts = [int(np.prod(shape)) for shape in shapes]
        flat = jr.normal(key, (sum(counts),), jnp.float32) * noise_std
        chunks = jnp.split(flat, np.cumsum(counts)[:-1])
        noise = {
            name: chunk.reshape(shape)
            for name, chunk, shape in zip(names, chunks, shapes, strict=True)
        }
        return lean_upgd_w_update(params, state, grads, noise, hp)

    return init_fn, step_fn


def _make_adamw_learner(hp: dict[str, float]) -> tuple[LearnerInitFn, LearnerStepFn]:
    """AdamW via the baseline Adam with decoupled weight decay."""
    optimizer = Adam(
        step_size=hp["step_size"],
        beta1=hp["beta1"],
        beta2=hp["beta2"],
        eps=hp["eps"],
        weight_decay=hp["weight_decay"],
    )

    def init_fn(params: dict[str, Array]) -> dict[str, Any]:
        return {name: optimizer.init_for_shape(value.shape) for name, value in params.items()}

    def step_fn(
        params: dict[str, Array], state: dict[str, Any], grads: dict[str, Array], key: Array
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        del key  # AdamW is deterministic
        new_params: dict[str, Array] = {}
        new_state: dict[str, Any] = {}
        for name, value in params.items():
            step_arr, leaf_state = optimizer.update_from_gradient(
                state[name], grads[name], error=None, param=value
            )
            new_params[name] = value - step_arr
            new_state[name] = leaf_state
        return new_params, new_state

    return init_fn, step_fn


_LEARNER_FACTORIES: dict[str, Callable[[dict[str, float]], tuple[LearnerInitFn, LearnerStepFn]]] = {
    "upgd_w": _make_upgd_w_learner,
    "adamw": _make_adamw_learner,
}
_LEARNER_DEFAULT_HYPERPARAMETERS: dict[str, dict[str, float]] = {
    "upgd_w": UPGD_W_PROTOCOL_HYPERPARAMETERS,
    "adamw": ADAMW_PROTOCOL_HYPERPARAMETERS,
}


@dataclass(frozen=True)
class IPMNISTRunResult:
    """Host-side result of one learner's multi-seed run.

    Per-task arrays have shape ``(n_seeds, n_tasks)``. ``per_step_accuracy``
    (``(n_seeds, n_tasks, task_length)``), ``initial_params``,
    ``permutations``, and ``example_indices`` are populated only when the run
    was invoked with ``return_per_step=True`` (debug/testing scale).
    """

    learner: str
    hyperparameters: dict[str, float]
    seeds: tuple[int, ...]
    config: IPMNISTConfig
    per_task_accuracy: np.ndarray
    per_task_loss: np.ndarray
    per_task_plasticity: np.ndarray
    average_online_accuracy: np.ndarray
    wall_clock_seconds: float
    per_step_accuracy: np.ndarray | None = None
    initial_params: dict[str, np.ndarray] | None = None
    permutations: np.ndarray | None = None
    example_indices: np.ndarray | None = None


def resolve_hyperparameters(
    learner: str, overrides: dict[str, float] | None = None
) -> dict[str, float]:
    """Merge overrides into the learner's published defaults, rejecting unknown keys."""
    if learner not in _LEARNER_DEFAULT_HYPERPARAMETERS:
        raise ValueError(f"unknown learner {learner!r}; expected one of "
                         f"{sorted(_LEARNER_DEFAULT_HYPERPARAMETERS)}")
    merged = dict(_LEARNER_DEFAULT_HYPERPARAMETERS[learner])
    if overrides:
        unknown = set(overrides) - set(merged)
        if unknown:
            raise ValueError(f"unknown hyperparameters for {learner}: {sorted(unknown)}")
        merged.update({name: float(value) for name, value in overrides.items()})
    return merged


def run_ipmnist(
    data_x: np.ndarray | Array,
    data_y: np.ndarray | Array,
    learner: IPMNISTLearner,
    seeds: Sequence[int],
    config: IPMNISTConfig | None = None,
    hyperparameters: dict[str, float] | None = None,
    return_per_step: bool = False,
    progress_every: int | None = None,
) -> IPMNISTRunResult:
    """Run the online Input-permuted MNIST protocol for one learner.

    Args:
        data_x: ``(n_train, input_dim)`` float32 inputs, already normalized
            to ``[-1, 1]`` (see :func:`load_mnist_train`).
        data_y: ``(n_train,)`` integer class labels.
        learner: ``"upgd_w"`` or ``"adamw"``.
        seeds: Run seeds; all seeds execute in parallel under ``vmap``.
        config: Protocol configuration (defaults to the published protocol).
        hyperparameters: Optional overrides of the published defaults.
        return_per_step: Also return per-step accuracies plus the initial
            parameters and schedules (debug/testing scale only -- the full
            protocol would materialize ``n_seeds x 1e6`` floats).
        progress_every: Log progress every N tasks (None = silent).

    Returns:
        Host-side result arrays; see :class:`IPMNISTRunResult`.
    """
    if config is None:
        config = IPMNISTConfig()
    hp = resolve_hyperparameters(learner, hyperparameters)
    init_fn, step_fn = _LEARNER_FACTORIES[learner](hp)

    data_x = jnp.asarray(data_x, dtype=jnp.float32)
    data_y = jnp.asarray(data_y, dtype=jnp.int32)
    if data_x.ndim != 2 or data_x.shape[1] != config.input_dim:
        raise ValueError(
            f"data_x must have shape (n_train, {config.input_dim}), got {data_x.shape}"
        )
    if data_y.shape != (data_x.shape[0],):
        raise ValueError("data_y must be (n_train,) aligned with data_x")
    n_train = int(data_x.shape[0])
    if n_train < config.task_length:
        raise ValueError("dataset smaller than task_length; cannot sample without replacement")

    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError("at least one seed is required")
    seeds_array = jnp.asarray(seed_tuple, dtype=jnp.uint32)

    def init_seed(seed: Array) -> tuple[dict[str, Array], Any, IPMNISTSchedule, Array]:
        root = jr.key(seed)
        key_init, key_schedule, key_noise = jr.split(root, 3)
        params = init_mlp_params(key_init, config)
        return params, init_fn(params), build_schedule(key_schedule, config, n_train), key_noise

    params, opt_state, schedules, noise_keys = jax.jit(jax.vmap(init_seed))(seeds_array)

    initial_params_host: dict[str, np.ndarray] | None = None
    if return_per_step:
        initial_params_host = {name: np.asarray(value) for name, value in params.items()}

    def run_task(
        params: dict[str, Array],
        opt_state: Any,
        noise_key: Array,
        permutation: Array,
        examples: Array,
    ) -> tuple[dict[str, Array], Any, Array, Array, Array, Array]:
        def one_step(
            carry: tuple[dict[str, Array], Any, Array], example: Array
        ) -> tuple[tuple[dict[str, Array], Any, Array], tuple[Array, Array, Array]]:
            step_params, step_state, key = carry
            x = data_x[example][permutation]
            y = data_y[example]
            (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
                step_params, x, y
            )
            accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
            key, step_key = jr.split(key)
            new_params, new_state = step_fn(step_params, step_state, grads, step_key)
            loss_after, _ = cross_entropy_loss(new_params, x, y)
            plasticity = jnp.clip(
                1.0 - loss_after / jnp.maximum(loss, _PLASTICITY_LOSS_FLOOR), 0.0, 1.0
            )
            return (new_params, new_state, key), (accuracy, loss, plasticity)

        (params, opt_state, noise_key), (accuracies, losses, plasticities) = jax.lax.scan(
            one_step, (params, opt_state, noise_key), examples
        )
        return params, opt_state, noise_key, accuracies, losses, plasticities

    run_task_batched = jax.jit(jax.vmap(run_task))

    task_accuracy: list[np.ndarray] = []
    task_loss: list[np.ndarray] = []
    task_plasticity: list[np.ndarray] = []
    step_accuracy: list[np.ndarray] = []
    started = time.monotonic()
    for task in range(config.n_tasks):
        params, opt_state, noise_keys, accuracies, losses, plasticities = run_task_batched(
            params,
            opt_state,
            noise_keys,
            schedules.permutations[:, task],
            schedules.example_indices[:, task],
        )
        task_accuracy.append(np.asarray(accuracies.mean(axis=1)))
        task_loss.append(np.asarray(losses.mean(axis=1)))
        task_plasticity.append(np.asarray(plasticities.mean(axis=1)))
        if return_per_step:
            step_accuracy.append(np.asarray(accuracies))
        if progress_every is not None and (task + 1) % progress_every == 0:
            elapsed = time.monotonic() - started
            logger.info(
                "%s task %d/%d online_acc=%.4f elapsed=%.1fs",
                learner,
                task + 1,
                config.n_tasks,
                float(task_accuracy[-1].mean()),
                elapsed,
            )

    per_task_accuracy = np.stack(task_accuracy, axis=1)
    per_task_loss = np.stack(task_loss, axis=1)
    per_task_plasticity = np.stack(task_plasticity, axis=1)
    return IPMNISTRunResult(
        learner=learner,
        hyperparameters=hp,
        seeds=seed_tuple,
        config=config,
        per_task_accuracy=per_task_accuracy,
        per_task_loss=per_task_loss,
        per_task_plasticity=per_task_plasticity,
        average_online_accuracy=per_task_accuracy.mean(axis=1),
        wall_clock_seconds=time.monotonic() - started,
        per_step_accuracy=np.stack(step_accuracy, axis=1) if return_per_step else None,
        initial_params=initial_params_host,
        permutations=np.asarray(schedules.permutations) if return_per_step else None,
        example_indices=np.asarray(schedules.example_indices) if return_per_step else None,
    )


# =============================================================================
# MNIST loading (same OpenML plumbing as the step2 runners)
# =============================================================================

_STEP2_CACHE_NAME = "step2_published_mnist_openml_cache"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_openml_data_home() -> Path:
    """Return the ``fetch_openml`` ``data_home`` to use.

    Preference order: an existing step2 MNIST cache in this repository's
    ``outputs/``; an existing step2 cache in a sibling full upstream clone
    (``../../alberta/outputs/``); otherwise a fresh cache under
    ``outputs/upgd_ipmnist/openml_cache`` (triggers one download).
    """
    root = _repo_root()
    candidates = (
        root / "outputs" / _STEP2_CACHE_NAME,
        root.parent.parent / "alberta" / "outputs" / _STEP2_CACHE_NAME,
    )
    for candidate in candidates:
        if (candidate / "openml").is_dir():
            return candidate
    return root / "outputs" / "upgd_ipmnist" / "openml_cache"


def load_mnist_train(data_home: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load the canonical 60,000-example MNIST train split, scaled to ``[-1, 1]``.

    Uses ``sklearn.datasets.fetch_openml("mnist_784", version=1)`` -- the same
    plumbing as the step2 runners -- whose first 60,000 rows are the
    torchvision train split the published protocol uses. The transform
    matches ``ToTensor`` + ``Normalize((0.5,), (0.5,))``.
    """
    try:
        from sklearn.datasets import fetch_openml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scikit-learn is required to load OpenML MNIST") from exc

    home = data_home if data_home is not None else default_openml_data_home()
    raw = fetch_openml(
        "mnist_784", version=1, as_frame=False, data_home=str(home), n_retries=3, delay=2.0
    )
    x = np.asarray(raw.data, dtype=np.float32)[:60_000]
    y = np.asarray(raw.target, dtype=np.int32)[:60_000]
    x = (x / 255.0 - 0.5) / 0.5
    return x, y


# =============================================================================
# Summaries, comparison, artifact
# =============================================================================


def summarize_result(result: IPMNISTRunResult) -> dict[str, Any]:
    """Reduce one learner's run to JSON-serializable summary statistics."""
    accuracy = result.average_online_accuracy
    n_seeds = accuracy.shape[0]
    quarter = max(1, result.config.n_tasks // 4)
    per_seed_first = result.per_task_accuracy[:, :quarter].mean(axis=1)
    per_seed_last = result.per_task_accuracy[:, -quarter:].mean(axis=1)

    def _stderr(values: np.ndarray) -> float:
        if values.shape[0] < 2:
            return 0.0
        return float(values.std(ddof=1) / math.sqrt(values.shape[0]))

    return {
        "learner": result.learner,
        "hyperparameters": result.hyperparameters,
        "seeds": list(result.seeds),
        "n_seeds": n_seeds,
        "average_online_accuracy_mean": float(accuracy.mean()),
        "average_online_accuracy_stderr": _stderr(accuracy),
        "per_seed_average_online_accuracy": [round(float(v), 6) for v in accuracy],
        "first_quarter_accuracy_mean": float(per_seed_first.mean()),
        "last_quarter_accuracy_mean": float(per_seed_last.mean()),
        "accuracy_drift_last_minus_first": float(per_seed_last.mean() - per_seed_first.mean()),
        "average_plasticity_mean": float(result.per_task_plasticity.mean()),
        "per_task_accuracy_mean": [
            round(float(v), 6) for v in result.per_task_accuracy.mean(axis=0)
        ],
        "per_task_accuracy_stderr": [
            round(_stderr(result.per_task_accuracy[:, t]), 6)
            for t in range(result.config.n_tasks)
        ],
        "per_task_plasticity_mean": [
            round(float(v), 6) for v in result.per_task_plasticity.mean(axis=0)
        ],
        "per_seed_per_task_accuracy": [
            [round(float(v), 6) for v in row] for row in result.per_task_accuracy
        ],
        "wall_clock_seconds": round(result.wall_clock_seconds, 2),
    }


def build_comparison(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare run summaries against the published reference numbers.

    Any absolute gap above :data:`REPRODUCTION_GAP_THRESHOLD` is flagged as a
    reproduction gap to investigate -- never silently absorbed.
    """
    reference = PAPER_REFERENCE["approximate_average_online_accuracy"]
    comparison: dict[str, Any] = {
        "reference": PAPER_REFERENCE,
        "gap_threshold": REPRODUCTION_GAP_THRESHOLD,
        "learners": {},
    }
    for learner, summary in summaries.items():
        if learner not in reference:
            continue
        ours = summary["average_online_accuracy_mean"]
        published = float(reference[learner])
        gap = ours - published
        comparison["learners"][learner] = {
            "ours": round(ours, 6),
            "published_approximate": published,
            "gap": round(gap, 6),
            "reproduction_gap_flagged": bool(abs(gap) > REPRODUCTION_GAP_THRESHOLD),
        }
    if "upgd_w" in summaries and "adamw" in summaries:
        comparison["upgd_w_beats_adamw"] = bool(
            summaries["upgd_w"]["average_online_accuracy_mean"]
            > summaries["adamw"]["average_online_accuracy_mean"]
        )
    return comparison


def build_artifact(
    results: dict[str, IPMNISTRunResult],
    config: IPMNISTConfig,
    data_home: Path,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble the artifact JSON payload for a set of learner runs."""
    summaries = {learner: summarize_result(result) for learner, result in results.items()}
    return {
        "benchmark": "upgd_input_permuted_mnist",
        "schema_version": 1,
        "created_unix": time.time(),
        "protocol": {
            **config.to_config(),
            "n_steps": config.n_steps,
            "is_protocol_exact": config.is_protocol_exact,
            "dataset": "OpenML mnist_784 v1, first 60000 rows (torchvision train split)",
            "input_scaling": "(x/255 - 0.5) / 0.5",
            "loss": "softmax cross-entropy, one example per step",
            "metric": "online accuracy of the pre-update prediction, averaged per task",
            "plasticity": "clip(1 - loss_after/max(loss, 1e-8), 0, 1) per step",
        },
        "provenance": {
            "openml_data_home": str(data_home),
            "deviations": [
                "seeded permutation/example streams (upstream permutations are unseeded)",
                "exact task blocks [t*L, (t+1)*L) (upstream logging is shifted one step)",
                "float32 bias corrections (upstream mixes float64 scalars)",
            ],
        },
        "learners": summaries,
        "comparison": build_comparison(summaries),
        "environment": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(jax.devices()[0]),
        },
        "notes": list(notes),
    }


# =============================================================================
# Partial (per-seed shard) results -- one process per seed parallelism
# =============================================================================

PARTIAL_SCHEMA = "upgd_ipmnist.partial.v1"


def partial_payload(result: IPMNISTRunResult) -> dict[str, Any]:
    """Serialize one shard's run so shards can be merged later."""
    return {
        "schema": PARTIAL_SCHEMA,
        "learner": result.learner,
        "hyperparameters": result.hyperparameters,
        "seeds": list(result.seeds),
        "config": result.config.to_config(),
        "per_task_accuracy": result.per_task_accuracy.tolist(),
        "per_task_loss": result.per_task_loss.tolist(),
        "per_task_plasticity": result.per_task_plasticity.tolist(),
        "wall_clock_seconds": result.wall_clock_seconds,
    }


def merge_partial_results(paths: Sequence[Path]) -> dict[str, IPMNISTRunResult]:
    """Merge per-seed shard JSONs into one ``IPMNISTRunResult`` per learner.

    All shards of a learner must share config and hyperparameters; seeds must
    not repeat. Rows are ordered by seed.
    """
    if not paths:
        raise ValueError("no partial result files given")
    by_learner: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != PARTIAL_SCHEMA:
            raise ValueError(f"{path} is not a {PARTIAL_SCHEMA} payload")
        by_learner.setdefault(payload["learner"], []).append(payload)

    merged: dict[str, IPMNISTRunResult] = {}
    for learner, shards in by_learner.items():
        reference = shards[0]
        for shard in shards[1:]:
            if shard["config"] != reference["config"]:
                raise ValueError(f"{learner}: shards disagree on config")
            if shard["hyperparameters"] != reference["hyperparameters"]:
                raise ValueError(f"{learner}: shards disagree on hyperparameters")
        rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        for shard in shards:
            accuracy = np.asarray(shard["per_task_accuracy"], dtype=np.float64)
            loss = np.asarray(shard["per_task_loss"], dtype=np.float64)
            plasticity = np.asarray(shard["per_task_plasticity"], dtype=np.float64)
            for row_index, seed in enumerate(shard["seeds"]):
                rows.append(
                    (int(seed), accuracy[row_index], loss[row_index], plasticity[row_index])
                )
        seeds_seen = [row[0] for row in rows]
        if len(set(seeds_seen)) != len(seeds_seen):
            raise ValueError(f"{learner}: duplicate seeds across shards: {sorted(seeds_seen)}")
        rows.sort(key=lambda row: row[0])
        per_task_accuracy = np.stack([row[1] for row in rows])
        merged[learner] = IPMNISTRunResult(
            learner=learner,
            hyperparameters=dict(reference["hyperparameters"]),
            seeds=tuple(row[0] for row in rows),
            config=IPMNISTConfig(**reference["config"]),
            per_task_accuracy=per_task_accuracy,
            per_task_loss=np.stack([row[2] for row in rows]),
            per_task_plasticity=np.stack([row[3] for row in rows]),
            average_online_accuracy=per_task_accuracy.mean(axis=1),
            wall_clock_seconds=float(sum(shard["wall_clock_seconds"] for shard in shards)),
        )
    return merged


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the protocol-exact benchmark run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--learners", default="upgd_w,adamw",
                        help="comma-separated subset of: upgd_w, adamw")
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed value")
    parser.add_argument("--seed-list", default=None,
                        help="explicit comma-separated seeds (overrides --seeds/--seed-start)")
    parser.add_argument("--n-tasks", type=int, default=200)
    parser.add_argument("--task-length", type=int, default=5000)
    parser.add_argument("--data-home", type=Path, default=None,
                        help="fetch_openml data_home (defaults to the step2 cache when present)")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/upgd_ipmnist/results.v1.json"))
    parser.add_argument("--partial-out", type=Path, default=None,
                        help="write a mergeable per-seed shard JSON here instead of an artifact")
    parser.add_argument("--merge-partials", type=Path, nargs="+", default=None,
                        help="skip running; merge these shard JSONs into --output")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--note", action="append", default=[],
                        help="free-form provenance note recorded in the artifact")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )

    config = IPMNISTConfig(n_tasks=args.n_tasks, task_length=args.task_length)
    data_home = args.data_home if args.data_home is not None else default_openml_data_home()

    if args.merge_partials is not None:
        results = merge_partial_results(args.merge_partials)
        merged_configs = {result.config for result in results.values()}
        if len(merged_configs) != 1:
            raise SystemExit("shards span multiple protocol configs; merge them separately")
        artifact = build_artifact(results, merged_configs.pop(), data_home, notes=args.note)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        logger.info("merged %d shards -> %s", len(args.merge_partials), args.output)
        return

    learners = [name.strip() for name in args.learners.split(",") if name.strip()]
    for name in learners:
        if name not in _LEARNER_FACTORIES:
            raise SystemExit(f"unknown learner {name!r}")

    logger.info("loading MNIST from data_home=%s", data_home)
    data_x, data_y = load_mnist_train(data_home)
    logger.info("train split: x=%s y=%s", data_x.shape, data_y.shape)

    if args.seed_list is not None:
        seeds = [int(part) for part in args.seed_list.split(",") if part.strip()]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    results = {}
    for name in learners:
        logger.info("running %s for %d seeds x %d steps", name, len(seeds), config.n_steps)
        results[name] = run_ipmnist(
            data_x,
            data_y,
            name,
            seeds,
            config=config,
            progress_every=args.progress_every,
        )
        logger.info(
            "%s done: average online accuracy %.4f (wall %.1fs)",
            name,
            float(results[name].average_online_accuracy.mean()),
            results[name].wall_clock_seconds,
        )

    if args.partial_out is not None:
        if len(learners) != 1:
            raise SystemExit("--partial-out expects exactly one learner per shard")
        args.partial_out.parent.mkdir(parents=True, exist_ok=True)
        args.partial_out.write_text(
            json.dumps(partial_payload(results[learners[0]]), indent=2, sort_keys=True) + "\n"
        )
        logger.info("wrote shard %s", args.partial_out)
        return

    artifact = build_artifact(results, config, data_home, notes=args.note)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
