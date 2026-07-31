"""In-repo catastrophic-forgetting gate for the UPGD learner.

Streams three class-blocked synthetic regression tasks (distinct tanh ridge
functions over a shared 8-d Gaussian input space, one task head active per
1200-step block) through a UPGD learner and an identical twin with the
utility-gated perturbation disabled (``perturbation_sigma=0``), paired over
24 seeds via ``jax.vmap`` (same init key, same data per seed).

Gate configuration: the Step 2 production trunk/optimizer settings
(one hidden layer, ``step_size=0.03``, ``ObGDBounding(kappa=0.5)``, layer
norm) with the documented ``UPGDLearner`` constructor-default perturbation
mechanism (``perturbation_sigma=1e-3``, Gaussian noise every step,
``sparsity=0.9`` sparse init) and a 64-unit hidden layer. The
:meth:`UPGDLearner.step2_default` factory itself was measured first and its
lean perturbation (``sigma=1e-4``, Rademacher noise every 16 steps) is below
the resolvable effect size for a 3-task gate: paired retention advantage
+0.001..+0.005 MSE (|t| <= 1.7) at 1500-step blocks and sign-unstable at
2000-step blocks. The constructor-default perturbation strength is the
smallest configuration whose retention effect is CI-stable here.

Measured results (seeds 0-23; ranges over seed bases 0/500/4242, 24 seeds
each; loss values are the online ``0.5 * err^2`` metric, retention values
are plain MSE on a held-out task-1 batch; zero-predictor MSE ~= 0.635):

- (a) Retention: after training tasks 1->2->3, retained task-1 MSE is
  0.306-0.312 for UPGD vs 0.349-0.357 for the sigma=0 twin. Paired
  advantage +0.041..+0.045 (se 0.006-0.009, t = +5.3..+7.1), positive for
  92-96% of seeds. Gate: mean advantage > 0.015 and > 75% of seeds positive.
- (b) Adaptation: final-window (last 100 steps) task loss 0.014-0.021 for
  UPGD vs 0.011-0.014 for the twin (ratio <= 1.5; the perturbation noise
  floor), both far below the ~0.3 start-of-task loss. Steps to smoothed
  loss <= 0.05: UPGD within 1.02x-1.39x of the twin, every seed under 480
  steps of the 1200-step block. Gate: per-task final-window mean <= 2x twin
  and < 0.04 absolute; steps-to-criterion mean <= 1.6x twin, max <= 800.
- (c) Recurrence savings: revisiting task 1 after task 3, UPGD reaches
  smoothed loss <= 0.10 in 61-67 steps vs 83-93 steps at first exposure:
  recurrence savings ratio (first / revisit) 1.28-1.52, revisit strictly
  faster for 67-92% of seeds (the smoothing window floors this ratio: a
  first exposure can cross no earlier than step 50). Gate: ratio > 1.08,
  strict mean improvement, >= 50% of seeds strictly faster, every revisit
  seed reaches criterion.

Total module runtime ~15-20 s on CPU (both arms jit+vmap compiled once).
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array
from jaxtyping import Float

from alberta_framework.core.optimizers import ObGDBounding
from alberta_framework.core.upgd import UPGDLearner, run_upgd_arrays

FEATURE_DIM = 8
N_TASKS = 3
BLOCK_STEPS = 1200
N_SEEDS = 24
EVAL_BATCH = 256
SMOOTH_WINDOW = 50
# Smoothed online-loss levels: moderate criterion for within-task adaptation
# speed, and a looser one for the first-exposure vs revisit comparison.
ADAPT_CRITERION = 0.05
RECURRENCE_CRITERION = 0.10

# Fixed unit directions defining the three task target functions
# ``y_k(x) = tanh(2 * u_k . x)`` over the shared input space.
_TASK_DIRECTIONS: Float[Array, "n_tasks feature_dim"] = jnp.stack(
    [
        jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / jnp.sqrt(2.0),
        jnp.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]) / jnp.sqrt(2.0),
        jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0]) / jnp.sqrt(3.0),
    ]
)


def task_targets(x: Float[Array, "batch feature_dim"], task: int) -> Float[Array, " batch"]:
    """Task-``task`` regression targets for a batch of observations."""
    return jnp.tanh(2.0 * x @ _TASK_DIRECTIONS[task])


def make_gate_learner() -> UPGDLearner:
    """UPGD gate configuration (see module docstring for the rationale)."""
    return UPGDLearner(
        n_heads=N_TASKS,
        hidden_sizes=(64,),
        step_size=0.03,
        bounder=ObGDBounding(kappa=0.5),
    )


def make_sigma_zero_twin(learner: UPGDLearner) -> UPGDLearner:
    """Identical learner with the utility-gated perturbation disabled."""
    config = learner.to_config()
    config["perturbation_sigma"] = 0.0
    return UPGDLearner.from_config(config)


def _task_block(
    key: Array, task: int
) -> tuple[Float[Array, "block feature_dim"], Float[Array, "block n_tasks"]]:
    """One class-blocked task segment: only the task's head is active."""
    obs = jr.normal(key, (BLOCK_STEPS, FEATURE_DIM))
    targets = jnp.full((BLOCK_STEPS, N_TASKS), jnp.nan)
    targets = targets.at[:, task].set(task_targets(obs, task))
    return obs, targets


def _make_seed_run(learner: UPGDLearner):
    """Build the per-seed experiment: 3 task blocks, task-1 eval, task-1 revisit."""

    def run(seed: Array) -> tuple[Array, Array, Array]:
        key = jr.key(seed)
        k_init, k_b0, k_b1, k_b2, k_eval, k_revisit = jr.split(key, 6)
        state = learner.init(FEATURE_DIM, k_init)

        blocks = [_task_block(k, task) for k, task in ((k_b0, 0), (k_b1, 1), (k_b2, 2))]
        observations = jnp.concatenate([obs for obs, _ in blocks])
        targets = jnp.concatenate([tgt for _, tgt in blocks])
        result = run_upgd_arrays(learner, state, observations, targets)

        # Retained task-1 error on held-out task-1 data after task-3 training.
        eval_x = jr.normal(k_eval, (EVAL_BATCH, FEATURE_DIM))
        preds = jax.vmap(lambda x: learner.predict(result.state, x))(eval_x)
        retained_mse = jnp.mean((preds[:, 0] - task_targets(eval_x, 0)) ** 2)

        # Revisit task 1 with fresh data from the same distribution.
        revisit_obs, revisit_targets = _task_block(k_revisit, 0)
        revisit = run_upgd_arrays(learner, result.state, revisit_obs, revisit_targets)

        return retained_mse, result.metrics[:, 0], revisit.metrics[:, 0]

    return run


def _steps_to_criterion(losses: Float[Array, " steps"], threshold: float) -> Array:
    """Steps until the window-smoothed online loss first reaches ``threshold``.

    Returns ``len(losses) + 1`` when the criterion is never reached. The
    smoothing implies a floor of ``SMOOTH_WINDOW`` steps.
    """
    kernel = jnp.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    smoothed = jnp.convolve(losses, kernel, mode="valid")
    below = smoothed <= threshold
    first = jnp.argmax(below) + SMOOTH_WINDOW
    return jnp.where(jnp.any(below), first, losses.shape[0] + 1).astype(jnp.float32)


@pytest.fixture(scope="module")
def gate_results() -> dict[str, Array]:
    """Run both arms once, paired over seeds, and cache all per-seed traces."""
    learner = make_gate_learner()
    twin = make_sigma_zero_twin(learner)
    seeds = jnp.arange(N_SEEDS)

    run_upgd = jax.jit(jax.vmap(_make_seed_run(learner)))
    run_twin = jax.jit(jax.vmap(_make_seed_run(twin)))
    retained_upgd, losses_upgd, revisit_upgd = run_upgd(seeds)
    retained_twin, losses_twin, revisit_twin = run_twin(seeds)

    return {
        "retained_upgd": retained_upgd,
        "retained_twin": retained_twin,
        "losses_upgd": losses_upgd,
        "losses_twin": losses_twin,
        "revisit_upgd": revisit_upgd,
    }


def test_twin_matches_except_perturbation_sigma() -> None:
    learner = make_gate_learner()
    twin = make_sigma_zero_twin(learner)

    config = learner.to_config()
    twin_config = twin.to_config()
    assert config.pop("perturbation_sigma") == 1e-3
    assert twin_config.pop("perturbation_sigma") == 0.0
    assert config == twin_config

    # Identical init from the same key, so seed runs are exactly paired.
    state = learner.init(FEATURE_DIM, jr.key(0))
    twin_state = twin.init(FEATURE_DIM, jr.key(0))
    chex.assert_trees_all_equal(state.trunk_params, twin_state.trunk_params)
    chex.assert_trees_all_equal(state.head_params, twin_state.head_params)


def test_task1_retention_beats_sigma_zero_twin(gate_results: dict[str, Array]) -> None:
    """(a) Retained task-1 error after task-3 training: UPGD < sigma=0 twin."""
    advantage = gate_results["retained_twin"] - gate_results["retained_upgd"]
    mean_advantage = float(jnp.mean(advantage))
    frac_positive = float(jnp.mean((advantage > 0).astype(jnp.float32)))

    # Measured +0.041..+0.045 (t = +5.3..+7.1); gate at ~3x margin below.
    assert mean_advantage > 0.015, (
        f"paired retention advantage {mean_advantage:+.4f} (UPGD "
        f"{float(jnp.mean(gate_results['retained_upgd'])):.4f} vs twin "
        f"{float(jnp.mean(gate_results['retained_twin'])):.4f})"
    )
    # Measured 0.92-0.96 of seeds positive.
    assert frac_positive >= 0.75, f"only {frac_positive:.2f} of seeds favor UPGD"


def test_forgetting_is_present_and_upgd_retention_nontrivial(
    gate_results: dict[str, Array],
) -> None:
    """The stream induces real forgetting; UPGD retention stays well above chance."""
    retained_twin = float(jnp.mean(gate_results["retained_twin"]))
    retained_upgd = float(jnp.mean(gate_results["retained_upgd"]))

    # Twin forgets substantially (measured ~0.35 vs ~0.03 end-of-task error),
    # so the retention comparison is about a real effect.
    assert retained_twin > 0.15, f"twin retained MSE {retained_twin:.4f}: no forgetting to gate"
    # UPGD keeps clearly more than half of the naive zero-predictor gap
    # (zero-predictor MSE ~0.635; measured ~0.31).
    assert retained_upgd < 0.45, f"UPGD retained MSE {retained_upgd:.4f} near chance level"


def test_within_task_adaptation_not_slower(gate_results: dict[str, Array]) -> None:
    """(b) UPGD tracks each task about as fast and as well as the twin."""
    losses_upgd = gate_results["losses_upgd"]
    losses_twin = gate_results["losses_twin"]

    for task in range(N_TASKS):
        start, end = task * BLOCK_STEPS, (task + 1) * BLOCK_STEPS
        final_upgd = float(jnp.mean(losses_upgd[:, end - 100 : end]))
        final_twin = float(jnp.mean(losses_twin[:, end - 100 : end]))
        # Measured ratio <= 1.5 (perturbation noise floor), absolute <= 0.021.
        assert final_upgd < 2.0 * final_twin, (
            f"task {task}: UPGD final-window loss {final_upgd:.4f} vs twin {final_twin:.4f}"
        )
        assert final_upgd < 0.04, f"task {task}: UPGD final-window loss {final_upgd:.4f}"

        steps_upgd = jax.vmap(lambda x: _steps_to_criterion(x, ADAPT_CRITERION))(
            losses_upgd[:, start:end]
        )
        steps_twin = jax.vmap(lambda x: _steps_to_criterion(x, ADAPT_CRITERION))(
            losses_twin[:, start:end]
        )
        mean_upgd = float(jnp.mean(steps_upgd))
        mean_twin = float(jnp.mean(steps_twin))
        # Measured 1.02x-1.39x; every seed reaches criterion in < 480 steps.
        assert mean_upgd <= 1.6 * mean_twin, (
            f"task {task}: UPGD steps-to-criterion {mean_upgd:.1f} vs twin {mean_twin:.1f}"
        )
        assert float(jnp.max(steps_upgd)) <= 800.0, (
            f"task {task}: slowest UPGD seed took {float(jnp.max(steps_upgd)):.0f} steps"
        )


def test_task1_revisit_reaches_criterion_faster(gate_results: dict[str, Array]) -> None:
    """(c) Recurrence savings: relearning task 1 is faster than first exposure."""
    first = jax.vmap(lambda x: _steps_to_criterion(x, RECURRENCE_CRITERION))(
        gate_results["losses_upgd"][:, :BLOCK_STEPS]
    )
    revisit = jax.vmap(lambda x: _steps_to_criterion(x, RECURRENCE_CRITERION))(
        gate_results["revisit_upgd"]
    )

    # Every revisit seed re-reaches criterion (measured max 133 of 1200 steps).
    assert float(jnp.max(revisit)) <= BLOCK_STEPS, "a revisit seed never re-reached criterion"

    mean_first = float(jnp.mean(first))
    mean_revisit = float(jnp.mean(revisit))
    savings_ratio = mean_first / mean_revisit
    frac_faster = float(jnp.mean((revisit < first).astype(jnp.float32)))

    assert mean_revisit < mean_first, (
        f"revisit {mean_revisit:.1f} steps not faster than first exposure {mean_first:.1f}"
    )
    # Measured 1.28-1.52; the 50-step smoothing floor bounds this ratio well
    # below the raw retention advantage, so gate modestly above 1.
    assert savings_ratio > 1.08, f"recurrence savings ratio {savings_ratio:.3f}"
    # Measured 0.67-0.92 of seeds strictly faster on revisit.
    assert frac_faster >= 0.5, f"only {frac_faster:.2f} of seeds relearned faster"
