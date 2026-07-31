"""First learner-on-Pavlovian-stream tests: delta-rule demons on conditioning.

Puts actual learners on ``streams/pavlovian.py`` outputs (previous tests
only validated the stream itself). Each demon is a ``LinearLearner``
trained on the delay-shifted US cumulant: at step ``t`` the input is the
CS indicator observation and the target is the US indicator at
``t + cs_us_delay`` — the gamma=0 GVF "will the US fire ``delay`` steps
from now?". At CS-onset steps this reduces exactly to the
Rescorla-Wagner delta rule (prediction error times CS indicators), and
the abundant ITI steps (all-zero features, target 0) keep the bias
pinned near zero so per-CS weights carry the associative strength.

Demonstrations:

- Kamin blocking (LMS demon on ``blocking_scenario``): after CS0
  pretraining, the compound (CS0, CS1) phase leaves w_CS0 ~ 1 and
  w_CS1 near 0, while a control demon trained on the same compound
  trials WITHOUT pretraining splits credit between the cues.
- Savings (IDBD demon on ``reacquisition_scenario``): reacquisition
  reaches criterion in fewer steps than initial acquisition even though
  extinction drove the prediction back near zero — the memory lives in
  IDBD's meta-learned per-feature step-sizes.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.optimizers import IDBD, LMS
from alberta_framework.streams.pavlovian import (
    ClassicalConditioningStream,
    blocking_scenario,
    reacquisition_scenario,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(
    stream: ClassicalConditioningStream, key: jnp.ndarray, n_steps: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a stream for ``n_steps`` under ``lax.scan``; return (obs, targets)."""
    state = stream.init(key)

    def step_fn(carry, idx):
        ts, new_state = stream.step(carry, idx)
        return new_state, (ts.observation, ts.target)

    _, (obs, tgt) = jax.lax.scan(step_fn, state, jnp.arange(n_steps))
    return obs, tgt


def _shifted_pairs(
    obs: jnp.ndarray, tgt: jnp.ndarray, delay: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pair each observation with the US indicator ``delay`` steps later."""
    return obs[:-delay], tgt[delay:, 0]


def _run_learner(learner: LinearLearner, xs: jnp.ndarray, ys: jnp.ndarray):
    """Scan a LinearLearner over (x, y) pairs.

    Returns ``(final_state, predictions, weights)`` where ``predictions``
    and ``weights`` are the PRE-update values at every step.
    """
    init_state = learner.init(xs.shape[1])

    def step_fn(state, xy):
        x, y = xy
        result = learner.update(state, x, y)
        return result.state, (jnp.squeeze(result.prediction), state.weights)

    final_state, (preds, weights) = jax.lax.scan(step_fn, init_state, (xs, ys))
    return final_state, preds, weights


# ---------------------------------------------------------------------------
# Kamin blocking
# ---------------------------------------------------------------------------


class TestKaminBlocking:
    """Pretrained CS0 blocks CS1 from acquiring the association."""

    def test_blocking_delta_rule_demon(self) -> None:
        delay = 5
        n_pre, n_comp = 1500, 1500
        stream = blocking_scenario(
            n_pretrain=n_pre,
            n_compound=n_comp,
            cs_us_delay=delay,
            cs_duration=1,
            iti_min=5,
            iti_max=10,
            noise_std=0.0,
            distractor_prob=0.0,
        )
        obs, tgt = _collect(stream, jr.key(0), n_pre + n_comp)
        xs, ys = _shifted_pairs(obs, tgt, delay)

        learner = LinearLearner(optimizer=LMS(step_size=0.1))
        final_state, _, weights = _run_learner(learner, xs, ys)

        # Pretraining established CS0 -> US: w_CS0 ~ 1. CS1 never fired,
        # so its weight is still exactly zero (noiseless stream).
        w_end_pretrain = weights[n_pre - 50]
        assert 0.7 < float(w_end_pretrain[0]) < 1.3, (
            f"pretraining failed: w_CS0={float(w_end_pretrain[0]):.3f}"
        )
        assert abs(float(w_end_pretrain[1])) < 1e-6

        # Blocking: through the compound phase CS0 keeps the association
        # and CS1 acquires almost none (delta-rule prediction: the
        # compound's prediction error is already ~0).
        w_final = final_state.weights
        assert 0.7 < float(w_final[0]) < 1.3, (
            f"w_CS0 should stay ~1 through compound: {float(w_final[0]):.3f}"
        )
        assert abs(float(w_final[1])) < 0.15, (
            f"CS1 should be blocked: w_CS1={float(w_final[1]):.3f}"
        )
        assert abs(float(final_state.bias)) < 0.1

        # Control: the same compound trials WITHOUT pretraining split
        # credit between CS0 and CS1 — the blocking above is caused by
        # the pretraining, not by a stream artifact.
        ctrl_start = n_pre + 60  # safely inside the compound phase
        xs_ctrl, ys_ctrl = xs[ctrl_start:], ys[ctrl_start:]
        cs0_on = xs_ctrl[:, 0] > 0.5
        cs1_on = xs_ctrl[:, 1] > 0.5
        assert bool(jnp.all(cs0_on == cs1_on)), "compound segment impure"
        assert int(jnp.sum(cs0_on)) > 50

        control = LinearLearner(optimizer=LMS(step_size=0.1))
        ctrl_state, _, _ = _run_learner(control, xs_ctrl, ys_ctrl)
        w_ctrl = ctrl_state.weights
        assert float(w_ctrl[1]) > 0.3, (
            f"control CS1 should acquire ~0.5: w_CS1={float(w_ctrl[1]):.3f}"
        )
        assert abs(float(w_final[1])) < 0.5 * float(w_ctrl[1]), (
            f"blocked w_CS1={float(w_final[1]):.3f} should be well below "
            f"control w_CS1={float(w_ctrl[1]):.3f}"
        )


# ---------------------------------------------------------------------------
# Reacquisition savings
# ---------------------------------------------------------------------------


class TestReacquisitionSavings:
    """Reacquisition is faster than initial acquisition (savings)."""

    def test_savings_with_idbd_demon(self) -> None:
        delay = 5
        n_acq = n_ext = n_re = 2000
        initial_step_size = 0.02
        stream = reacquisition_scenario(
            n_acquisition=n_acq,
            n_extinction=n_ext,
            n_reacquisition=n_re,
            cs_us_delay=delay,
            cs_duration=1,
            iti_min=5,
            iti_max=10,
            noise_std=0.0,
            distractor_prob=0.0,
        )
        obs, tgt = _collect(stream, jr.key(1), n_acq + n_ext + n_re)
        xs, ys = _shifted_pairs(obs, tgt, delay)

        learner = LinearLearner(
            optimizer=IDBD(initial_step_size=initial_step_size, meta_step_size=0.05)
        )
        final_state, preds, _ = _run_learner(learner, xs, ys)

        cs_on = xs[:, 0] > 0.5
        criterion = 0.6
        n_total = int(xs.shape[0])

        def steps_to_criterion(start: int, end: int) -> int:
            """First CS-onset step in [start, end) whose prediction >= criterion."""
            idx = jnp.arange(n_total)
            hit = cs_on & (idx >= start) & (idx < end) & (preds >= criterion)
            hits = jnp.where(hit)[0]
            assert int(hits.shape[0]) > 0, (
                f"criterion {criterion} never reached in [{start}, {end})"
            )
            return int(hits[0]) - start

        acq_steps = steps_to_criterion(0, n_acq)
        re_steps = steps_to_criterion(n_acq + n_ext, n_total)

        # Extinction genuinely extinguished the prediction: late-extinction
        # CS-onset predictions are back near zero, so the reacquisition
        # speed-up is not just residual associative strength.
        idx = jnp.arange(n_total)
        late_ext = cs_on & (idx >= n_acq + n_ext - 500) & (idx < n_acq + n_ext)
        assert int(jnp.sum(late_ext)) > 10
        late_ext_pred = float(jnp.mean(preds[late_ext]))
        assert late_ext_pred < 0.3, (
            f"extinction incomplete: mean late-extinction prediction "
            f"{late_ext_pred:.3f}"
        )

        # Savings: reacquisition reaches criterion in fewer steps.
        assert re_steps < acq_steps, (
            f"no savings: reacquisition took {re_steps} steps vs "
            f"{acq_steps} for initial acquisition"
        )
        assert re_steps < 0.7 * acq_steps, (
            f"savings too weak: reacquisition {re_steps} steps vs "
            f"acquisition {acq_steps}"
        )

        # The memory mechanism: IDBD's per-feature step-size for the CS
        # grew across acquisition/extinction, so relearning is faster.
        final_cs_step_size = float(jnp.exp(final_state.optimizer_state.log_step_sizes[0]))
        assert final_cs_step_size > initial_step_size
