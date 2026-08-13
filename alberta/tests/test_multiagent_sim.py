"""Development diagnostics for the multi-agent continual-learning streams.

Three ascending stages, all pure JAX (calibration medians from 10-12-seed
runs are quoted in comments).  These are useful mechanism probes, not
held-out certification: Stage 2 supplies the active rule as an oracle context
feature, uses the same seeds for calibration and assertions, and reports no
confidence intervals.

- Stage 1a (:class:`LearningOpponentStream`): the non-stationarity IS a
  learning process.  The stream's internal LMS opponent converges toward a
  hidden relation and is periodically reset; an external IDBD learner tracks
  it, and its meta-learned step-sizes rise in response to the opponent's
  motion — step-size adaptation reacting to another learner's learning.

- Stage 1b (:class:`AdversarialPursuitStream`): worst-case closed-loop drift.
  The target moves within a budget in exactly the direction that hurts the
  current prediction.  A frozen predictor is driven away without bound
  (measured 48x error growth); every continual learner stays bounded, and
  meta-step-size learners track ~50x tighter than a slow fixed step-size.

- Stage 2 (:class:`RecurringConventionGame`): two independent learning agents
  must jointly discover conventions under recurring rules.  With rule context
  in the observation, conventions live in disjoint Q-blocks: recurrence is
  met with immediate re-coordination (time-to-coordination at the metric
  floor, early reward ~0.94) while the context-free twin relearns every
  phase (~50 steps, early reward ~0.78).  This isolates the value of a
  supplied task-separating representation in continual multi-agent control;
  it does not demonstrate autonomous context or feature discovery.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)
from alberta_framework.core.learners import LinearLearner, run_learning_loop
from alberta_framework.core.optimizers import IDBD, LMS, Autostep
from alberta_framework.core.types import StepSizeTrackingConfig
from alberta_framework.streams.matrix_game import (
    ConventionGameConfig,
    RecurringConventionGame,
    phase_reward_profile,
    run_matrix_game,
    time_to_coordination,
)
from alberta_framework.streams.opponent import (
    AdversarialPursuitStream,
    LearningOpponentStream,
    run_pursuit_loop,
)

# =============================================================================
# Stage 1a: LearningOpponentStream
# =============================================================================


class TestLearningOpponentStream:
    def test_shapes_and_protocol(self):
        stream = LearningOpponentStream(feature_dim=6)
        state = stream.init(jr.key(0))
        timestep, new_state = stream.step(state, jnp.array(0))
        assert timestep.observation.shape == (6,)
        assert timestep.target.shape == (1,)
        assert int(new_state.step_count) == 1

    def test_opponent_learns_then_resets(self):
        """The internal opponent closes most of its distance to w_star within
        an epoch, and the reset snaps it back to zero."""
        stream = LearningOpponentStream(feature_dim=8, reset_interval=2000)
        state = stream.init(jr.key(1))

        def dist(s):
            return float(jnp.linalg.norm(s.w_opp - s.w_star))

        d0 = dist(state)

        def scan_to(state, n):
            def body(s, i):
                _, s = stream.step(s, i)
                return s, None

            s, _ = jax.lax.scan(body, state, jnp.arange(n))
            return s

        state_late = scan_to(state, 1900)
        assert dist(state_late) < 0.25 * d0
        # Crossing the reset boundary re-initializes the opponent.
        state_reset = scan_to(state_late, 200)  # crosses step 2000
        assert dist(state_reset) > 2.0 * dist(state_late)

    def test_idbd_step_sizes_rise_with_opponent_motion(self):
        """Meta-learned step-sizes respond to the opponent's learning: the
        mean IDBD step-size grows from its init as the stream keeps moving
        (calibration: 0.010 -> 0.016 over 12k steps, monotone)."""
        stream = LearningOpponentStream(feature_dim=10, reset_interval=4000)
        result = run_learning_loop(
            LinearLearner(optimizer=IDBD()),
            stream,
            num_steps=12000,
            key=jr.key(2),
            step_size_tracking=StepSizeTrackingConfig(interval=3000),
        )
        history = result[2]
        mean_alphas = jnp.mean(history.step_sizes, axis=1)
        assert float(mean_alphas[-1]) > 1.3 * float(mean_alphas[0])
        # And the rise is (weakly) monotone across recordings.
        assert bool(jnp.all(jnp.diff(mean_alphas) >= -1e-6))

    def test_idbd_tracks_opponent_beats_default_fixed_alpha(self):
        """IDBD beats an untuned fixed step-size overall and reaches the
        noise floor once the opponent converges (calibration: overall 0.063
        vs 0.075; late-epoch 0.0036).  A hand-tuned LMS(0.1) is still better
        on this short horizon (0.036) — the price of generality that
        meta-learning pays down as the lifetime grows."""
        stream = LearningOpponentStream(feature_dim=10, reset_interval=4000)

        def overall(learner, seed):
            _, metrics = run_learning_loop(learner, stream, 12000, jr.key(seed))
            return metrics[:, 0]

        idbd = jnp.stack(
            [overall(LinearLearner(optimizer=IDBD()), s) for s in range(5)]
        )
        lms = jnp.stack(
            [overall(LinearLearner(optimizer=LMS(step_size=0.01)), s) for s in range(5)]
        )
        assert float(jnp.mean(idbd)) < float(jnp.mean(lms))
        # Late-epoch (opponent converged): IDBD sits near the noise floor.
        assert float(jnp.mean(idbd[:, 3000:4000])) < 0.01


# =============================================================================
# Stage 1b: AdversarialPursuitStream
# =============================================================================


@pytest.fixture(scope="module")
def stream():
    return AdversarialPursuitStream(feature_dim=10, drift_budget=0.02)


class TestAdversarialPursuit:
    def _sq(self, stream, learner, frozen=False, seeds=4, steps=6000):
        return jnp.stack(
            [
                run_pursuit_loop(learner, stream, steps, jr.key(s), frozen=frozen)[1]
                for s in range(seeds)
            ]
        )

    def test_frozen_predictor_is_driven_away(self, stream):
        """Without adaptation the adversary compounds: error grows without
        bound (calibration: 48x growth from first to last thousand steps)."""
        sq = self._sq(stream, LinearLearner(optimizer=LMS(step_size=0.0)), frozen=True)
        first = float(jnp.mean(sq[:, :1000]))
        last = float(jnp.mean(sq[:, -1000:]))
        assert last > 5.0 * first

    def test_continual_learners_stay_bounded(self, stream):
        """IDBD and Autostep keep the adversary within a small pursuit
        distance: late error is small and not growing (calibration: last-1k
        0.06 / 0.12 with ratios < 0.2)."""
        for optimizer in (IDBD(), Autostep()):
            sq = self._sq(stream, LinearLearner(optimizer=optimizer))
            first = float(jnp.mean(sq[:, :1000]))
            last = float(jnp.mean(sq[:, -1000:]))
            assert last < 0.5
            assert last < first  # converging toward the pursuit equilibrium

    def test_adaptive_beats_slow_fixed_alpha_under_adversarial_drift(self, stream):
        """A too-slow fixed step-size cannot keep up with worst-case drift
        (calibration: last-1k 3.0 vs 0.06 for IDBD — a ~50x gap)."""
        slow = self._sq(stream, LinearLearner(optimizer=LMS(step_size=0.003)))
        idbd = self._sq(stream, LinearLearner(optimizer=IDBD()))
        assert float(jnp.mean(slow[:, -1000:])) > 5.0 * float(
            jnp.mean(idbd[:, -1000:])
        )


# =============================================================================
# Stage 2: RecurringConventionGame
# =============================================================================


def _agent(use_bias: bool = False) -> DifferentialSARSAAgent:
    return DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=12,
            q_step_size=0.15,
            average_reward_step_size=0.01,
            epsilon_start=0.03,
            epsilon_end=0.03,
            use_bias=use_bias,
        )
    )


N_SEEDS = 10
PHASE = 2000
STEPS = 12000  # six phases: A B A B A B


@pytest.fixture(scope="module")
def context_runs():
    game = RecurringConventionGame(ConventionGameConfig(feature_mode="context"))
    return [run_matrix_game(_agent(), game, STEPS, jr.key(s)) for s in range(N_SEEDS)]


@pytest.fixture(scope="module")
def plain_runs():
    game = RecurringConventionGame(ConventionGameConfig(feature_mode="plain"))
    return [run_matrix_game(_agent(), game, STEPS, jr.key(s)) for s in range(N_SEEDS)]


def _t_coords(runs):
    """(n_seeds, n_phases) time-to-coordination matrix."""
    out = []
    for r in runs:
        out.append(
            [
                int(
                    time_to_coordination(
                        r.rewards[p * PHASE : (p + 1) * PHASE]
                    )
                )
                for p in range(STEPS // PHASE)
            ]
        )
    return jnp.asarray(out)


def _early(runs):
    return jnp.stack([phase_reward_profile(r.rewards, PHASE)[0] for r in runs])


class TestConventionGameMechanics:
    def test_reward_rule(self):
        game = RecurringConventionGame(
            ConventionGameConfig(n_actions=12, offsets=(0, 3))
        )
        state = game.init(jr.key(0))
        r, _ = game.step(state, jnp.array(5), jnp.array(5))
        assert float(r) == 1.0  # rule A: offset 0
        r, _ = game.step(state, jnp.array(5), jnp.array(4))
        assert float(r) == 0.0
        # Jump to the second phase: offset 3 required.
        state_b = state.replace(
            step_count=jnp.array(2000, dtype=jnp.int32),
            step_words=jnp.array((0, 2000), dtype=jnp.uint32),
        )
        r, _ = game.step(state_b, jnp.array(7), jnp.array(4))
        assert float(r) == 1.0
        r, _ = game.step(state_b, jnp.array(7), jnp.array(7))
        assert float(r) == 0.0

    def test_observation_modes(self):
        plain = RecurringConventionGame(ConventionGameConfig(feature_mode="plain"))
        ctx = RecurringConventionGame(ConventionGameConfig(feature_mode="context"))
        assert plain.observation_dim == 1
        assert ctx.observation_dim == 2
        state = ctx.init(jr.key(0))
        assert jnp.array_equal(ctx.observe(state), jnp.array([1.0, 0.0]))
        state_b = state.replace(
            step_count=jnp.array(2500, dtype=jnp.int32),
            step_words=jnp.array((0, 2500), dtype=jnp.uint32),
        )
        assert jnp.array_equal(ctx.observe(state_b), jnp.array([0.0, 1.0]))

    def test_run_shapes(self):
        game = RecurringConventionGame(ConventionGameConfig())
        result = run_matrix_game(_agent(), game, 300, jr.key(0))
        assert result.rewards.shape == (300,)
        assert result.actions.shape == (300, 2)

    def test_conventions_are_emergent_not_fixed(self, context_runs):
        """Different seeds converge to different rule-A conventions — the
        coordination point is discovered by joint symmetry breaking, not
        baked into the game or the agents."""
        conventions = set()
        for r in context_runs:
            # Modal agent-0 action over the tail of phase 0.
            tail_actions = r.actions[PHASE - 200 : PHASE, 0]
            conventions.add(int(jnp.bincount(tail_actions, length=12).argmax()))
        assert len(conventions) >= 3


class TestConventionDiagnostics:
    def test_both_modes_solve_every_phase(self, context_runs, plain_runs):
        """Plasticity holds in both representations: whatever the rule, the
        team ends every phase coordinated (tail reward >= 0.85 vs the 0.94
        epsilon-ceiling; chance is 1/12)."""
        for runs in (context_runs, plain_runs):
            tails = jnp.stack(
                [phase_reward_profile(r.rewards, PHASE)[1] for r in runs]
            )
            assert float(jnp.min(jnp.median(tails, axis=0))) >= 0.85

    def test_context_recalls_conventions_instantly(self, context_runs):
        """On every rule recurrence (phases 2-5) the context-gated agents
        re-coordinate at the metric floor (calibration: t_coord 20 = floor,
        early reward 0.94-0.95)."""
        tc = _t_coords(context_runs)
        early = _early(context_runs)
        for phase in (2, 3, 4, 5):
            assert float(jnp.median(tc[:, phase])) <= 25.0
            assert float(jnp.median(early[:, phase])) >= 0.90

    def test_plain_relearns_every_phase(self, plain_runs):
        """Without rule context the same agents must rediscover a convention
        at every flip, forever (calibration: t_coord ~50-60, early ~0.73-0.80
        on recurrences)."""
        tc = _t_coords(plain_runs)
        early = _early(plain_runs)
        recur_tc = jnp.median(tc[:, 2:], axis=0)
        recur_early = jnp.median(early[:, 2:], axis=0)
        assert float(jnp.min(recur_tc)) >= 30.0
        assert float(jnp.max(recur_early)) <= 0.88

    def test_representation_gap_is_large(self, context_runs, plain_runs):
        """The memory gap between representations: context re-coordination is
        at least 1.5x faster and its early reward at least 0.05 higher on
        every recurrence (measured gaps are ~2.5x and ~0.15)."""
        tc_gap = jnp.median(_t_coords(plain_runs)[:, 2:]) / jnp.maximum(
            jnp.median(_t_coords(context_runs)[:, 2:]), 1
        )
        early_gap = jnp.median(_early(context_runs)[:, 2:]) - jnp.median(
            _early(plain_runs)[:, 2:]
        )
        assert float(tc_gap) >= 1.5
        assert float(early_gap) >= 0.05
