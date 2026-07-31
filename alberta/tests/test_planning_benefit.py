"""Development diagnostics for imagined-experience control effects.

These configurations, thresholds, and several eight-seed sets were inspected
during calibration. They have no frozen held-out schedule, paired confidence
interval, versioned artifact, or equal total-update budget: planning conditions
intentionally receive additional backups. Passing the tests supports promotion
to a preregistered comparison; it is not itself L2 evidence for Steps 7 or 9.

The upstream CartPole benchmark for Dyna planning was a ceiling-effect null:
the baseline already solved the task, so planning had no room to help. These
tests instead close the loop on the in-tree micro-MDPs where model-based
backups have real work to do, and add a model-corruption stress for the Step 9
error-EMA dream gate.

Protocol (all runs are paired across 8 seeds, vmapped, CPU-only):

1. **Step 7 planning ablation** on a hard-exploration RiverSwim-4 chain
   (``p_right_up=0.55``, ``reward_left=0.01``): epsilon-greedy differential
   SARSA alone almost never propagates the right-end reward within 2000 steps
   (cumulative reward ~17-300 across seed sets), while the same agent with 4
   random-anchor Dyna backups per real step reaches ~650-1000. Calibration
   over five disjoint 8-seed sets measured planning-minus-baseline cumulative
   reward diffs with means {965, 672, 806, 538, 643} (wins >= 6/8 in every
   set) for ``planning_steps=4`` and means {450, 516, 482, 487, 487} for
   ``planning_steps=1``. Thresholds (mean > 250 / > 150, wins >= 5/8) sit
   >= 2x below every measured set mean.

2. **Step 9 dreaming ablation** on the same chain: guarded dreams (budget 4)
   vs a zero dream budget. Measured paired diffs across the same five seed
   sets had means {798, 279, 653, 429, 636} and wins >= 6/8; the dream
   acceptance rate was 0.976 (the only rejections are the 50-step warmup).
   Thresholds: mean > 120 (>= 2.3x below the worst set), wins >= 5/8,
   acceptance > 0.8.

3. **Model-corruption stress** on ``SwitchingTwoStateMDP`` (phase length 600,
   payoff flip at t=600 inverts every reward while dynamics stay fixed): the
   world model's reward head is suddenly wrong on every transition, so its
   error EMA (decay 0.9) jumps from ~0.05 to a ~0.4-0.6 peak and the gated
   config (``dreaming_max_model_error=0.2``) closes the dream gate, then
   reopens once the model relearns. Calibration over three 8-seed sets
   measured: per-seed close lag (first fully-rejected step after the flip)
   max 4, per-seed pre-flip acceptance min 0.66, seed-mean acute acceptance
   over [flip+2, flip+12) of {0.04, 0.11, 0.18}, seed-mean count of fully
   closed steps in [flip, flip+50) of {22, 26, 23}, while the permissive
   config (``dreaming_max_model_error=1e30``) accepted 100% of dreams
   throughout. In THIS configuration post-flip regret is statistically
   indistinguishable between gated and ungated, so the regret assertion is
   calibrated non-inferiority (tolerance 30). The corruption self-neutralizes
   through two channels traced numerically: (a) with ``model_hidden_sizes=()``
   and no interaction features the model is *additive* in state and action, so
   it cannot represent the XOR payoff structure — its dream rewards are nearly
   action-flat per state, and corrupt dreams shift Q *levels*, which
   epsilon-greedy control ignores; (b) right after the flip the corrupt dream
   rewards (~0.9) approximately equal the still-high reward-rate estimate
   rbar, so dream TD errors are ~0 exactly during the window in which real
   unlearning (alpha=0.05, 5x faster than rbar decay beta=0.01) flips the
   policy.

4. **Strict gate superiority under amplified corruption**: closing both
   channels — ``model_include_action_interactions=True`` (the model can now
   represent per-(s,a) rewards, so post-flip dream rewards are genuinely
   inverted) and ``dreams_update_average_reward=False`` (imagined rewards
   cannot move rbar) — with dream budget 16 and model step-size 0.005 (long
   corrupt window), dreaming through the corruption costs real regret.
   Calibration over three disjoint 48-seed sets (seeds 100-147, 200-247,
   300-347) measured permissive-minus-gated post-flip regret mean gaps of
   {35.6, 29.5, 35.7} (pooled 33.6) and median gaps of {12, 18, 19}; 11/144
   permissive seeds locked into the stale policy (regret 203-558) while the
   gated agent's worst seed across all 192 calibration+test runs was 145.
   Thresholds (mean gap > 12, median gap > 5, gated mean <= 150, gated max
   <= 300) sit >= 2x below/above every measured set; bootstrapping the pooled
   144-seed distribution puts each assertion's failure rate below 5e-5 at
   n=96 seeds. The legacy coupling (``dreams_update_average_reward=True``)
   measured in the same regime is strictly worse for both arms (gated mean
   regret 113.5-115.6 vs ~97; permissive 239.3-251.5 with 24-28/48 locks).

Total runtime ~35s on CPU (9 jit compiles dominate).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step7 import (
    Step7DynaConfig,
    init_step7_state,
    make_step7_components,
    step7_update,
)
from alberta_framework.steps.step8 import Step8WorldModelConfig
from alberta_framework.steps.step9 import (
    Step9DreamingConfig,
    init_step9_state,
    make_step9_components,
    step9_update,
)
from alberta_framework.streams.closed_loop import (
    RiverSwimConfig,
    RiverSwimMDP,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

SEEDS = jnp.arange(8, dtype=jnp.uint32)

# Hard-exploration RiverSwim-4: swimming right fights the drift toward a
# 1.0 reward, swimming left pays a distractor 0.01. Calibrated so the
# no-planning baseline rarely solves it within the step budget.
RIVERSWIM_CONFIG = RiverSwimConfig(
    n_states=4,
    p_right_up=0.55,
    p_right_down=0.05,
    reward_left=0.01,
)
RIVERSWIM_STEPS = 2000

# Constant-epsilon control head shared by both ablations.
CONTROL_CONFIG = Step6DifferentialSARSAConfig(epsilon_start=0.25, epsilon_end=0.25)


def _run_step7_closed_loop(
    env: RiverSwimMDP,
    cfg: Step7DynaConfig,
    seeds: jnp.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Roll the Step 7 facade closed-loop on ``env``; returns (n_seeds, n_steps) rewards.

    The action executed in the environment at each step is exactly the action
    the SARSA head committed to (``control_state.last_action``), so the loop
    is genuinely closed: actions determine the next observation.
    """
    agent, model = make_step7_components(cfg)

    def run_one(seed: jnp.ndarray) -> jnp.ndarray:
        key = jr.key(seed)
        env_key, state_key, roll_key = jr.split(key, 3)
        env_state = env.init(env_key)
        state = init_step7_state(
            agent,
            model,
            key=state_key,
            initial_observation=env.observe(env_state),
            memory_size=cfg.planning_memory_size,
        )

        def scan_fn(carry, step_key):
            step7_state, env_carry = carry
            action = step7_state.control_state.last_action
            obs, reward, env_carry = env.step(env_carry, action, step_key)
            result = step7_update(cfg, agent, model, step7_state, reward, obs)
            return (result.state, env_carry), reward

        keys = jr.split(roll_key, n_steps)
        _, rewards = jax.lax.scan(scan_fn, (state, env_state), keys)
        return rewards

    return np.asarray(jax.jit(jax.vmap(run_one))(seeds))


def _run_step9_closed_loop(
    env: RiverSwimMDP | SwitchingTwoStateMDP,
    cfg: Step9DreamingConfig,
    seeds: jnp.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the Step 9 facade closed-loop; returns (rewards, dream_acceptance, error_ema).

    All outputs have shape (n_seeds, n_steps); ``dream_acceptance`` is the
    per-step mean of the accepted-dream mask (0.0 when the budget is zero).
    """
    agent, model, buffer = make_step9_components(cfg)

    def run_one(seed: jnp.ndarray):
        key = jr.key(seed)
        env_key, state_key, roll_key = jr.split(key, 3)
        env_state = env.init(env_key)
        state = init_step9_state(
            agent,
            model,
            buffer,
            key=state_key,
            initial_observation=env.observe(env_state),
        )

        def scan_fn(carry, step_key):
            step9_state, env_carry = carry
            action = step9_state.control_state.last_action
            obs, reward, env_carry = env.step(env_carry, action, step_key)
            result = step9_update(cfg, agent, model, buffer, step9_state, reward, obs)
            if cfg.planning_budget > 0:
                accepted = jnp.mean(result.dream_accepted.astype(jnp.float32))
            else:
                accepted = jnp.array(0.0, dtype=jnp.float32)
            ema = result.state.world_model_state.model_error_ema
            return (result.state, env_carry), (reward, accepted, ema)

        keys = jr.split(roll_key, n_steps)
        _, (rewards, accepted, emas) = jax.lax.scan(scan_fn, (state, env_state), keys)
        return rewards, accepted, emas

    rewards, accepted, emas = jax.jit(jax.vmap(run_one))(seeds)
    return np.asarray(rewards), np.asarray(accepted), np.asarray(emas)


def _riverswim_step7_config(planning_steps: int) -> Step7DynaConfig:
    return Step7DynaConfig(
        control=CONTROL_CONFIG,
        world_model=Step8WorldModelConfig(
            observation_dim=RIVERSWIM_CONFIG.n_states,
            n_actions=2,
            hidden_sizes=(),
            use_layer_norm=False,
            sparsity=0.0,
            step_size=0.2,
        ),
        planning_steps=planning_steps,
        planning_strategy="random",
    )


def test_step7_planning_accelerates_riverswim_control() -> None:
    """One-step Dyna planning strictly accelerates average-reward growth.

    Calibration (this exact config, seeds 0-7): cumulative-reward means were
    38.9 (planning 0), 488.5 (planning 1), and 1004.0 (planning 4); paired
    diffs vs the no-planning baseline had mean 449.7 (wins 7/8) and 965.1
    (min 551.7, wins 8/8). Across four more disjoint 8-seed sets the
    planning-4 diff mean never fell below 538 and wins never below 6/8, so
    the thresholds below carry >= 2x margins.
    """
    env = RiverSwimMDP(RIVERSWIM_CONFIG)
    cumulative = {}
    for planning_steps in (0, 1, 4):
        rewards = _run_step7_closed_loop(
            env,
            _riverswim_step7_config(planning_steps),
            SEEDS,
            RIVERSWIM_STEPS,
        )
        assert np.all(np.isfinite(rewards)), f"non-finite rewards at p={planning_steps}"
        cumulative[planning_steps] = rewards.sum(axis=1)

    diff_4 = cumulative[4] - cumulative[0]
    assert diff_4.mean() > 250.0, (
        f"planning_steps=4 paired cumulative-reward gain {diff_4.mean():.1f} <= 250 "
        f"(calibrated mean 965, worst disjoint-set mean 538)"
    )
    assert int((diff_4 > 0).sum()) >= 5, (
        f"planning_steps=4 won only {int((diff_4 > 0).sum())}/8 paired seeds "
        f"(calibrated 8/8, worst disjoint-set 6/8)"
    )

    diff_1 = cumulative[1] - cumulative[0]
    assert diff_1.mean() > 150.0, (
        f"planning_steps=1 paired cumulative-reward gain {diff_1.mean():.1f} <= 150 "
        f"(calibrated mean 450, worst disjoint-set mean 450)"
    )
    assert int((diff_1 > 0).sum()) >= 5, (
        f"planning_steps=1 won only {int((diff_1 > 0).sum())}/8 paired seeds "
        f"(calibrated 7/8, worst disjoint-set 6/8)"
    )


def _riverswim_step9_config(planning_budget: int) -> Step9DreamingConfig:
    return Step9DreamingConfig(
        control=CONTROL_CONFIG,
        observation_dim=RIVERSWIM_CONFIG.n_states,
        n_actions=2,
        model_hidden_sizes=(),
        model_step_size=0.2,
        model_sparsity=0.0,
        model_use_layer_norm=False,
        model_gamma=0.99,
        dreaming_warmup_steps=50,
        dreaming_max_model_error=10.0,
        planning_budget=planning_budget,
    )


def test_step9_dreaming_accelerates_riverswim_control() -> None:
    """Guarded dreaming (budget 4) beats a zero dream budget on RiverSwim-4.

    Calibration (this exact config, seeds 0-7): cumulative-reward means were
    38.9 (budget 0) and 837.3 (budget 4); the paired diff had mean 798.4
    (min 105.2, wins 8/8) and the dream acceptance rate was 0.976 (only the
    50-step warmup rejects). Across four more disjoint 8-seed sets the diff
    mean never fell below 279 and wins never below 6/8.
    """
    env = RiverSwimMDP(RIVERSWIM_CONFIG)
    rewards_dream, accepted, _ = _run_step9_closed_loop(
        env, _riverswim_step9_config(4), SEEDS, RIVERSWIM_STEPS
    )
    rewards_base, _, _ = _run_step9_closed_loop(
        env, _riverswim_step9_config(0), SEEDS, RIVERSWIM_STEPS
    )
    assert np.all(np.isfinite(rewards_dream))
    assert np.all(np.isfinite(rewards_base))

    diff = rewards_dream.sum(axis=1) - rewards_base.sum(axis=1)
    assert diff.mean() > 120.0, (
        f"dreaming paired cumulative-reward gain {diff.mean():.1f} <= 120 "
        f"(calibrated mean 798, worst disjoint-set mean 279)"
    )
    assert int((diff > 0).sum()) >= 5, (
        f"dreaming won only {int((diff > 0).sum())}/8 paired seeds "
        f"(calibrated 8/8, worst disjoint-set 6/8)"
    )
    # The benefit must come from dreams that actually fired.
    acceptance_rate = accepted.mean()
    assert acceptance_rate > 0.8, (
        f"dream acceptance rate {acceptance_rate:.3f} <= 0.8 (calibrated 0.976; "
        f"the warmup alone caps it at 0.975)"
    )


# --- Dream / average-reward separation ---------------------------------------


def test_step9_dreams_do_not_move_average_reward_when_disabled() -> None:
    """Dream updates leave rbar bit-identical when dreams_update_average_reward=False.

    Dyna doctrine: planning backups improve *value* estimates; the reward-rate
    estimate rbar is a property of actual behavior in the real environment, so
    imagined experience should not move it. With the flag disabled (the
    default), the post-dream rbar must equal the post-real-update rbar exactly
    (bit-identical), while accepted dreams still move the Q parameters. With
    the flag enabled (legacy behavior), the same accepted dreams move rbar.
    """
    def make_cfg(dreams_update_average_reward: bool) -> Step9DreamingConfig:
        return Step9DreamingConfig(
            control=Step6DifferentialSARSAConfig(),
            observation_dim=3,
            n_actions=2,
            model_hidden_sizes=(8,),
            dreaming_warmup_steps=0,
            dreaming_max_model_error=1.0e30,
            planning_budget=4,
            dreams_update_average_reward=dreams_update_average_reward,
        )

    def run_updates(cfg: Step9DreamingConfig) -> tuple[bool, bool, bool]:
        agent, model, buffer = make_step9_components(cfg)
        key = jr.key(7)
        state_key, data_key = jr.split(key)
        observations = jr.normal(data_key, (6, cfg.observation_dim), dtype=jnp.float32)
        rewards = jnp.tanh(observations[1:, 0])
        state = init_step9_state(
            agent, model, buffer, key=state_key, initial_observation=observations[0]
        )
        any_accepted = False
        rbar_moved_by_dreams = False
        q_moved_by_dreams = False
        for t in range(5):
            result = step9_update(
                cfg, agent, model, buffer, state, rewards[t], observations[t + 1]
            )
            accepted = bool(jnp.any(result.dream_accepted))
            any_accepted = any_accepted or accepted
            post_real_rbar = result.real_control_result.state.average_reward
            post_dream_rbar = result.state.control_state.average_reward
            if bool(post_dream_rbar != post_real_rbar):
                rbar_moved_by_dreams = True
            if accepted and bool(
                jnp.any(
                    result.state.control_state.q_weights
                    != result.real_control_result.state.q_weights
                )
            ):
                q_moved_by_dreams = True
            state = result.state
        return any_accepted, rbar_moved_by_dreams, q_moved_by_dreams

    accepted_off, rbar_moved_off, q_moved_off = run_updates(make_cfg(False))
    assert accepted_off, "no dreams accepted with the gate wide open (setup broken)"
    assert q_moved_off, "accepted dreams failed to move Q parameters (setup broken)"
    assert not rbar_moved_off, (
        "dreams moved average_reward despite dreams_update_average_reward=False"
    )

    accepted_on, rbar_moved_on, _ = run_updates(make_cfg(True))
    assert accepted_on, "no dreams accepted with the gate wide open (setup broken)"
    assert rbar_moved_on, (
        "legacy path (dreams_update_average_reward=True) failed to move rbar; "
        "the flag is not exercising a real behavioral difference"
    )

    # The default must be the protected path.
    assert Step9DreamingConfig(
        observation_dim=3, n_actions=2
    ).dreams_update_average_reward is False


def test_step9_config_passes_action_interactions_to_world_model() -> None:
    """model_include_action_interactions reaches the core world-model config.

    With ``model_hidden_sizes=()`` the world model is linear in
    ``concat(obs, one_hot(action))`` — additive in state and action — and
    cannot represent any payoff with state-action interaction structure (e.g.
    the XOR payoffs of ``SwitchingTwoStateMDP``). The core config's
    ``include_action_interactions`` fixes that with obs-by-action slope
    features; Step 9 must be able to reach it.
    """
    cfg = Step9DreamingConfig(model_include_action_interactions=True)
    assert cfg.to_world_model_config().include_action_interactions is True
    # Default off: preserves the legacy additive linear model.
    assert (
        Step9DreamingConfig().to_world_model_config().include_action_interactions
        is False
    )
    # Round-trips through serialization.
    restored = Step9DreamingConfig.from_dict(cfg.to_dict())
    assert restored.model_include_action_interactions is True


# --- Model-corruption stress -------------------------------------------------

FLIP_PHASE = 600
FLIP_STEPS = 2 * FLIP_PHASE
GATE_THRESHOLD = 0.2


def _flip_step9_config(max_model_error: float) -> Step9DreamingConfig:
    return Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(epsilon_start=0.15, epsilon_end=0.15),
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_step_size=0.01,
        model_sparsity=0.0,
        model_use_layer_norm=False,
        model_gamma=0.99,
        dreaming_warmup_steps=50,
        dreaming_max_model_error=max_model_error,
        model_error_decay=0.9,
        planning_budget=4,
    )


def test_step9_error_gate_closes_after_payoff_flip() -> None:
    """The error-EMA gate collapses dream acceptance right after a payoff flip.

    ``SwitchingTwoStateMDP`` inverts every payoff at t=600 while dynamics stay
    fixed, so the world model's reward head is wrong on every transition and
    the error EMA (decay 0.9, pre-flip steady state ~0.05) must cross the
    calibrated gate threshold 0.2 within a few steps. Calibration (this exact
    config, seeds 0-7): per-seed close lag [0..3], per-seed pre-flip
    acceptance min 0.74, seed-mean acute acceptance over [flip+2, flip+12)
    0.04, seed-mean fully-closed steps in [flip, flip+50) 21.9, gate fully
    reopened by flip+300 (model relearned). Two more disjoint 8-seed sets
    measured close lag max 4, pre-flip acceptance min 0.66, acute acceptance
    means 0.11/0.18, closed-step means 25.5/23.3. The permissive control
    config (gate 1e30) accepted 100% of dreams throughout.

    Post-flip regret: calibration across all measured variants found gated
    minus ungated seed-set mean gaps in [-13.1, +5.2] (noise around zero;
    corrupt imagined rewards are neutralized by the average-reward baseline),
    so the assertion is non-inferiority with a calibrated tolerance of 30:
    closing the gate — losing every dream update during the corrupt window —
    must not cost measurable control performance.
    """
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=FLIP_PHASE))
    rewards_gated, acc_gated, ema_gated = _run_step9_closed_loop(
        env, _flip_step9_config(GATE_THRESHOLD), SEEDS, FLIP_STEPS
    )
    rewards_ungated, acc_ungated, _ = _run_step9_closed_loop(
        env, _flip_step9_config(1.0e30), SEEDS, FLIP_STEPS
    )
    assert np.all(np.isfinite(rewards_gated))
    assert np.all(np.isfinite(rewards_ungated))

    # (a) Pre-flip the gate is open: the model has converged and dreams flow.
    pre_acceptance = acc_gated[:, FLIP_PHASE - 100 : FLIP_PHASE].mean(axis=1)
    assert float(pre_acceptance.min()) >= 0.4, (
        f"pre-flip acceptance min {pre_acceptance.min():.2f} < 0.4 "
        f"(calibrated per-seed min 0.66 across three seed sets)"
    )
    assert float(pre_acceptance.mean()) >= 0.8, (
        f"pre-flip acceptance mean {pre_acceptance.mean():.2f} < 0.8 (calibrated ~0.95+)"
    )

    # (b) The EMA crosses the gate threshold shortly after the flip...
    ema_peak = ema_gated[:, FLIP_PHASE : FLIP_PHASE + 15].max(axis=1)
    assert np.all(ema_peak > GATE_THRESHOLD), (
        f"error EMA failed to cross the gate threshold within 15 post-flip steps "
        f"(per-seed peaks {np.round(ema_peak, 3)}; calibrated peaks ~0.4-0.6)"
    )

    # ... and every seed's gate fully closes (all dreams rejected) within 15
    # steps of the flip (calibrated close lag <= 4 across 24 seeds).
    post_window = acc_gated[:, FLIP_PHASE : FLIP_PHASE + 15]
    fully_closed_fast = (post_window == 0.0).any(axis=1)
    assert bool(fully_closed_fast.all()), (
        f"gate failed to fully close within 15 post-flip steps for seeds "
        f"{np.where(~fully_closed_fast)[0].tolist()} (calibrated close lag <= 4)"
    )

    # Acceptance collapses in the acute window and the gate stays closed for
    # a sustained stretch while the model error remains high.
    acute_acceptance = acc_gated[:, FLIP_PHASE + 2 : FLIP_PHASE + 12].mean()
    assert acute_acceptance <= 0.35, (
        f"acute post-flip acceptance {acute_acceptance:.3f} > 0.35 "
        f"(calibrated seed-means 0.04-0.18)"
    )
    closed_steps = (acc_gated[:, FLIP_PHASE : FLIP_PHASE + 50] == 0.0).sum(axis=1)
    assert float(closed_steps.mean()) >= 8.0, (
        f"gate closed for only {closed_steps.mean():.1f} steps on average in the 50 "
        f"steps after the flip (calibrated seed-means 21.9-25.5)"
    )

    # (c) The permissive config keeps dreaming straight through the corruption
    # (this is what makes the gated collapse an error-gate effect, not a
    # side effect of the flip itself).
    ungated_post = acc_ungated[:, FLIP_PHASE : FLIP_PHASE + 50].mean(axis=1)
    assert float(ungated_post.min()) >= 0.95, (
        f"permissive-gate acceptance dropped to {ungated_post.min():.2f} post-flip "
        f"(calibrated exactly 1.0)"
    )

    # (d) Non-inferiority: shutting off dreams during the corrupt window must
    # not cost control performance vs dreaming through it.
    optimal_rate = env.optimal_average_reward(1)  # 1.0 in both phases
    regret_gated = FLIP_PHASE * optimal_rate - rewards_gated[:, FLIP_PHASE:].sum(axis=1)
    regret_ungated = FLIP_PHASE * optimal_rate - rewards_ungated[:, FLIP_PHASE:].sum(axis=1)
    assert regret_gated.mean() <= regret_ungated.mean() + 30.0, (
        f"gated post-flip regret {regret_gated.mean():.1f} exceeds ungated "
        f"{regret_ungated.mean():.1f} by more than the calibrated 30-step tolerance "
        f"(measured gaps were noise in [-13.1, +5.2])"
    )
    # Both arms recover: regret stays far below the ~600 worst case.
    assert regret_gated.mean() <= 150.0, (
        f"gated post-flip regret {regret_gated.mean():.1f} > 150 (calibrated ~83-94)"
    )
    assert regret_ungated.mean() <= 150.0, (
        f"ungated post-flip regret {regret_ungated.mean():.1f} > 150 (calibrated ~83-94)"
    )


# --- Strict gate superiority under amplified corruption ----------------------

CORRUPTION_SEEDS = jnp.arange(96, dtype=jnp.uint32)


def _corruption_step9_config(max_model_error: float) -> Step9DreamingConfig:
    """Amplified-corruption regime: see the module docstring, item 4."""
    return Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(epsilon_start=0.15, epsilon_end=0.15),
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_step_size=0.005,
        model_sparsity=0.0,
        model_include_action_interactions=True,
        model_use_layer_norm=False,
        model_gamma=0.99,
        dreaming_warmup_steps=50,
        dreaming_max_model_error=max_model_error,
        model_error_decay=0.9,
        planning_budget=16,
        dreams_update_average_reward=False,
    )


def test_step9_gate_strictly_reduces_corruption_regret() -> None:
    """The error gate strictly beats dreaming through a payoff flip on regret.

    This is the strict-superiority companion to the non-inferiority assertion
    in :func:`test_step9_error_gate_closes_after_payoff_flip`. Two changes
    make the corruption bite (both channels are documented in the module
    docstring): the world model gets obs-by-action interaction features, so
    its post-flip dream rewards are genuinely inverted per (state, action)
    instead of action-flat, and the dream budget is 16 with a slow model
    step-size (0.005) that extends the corrupt window. rbar is protected from
    dreams (``dreams_update_average_reward=False``, the default).

    Calibration (three disjoint 48-seed sets: 100-147, 200-247, 300-347):
    permissive-minus-gated post-flip regret mean gaps {35.6, 29.5, 35.7}
    (pooled 33.6), median gaps {12, 18, 19}; permissive locks (regret > 200,
    stale-policy lock-in) 11/144 reaching up to 558, gated max regret 127.
    Test-seed verification (seeds 0-95, disjoint from calibration): mean gap
    40.4, median gap 19.0, gated mean 97.3, gated max 145, 7/96 permissive
    locks. Bootstrapped from the pooled calibration distribution, each
    assertion below fails with probability < 5e-5 at n=96. Gate mechanics
    (measured over all 192 calibration+test seeds): gated pre-flip acceptance
    mean 0.99 / min 0.81, gated acute post-flip acceptance mean <= 0.03,
    permissive acceptance 1.0 throughout.

    The legacy rbar coupling (``dreams_update_average_reward=True``) measured
    in this same regime is strictly worse for both arms (gated mean regret
    113.5-115.6, permissive 239.3-251.5 with 24-28/48 locks), so the
    protected default is also the stronger baseline.
    """
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=FLIP_PHASE))
    rewards_gated, acc_gated, _ = _run_step9_closed_loop(
        env, _corruption_step9_config(GATE_THRESHOLD), CORRUPTION_SEEDS, FLIP_STEPS
    )
    rewards_perm, acc_perm, _ = _run_step9_closed_loop(
        env, _corruption_step9_config(1.0e30), CORRUPTION_SEEDS, FLIP_STEPS
    )
    assert np.all(np.isfinite(rewards_gated))
    assert np.all(np.isfinite(rewards_perm))

    optimal_rate = env.optimal_average_reward(1)  # 1.0 in both phases
    regret_gated = FLIP_PHASE * optimal_rate - rewards_gated[:, FLIP_PHASE:].sum(axis=1)
    regret_perm = FLIP_PHASE * optimal_rate - rewards_perm[:, FLIP_PHASE:].sum(axis=1)

    # (a) Strict superiority in the mean: dreaming through the corruption
    # costs regret (calibrated set gaps 29.5-35.7, threshold >= 2.4x below).
    mean_gap = regret_perm.mean() - regret_gated.mean()
    assert mean_gap > 12.0, (
        f"permissive-minus-gated post-flip regret gap {mean_gap:.1f} <= 12 "
        f"(calibrated disjoint-set gaps 29.5-35.7, verification 40.4)"
    )

    # (b) The typical seed is also hurt, not just the locked tail (calibrated
    # median gaps 12-19, threshold >= 2.4x below).
    median_gap = float(np.median(regret_perm) - np.median(regret_gated))
    assert median_gap > 5.0, (
        f"median permissive-minus-gated regret gap {median_gap:.1f} <= 5 "
        f"(calibrated 12-19)"
    )

    # (c) The gate keeps every seed out of catastrophic stale-policy lock-in:
    # calibrated gated mean ~97 (max over 192 seeds: 145) vs permissive locks
    # of 200-558 on ~8% of seeds.
    assert regret_gated.mean() <= 150.0, (
        f"gated mean post-flip regret {regret_gated.mean():.1f} > 150 (calibrated ~97)"
    )
    assert regret_gated.max() <= 300.0, (
        f"gated worst-seed post-flip regret {regret_gated.max():.1f} > 300 "
        f"(calibrated max 145 over 192 seeds; permissive locks reach 558)"
    )

    # (d) Mechanism: the gated benefit comes from dreams that flowed before
    # the flip and were rejected right after it, while the permissive arm
    # dreamed straight through the corruption.
    pre_acc = acc_gated[:, FLIP_PHASE - 100 : FLIP_PHASE].mean(axis=1)
    assert float(pre_acc.mean()) >= 0.9, (
        f"gated pre-flip acceptance mean {pre_acc.mean():.3f} < 0.9 (calibrated 0.99)"
    )
    assert float(pre_acc.min()) >= 0.6, (
        f"gated pre-flip acceptance min {pre_acc.min():.3f} < 0.6 (calibrated 0.81)"
    )
    acute_acc = acc_gated[:, FLIP_PHASE + 2 : FLIP_PHASE + 12].mean()
    assert float(acute_acc) <= 0.2, (
        f"gated acute post-flip acceptance {acute_acc:.3f} > 0.2 (calibrated <= 0.03)"
    )
    perm_post = acc_perm[:, FLIP_PHASE : FLIP_PHASE + 50].mean(axis=1)
    assert float(perm_post.min()) >= 0.95, (
        f"permissive acceptance dropped to {perm_post.min():.2f} post-flip "
        f"(calibrated exactly 1.0)"
    )
