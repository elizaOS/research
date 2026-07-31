"""Development diagnostic for selected PrototypeAgent mechanisms.

CartPole-style benchmarks saturate: once any agent balances the pole, no integration
benefit is measurable.  The two closed-loop micro-MDPs in ``streams/closed_loop.py``
do not saturate — :class:`SwitchingTwoStateMDP` inverts its payoff matrix on a fixed
schedule forever, and :class:`RiverSwimMDP` hides most of its achievable reward
behind a hard exploration barrier. This module compares one selected
:class:`PrototypeAgent` composition (OaK options + world model + guarded
dreaming + curation) against the flat :class:`DifferentialSARSAAgent` baseline
on both, with paired development seeds.

This is not a promoted integration result. The same eight seeds were used to
calibrate configurations and assertion margins; there is no held-out seed
schedule, confidence interval, versioned artifact, or matched compute budget.
The option subgoals are supplied observation indices, and the composition does
not enable Horde, IA, GRU state, or autonomous feature discovery. A pass
therefore establishes a useful development direction for selected mechanisms,
not a full PrototypeAgent or Alberta Plan demonstration.

Protocol (8 paired seeds each; seed s -> ``jr.key(s)`` split identically per agent):

* **Switching domain**: phase_length=400, 4000 steps (9 payoff switches). Optimal
  average reward is 1.0 in both phases; uniform random earns 0.5. The selected
  agent runs with 2 guarded dreams/step and curation fired at steps 1000/2000/3000.
* **RiverSwim domain**: 6 states, 5000 steps. Optimal average reward is 0.857
  (all-RIGHT policy); the all-LEFT trap pays 0.005; uniform random earns 0.003.
  The selected composition runs with two supplied option subtasks (features 2
  and 5 — a mid-chain subgoal and the far right end), 2 guarded dreams/step,
  curation fired at step 2500.

Both flat epsilon schedules were measured and the baseline was never tuned down:
the repo-proven control-gate schedule (0.5 -> 0.02 over 2500 steps, from
``test_control_learning_gates.py``) and a constant-0.1 schedule matching the
selected composition's own ``epsilon_base``. The constant schedule is the stronger
flat baseline under recurring nonstationarity, so the selected composition is asserted
to *dominate* flat_gate and to be *no worse than* flat_fixed on the switching
domain; on RiverSwim both schedules fail identically.

Calibration (2026-07-30, CPU, 8 seeds).  Numbers were measured on two snapshots of
this actively-developed library (STOMP/dream semantics changed mid-calibration);
every asserted margin is >= ~2.5x looser than the weakest measured gap across both
snapshots.

Switching (means over seeds; post_switch = mean reward in the 120 steps after each
switch; recovery = steps for a trailing 25-step mean to regain 0.8):

    variant       life         post_switch   recovery
    flat_gate     0.859        0.732         51.1
    flat_fixed    0.882        0.771         43.2
    proto_dream   0.878-0.894  0.779-0.854   31.5-44.8   (across snapshots)
    proto_nodream 0.881-0.882  0.794-0.795   40.5-40.9

    paired proto_dream - flat_gate : life +0.021..+0.035 (8/8 seeds better in
        both snapshots), post_switch +0.051..+0.122 (8/8), recovery -6.4..-19.6
    paired proto_dream - flat_fixed: life -0.002..+0.013, post_switch
        +0.011..+0.083, recovery -11.7..+1.5  (parity, asserted as no-worse)
    paired proto_nodream - flat_gate: post_switch +0.060..+0.064 (8/8)

RiverSwim (found_right = steps receiving the right-end reward 1.0):

    variant     life            final1000       seeds with found_right>0
    flat (both) 0.0045-0.0047   0.0047-0.0050   0 of 8  (all 16 runs stuck)
    proto_selected 0.062-0.064  0.054-0.087     7-8 of 8
    (ablation: options-without-dreams reaches 0.27-0.32 mean life, 8/8 find
     the goal, min paired life gain +0.079..+0.106)

    paired proto_selected - flat_gate: life +0.058 mean in both snapshots; worst
        per-seed diff -0.003 (snapshot 1) / +0.017 (snapshot 2)

The flat agent NEVER earned the right-end reward in any run of either schedule —
it converges to the all-LEFT trap (life ~ 0.005, an order of magnitude below the
0.857 optimum). The selected composition's temporally-extended options repeatedly
reach the far end: that exploration benefit is the RiverSwim claim.

Honest findings that did NOT survive as assertions (documented only):

* Dream attribution on the switching domain was snapshot-dependent: on the first
  snapshot dreams-on beat the dreams-off ablation post-switch by +0.058 (8/8
  seeds); after concurrent STOMP/dream-semantics changes the two variants became
  statistically indistinguishable there (+/-0.02).  The integrated-vs-flat gaps
  survived both snapshots; the component attribution did not, so it is not gated.
* Earlier snapshots showed expectation-valued dreams hurting RiverSwim
  exploration relative to options-only control because a soft stochastic
  next-state blend dilutes the one-hot Q-function. On the current source, an
  eight-seed consumed-seed development rerun compared the same two-proposal
  budget with expectation-valued versus sampled-one-hot next observations:
  mean lifetime reward was 0.1930 versus 0.2475 (paired +0.0545), with 6/8
  positive differences and a per-seed range of -0.0769 to +0.1670. Because
  this arm was designed after inspecting these development seeds, it is a
  nonpromoting diagnostic with no efficacy gate or held-out claim.
* Curation can evict the useful right-end option on RiverSwim (2 of 8 seeds
  collapsed in a no-dream + curation ablation on snapshot 1); with the mid-chain
  subgoal option present the selected composition stayed ahead of flat on every seed.
* The selected composition's constant epsilon_base=0.1 costs it the *converged* tail
  windows vs the decayed flat_gate schedule (final-500 ~0.89 vs ~0.93) — the
  price of retained adaptability. Lifetime reward still favours the selected
  agent because post-switch losses dominate this stream.

Runtime: ~45-55s total on CPU (module-scoped fixtures share rollouts across tests;
jitted chunk runners are cached per agent-config via ``_dedup``).
"""

from __future__ import annotations

import functools
from typing import Literal

import jax
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)
from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import PrototypeAgent, PrototypeAgentConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.streams import (
    RiverSwimConfig,
    RiverSwimMDP,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

pytestmark = [pytest.mark.slow, pytest.mark.development]

NUM_SEEDS = 8

# Switching domain protocol
SW_STEPS = 4000
SW_PHASE = 400
SW_CHUNK = 250
SW_CURATE_AT = frozenset({1000, 2000, 3000})
SW_POST_WINDOW = 120
SW_RECOVERY_LEVEL = 0.8
SW_RECOVERY_MA = 25

# RiverSwim protocol
RS_STEPS = 5000
RS_CHUNK = 500
RS_STATES = 6
RS_CURATE_AT = frozenset({2500})

# Calibrated margins.  Each constant is >= ~2.5x looser than the weakest gap
# measured across both library snapshots (see module docstring for the tables).
SW_POST_SWITCH_MARGIN = 0.02  # measured +0.051..+0.122 vs flat_gate
SW_RECOVERY_MARGIN = 2.0  # measured 6.4..19.6 steps faster than flat_gate
SW_LIFE_GATE_MARGIN = 0.005  # measured +0.021..+0.035 vs flat_gate
SW_NO_WORSE_LIFE = 0.02  # measured -0.002..+0.013 vs flat_fixed
SW_NO_WORSE_POST = 0.02  # measured +0.011..+0.083 vs flat_fixed
SW_NO_WORSE_RECOVERY = 8.0  # measured -11.7..+1.5 steps vs flat_fixed
SW_NODREAM_POST_MARGIN = 0.02  # measured +0.060..+0.064 vs flat_gate
RS_FLAT_STUCK_CEILING = 0.05  # measured 0.0045 (all-LEFT trap pays 0.005)
RS_LIFE_MARGIN = 0.02  # measured +0.058 mean paired gain (both snapshots)
RS_PER_SEED_NO_WORSE = 0.01  # measured worst per-seed diff -0.003
RS_MIN_SEEDS_FINDING_GOAL = 4  # measured 7-8 of 8 seeds with found_right >= 10


# ---------------------------------------------------------------------------
# Flat baseline runner
# ---------------------------------------------------------------------------


def _run_flat(
    env: SwitchingTwoStateMDP | RiverSwimMDP,
    seed: int,
    num_steps: int,
    *,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> np.ndarray:
    """Closed-loop rollout of a fresh flat DifferentialSARSAAgent; returns rewards.

    Hyperparameters mirror ``tests/test_control_learning_gates.py`` — the repo's
    proven flat-control configuration for these environments.
    """
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=env.n_actions,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        )
    )
    env_key, agent_key, scan_key = jr.split(jr.key(seed), 3)
    env_state = env.init(env_key)
    agent_state = agent.init(env.feature_dim, agent_key)
    agent_state, _ = agent.start(agent_state, env.observe(env_state))

    def scan_fn(carry, step_key):
        a_state, e_state = carry
        obs, reward, new_e_state = env.step(e_state, a_state.last_action, step_key)
        result = agent.update(a_state, reward, obs)
        return (result.state, new_e_state), reward

    _final, rewards = jax.lax.scan(
        scan_fn, (agent_state, env_state), jr.split(scan_key, num_steps)
    )
    return np.asarray(rewards)


# ---------------------------------------------------------------------------
# Integrated PrototypeAgent runner
# ---------------------------------------------------------------------------


def _make_proto_agent(
    *,
    obs_dim: int,
    feature_indices: tuple[int, ...],
    n_dreams: int,
    max_option_steps: int,
    epsilon_option: float,
    buffer_capacity: int,
    wm_error_decay: float,
    dream_warmup: int,
    dream_max_error_ema: float,
    min_steps_before_curation: int,
    dream_next_observation_mode: Literal[
        "model_prediction",
        "sample_one_hot",
    ] = "model_prediction",
) -> PrototypeAgent:
    """Full-stack PrototypeAgent: OaK options + linear world model (+ dreaming)."""
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=tuple(
                SubtaskSpec(feature_index=f, max_option_steps=max_option_steps)
                for f in feature_indices
            ),
            observation_dim=obs_dim,
            n_primitive_actions=2,
            base_step_size=0.1,
            base_avg_reward_step_size=0.01,
            epsilon_base=0.1,
            epsilon_option=epsilon_option,
        ),
        min_steps_before_curation=min_steps_before_curation,
    )
    # include_action_interactions lets the linear model represent the payoff
    # matrix exactly on one-hot states (reward depends on state x action jointly).
    wm = ActionConditionedWorldModelConfig(
        observation_dim=obs_dim,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.2,
        error_decay=wm_error_decay,
        include_action_interactions=True,
    )
    dreaming = None
    if n_dreams > 0:
        dreaming = DreamingConfig(
            warmup_steps=dream_warmup, max_model_error_ema=dream_max_error_ema
        )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=oak,
            world_model=wm,
            dreaming=dreaming,
            buffer_capacity=buffer_capacity,
            n_dreams_per_step=n_dreams,
            dream_next_observation_mode=dream_next_observation_mode,
        )
    )


@functools.partial(jax.jit, static_argnums=(0, 1))
def _proto_chunk(agent, env, carry, keys):
    """One jitted closed-loop chunk; cached per (agent, env) instance pair."""

    def step(c, k):
        p_state, e_state, action = c
        obs, reward, e2 = env.step(e_state, action, k)
        res = agent.update(p_state, reward, obs)
        return (res.state, e2, res.action), reward

    return jax.lax.scan(step, carry, keys)


# Curation returns a fresh PrototypeAgent instance even for an already-seen config
# (on these tiny observation spaces the replacement feature pool is small), which
# would defeat the per-instance jit cache.  Dedup instances by static config so
# each distinct config compiles exactly once across all seeds and variants.
_PROTO_CACHE: dict[tuple, PrototypeAgent] = {}


def _dedup(agent: PrototypeAgent, domain: str, n_dreams: int) -> PrototypeAgent:
    key = (
        domain,
        n_dreams,
        agent.config.dream_next_observation_mode,
        tuple(s.feature_index for s in agent.config.oak.stomp.subtask_specs),
    )
    return _PROTO_CACHE.setdefault(key, agent)


def _run_proto(
    env: SwitchingTwoStateMDP | RiverSwimMDP,
    template: PrototypeAgent,
    seed: int,
    num_steps: int,
    chunk: int,
    curate_at: frozenset[int],
    domain: str,
    n_dreams: int,
) -> np.ndarray:
    """Closed-loop chunked rollout with Python-level curation at chunk boundaries."""
    agent = _dedup(template, domain, n_dreams)
    env_key, agent_key, run_key = jr.split(jr.key(seed), 3)
    e_state = env.init(env_key)
    obs = env.observe(e_state)
    p_state = agent.start(agent.init(agent_key), obs)
    action = agent.act(p_state, obs)

    rewards: list[np.ndarray] = []
    step = 0
    carry = (p_state, e_state, action)
    while step < num_steps:
        run_key, chunk_key = jr.split(run_key)
        carry, r = _proto_chunk(agent, env, carry, jr.split(chunk_key, chunk))
        rewards.append(np.asarray(r))
        step += chunk
        if step in curate_at:
            run_key, c_key = jr.split(run_key)
            new_agent, new_p_state = agent.curate(carry[0], c_key)
            agent = _dedup(new_agent, domain, n_dreams)
            carry = (new_p_state, carry[1], carry[2])
    return np.concatenate(rewards)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _switching_metrics(rewards: np.ndarray) -> dict[str, float]:
    switches = range(SW_PHASE, SW_STEPS, SW_PHASE)
    post = float(np.mean([rewards[t : t + SW_POST_WINDOW].mean() for t in switches]))
    recoveries = []
    for t in switches:
        segment = rewards[t : t + SW_PHASE]
        trailing = np.convolve(
            segment, np.ones(SW_RECOVERY_MA) / SW_RECOVERY_MA, mode="valid"
        )
        if trailing.max() >= SW_RECOVERY_LEVEL:
            recoveries.append(float(np.argmax(trailing >= SW_RECOVERY_LEVEL) + SW_RECOVERY_MA))
        else:
            recoveries.append(float(SW_PHASE))
    return {
        "life": float(rewards.mean()),
        "post_switch": post,
        "recovery": float(np.mean(recoveries)),
    }


def _mean(per_seed: list[dict[str, float]], key: str) -> float:
    return float(np.mean([m[key] for m in per_seed]))


# ---------------------------------------------------------------------------
# Module-scoped rollouts (shared across tests)
# ---------------------------------------------------------------------------


def _switching_proto_template(n_dreams: int) -> PrototypeAgent:
    return _make_proto_agent(
        obs_dim=2,
        feature_indices=(0,),
        n_dreams=n_dreams,
        max_option_steps=4,
        epsilon_option=0.1,
        buffer_capacity=32,
        wm_error_decay=0.9,
        dream_warmup=20,
        dream_max_error_ema=0.3,
        min_steps_before_curation=200,
    )


@pytest.fixture(scope="module")
def switching_runs() -> dict[str, list[dict[str, float]]]:
    """Per-seed switching-domain metrics for both flat schedules and both proto variants."""
    env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=SW_PHASE))
    raw: dict[str, list[np.ndarray]] = {
        "flat_gate": [
            _run_flat(
                env, s, SW_STEPS, epsilon_start=0.5, epsilon_end=0.02, epsilon_decay_steps=2500
            )
            for s in range(NUM_SEEDS)
        ],
        "flat_fixed": [
            _run_flat(
                env, s, SW_STEPS, epsilon_start=0.1, epsilon_end=0.1, epsilon_decay_steps=0
            )
            for s in range(NUM_SEEDS)
        ],
        "proto_dream": [
            _run_proto(
                env, _switching_proto_template(2), s, SW_STEPS, SW_CHUNK, SW_CURATE_AT, "sw", 2
            )
            for s in range(NUM_SEEDS)
        ],
        "proto_nodream": [
            _run_proto(
                env, _switching_proto_template(0), s, SW_STEPS, SW_CHUNK, SW_CURATE_AT, "sw", 0
            )
            for s in range(NUM_SEEDS)
        ],
    }
    for name, arrays in raw.items():
        for s, r in enumerate(arrays):
            assert np.isfinite(r).all(), f"non-finite rewards in {name} seed {s}"
    return {name: [_switching_metrics(r) for r in arrays] for name, arrays in raw.items()}


@pytest.fixture(scope="module")
def riverswim_runs() -> dict[str, list[np.ndarray]]:
    """Per-seed RiverSwim reward arrays for the flat baseline and full proto agent."""
    env = RiverSwimMDP(RiverSwimConfig(n_states=RS_STATES))

    def proto_template(
        dream_next_observation_mode: Literal[
            "model_prediction",
            "sample_one_hot",
        ] = "model_prediction",
    ) -> PrototypeAgent:
        return _make_proto_agent(
            obs_dim=RS_STATES,
            feature_indices=(2, 5),
            n_dreams=2,
            max_option_steps=8,
            epsilon_option=0.2,
            buffer_capacity=256,
            wm_error_decay=0.95,
            dream_warmup=50,
            dream_max_error_ema=0.5,
            min_steps_before_curation=500,
            dream_next_observation_mode=dream_next_observation_mode,
        )

    raw: dict[str, list[np.ndarray]] = {
        "flat_gate": [
            _run_flat(
                env, s, RS_STEPS, epsilon_start=0.5, epsilon_end=0.02, epsilon_decay_steps=2500
            )
            for s in range(NUM_SEEDS)
        ],
        "proto_full": [
            _run_proto(env, proto_template(), s, RS_STEPS, RS_CHUNK, RS_CURATE_AT, "rs", 2)
            for s in range(NUM_SEEDS)
        ],
        "proto_sample_one_hot": [
            _run_proto(
                env,
                proto_template("sample_one_hot"),
                s,
                RS_STEPS,
                RS_CHUNK,
                RS_CURATE_AT,
                "rs",
                2,
            )
            for s in range(NUM_SEEDS)
        ],
    }
    for name, arrays in raw.items():
        for s, r in enumerate(arrays):
            assert np.isfinite(r).all(), f"non-finite rewards in {name} seed {s}"
    return raw


# ---------------------------------------------------------------------------
# Switching domain: post-switch recovery and aggregate reward
# ---------------------------------------------------------------------------


class TestSwitchingDomain:
    def test_integrated_beats_flat_gate_on_post_switch_reward(self, switching_runs) -> None:
        """Mean reward in the 120 steps after each payoff switch beats flat_gate.

        Measured paired gap +0.051..+0.122 across snapshots (every seed positive
        in both); asserted margin 0.02.
        """
        proto = _mean(switching_runs["proto_dream"], "post_switch")
        flat = _mean(switching_runs["flat_gate"], "post_switch")
        assert proto >= flat + SW_POST_SWITCH_MARGIN, (
            f"post-switch reward: proto {proto:.4f} does not beat "
            f"flat_gate {flat:.4f} by {SW_POST_SWITCH_MARGIN}"
        )

    def test_integrated_recovers_faster_than_flat_gate(self, switching_runs) -> None:
        """Steps to regain a 0.8 trailing mean after a switch beat flat_gate.

        Measured: proto 6.4..19.6 steps faster across snapshots; asserted
        margin 2 steps.
        """
        proto = _mean(switching_runs["proto_dream"], "recovery")
        flat = _mean(switching_runs["flat_gate"], "recovery")
        assert proto <= flat - SW_RECOVERY_MARGIN, (
            f"recovery: proto {proto:.1f} steps is not at least "
            f"{SW_RECOVERY_MARGIN} steps faster than flat_gate {flat:.1f}"
        )

    def test_integrated_lifetime_reward_beats_flat_gate(self, switching_runs) -> None:
        """Lifetime mean reward beats flat_gate and clearly beats uniform random.

        Measured: +0.021..+0.035 vs flat_gate across snapshots (every seed
        positive in both); asserted margin 0.005.
        """
        proto = _mean(switching_runs["proto_dream"], "life")
        flat = _mean(switching_runs["flat_gate"], "life")
        uniform_random = 0.5  # exact: SwitchingTwoStateMDP.uniform_random_average_reward
        assert proto >= flat + SW_LIFE_GATE_MARGIN, (
            f"lifetime reward: proto {proto:.4f} does not beat flat_gate "
            f"{flat:.4f} by {SW_LIFE_GATE_MARGIN}"
        )
        assert proto >= uniform_random + 0.2, (
            f"lifetime reward {proto:.4f} does not clearly beat uniform random 0.5"
        )

    def test_integrated_no_worse_than_strongest_flat_schedule(self, switching_runs) -> None:
        """Parity-with-margin against the constant-epsilon flat schedule.

        flat_fixed is the strongest flat variant under recurring nonstationarity
        (its epsilon never decays away). Across snapshots the selected composition
        measured -0.002..+0.013 lifetime, +0.011..+0.083 post-switch, and
        -11.7..+1.5 recovery steps against it — parity or better, asserted as
        no-worse within generous margins rather than dominance.
        """
        proto = switching_runs["proto_dream"]
        flat = switching_runs["flat_fixed"]
        assert _mean(proto, "life") >= _mean(flat, "life") - SW_NO_WORSE_LIFE, (
            f"lifetime: proto {_mean(proto, 'life'):.4f} worse than flat_fixed "
            f"{_mean(flat, 'life'):.4f} beyond {SW_NO_WORSE_LIFE}"
        )
        assert _mean(proto, "post_switch") >= _mean(flat, "post_switch") - SW_NO_WORSE_POST, (
            f"post-switch: proto {_mean(proto, 'post_switch'):.4f} worse than "
            f"flat_fixed {_mean(flat, 'post_switch'):.4f} beyond {SW_NO_WORSE_POST}"
        )
        assert _mean(proto, "recovery") <= _mean(flat, "recovery") + SW_NO_WORSE_RECOVERY, (
            f"recovery: proto {_mean(proto, 'recovery'):.1f} slower than flat_fixed "
            f"{_mean(flat, 'recovery'):.1f} beyond {SW_NO_WORSE_RECOVERY} steps"
        )

    def test_options_and_curation_alone_also_beat_flat_gate(self, switching_runs) -> None:
        """The dreams-off integrated variant still beats flat_gate post-switch.

        Shows the OaK-options + curation stack carries a benefit independent of
        dreaming.  Measured +0.060..+0.064 across snapshots (every seed
        positive); asserted margin 0.02.  Dream attribution itself was
        snapshot-dependent and is documented, not asserted (module docstring).
        """
        nodream = _mean(switching_runs["proto_nodream"], "post_switch")
        flat = _mean(switching_runs["flat_gate"], "post_switch")
        assert nodream >= flat + SW_NODREAM_POST_MARGIN, (
            f"post-switch reward: proto_nodream {nodream:.4f} does not beat "
            f"flat_gate {flat:.4f} by {SW_NODREAM_POST_MARGIN}"
        )


# ---------------------------------------------------------------------------
# RiverSwim: exploration via temporally-extended options
# ---------------------------------------------------------------------------


class TestRiverSwimExploration:
    def test_sampled_one_hot_arm_is_a_finite_development_diagnostic(
        self,
        riverswim_runs,
    ) -> None:
        """Exercise the consumed-seed arm without turning it into a gate."""
        sampled = riverswim_runs["proto_sample_one_hot"]
        assert len(sampled) == NUM_SEEDS
        assert all(
            rewards.shape == (RS_STEPS,) and np.isfinite(rewards).all()
            for rewards in sampled
        )

    def test_flat_baseline_is_stuck_in_the_left_trap(self, riverswim_runs) -> None:
        """The flat agent stays near the all-LEFT gain (0.005) on every seed.

        Measured mean lifetime reward 0.0045 across 8 seeds with 0 steps of
        right-end reward in all runs (also measured for the constant-0.1
        schedule).  Asserted ceiling 0.05 — a flat agent that found and kept the
        right-end reward would earn an order of magnitude more (optimum 0.857).
        """
        flat_life = np.array([r.mean() for r in riverswim_runs["flat_gate"]])
        assert float(flat_life.mean()) < RS_FLAT_STUCK_CEILING, (
            f"flat baseline unexpectedly escaped the left trap: mean lifetime "
            f"reward {flat_life.mean():.4f} >= {RS_FLAT_STUCK_CEILING}"
        )

    def test_integrated_agent_finds_the_distant_reward(self, riverswim_runs) -> None:
        """The selected composition's options reach the far-right reward; flat never does.

        Measured across snapshots: 7-8 of 8 seeds collect the right-end reward
        (>= 100 rewarded steps each); mean paired lifetime gain +0.058 with
        worst per-seed diff -0.003.  Asserted: >= 4 seeds with >= 10 rewarded
        steps, mean gain >= 0.02, and per-seed no-worse within 0.01.
        """
        flat = riverswim_runs["flat_gate"]
        proto = riverswim_runs["proto_full"]

        seeds_finding_goal = sum(1 for r in proto if int((r >= 0.99).sum()) >= 10)
        assert seeds_finding_goal >= RS_MIN_SEEDS_FINDING_GOAL, (
            f"only {seeds_finding_goal}/{NUM_SEEDS} seeds reached the right-end "
            f"reward at least 10 times (need {RS_MIN_SEEDS_FINDING_GOAL})"
        )

        proto_life = np.array([r.mean() for r in proto])
        flat_life = np.array([r.mean() for r in flat])
        assert float(proto_life.mean()) >= float(flat_life.mean()) + RS_LIFE_MARGIN, (
            f"lifetime reward: proto {proto_life.mean():.4f} does not beat flat "
            f"{flat_life.mean():.4f} by {RS_LIFE_MARGIN}"
        )
        worst_paired = float((proto_life - flat_life).min())
        assert worst_paired >= -RS_PER_SEED_NO_WORSE, (
            f"a seed regressed beyond the no-worse margin: worst paired lifetime "
            f"diff {worst_paired:.4f} < -{RS_PER_SEED_NO_WORSE}"
        )
