"""Development single-life integration diagnostic over recurring control.

RESEARCH_STATUS.md defines **L3 — integration** as one uninterrupted agent life
that demonstrates the required interactions, retention, recovery, bounded
resource use, and causal ablations. This module exercises a useful subset of
that shape: a continuing agent lives through a 48,000-step closed-loop
nonstationary world with no learner reset, episode, or mid-life hyperparameter
change. Control learning, representation-dependent memory, plasticity,
stability, and stressor recovery are measured on the same life, with a
per-seed-paired representation ablation.

It is not promoted L3 evidence. The context is directly observable, the
strong-memory representation is hand constructed, all eight seeds were used
for calibration, there is no immutable held-out artifact or confidence
interval, and the full-stack arm has no matched component ablations. It also
does not demonstrate autonomous learned state, general feature discovery,
partner learning, a causally useful model/planner, option discovery, or
intelligence amplification. The test is retained as a development integration
and long-run regression lane.

The life
========

:class:`SwitchingTwoStateMDP` (phase_length=400) inverts its 2x2 payoff matrix
every 400 steps forever: phase A rewards toggling, phase B rewards staying, the
optimal average reward is exactly 1.0 in both phases and uniform random earns
exactly 0.5 (both computed analytically by the env).  48,000 steps = 120 phase
occurrences = **60 recurrences of each rule** (task requires >= 10).

The base env hides its phase, so the test wraps its observation: the agent sees
``[one_hot(latent_state), one_hot(active_phase)]`` (dim 4) — the recurring rule
context is *observable*, which is the precondition for context-keyed memory.
Mid-life stressor: for steps [22000, 24000) — exactly phase occurrences 55-59 —
iid Gaussian noise (std 0.5) is added to **all four** observation channels,
corrupting both the percept and the context signal for 2,000 consecutive steps.

Rung 1 — certified memory mechanism (primary)
=============================================

:class:`DifferentialSARSAAgent` with ``use_bias=False`` over **exclusive
context-gated features** ``[c_A * x, c_B * x]`` — the memory mechanism
certified by ``streams/matrix_game.py`` + ``streams/gauntlet.py``: each rule
occupies a disjoint Q-weight block, the inactive rule's policy is untouchable
while the other rule is in force, and disabling the bias closes the one
always-on shared-parameter forgetting channel.  Constant epsilon 0.05, constant
step-sizes (q 0.1, rbar 0.01): every update is identical from birth to death.

The **causal ablation twin** is the same agent, same seeds, same life protocol,
with the context channels withheld from its features (it sees only ``x``): both
rules then compete for one weight block — perfect plasticity, memory impossible
in principle.

Rung 2 — multi-component prototype
==================================

The full :class:`PrototypeAgent` (OaK options + linear action-conditioned world
model + 2 guarded dreams/step + periodic curation) from
``tests/test_prototype_nonsaturating.py``'s switching-domain template, adapted
only to the 4-dim observation (obs_dim 2 -> 4) and to a uniform curation cadence
(every 4000 steps, 11 events) for the 12x-longer life.  Same env stream
protocol, same seeds, same metrics.

Calibration (2026-07-30, CPU, 8 seeds; every asserted margin is >= 2x the
distance to the weakest measured value)
=======================================

Per-phase early-window = mean reward over the first 50 steps of a phase
occurrence (the re-coordination / savings signal); recovery = steps for a
trailing-25 mean to regain 0.8 (floor 25); "settled" = phases >= 6 excluding
stressor phases 55-60.

Rung 1, means over 8 seeds (worst seed in parens):

    variant   life            late8000        settled-early    settled-early-min
    gated     0.9697 (0.9688) 0.9750 (0.9736) 0.974  (0.964)   0.898 (0.860)
    ablation  0.8921 (0.8906) 0.8946 (0.8882) 0.30-0.40        0.140 (max 0.220)

    recovery: gated 25.0 early / 25.0 late (the floor — instant, every switch,
    all life); ablation 54.4 early / 52.5 late (relearns every switch, ~52-56
    steps, no upward trend).  Paired settled-early gap gated-ablation:
    mean +0.598, per-seed min +0.586.  Stressor: gated during 0.847
    (min 0.832), post-vs-pre per-seed worst -0.0045; final |q_weights| max
    0.908 (gated) / 2.119 (ablation); rbar 0.920..0.985; q_bias identically 0.

Early-window trajectory by recurrence k of each rule (means over 8 seeds) —
"stays at its best for the rest of the life after the first few exposures":

    gated  rule A: k0 0.92 | k1 0.99 | k5 0.96 | k20 0.98 | k40 0.97 | k59 0.97
    gated  rule B: k0 0.95 | k1 0.97 | k5 0.97 | k20 0.98 | k40 0.96 | k59 0.98
      (only stressor-recurrence dips: e.g. rule A k29 = phase 58 -> 0.77 under
       noise, back to 0.97 immediately after — the memory survives the stressor)
    ablation rule A: k0 0.92 | k1 0.11 | k5 0.37 | k20 0.36 | k40 0.38 | k59 0.35
    ablation rule B: k0 0.02 | k1 0.20 | k5 0.39 | k20 0.40 | k40 0.42 | k59 0.40
      (k0 rule A is high because it is the from-scratch learning phase, not a
       recurrence; every true recurrence pays the full relearning cost forever)

Rung 2, means over 8 seeds (worst seed in parens):

    life 0.9003 (0.8849); late8000 0.8990 (0.8764); recovery 38.9 early /
    37.8 late, per-seed late-early diff in [-3.67, +1.45]; early-window
    first-6 phases 0.659 -> late-third 0.710, per-seed retention diff
    (late - first) in [-0.0017, +0.121], mean +0.051; stressor during 0.846
    (min 0.824), post-vs-pre per-seed diff in [-0.0585, +0.0275]; state pytree
    440 elements at birth and 440 at death, all finite.

Honest scope notes
==================

* The rule context is *oracle-observable* (a channel in the observation), and
  rung 1's gating is the certified hand-built feature map.  Autonomous
  discovery of ``ctx*x`` features from raw observations is exercised in
  ``test_gauntlet_discovery.py`` and remains gated by its known scale-shock gap.
* Rung 2's retention is genuinely *weak*: its settled early-window (~0.71) sits
  far below the gated rung (~0.97) because its linear Q over ``[x, ctx]``
  cannot represent both rules simultaneously without interaction features.  Its
  retention is therefore asserted only as "not degrading over the life"
  (measured mildly positive, +0.05 mean); the *memory* claim of this module
  rests on rung 1 and its paired ablation.
* The ablation twin's during-stress reward (0.867) exceeds its own settled
  post-switch early-window — input noise incidentally dithers a memoryless
  agent — which is why the memory metrics exclude the stressor phases.

Runtime: ~70-80s on CPU (8 seeds x (2 SARSA lives + 1 prototype life) of 48k
steps each; module-scoped fixtures share the rollouts across tests).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
)
from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import PrototypeAgent, PrototypeAgentConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.streams import SwitchingTwoStateConfig, SwitchingTwoStateMDP
from alberta_framework.streams.closed_loop import PHASE_A

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Life protocol
# ---------------------------------------------------------------------------

NUM_SEEDS = 8
LIFE_STEPS = 48_000
PHASE = 400
N_PHASES = LIFE_STEPS // PHASE  # 120 -> 60 recurrences of each of the 2 rules
STRESS_START = 22_000  # phase 55 begins exactly here
STRESS_END = 24_000  # ... and phase 60 begins exactly here
STRESS_STD = 0.5

EARLY_W = 50  # early-window width (re-coordination / savings signal)
REC_MA = 25  # trailing-mean width for recovery (also the metric floor)
REC_LEVEL = 0.8  # recovery criterion on the trailing mean
SETTLE_PHASE = 6  # first 3 exposures per rule are the learning transient
LATE_START = 80  # late-life third begins at this phase occurrence
LATE_WINDOW = 8_000  # trailing steps for the "approaches optimum" check

_STRESS_PHASES = frozenset(range(STRESS_START // PHASE, STRESS_END // PHASE))
# Non-stressor phases (the phase straddling the stressor's end is excluded too).
_CLEAN = np.array(
    [p not in _STRESS_PHASES and p != STRESS_END // PHASE for p in range(N_PHASES)]
)
_SETTLED = _CLEAN & (np.arange(N_PHASES) >= SETTLE_PHASE)
_LATE = _CLEAN & (np.arange(N_PHASES) >= LATE_START)

_ENV = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=PHASE))
OPTIMAL = _ENV.optimal_average_reward(PHASE_A)  # 1.0 (same in PHASE_B)
RANDOM = _ENV.uniform_random_average_reward(PHASE_A)  # 0.5 (same in PHASE_B)

# ---------------------------------------------------------------------------
# Calibrated margins (see module docstring for the measured tables)
# ---------------------------------------------------------------------------

# (a) control learning — gated rung
LIFE_FLOOR = 0.85  # measured per-seed min 0.9688
LIFE_VS_RANDOM = 0.35  # measured mean gap over random +0.4697
NEAR_OPT_SLACK = 0.08  # measured late-8000 shortfall from optimum <= 0.0264
# (b) memory — gated rung
MEM_SETTLED_MIN = 0.60  # measured per-seed min 0.860; ablation max 0.220
MEM_LATE_MEAN = 0.90  # measured per-seed min 0.9635
MEM_NO_DEGRADE = 0.05  # measured worst late-vs-first diff -0.0132
MEM_ABLATION_GAP = 0.30  # measured per-seed min paired gap +0.586
ABLATION_LIFE_FLOOR = 0.80  # measured per-seed min 0.8906 (twin is no straw man)
ABLATION_MEMORYLESS_CEIL = 0.60  # measured per-seed settled-early mean max 0.4030
# (c) plasticity
REC_TREND_GATED = 3.0  # measured late-vs-early diff 0.00 on every seed
REC_SETTLED_CEIL = REC_MA + 5  # measured settled mean exactly 25.0 (the floor)
REC_TREND_ABLATION = 8.0  # measured worst-case bound +0.81
# (d) stability
Q_BOUND_GATED = 4.0  # measured max |q_weights| 0.908
Q_BOUND_ABLATION = 6.0  # measured max |q_weights| 2.119
RBAR_LO, RBAR_HI = 0.4, 1.2  # measured 0.920..0.985 (optimum 1.0, random 0.5)
# (e) stressor
STRESS_DROP = 0.05  # measured per-seed worst post-vs-pre -0.0045
STRESS_DURING_FLOOR = 0.65  # measured per-seed min during-stress mean 0.832

# Prototype rung (margins loosened further: core/oak|options|prototype_agent
# are under active concurrent development; nonsaturating measured ~0.02
# snapshot-to-snapshot drift in life means)
P_LIFE_FLOOR = 0.70  # measured per-seed min 0.8849
P_LIFE_MEAN = 0.75  # measured mean 0.9003
P_LATE_FLOOR = 0.70  # measured per-seed min late-8000 0.8764
P_REC_TREND_SEED = 8.0  # measured per-seed late-vs-early diff in [-3.67, +1.45]
P_REC_TREND_MEAN = 4.0  # measured mean diff -1.15
P_RET_SEED_FLOOR = -0.08  # measured per-seed retention diff worst -0.0017
P_RET_MEAN_FLOOR = -0.02  # measured mean retention diff +0.051
P_STRESS_DROP_SEED = 0.12  # measured per-seed worst post-vs-pre -0.0585
P_STRESS_DROP_MEAN = 0.05  # measured mean -0.0085
P_STRESS_DURING_FLOOR = 0.60  # measured per-seed min 0.8240
P_STATE_GROWTH = 2.0  # measured exactly 1.0 (440 elements at birth and death)

# Prototype protocol adaptation
P_CHUNK = 1_000
P_CURATE_EVERY = 4_000


# ---------------------------------------------------------------------------
# Shared life machinery
# ---------------------------------------------------------------------------


def _context_observation(env_state) -> jax.Array:
    """Raw 4-dim observation: latent one-hot plus observable phase one-hot."""
    return jnp.concatenate(
        [
            _ENV.observe(env_state),
            jax.nn.one_hot(_ENV.phase_id(env_state), 2, dtype=jnp.float32),
        ]
    )


def _perturbed(obs4: jax.Array, env_state, noise_key: jax.Array) -> jax.Array:
    """Apply the mid-life input stressor to all four observation channels."""
    in_window = (
        (env_state.step_count >= STRESS_START) & (env_state.step_count < STRESS_END)
    ).astype(jnp.float32)
    return obs4 + in_window * STRESS_STD * jr.normal(noise_key, (4,), dtype=jnp.float32)


def _gated_features(obs4: jax.Array) -> jax.Array:
    """Exclusive context-gated features [c_A * x, c_B * x] (dim 4)."""
    x, ctx = obs4[:2], obs4[2:]
    return jnp.concatenate([ctx[0] * x, ctx[1] * x])


def _plain_features(obs4: jax.Array) -> jax.Array:
    """No-context ablation features: the latent one-hot only (dim 2)."""
    return obs4[:2]


class LifeRun(NamedTuple):
    """One completed life: per-step rewards plus birth/death agent states."""

    rewards: np.ndarray
    init_state: object
    final_state: object
    metrics: dict[str, float]


def _phase_stats(rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Early-window mean and recovery time per phase occurrence."""
    phases = rewards.reshape(N_PHASES, PHASE)
    early = phases[:, :EARLY_W].mean(axis=1)
    kernel = np.ones(REC_MA) / REC_MA
    recov = np.empty(N_PHASES)
    for p in range(N_PHASES):
        trailing = np.convolve(phases[p], kernel, mode="valid")
        hit = trailing >= REC_LEVEL
        recov[p] = (np.argmax(hit) + REC_MA) if hit.any() else PHASE
    return early, recov


def _life_metrics(rewards: np.ndarray) -> dict[str, float]:
    """All single-life scalar metrics used by the assertions."""
    early, recov = _phase_stats(rewards)
    return {
        "life": float(rewards.mean()),
        "late": float(rewards[-LATE_WINDOW:].mean()),
        "early_first": float(early[:SETTLE_PHASE].mean()),
        "early_settled_min": float(early[_SETTLED].min()),
        "early_settled_mean": float(early[_SETTLED].mean()),
        "early_late_mean": float(early[_LATE].mean()),
        "recov_early_mean": float(recov[1 : LATE_START // 2].mean()),
        "recov_late_mean": float(recov[_LATE].mean()),
        "recov_settled_mean": float(recov[_SETTLED].mean()),
        "pre_stress": float(rewards[STRESS_START - 4_000 : STRESS_START].mean()),
        "during_stress": float(rewards[STRESS_START:STRESS_END].mean()),
        "post_stress": float(rewards[STRESS_END + PHASE : STRESS_END + PHASE + 4_000].mean()),
    }


# ---------------------------------------------------------------------------
# Rung 1 runner: DifferentialSARSA over a fixed feature map, one lax.scan life
# ---------------------------------------------------------------------------


def _run_sarsa_life(seed: int, feature_map, feature_dim: int) -> LifeRun:
    """One uninterrupted 48k-step life; temporally uniform updates throughout."""
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=0.05,
            epsilon_end=0.05,
            epsilon_decay_steps=0,  # constant epsilon: uniform updates for life
            use_bias=False,  # the certified continual-memory configuration
        )
    )
    env_key, agent_key, scan_key = jr.split(jr.key(seed), 3)
    e_state = _ENV.init(env_key)
    init_state = agent.init(feature_dim, agent_key)
    a_state, _ = agent.start(init_state, feature_map(_context_observation(e_state)))

    def scan_fn(carry, step_key):
        a_st, e_st = carry
        k_env, k_noise = jr.split(step_key)
        _, reward, e2 = _ENV.step(e_st, a_st.last_action, k_env)
        obs4 = _perturbed(_context_observation(e2), e2, k_noise)
        result = agent.update(a_st, reward, feature_map(obs4))
        return (result.state, e2), reward

    (final_state, _), rewards = jax.lax.scan(
        scan_fn, (a_state, e_state), jr.split(scan_key, LIFE_STEPS)
    )
    rewards = np.asarray(rewards)
    return LifeRun(rewards, init_state, final_state, _life_metrics(rewards))


# ---------------------------------------------------------------------------
# Rung 2 runner: full PrototypeAgent, chunked scan with periodic curation
# ---------------------------------------------------------------------------


def _make_proto() -> PrototypeAgent:
    """The nonsaturating switching-domain template, adapted to obs_dim=4."""
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, max_option_steps=4),),
            observation_dim=4,
            n_primitive_actions=2,
            base_step_size=0.1,
            base_avg_reward_step_size=0.01,
            epsilon_base=0.1,
            epsilon_option=0.1,
        ),
        min_steps_before_curation=200,
    )
    world_model = ActionConditionedWorldModelConfig(
        observation_dim=4,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.2,
        error_decay=0.9,
        include_action_interactions=True,
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=oak,
            world_model=world_model,
            dreaming=DreamingConfig(warmup_steps=20, max_model_error_ema=0.3),
            buffer_capacity=32,
            n_dreams_per_step=2,
        )
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _proto_chunk(agent, carry, keys):
    """One jitted closed-loop chunk of the prototype life."""

    def step(c, k):
        p_state, e_state, action = c
        k_env, k_noise = jr.split(k)
        _, reward, e2 = _ENV.step(e_state, action, k_env)
        obs4 = _perturbed(_context_observation(e2), e2, k_noise)
        result = agent.update(p_state, reward, obs4)
        return (result.state, e2, result.action), reward

    return jax.lax.scan(step, carry, keys)


# Curation returns a fresh PrototypeAgent instance; dedup by static config so
# each distinct config compiles exactly once across seeds (as in
# test_prototype_nonsaturating.py).
_PROTO_CACHE: dict[tuple, PrototypeAgent] = {}


def _dedup(agent: PrototypeAgent) -> PrototypeAgent:
    key = tuple(s.feature_index for s in agent.config.oak.stomp.subtask_specs)
    return _PROTO_CACHE.setdefault(key, agent)


def _run_proto_life(seed: int) -> tuple[LifeRun, int, int]:
    """One uninterrupted prototype life; uniform curation cadence every 4k steps.

    Returns the life plus the state pytree element counts at birth and death.
    """
    agent = _dedup(_make_proto())
    env_key, agent_key, run_key = jr.split(jr.key(seed), 3)
    e_state = _ENV.init(env_key)
    obs4 = _context_observation(e_state)
    p_state = agent.start(agent.init(agent_key), obs4)
    action = agent.act(p_state, obs4)
    n_birth = sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(p_state))

    rewards: list[np.ndarray] = []
    step = 0
    carry = (p_state, e_state, action)
    while step < LIFE_STEPS:
        run_key, chunk_key = jr.split(run_key)
        carry, r = _proto_chunk(agent, carry, jr.split(chunk_key, P_CHUNK))
        rewards.append(np.asarray(r))
        step += P_CHUNK
        if step % P_CURATE_EVERY == 0 and step < LIFE_STEPS:
            run_key, curate_key = jr.split(run_key)
            new_agent, new_p_state = agent.curate(carry[0], curate_key)
            agent = _dedup(new_agent)
            carry = (new_p_state, carry[1], carry[2])
    all_rewards = np.concatenate(rewards)
    n_death = sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(carry[0]))
    return (
        LifeRun(all_rewards, p_state, carry[0], _life_metrics(all_rewards)),
        n_birth,
        n_death,
    )


# ---------------------------------------------------------------------------
# Module-scoped rollouts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sarsa_lives() -> dict[str, list[LifeRun]]:
    """Paired gated / no-context lives for all seeds (rung 1 + its ablation)."""
    assert N_PHASES * PHASE == LIFE_STEPS
    assert N_PHASES // 2 >= 10  # >= 10 recurrences of each rule (actual: 60)
    assert LIFE_STEPS >= 40_000
    out: dict[str, list[LifeRun]] = {}
    for name, feature_map, feature_dim in (
        ("gated", _gated_features, 4),
        ("ablation", _plain_features, 2),
    ):
        runs = []
        for seed in range(NUM_SEEDS):
            run = _run_sarsa_life(seed, feature_map, feature_dim)
            assert np.isfinite(run.rewards).all(), f"non-finite rewards: {name} seed {seed}"
            runs.append(run)
        out[name] = runs
    return out


@pytest.fixture(scope="module")
def proto_lives() -> list[tuple[LifeRun, int, int]]:
    """Full-stack PrototypeAgent lives for all seeds (rung 2)."""
    runs = []
    for seed in range(NUM_SEEDS):
        run, n_birth, n_death = _run_proto_life(seed)
        assert np.isfinite(run.rewards).all(), f"non-finite rewards: proto seed {seed}"
        runs.append((run, n_birth, n_death))
    return runs


def _metric(runs: list[LifeRun], key: str) -> np.ndarray:
    return np.array([r.metrics[key] for r in runs])


# ---------------------------------------------------------------------------
# Rung 1: the certified memory mechanism, all properties on one life
# ---------------------------------------------------------------------------


class TestCertifiedMemoryLife:
    """DifferentialSARSA(use_bias=False) over exclusive context-gated features."""

    def test_control_learning_above_random_toward_optimum(self, sarsa_lives) -> None:
        """(a) Lifetime reward far above random; late life approaches the optimum.

        Measured: life per-seed min 0.9688 (random is exactly 0.5); final-8000
        mean per-seed min 0.9736 against the exact optimum 1.0.
        """
        life = _metric(sarsa_lives["gated"], "life")
        late = _metric(sarsa_lives["gated"], "late")
        assert life.min() >= LIFE_FLOOR, f"lifetime reward {life.min():.4f} < {LIFE_FLOOR}"
        assert life.mean() >= RANDOM + LIFE_VS_RANDOM, (
            f"lifetime reward {life.mean():.4f} does not clear random "
            f"{RANDOM} by {LIFE_VS_RANDOM}"
        )
        assert late.min() >= OPTIMAL - NEAR_OPT_SLACK, (
            f"late-life reward {late.min():.4f} not within {NEAR_OPT_SLACK} "
            f"of the analytic optimum {OPTIMAL}"
        )

    def test_memory_early_window_stays_at_best_for_life(self, sarsa_lives) -> None:
        """(b) After the first 3 exposures per rule, every re-coordination is instant.

        The early-window (first 50 steps) reward of every settled recurrence of
        both rules stays >= 0.60 on every seed (measured min 0.860; the
        memoryless twin's best seed never exceeds 0.220), the late-life-third
        mean stays >= 0.90 (measured min 0.9635), and late life is no worse
        than the first exposures (measured worst diff -0.0132, i.e. the k-th
        recurrence stays at — mostly above — its early best for k up to 59).
        """
        gated = sarsa_lives["gated"]
        settled_min = _metric(gated, "early_settled_min")
        late_mean = _metric(gated, "early_late_mean")
        first = _metric(gated, "early_first")
        assert settled_min.min() >= MEM_SETTLED_MIN, (
            f"a settled recurrence lost its convention: worst early-window "
            f"{settled_min.min():.4f} < {MEM_SETTLED_MIN}"
        )
        assert late_mean.min() >= MEM_LATE_MEAN, (
            f"late-life re-coordination degraded: {late_mean.min():.4f} < {MEM_LATE_MEAN}"
        )
        worst_trend = float((late_mean - first).min())
        assert worst_trend >= -MEM_NO_DEGRADE, (
            f"early-window drifted down over the life: worst late-vs-first "
            f"diff {worst_trend:.4f} < -{MEM_NO_DEGRADE}"
        )

    def test_memory_beats_no_context_ablation_twin(self, sarsa_lives) -> None:
        """(b) Causal ablation: withholding context erases the savings, per seed.

        Same agent, same seeds, same env stream protocol; only the context
        channels are withheld from the features. Measured paired settled
        early-window gap: mean +0.598, per-seed min +0.586; asserted >= 0.30.
        """
        gated = _metric(sarsa_lives["gated"], "early_settled_mean")
        plain = _metric(sarsa_lives["ablation"], "early_settled_mean")
        diffs = gated - plain
        assert diffs.min() >= MEM_ABLATION_GAP, (
            f"paired memory gap collapsed on a seed: min gated-ablation "
            f"settled early-window diff {diffs.min():.4f} < {MEM_ABLATION_GAP}"
        )
        assert diffs.mean() >= MEM_ABLATION_GAP, (
            f"mean paired memory gap {diffs.mean():.4f} < {MEM_ABLATION_GAP}"
        )

    def test_ablation_twin_is_plastic_but_memoryless(self, sarsa_lives) -> None:
        """The twin is no straw man: it learns and relearns fine — it just forgets.

        Measured: ablation lifetime per-seed min 0.8906 (strong control); its
        settled early-window mean never exceeds 0.4030 per seed (every
        recurrence pays the relearning cost); its recovery does not trend up
        (early 54.4 -> late 52.5).
        """
        ablation = sarsa_lives["ablation"]
        life = _metric(ablation, "life")
        settled = _metric(ablation, "early_settled_mean")
        trend = _metric(ablation, "recov_late_mean") - _metric(ablation, "recov_early_mean")
        assert life.min() >= ABLATION_LIFE_FLOOR, (
            f"ablation twin failed to learn: lifetime {life.min():.4f} < {ABLATION_LIFE_FLOOR}"
        )
        assert settled.max() <= ABLATION_MEMORYLESS_CEIL, (
            f"ablation twin unexpectedly retained conventions: settled "
            f"early-window {settled.max():.4f} > {ABLATION_MEMORYLESS_CEIL}"
        )
        assert trend.max() <= REC_TREND_ABLATION, (
            f"ablation twin lost plasticity: recovery slowed by {trend.max():.2f} steps"
        )

    def test_plasticity_recovery_not_trending_up(self, sarsa_lives) -> None:
        """(c) Post-switch recovery late in life is as fast as early in life.

        Measured: settled recovery sits at the metric floor (25.0 = trailing
        window width) on every seed, early and late — recovery is instant for
        the whole life, so there is no loss-of-plasticity trend to find.
        """
        gated = sarsa_lives["gated"]
        early = _metric(gated, "recov_early_mean")
        late = _metric(gated, "recov_late_mean")
        settled = _metric(gated, "recov_settled_mean")
        assert (late - early).max() <= REC_TREND_GATED, (
            f"recovery is slowing down across the life: worst late-vs-early "
            f"diff {(late - early).max():.2f} > {REC_TREND_GATED}"
        )
        assert settled.max() <= REC_SETTLED_CEIL, (
            f"settled recovery {settled.max():.2f} above ceiling {REC_SETTLED_CEIL}"
        )

    def test_stability_bounded_state_constant_shapes(self, sarsa_lives) -> None:
        """(d) No NaN anywhere; weights bounded; pytree structure constant for life.

        Measured final max |q_weights|: gated 0.908, ablation 2.119; rbar in
        [0.920, 0.985]; the bias — the one shared forgetting channel — stays
        identically zero under use_bias=False.
        """
        for name, q_bound in (("gated", Q_BOUND_GATED), ("ablation", Q_BOUND_ABLATION)):
            for seed, run in enumerate(sarsa_lives[name]):
                final: DifferentialSARSAState = run.final_state
                init_def = jax.tree_util.tree_structure(run.init_state)
                final_def = jax.tree_util.tree_structure(final)
                assert init_def == final_def, f"{name} seed {seed}: pytree structure changed"
                for a, b in zip(
                    jax.tree_util.tree_leaves(run.init_state),
                    jax.tree_util.tree_leaves(final),
                ):
                    assert np.shape(a) == np.shape(b), (
                        f"{name} seed {seed}: leaf shape changed {np.shape(a)} -> {np.shape(b)}"
                    )
                q_weights = np.asarray(final.q_weights)
                assert np.isfinite(q_weights).all(), f"{name} seed {seed}: non-finite weights"
                assert np.abs(q_weights).max() <= q_bound, (
                    f"{name} seed {seed}: |q_weights| {np.abs(q_weights).max():.3f} > {q_bound}"
                )
                assert np.all(np.asarray(final.q_bias) == 0.0), (
                    f"{name} seed {seed}: the bias forgetting channel opened"
                )
                rbar = float(final.average_reward)
                assert RBAR_LO <= rbar <= RBAR_HI, f"{name} seed {seed}: rbar {rbar:.3f}"

    def test_stressor_survived(self, sarsa_lives) -> None:
        """(e) The 2000-step all-channel noise stressor is survived on every seed.

        Measured: reward during the stressor stays >= 0.832 (still far above
        random 0.5) and the first 4000 post-stressor steps recover to within
        0.0045 of the pre-stressor level on the worst seed — the conventions
        stored in the gated blocks come through the noise intact.
        """
        gated = sarsa_lives["gated"]
        during = _metric(gated, "during_stress")
        drop = _metric(gated, "pre_stress") - _metric(gated, "post_stress")
        assert during.min() >= STRESS_DURING_FLOOR, (
            f"agent collapsed during the stressor: {during.min():.4f} < {STRESS_DURING_FLOOR}"
        )
        assert drop.max() <= STRESS_DROP, (
            f"post-stressor reward did not recover: worst drop {drop.max():.4f} > {STRESS_DROP}"
        )


# ---------------------------------------------------------------------------
# Rung 2: the multi-component prototype on the same life
# ---------------------------------------------------------------------------


class TestIntegratedPrototypeLife:
    """Full PrototypeAgent (options + world model + dreams + curation), one life."""

    def test_learning_above_random(self, proto_lives) -> None:
        """Life-level control learning holds for the full stack on the long life.

        Measured: lifetime per-seed min 0.8849, mean 0.9003 (random 0.5,
        optimum 1.0); final-8000 per-seed min 0.8764.
        """
        runs = [r for r, _, _ in proto_lives]
        life = _metric(runs, "life")
        late = _metric(runs, "late")
        assert life.min() >= P_LIFE_FLOOR, f"proto lifetime {life.min():.4f} < {P_LIFE_FLOOR}"
        assert life.mean() >= P_LIFE_MEAN, f"proto mean lifetime {life.mean():.4f} < {P_LIFE_MEAN}"
        assert late.min() >= P_LATE_FLOOR, f"proto late-life {late.min():.4f} < {P_LATE_FLOOR}"

    def test_recovery_not_trending_up(self, proto_lives) -> None:
        """Post-switch recovery does not slow down across the 119-switch life.

        Measured per-seed late-vs-early recovery diff in [-3.67, +1.45] steps
        (mean -1.15, i.e. recovery mildly speeds up); asserted <= +8 per seed
        and <= +4 in the mean.
        """
        runs = [r for r, _, _ in proto_lives]
        diff = _metric(runs, "recov_late_mean") - _metric(runs, "recov_early_mean")
        assert diff.max() <= P_REC_TREND_SEED, (
            f"a seed lost plasticity: recovery slowed by {diff.max():.2f} steps"
        )
        assert diff.mean() <= P_REC_TREND_MEAN, (
            f"mean recovery slowed by {diff.mean():.2f} steps"
        )

    def test_retention_measured_not_degrading(self, proto_lives) -> None:
        """Retention is measured and asserted only in its robust weak form.

        Measured per-seed retention diff (late-third early-window minus
        first-6-phases early-window) in [-0.0017, +0.121], mean +0.051 — the
        stack's re-coordination mildly improves over the life and never
        degrades.  Strong context-keyed memory is NOT claimed for this rung
        (settled early-window ~0.71 vs the gated rung's ~0.97): its linear Q
        over [x, ctx] cannot represent both rules at once, so the memory claim
        of this module rests on the certified gated rung.
        """
        runs = [r for r, _, _ in proto_lives]
        diff = _metric(runs, "early_late_mean") - _metric(runs, "early_first")
        assert diff.min() >= P_RET_SEED_FLOOR, (
            f"a seed's re-coordination degraded over the life: {diff.min():.4f} "
            f"< {P_RET_SEED_FLOOR}"
        )
        assert diff.mean() >= P_RET_MEAN_FLOOR, (
            f"mean re-coordination trend {diff.mean():.4f} < {P_RET_MEAN_FLOOR}"
        )

    def test_stability_bounded_state(self, proto_lives) -> None:
        """No NaN over 48k steps and 11 curation events; state stays bounded.

        Measured: the state pytree holds exactly 440 elements at birth and at
        death on every seed (curation replaces, never grows), all finite.
        """
        for seed, (run, n_birth, n_death) in enumerate(proto_lives):
            assert n_death <= P_STATE_GROWTH * n_birth, (
                f"proto seed {seed}: state grew {n_birth} -> {n_death} elements"
            )
            for leaf in jax.tree_util.tree_leaves(run.final_state):
                dtype = getattr(leaf, "dtype", None)
                # Skip non-float leaves (counters, PRNG keys) before converting.
                if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
                    assert np.isfinite(np.asarray(leaf)).all(), (
                        f"proto seed {seed}: non-finite state leaf"
                    )

    def test_stressor_survived(self, proto_lives) -> None:
        """The full stack also survives the mid-life input-noise stressor.

        Measured: during-stressor reward per-seed min 0.8240; post-vs-pre
        per-seed diff in [-0.0585, +0.0275] (mean -0.0085).
        """
        runs = [r for r, _, _ in proto_lives]
        during = _metric(runs, "during_stress")
        drop = _metric(runs, "pre_stress") - _metric(runs, "post_stress")
        assert during.min() >= P_STRESS_DURING_FLOOR, (
            f"proto collapsed during the stressor: {during.min():.4f} "
            f"< {P_STRESS_DURING_FLOOR}"
        )
        assert drop.max() <= P_STRESS_DROP_SEED, (
            f"proto post-stressor reward did not recover: worst drop "
            f"{drop.max():.4f} > {P_STRESS_DROP_SEED}"
        )
        assert drop.mean() <= P_STRESS_DROP_MEAN, (
            f"proto mean post-stressor drop {drop.mean():.4f} > {P_STRESS_DROP_MEAN}"
        )
