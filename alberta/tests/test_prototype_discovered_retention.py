"""PrototypeAgent-level retention: the discovery bank supplies the rule-isolating features.

``tests/test_integrated_life.py`` measured the full :class:`PrototypeAgent`'s
retention as honestly weak (settled recurrence early-window ~0.71 on the 48k
life vs ~0.97 for oracle-gated DifferentialSARSA) because its linear Q over
``[x, ctx]`` cannot represent both recurring rules at once.  This module closes
that gap at the *integrated-agent* level, in two rungs, with no library change:

1. **Oracle-gated rung** — the same full stack (OaK options + action-conditioned
   world model + 2 guarded dreams/step + periodic curation) run over the
   certified exclusive context-gated feature map ``[c_A * x, c_B * x]`` (the
   rung-1 map of ``test_integrated_life.py``).  Its settled early-window jumps
   from ~0.74 (plain twin, this protocol) to ~0.92, proving the integrated
   stack's retention deficit is purely representational: same components, same
   hyperparameters, same seeds — only the feature path differs.
2. **Discovered rung** — the feature map is instead the
   :class:`FixedBudgetInteractionLearner` bank (the frozen scale-robust
   configuration certified by ``test_gauntlet_discovery.py`` and
   ``test_discovery_control_life.py``), trained *online during the first 4,000
   steps of the same life* on action-conditioned next-reward prediction from
   raw observations that include distractor channels, then **frozen** for the
   remaining 20,000 steps.  The frozen bank holds >= 3 of the 4 oracle
   context-binding products ``s_i * c_k`` on every calibration seed, and the
   stack's settled early-window (~0.92) lands within ~0.01 of the oracle-gated
   rung — the discovery module supplies the rule-isolating representation
   autonomously.

Life protocol (reduced L3 shape)
================================

:class:`SwitchingTwoStateMDP` (phase_length=400) inverts its 2x2 payoff matrix
every 400 steps: the optimal average reward is exactly 1.0 in both phases,
uniform random earns exactly 0.5, and the dynamics are deterministic, so the
phase schedule is identical across arms and seeds.  24,000 steps = 60 phase
occurrences = 30 recurrences of each rule (task shape requires >= 10).  No
learner reset, no episode boundary, no mid-life hyperparameter change; the
mid-life stressor lane stays in ``test_integrated_life.py``.

Raw observation (all arms, dim 6): ``[one_hot(latent_state), one_hot(phase),
rademacher(2)]`` — context observable (the precondition for context-keyed
memory), plus two fresh Rademacher distractor channels that only the discovery
arm consumes (they enlarge its pair space to C(6,2)=15 candidates, 11 of them
distractor/degenerate).  Arms: **plain** = ``obs[:4]`` (the integrated-life
rung-2 baseline), **gated** = ``[c_A * x, c_B * x]`` (dim 4), **discovered** =
bank pair products ``obs[left] * obs[right]`` (8 slots).

Full-stack agent per arm: the ``test_prototype_nonsaturating.py`` switching
template as adapted by ``test_integrated_life.py`` (1 subtask option, linear
world model with action interactions, DreamingConfig(warmup=20, gate=0.3),
buffer 32, 2 dreams/step), curation every 4,000 steps (5 events/life) with the
replacement pool restricted to ``[1]`` so the number of distinct compiled agent
configs stays bounded (curation still evicts, resets, and re-specs the option
each event).

Interaction finding: world-model LMS stability under the wider bank
===================================================================

At the template's world-model step size 0.2, the discovered arm's linear
action-interaction world model **diverged to NaN on 4 of 8 development seeds**
(seeds 3/4/6/7): its input ``[phi, one_hot(a), phi x one_hot(a)]`` has squared
norm up to ``2*||phi||^2 + 1 = 17`` for the 8-slot bank (features in [-1, 1]),
exceeding the LMS stability bound ``2/0.2 = 10``, while the dim-4 arms stay
under it (norm <= 5, plain; <= 3, gated).  The failure was *silent at the
control level*: a NaN ``model_error_ema`` fails the dream gate comparison, so
dreams self-disable and rewards stay clean — only the state-finiteness check
caught it.  The fix asserted here is the norm-scaled step size 0.05 for the
dim-8 arm (bound 40 > 17), which keeps the model finite on all 16 calibration
seeds and slightly *improves* the discovered arm (dreams stay active;
settled-early +0.013, lifetime +0.012 vs the diverged configuration).

Calibration (2026-07-30, CPU; development batch seeds 0..7, disjoint
robustness batch seeds 100..107 — worst value across both in brackets; every
frozen threshold keeps >= ~2x margin on the worst value)
========================================================

Settled = phase occurrences >= 14 (past the 10-phase discovery segment plus a
re-learning transient after the bank freeze, uniform across arms);
early-window = mean reward over the first 50 steps of a phase occurrence.

    arm         life mean/min             settled-early mean (per-seed min)
    plain       0.9109 / 0.9061 [0.9061]  0.7391 (max 0.7509) [max 0.7578]
    gated       0.9249 / 0.9176 [0.9176]  0.9234 (min 0.9130) [min 0.9130]
    discovered  0.9093 / 0.8868 [0.8835]  0.9155 (min 0.8713) [min 0.8709]

    paired settled-early diffs        mean / per-seed min   [worst batch]
    gated - plain                     +0.1843 / +0.1622     [+0.1805 / +0.1622]
    discovered - plain                +0.1764 / +0.1287     [+0.1637 / +0.1287]
    paired lifetime diffs
    gated - plain                     +0.0140 / +0.0087     [+0.0140 / +0.0087]
    discovered - plain                -0.0017 / -0.0274     [-0.0061 / -0.0280]

    worst single settled phase: gated 0.72 [0.64], discovered 0.68 [0.58]
    oracle products in the frozen bank (of 4): per-seed min 3 [3],
    mean 3.875 [3.5]; state pytree elements birth -> death: plain/gated
    440 -> 440, discovered 1040 -> 1040 on every seed; zero non-finite values
    (with the norm-scaled world-model step — see the interaction finding).

Honest scope notes
==================

* The rule context is *oracle-observable*; the discovery learner binds it into
  products autonomously but does not infer it (context inference is
  ``test_step2_context_inference.py``'s open problem).
* The discovered rung is the **certified-static coupling**: bank frozen after
  the initial segment.  Slow refresh with Q carry-over inside the OaK base
  learner (the ``test_discovery_control_life.py`` mechanism) is not attempted
  here — prototype internals are under active development and the frozen-bank
  variant is the robust one.  The re-learning cost of the churning-bank
  segment is real and bounded: the discovered arm pays a small lifetime cost
  vs plain (measured mean -0.002/-0.006, worst seed -0.028), asserted as
  bounded rather than zero, while the gated arm's lifetime strictly improves.
* One adverse interaction was found and is characterized above (world-model
  divergence under the wider bank at the template step size); with the
  norm-scaled step, options, dreams, and curation all stay active in every
  arm and do not erase the gated-block memory (settled early-window 0.92 in
  both feature-path arms vs the 0.74 plain ceiling).  Per-component
  dream/option ablations are not run here; ``test_integrated_life.py``
  remains the stressor lane.
* All 8 development seeds were used for calibration; the disjoint batch
  100..107 was measured once to bound seed-batch drift (bracketed values).

Runtime: ~55-70s on CPU (8 seeds x 3 full-stack lives of 24k steps,
module-scoped fixture; jitted chunks cached per agent config via ``_dedup``).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.interaction_features import FixedBudgetInteractionLearner
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
LIFE_STEPS = 24_000
PHASE = 400
N_PHASES = LIFE_STEPS // PHASE  # 60 -> 30 recurrences of each of the 2 rules
EARLY_W = 50  # early-window width (re-coordination / savings signal)
SETTLE_PHASE = 14  # discovery segment (10 phases) + post-freeze transient
DISC_STEPS = 4_000  # discovery segment: 10 phases = 5 exposures to each rule
P_CHUNK = 1_000
P_CURATE_EVERY = 4_000
CURATE_POOL = [1]  # restricted pool: bounds distinct compiled agent configs

N_DISTRACTORS = 2
RAW_DIM = 4 + N_DISTRACTORS
N_SLOTS = 8
N_CANDIDATES = 15  # C(6, 2): the full pair space of the 6-dim raw observation
# The four context-binding products s_i * c_k: state channels {0, 1} bound to
# context channels {2, 3} (canonical left < right descriptor order).
ORACLE_PAIRS = frozenset({(0, 2), (1, 2), (0, 3), (1, 3)})

_ENV = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=PHASE))
OPTIMAL = _ENV.optimal_average_reward(PHASE_A)  # 1.0 (same in PHASE_B)
RANDOM = _ENV.uniform_random_average_reward(PHASE_A)  # 0.5 (same in PHASE_B)

# ---------------------------------------------------------------------------
# Calibrated margins (see module docstring for the measured tables)
# ---------------------------------------------------------------------------

# (a) control learning: all three arms learn far above random on the whole life
PLAIN_LIFE_FLOOR = 0.85  # measured per-seed min 0.9061 [0.9061]
GATED_LIFE_FLOOR = 0.86  # measured per-seed min 0.9176 [0.9176]
DISC_LIFE_FLOOR = 0.82  # measured per-seed min 0.8868 [0.8835]
# (b) retention: settled recurrence early-window, absolute and paired
PLAIN_SETTLED_CEIL = 0.85  # measured per-seed max 0.7509 [0.7578]
GATED_SETTLED_FLOOR = 0.85  # measured per-seed min 0.9130 [0.9130]
DISC_SETTLED_FLOOR = 0.80  # measured per-seed min 0.8713 [0.8709]
GATED_GAP_SEED = 0.08  # measured per-seed min +0.1622 [+0.1622]
GATED_GAP_MEAN = 0.09  # measured mean +0.1843 [+0.1805]
DISC_GAP_SEED = 0.06  # measured per-seed min +0.1287 [+0.1287]
DISC_GAP_MEAN = 0.08  # measured mean +0.1764 [+0.1637]
SETTLED_PHASE_FLOOR = 0.40  # worst single settled phase: gated 0.64, disc 0.58
# (c) lifetime pairing: gated improves; discovered pays a bounded segment cost
GATED_LIFE_GAP_FLOOR = -0.02  # measured paired per-seed min +0.0087 [+0.0087]
DISC_LIFE_GAP_SEED = -0.08  # measured paired per-seed min -0.0274 [-0.0280]
DISC_LIFE_GAP_MEAN = -0.03  # measured paired mean -0.0017 [-0.0061]
# (d) discovery: oracle context-binding products in the frozen bank (of 4)
BANK_ORACLE_MIN = 2  # measured per-seed min 3 [3]
BANK_ORACLE_MEAN = 2.5  # measured mean 3.875 [3.5]
# (f) world-model LMS stability (the interaction finding): input squared norm
# is 2*||phi||^2 + 1 <= 17 for the 8-slot bank, so its step must sit below
# 2/17 ~ 0.118; the dim-4 arms (norm <= 5) keep the 0.2 template value.
WM_STEP_DIM4 = 0.2
WM_STEP_DIM8 = 0.05
# (e) bounded state
STATE_GROWTH = 2.0  # measured exactly 1.0 (440->440 and 1040->1040 elements)


# ---------------------------------------------------------------------------
# Observation and feature maps
# ---------------------------------------------------------------------------


def _observe_raw(env_state, key: jax.Array) -> jax.Array:
    """Raw 6-dim observation: latent one-hot, phase one-hot, 2 distractors."""
    return jnp.concatenate(
        [
            _ENV.observe(env_state),
            jax.nn.one_hot(_ENV.phase_id(env_state), 2, dtype=jnp.float32),
            jr.rademacher(key, (N_DISTRACTORS,), dtype=jnp.float32),
        ]
    )


def _plain_map(obs: jax.Array) -> jax.Array:
    """Integrated-life rung-2 baseline features [x, ctx] (dim 4)."""
    return obs[:4]


def _gated_map(obs: jax.Array) -> jax.Array:
    """Exclusive context-gated features [c_A * x, c_B * x] (dim 4)."""
    x, ctx = obs[:2], obs[2:4]
    return jnp.concatenate([ctx[0] * x, ctx[1] * x])


def _pair_products(obs: jax.Array, left: jax.Array, right: jax.Array) -> jax.Array:
    """Bank pair-product features ``obs[left] * obs[right]`` (dim N_SLOTS)."""
    return obs[left] * obs[right]


# ---------------------------------------------------------------------------
# Full-stack agent template (test_integrated_life.py rung 2, obs_dim-generic)
# ---------------------------------------------------------------------------


def _make_proto(obs_dim: int, wm_step: float = WM_STEP_DIM4) -> PrototypeAgent:
    """The nonsaturating switching-domain template at the given feature dim.

    ``wm_step`` is the world-model LMS step size; it must respect the
    stability bound for the arm's feature norm (see the interaction finding
    in the module docstring).
    """
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0, max_option_steps=4),),
            observation_dim=obs_dim,
            n_primitive_actions=2,
            base_step_size=0.1,
            base_avg_reward_step_size=0.01,
            epsilon_base=0.1,
            epsilon_option=0.1,
        ),
        min_steps_before_curation=200,
    )
    world_model = ActionConditionedWorldModelConfig(
        observation_dim=obs_dim,
        n_actions=2,
        hidden_sizes=(),
        step_size=wm_step,
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


# Curation returns a fresh PrototypeAgent; dedup by static config so each
# distinct config compiles exactly once across seeds and arms.
_PROTO_CACHE: dict[tuple, PrototypeAgent] = {}


def _dedup(agent: PrototypeAgent) -> PrototypeAgent:
    key = (
        agent.config.world_model.step_size,
        agent.config.oak.stomp.observation_dim,
        tuple(s.feature_index for s in agent.config.oak.stomp.subtask_specs),
    )
    return _PROTO_CACHE.setdefault(key, agent)


# The frozen scale-robust discovery configuration certified by
# test_gauntlet_discovery.py / test_discovery_control_life.py, at this
# module's 6-dim raw observation (8 slots, full 15-pair candidate archive).
_DISC = FixedBudgetInteractionLearner(
    n_features=N_SLOTS,
    n_tasks=2,
    candidate_count=N_CANDIDATES,
    candidate_strategy="all_pairs",
    utility_retention_decay=0.9999,
    refresh_candidates=False,
    refresh_promoted_candidate=False,
    scale_robust=True,
    use_obgd=False,
    replacement_interval=50,
    min_feature_age=25,
    candidate_min_age=25,
)


# ---------------------------------------------------------------------------
# Jitted closed-loop chunks (Python-level composition with curation outside)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(0, 1))
def _fixed_chunk(agent, feature_fn, carry, keys):
    """One chunk of a fixed-feature-map full-stack life (plain / gated arms)."""

    def step(c, k):
        p_state, e_state, action = c
        k_env, k_obs = jr.split(k)
        _, reward, e2 = _ENV.step(e_state, action, k_env)
        obs = _observe_raw(e2, k_obs)
        result = agent.update(p_state, reward, feature_fn(obs))
        return (result.state, e2, result.action), reward

    return jax.lax.scan(step, carry, keys)


@functools.partial(jax.jit, static_argnums=(0,))
def _disc_chunk(agent, carry, keys):
    """Discovery-segment chunk: bank learns online on the taken action's reward.

    The discovery learner sees the pre-action raw observation with the
    observed reward on the taken action's head (the other head NaN-masked) —
    the action-conditioned next-reward coupling of
    ``test_discovery_control_life.py``.  The agent's features are the current
    bank's pair products, so descriptors may churn under the agent during the
    segment (bounded by the learner's replacement_interval); the re-learning
    cost is paid once and excluded from the settled window.
    """

    def step(c, k):
        p_state, e_state, action, d_state, raw_prev = c
        k_env, k_obs = jr.split(k)
        _, reward, e2 = _ENV.step(e_state, action, k_env)
        obs = _observe_raw(e2, k_obs)
        targets = jnp.full((2,), jnp.nan, dtype=jnp.float32).at[action].set(reward)
        d2 = _DISC.update(d_state, raw_prev, targets).state
        phi = _pair_products(obs, d2.feature_left, d2.feature_right)
        result = agent.update(p_state, reward, phi)
        return (result.state, e2, result.action, d2, obs), reward

    return jax.lax.scan(step, carry, keys)


@functools.partial(jax.jit, static_argnums=(0,))
def _frozen_chunk(agent, carry, keys):
    """Post-segment chunk: the bank descriptors are frozen for the rest of life."""

    def step(c, k):
        p_state, e_state, action, left, right = c
        k_env, k_obs = jr.split(k)
        _, reward, e2 = _ENV.step(e_state, action, k_env)
        obs = _observe_raw(e2, k_obs)
        result = agent.update(p_state, reward, _pair_products(obs, left, right))
        return (result.state, e2, result.action, left, right), reward

    return jax.lax.scan(step, carry, keys)


# ---------------------------------------------------------------------------
# Life runners
# ---------------------------------------------------------------------------


class LifeRun(NamedTuple):
    """One completed full-stack life."""

    rewards: np.ndarray
    n_birth: int
    n_death: int
    final_state: object
    n_oracle_in_bank: int  # -1 for the fixed-feature-map arms


def _n_elems(state) -> int:
    return sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(state))


def _maybe_curate(agent, carry, run_key, step):
    """Uniform curation cadence, restricted pool (see module docstring)."""
    if step % P_CURATE_EVERY == 0 and step < LIFE_STEPS:
        run_key, c_key = jr.split(run_key)
        new_agent, new_p = agent.curate(carry[0], c_key, CURATE_POOL)
        agent = _dedup(new_agent)
        carry = (new_p, *carry[1:])
    return agent, carry, run_key


def _run_fixed_life(seed: int, feature_fn) -> LifeRun:
    """One uninterrupted 24k-step life over a fixed feature map."""
    agent = _dedup(_make_proto(4))
    env_key, obs_key, agent_key, run_key = jr.split(jr.key(seed), 4)
    e_state = _ENV.init(env_key)
    phi = feature_fn(_observe_raw(e_state, obs_key))
    p_state = agent.start(agent.init(agent_key), phi)
    action = agent.act(p_state, phi)
    n_birth = _n_elems(p_state)

    rewards: list[np.ndarray] = []
    step = 0
    carry = (p_state, e_state, action)
    while step < LIFE_STEPS:
        run_key, chunk_key = jr.split(run_key)
        carry, r = _fixed_chunk(agent, feature_fn, carry, jr.split(chunk_key, P_CHUNK))
        rewards.append(np.asarray(r))
        step += P_CHUNK
        agent, carry, run_key = _maybe_curate(agent, carry, run_key, step)
    return LifeRun(np.concatenate(rewards), n_birth, _n_elems(carry[0]), carry[0], -1)


def _run_discovered_life(seed: int) -> LifeRun:
    """One 24k-step life: bank learned online for DISC_STEPS, then frozen."""
    agent = _dedup(_make_proto(N_SLOTS, wm_step=WM_STEP_DIM8))
    env_key, obs_key, agent_key, disc_key, run_key = jr.split(jr.key(seed), 5)
    e_state = _ENV.init(env_key)
    obs = _observe_raw(e_state, obs_key)
    d_state = _DISC.init(RAW_DIM, disc_key)
    phi = _pair_products(obs, d_state.feature_left, d_state.feature_right)
    p_state = agent.start(agent.init(agent_key), phi)
    action = agent.act(p_state, phi)
    n_birth = _n_elems(p_state)

    rewards: list[np.ndarray] = []
    step = 0
    carry = (p_state, e_state, action, d_state, obs)
    while step < DISC_STEPS:
        run_key, chunk_key = jr.split(run_key)
        carry, r = _disc_chunk(agent, carry, jr.split(chunk_key, P_CHUNK))
        rewards.append(np.asarray(r))
        step += P_CHUNK
        agent, carry, run_key = _maybe_curate(agent, carry, run_key, step)

    p_state, e_state, action, d_state, _ = carry
    left, right = d_state.feature_left, d_state.feature_right
    carry = (p_state, e_state, action, left, right)
    while step < LIFE_STEPS:
        run_key, chunk_key = jr.split(run_key)
        carry, r = _frozen_chunk(agent, carry, jr.split(chunk_key, P_CHUNK))
        rewards.append(np.asarray(r))
        step += P_CHUNK
        agent, carry, run_key = _maybe_curate(agent, carry, run_key, step)

    bank = {(int(a), int(b)) for a, b in zip(np.asarray(left), np.asarray(right))}
    n_oracle = len(bank & ORACLE_PAIRS)
    return LifeRun(np.concatenate(rewards), n_birth, _n_elems(carry[0]), carry[0], n_oracle)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _settled_early(rewards: np.ndarray) -> np.ndarray:
    """Early-window means of the settled phase occurrences (>= SETTLE_PHASE)."""
    phases = rewards.reshape(N_PHASES, PHASE)
    return phases[SETTLE_PHASE:, :EARLY_W].mean(axis=1)


def _life(runs: list[LifeRun]) -> np.ndarray:
    return np.array([run.rewards.mean() for run in runs])


def _settled_mean(runs: list[LifeRun]) -> np.ndarray:
    return np.array([_settled_early(run.rewards).mean() for run in runs])


def _settled_phase_min(runs: list[LifeRun]) -> np.ndarray:
    return np.array([_settled_early(run.rewards).min() for run in runs])


# ---------------------------------------------------------------------------
# Module-scoped rollouts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lives() -> dict[str, list[LifeRun]]:
    """Paired plain / gated / discovered full-stack lives for all seeds."""
    assert N_PHASES * PHASE == LIFE_STEPS
    assert (N_PHASES - SETTLE_PHASE) // 2 >= 10  # settled recurrences per rule: 23
    out: dict[str, list[LifeRun]] = {"plain": [], "gated": [], "discovered": []}
    for seed in range(NUM_SEEDS):
        for name, run in (
            ("plain", _run_fixed_life(seed, _plain_map)),
            ("gated", _run_fixed_life(seed, _gated_map)),
            ("discovered", _run_discovered_life(seed)),
        ):
            assert np.isfinite(run.rewards).all(), f"non-finite rewards: {name} seed {seed}"
            out[name].append(run)
    return out


# ---------------------------------------------------------------------------
# Rung 1: the retention deficit is representational — oracle gating fixes it
# ---------------------------------------------------------------------------


class TestOracleGatedIntegratedRetention:
    """Full PrototypeAgent over the certified gated map vs its plain twin."""

    def test_gated_retention_beats_plain_twin_per_seed(self, lives) -> None:
        """Settled recurrence early-window: gated far above the plain baseline.

        Measured: gated settled mean 0.9234 (per-seed min 0.9130), plain
        0.7391 (per-seed max 0.7578 across batches); paired gap mean +0.1843,
        per-seed min +0.1622.  Same stack, same seeds — only the feature path
        differs, so the plain arm's ~0.74 ceiling (the integrated-life rung-2
        finding) is purely representational.
        """
        gated = _settled_mean(lives["gated"])
        plain = _settled_mean(lives["plain"])
        diffs = gated - plain
        assert diffs.min() >= GATED_GAP_SEED, (
            f"paired gated-plain settled gap collapsed: min {diffs.min():.4f} < {GATED_GAP_SEED}"
        )
        assert diffs.mean() >= GATED_GAP_MEAN, (
            f"mean gated-plain settled gap {diffs.mean():.4f} < {GATED_GAP_MEAN}"
        )
        assert gated.min() >= GATED_SETTLED_FLOOR, (
            f"gated settled early-window {gated.min():.4f} < {GATED_SETTLED_FLOOR}"
        )
        assert plain.max() <= PLAIN_SETTLED_CEIL, (
            f"plain twin unexpectedly retained: settled {plain.max():.4f} > {PLAIN_SETTLED_CEIL}"
        )
        worst_phase = _settled_phase_min(lives["gated"])
        assert worst_phase.min() >= SETTLED_PHASE_FLOOR, (
            f"a gated settled recurrence collapsed: {worst_phase.min():.4f}"
            f" < {SETTLED_PHASE_FLOOR}"
        )

    def test_gated_lifetime_not_degraded(self, lives) -> None:
        """The gated feature path improves lifetime reward, never trades it away.

        Measured paired gated-plain lifetime diff: mean +0.0140, per-seed min
        +0.0087 across both calibration batches.
        """
        diffs = _life(lives["gated"]) - _life(lives["plain"])
        assert diffs.min() >= GATED_LIFE_GAP_FLOOR, (
            f"gated lifetime degraded vs plain: {diffs.min():.4f} < {GATED_LIFE_GAP_FLOOR}"
        )


# ---------------------------------------------------------------------------
# Rung 2: the discovery module supplies that representation autonomously
# ---------------------------------------------------------------------------


class TestDiscoveredBankIntegratedRetention:
    """Full PrototypeAgent over the online-discovered, then frozen, bank."""

    def test_frozen_bank_holds_context_binding_products(self, lives) -> None:
        """The bank frozen at step 4,000 holds distinct oracle products.

        Measured: 3 or 4 of the 4 products on every calibration seed in both
        batches (per-seed min 3, means 3.875 and 3.5) — promoted online from
        the 15-pair space against 11 distractor/degenerate pairs, from
        behavior data the churning full-stack agent itself generated.
        """
        counts = np.array([run.n_oracle_in_bank for run in lives["discovered"]])
        assert counts.min() >= BANK_ORACLE_MIN, (
            f"a seed's frozen bank holds {counts.min()} oracle products < {BANK_ORACLE_MIN}"
        )
        assert counts.mean() >= BANK_ORACLE_MEAN, (
            f"mean oracle products in frozen banks {counts.mean():.2f} < {BANK_ORACLE_MEAN}"
        )

    def test_discovered_retention_beats_plain_twin_per_seed(self, lives) -> None:
        """Settled recurrence early-window: discovered far above the plain baseline.

        Measured: discovered settled mean 0.9155 (per-seed min 0.8709 across
        batches) — within ~0.01 of the oracle-gated rung — against plain
        0.7391; paired gap mean +0.1764 (worst batch +0.1637), per-seed min
        +0.1287.
        """
        disc = _settled_mean(lives["discovered"])
        plain = _settled_mean(lives["plain"])
        diffs = disc - plain
        assert diffs.min() >= DISC_GAP_SEED, (
            f"paired discovered-plain settled gap collapsed: min {diffs.min():.4f}"
            f" < {DISC_GAP_SEED}"
        )
        assert diffs.mean() >= DISC_GAP_MEAN, (
            f"mean discovered-plain settled gap {diffs.mean():.4f} < {DISC_GAP_MEAN}"
        )
        assert disc.min() >= DISC_SETTLED_FLOOR, (
            f"discovered settled early-window {disc.min():.4f} < {DISC_SETTLED_FLOOR}"
        )
        worst_phase = _settled_phase_min(lives["discovered"])
        assert worst_phase.min() >= SETTLED_PHASE_FLOOR, (
            f"a discovered settled recurrence collapsed: {worst_phase.min():.4f}"
            f" < {SETTLED_PHASE_FLOOR}"
        )

    def test_discovery_segment_lifetime_cost_is_bounded(self, lives) -> None:
        """The one-time discovery/re-learning cost stays small at life scale.

        Measured paired discovered-plain lifetime diff: mean -0.0017 (worst
        batch -0.0061), per-seed min -0.0274 (worst batch -0.0280) — the
        churning-bank segment costs up to ~0.03 lifetime reward on the worst
        seed while buying a +0.13..+0.18 settled retention gain.  Asserted as
        bounded, not zero: this is the honest price of the frozen-bank
        coupling.
        """
        diffs = _life(lives["discovered"]) - _life(lives["plain"])
        assert diffs.min() >= DISC_LIFE_GAP_SEED, (
            f"discovered lifetime cost too large on a seed: {diffs.min():.4f}"
            f" < {DISC_LIFE_GAP_SEED}"
        )
        assert diffs.mean() >= DISC_LIFE_GAP_MEAN, (
            f"mean discovered lifetime cost {diffs.mean():.4f} < {DISC_LIFE_GAP_MEAN}"
        )


# ---------------------------------------------------------------------------
# Life-level properties, all arms
# ---------------------------------------------------------------------------


class TestLifeLevelProperties:
    """Control competence, zero NaN, and bounded state on every arm and seed."""

    def test_all_arms_learn_far_above_random(self, lives) -> None:
        """Every arm's lifetime reward clears random by a wide margin.

        Measured per-seed lifetime minima: plain 0.9061, gated 0.9176,
        discovered 0.8835 (worst batch), against random exactly 0.5 and the
        analytic optimum 1.0 — the plain baseline is competent (no straw man);
        the feature-path arms do not trade control for retention.
        """
        for name, floor in (
            ("plain", PLAIN_LIFE_FLOOR),
            ("gated", GATED_LIFE_FLOOR),
            ("discovered", DISC_LIFE_FLOOR),
        ):
            life = _life(lives[name])
            assert life.min() >= floor, f"{name} lifetime {life.min():.4f} < {floor}"
            assert floor > RANDOM + 0.3  # every floor sits far above random 0.5

    def test_bounded_state_and_finite_leaves(self, lives) -> None:
        """State never grows over 24k steps + 5 curation events; zero NaN.

        Measured: element counts identical at birth and death on every seed
        (440 for the dim-4 arms, 1040 for the dim-8 discovered arm), all
        float leaves finite.
        """
        for name, runs in lives.items():
            for seed, run in enumerate(runs):
                assert run.n_death <= STATE_GROWTH * run.n_birth, (
                    f"{name} seed {seed}: state grew {run.n_birth} -> {run.n_death} elements"
                )
                for leaf in jax.tree_util.tree_leaves(run.final_state):
                    dtype = getattr(leaf, "dtype", None)
                    # Skip non-float leaves (counters, PRNG keys) before checking.
                    if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
                        assert np.isfinite(np.asarray(leaf)).all(), (
                            f"{name} seed {seed}: non-finite state leaf"
                        )
