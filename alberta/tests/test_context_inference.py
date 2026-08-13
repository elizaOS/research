"""Latent context inference: the L3 memory life without the oracle context channel.

Every context-keyed memory demonstration so far observed the task context
directly (``tests/test_integrated_life.py`` appends the active-phase one-hot
to the observation; ``tests/test_discovery_control_life.py`` supplies it as an
observable channel).  This module closes that gap with
:class:`~alberta_framework.core.context_inference.ContextInference`: a bounded
bank of per-context reward-regime models that infers the active context from
``(observation, action, reward)`` alone — the environment's phase stays
hidden, exactly as :class:`SwitchingTwoStateMDP` emits it.

Two rungs
=========

**Mechanism** (scripted streams, no environment): on deterministic
(state, action, reward) sweeps the module detects rule flips within a
calibrated lag, re-identifies a previously seen rule by *reusing* its stored
slot (the memory-of-contexts property), allocates a fresh slot for a novel
rule, keeps inactive slot models bit-exactly frozen (exclusive gating — the
inactive model IS the memory), and stays within its slot bound via
least-recently-used eviction when rules outnumber slots.

**Headline** — the ``test_integrated_life`` L3 protocol with the context
HIDDEN: same env (phase_length=400), same 48,000-step / 120-switch life, same
mid-life stressor (obs noise, std 0.5, steps [22000, 24000)), same
:class:`DifferentialSARSAAgent` (``use_bias=False``, constant step-sizes and
epsilon).  The agent's features are ``kron(inferred_context_onehot, x)`` —
exclusive gating on the *inferred* slot, K = 4 slots (feature dim 8).  The
inference consumes the same noisy percept the agent sees; no oracle anywhere.
The paired ablation twin sees only ``x`` (memory impossible in principle);
a no-stressor arm isolates clean-life slot allocation.

Calibration (2026-07-30, CPU; dev batch ``jr.key(0..7)`` plus robustness
batches ``jr.key(100..107)`` and ``jr.key(7000..7007)``, worst value across
the three batches in brackets; every frozen threshold keeps a >= 2x margin
to the worst measured value)
===========================

Mechanism (module defaults: error_decay 0.8, switch_threshold 0.55):
2-rule detection lag exactly 4 steps on every flip; 3-state 3-rule lag 9;
slot sequences exactly [0,1,0,1,0] (2 rules, K=4), [0,1,2,0,1,2] (3 rules,
K=4: allocation then reuse of all three), bounded cycling at 2 slots for
3 rules with K=2 (LRU eviction); inactive slot models bit-identical across
a full foreign phase.

L3 life, 8 seeds paired (mean / per-seed worst):

    arm                life            settled-early     paired gap vs ablation
    inferred-gated     0.9578 (0.9542) 0.8954 (0.8841)   +0.519 (+0.493) mean/min
    ablation (no ctx)  0.8921 (0.8895) 0.376-0.382 (max 0.4002)
    (oracle-gated in test_integrated_life: settled-early 0.974, gap +0.598 —
     the inferred context recovers ~87% of the oracle's paired advantage)

    late8000 inferred min 0.9528 (optimum 1.0, random 0.5); late-third
    early-window mean 0.8962 (min 0.8855, dev batch).
    Context agreement with the hidden phase (slot->phase mapping fitted per
    era, settled clean steps, first 20 steps of each phase excluded): min
    0.9999 over all 24 seeds; no-stress arm min 1.0000.
    Slots: in_use == 2 on every seed at step 22k (pre-stress); no-stress arm
    holds exactly 2 slots and exactly 119 switches (= phase boundaries) for
    the whole 48k-step / 120-phase life on all 24 seeds.
    Stressor: during-stress reward min 0.7630 [0.7630]; post-vs-pre worst
    -0.0155; the noisy percept churns inference (~80 extra switches, bank
    fills to the K=4 bound) yet within 1-2 phases post-stress the memory
    re-forms.  Final |q_weights| max 1.706 [inferred] / 2.323 [ablation];
    rbar in [0.942, 0.992]; q_bias identically 0.

Honest scope notes
==================

* The stressor can permanently remap which slot serves which rule (a label
  permutation across the stress era, e.g. whole-life fixed-mapping agreement
  0.546 on one dev seed while both eras measure 1.0 under their own
  mappings).  Slot identities are arbitrary names: agreement is therefore
  measured with the mapping fitted per era, and the remap is exactly why the
  Q-block memory rebuild after the stressor costs 1-2 phases.
* Inference thresholds are calibrated for this env's {0, 1} reward scale
  (module defaults); other scales need recalibration.
* The regime model is a per-(state, action) reward table — sufficient for
  this family, not a general regime statistic.

Runtime: ~30-45s on CPU (three vmapped 48k-step lives over 8 seeds plus
Python-loop mechanism scripts; module-scoped fixtures).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)
from alberta_framework.core.context_inference import (
    ContextInference,
    ContextInferenceConfig,
)
from alberta_framework.streams import SwitchingTwoStateConfig, SwitchingTwoStateMDP
from alberta_framework.streams.closed_loop import PHASE_A

pytestmark = [pytest.mark.development, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Life protocol (mirrors tests/test_integrated_life.py, context hidden)
# ---------------------------------------------------------------------------

NUM_SEEDS = 8
LIFE_STEPS = 48_000
PHASE = 400
N_PHASES = LIFE_STEPS // PHASE  # 120 -> 60 recurrences of each of the 2 rules
STRESS_START = 22_000
STRESS_END = 24_000
STRESS_STD = 0.5
EARLY_W = 50
SETTLE_PHASE = 6
LATE_WINDOW = 8_000
K = 4  # bounded context slots; feature dim = K * obs_dim = 8
LAG_EXCLUDE = 20  # steps after each phase boundary excluded from agreement

_STRESS_PHASES = frozenset(range(STRESS_START // PHASE, STRESS_END // PHASE))
_CLEAN = np.array(
    [p not in _STRESS_PHASES and p != STRESS_END // PHASE for p in range(N_PHASES)]
)
_SETTLED = _CLEAN & (np.arange(N_PHASES) >= SETTLE_PHASE)

ENV = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=PHASE))
OPTIMAL = ENV.optimal_average_reward(PHASE_A)  # 1.0 (same in PHASE_B)
RANDOM = ENV.uniform_random_average_reward(PHASE_A)  # 0.5 (same in PHASE_B)

AGENT_CONFIG = DifferentialSARSAConfig(
    n_actions=2,
    q_step_size=0.1,
    average_reward_step_size=0.01,
    epsilon_start=0.05,
    epsilon_end=0.05,
    epsilon_decay_steps=0,
    use_bias=False,  # the certified continual-memory configuration
)

# ---------------------------------------------------------------------------
# Calibrated margins (see module docstring for measured tables)
# ---------------------------------------------------------------------------

# Mechanism
MECH_LAG_2RULE = 10  # measured exactly 4 on every flip
MECH_LAG_3RULE = 20  # measured exactly 9 on every flip
# (a) memory through inferred context
MEM_GAP = 0.25  # measured paired gap min +0.4930 (oracle-gated achieved +0.598)
MEM_SETTLED_MEAN = 0.70  # measured per-seed min 0.8841
MEM_LATE_MEAN = 0.75  # measured per-seed min 0.8855 (dev batch)
ABLATION_LIFE_FLOOR = 0.80  # measured per-seed min 0.8895 (twin is no straw man)
ABLATION_MEMORYLESS_CEIL = 0.60  # measured per-seed settled-early mean max 0.4002
# control learning
LIFE_FLOOR = 0.85  # measured per-seed min 0.9542
LIFE_VS_RANDOM = 0.35  # measured mean gap over random +0.458
NEAR_OPT_SLACK = 0.10  # measured late-8000 shortfall from optimum <= 0.0472
# (b) agreement with the hidden phase
AGREEMENT_FLOOR = 0.95  # measured per-seed per-era min 0.9999 over 24 seeds
# (c) bounded slots
CLEAN_SWITCH_CEIL = 140  # measured exactly 119 (= phase boundaries) on 24 seeds
# stability / stressor
Q_BOUND_INFERRED = 4.0  # measured max |q_weights| 1.706
Q_BOUND_ABLATION = 6.0  # measured max |q_weights| 2.323
RBAR_LO, RBAR_HI = 0.4, 1.2  # measured 0.942..0.992
STRESS_DURING_FLOOR = 0.60  # measured per-seed min 0.7630
STRESS_DROP = 0.05  # measured per-seed worst post-vs-pre -0.0155


def _context_config() -> ContextInferenceConfig:
    """Module defaults; calibrated for this env's {0, 1} reward scale."""
    return ContextInferenceConfig(n_actions=2, observation_dim=2, max_contexts=K)


# ---------------------------------------------------------------------------
# Mechanism-level scripted streams
# ---------------------------------------------------------------------------

RULE_A2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # env phase A
RULE_B2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # env phase B
# Three 3-state rules with pairwise table distance 4 of 6 cells; with the 2x2
# complementary pair above, any third binary table sits exactly at the
# reuse/allocate boundary (distance sums to 4), so novelty is exercised here.
RULE_R0 = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.float32)
RULE_R1 = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.float32)
RULE_R2 = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.float32)


def run_scripted(
    rules: list[np.ndarray],
    phase_len: int,
    config: ContextInferenceConfig,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Drive the module over a deterministic (state, action) sweep.

    Rewards come from the active rule table.  Returns per-step active slot,
    per-step allocated-slot count, and the reward-model bank snapshot at the
    end of each phase.
    """
    sweep = [(s, a) for s in range(n_states) for a in range(2)]
    module = ContextInference(config)
    state = module.init()
    actives, n_in_use, snapshots = [], [], []
    t = 0
    for rule in rules:
        for _ in range(phase_len):
            s, a = sweep[t % len(sweep)]
            obs = jnp.zeros(n_states, dtype=jnp.float32).at[s].set(1.0)
            state, _ = module.update(state, obs, jnp.int32(a), jnp.float32(rule[s, a]))
            actives.append(int(state.active_context))
            n_in_use.append(int(state.in_use.sum()))
            t += 1
        snapshots.append(np.asarray(state.reward_weights))
    return np.array(actives), np.array(n_in_use), snapshots


def detection_lags(actives: np.ndarray, phase_len: int, n_phases: int) -> list[int]:
    """Steps from each rule flip until the active slot first changes."""
    lags = []
    for p in range(1, n_phases):
        boundary = p * phase_len
        segment = actives[boundary : boundary + phase_len]
        changed = np.nonzero(segment != actives[boundary - 1])[0]
        assert changed.size > 0, f"flip at phase {p} never detected"
        lags.append(int(changed[0]) + 1)
    return lags


def end_of_phase_slots(actives: np.ndarray, phase_len: int, n_phases: int) -> list[int]:
    return [int(actives[(p + 1) * phase_len - 1]) for p in range(n_phases)]


class TestMechanism:
    """Scripted-stream properties of ContextInference."""

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError):
            ContextInferenceConfig(n_actions=0, observation_dim=2)
        with pytest.raises(ValueError):
            ContextInferenceConfig(n_actions=2, observation_dim=2, max_contexts=1)
        with pytest.raises(ValueError):
            ContextInferenceConfig(n_actions=2, observation_dim=2, error_decay=1.0)
        with pytest.raises(ValueError):
            # A gate below the fresh-model error scale would make freshly
            # allocated slots unable to learn.
            ContextInferenceConfig(
                n_actions=2, observation_dim=2, update_error_gate=0.3, novelty_prior_error=0.5
            )

    def test_detects_rule_flips_within_calibrated_lag(self) -> None:
        """Every flip between the two env rules is detected within 10 steps.

        Measured: the active slot changes exactly 4 steps after every flip
        (error_decay 0.8 needs (1 - 0.8^t) * 1.0 to cross switch_threshold
        0.55 under the maximal-mismatch per-step error of 1.0).
        """
        actives, _, _ = run_scripted(
            [RULE_A2, RULE_B2, RULE_A2, RULE_B2, RULE_A2], 60, _context_config(), 2
        )
        lags = detection_lags(actives, 60, 5)
        assert max(lags) <= MECH_LAG_2RULE, f"detection lags {lags} exceed {MECH_LAG_2RULE}"

    def test_reidentifies_previously_seen_rule_by_slot_reuse(self) -> None:
        """A returning rule re-activates its ORIGINAL slot — reuse, not allocation.

        Measured end-of-phase slots [0, 1, 0, 1, 0] with never more than 2
        slots allocated across 5 phases: the memory-of-contexts property.
        """
        actives, n_in_use, _ = run_scripted(
            [RULE_A2, RULE_B2, RULE_A2, RULE_B2, RULE_A2], 60, _context_config(), 2
        )
        assert end_of_phase_slots(actives, 60, 5) == [0, 1, 0, 1, 0]
        assert n_in_use.max() == 2, f"allocated {n_in_use.max()} slots for 2 rules"

    def test_novel_rule_allocates_fresh_slot_then_all_reused(self) -> None:
        """A genuinely novel rule allocates a new slot; all three then reuse.

        Three 3-state rules at pairwise distance 4/6: measured end-of-phase
        slots exactly [0, 1, 2, 0, 1, 2] (allocation on first exposure, slot
        reuse on every return), 3 slots total, detection lag 9 every flip.
        """
        config = ContextInferenceConfig(n_actions=2, observation_dim=3, max_contexts=4)
        actives, n_in_use, _ = run_scripted(
            [RULE_R0, RULE_R1, RULE_R2, RULE_R0, RULE_R1, RULE_R2], 80, config, 3
        )
        assert end_of_phase_slots(actives, 80, 6) == [0, 1, 2, 0, 1, 2]
        assert n_in_use.max() == 3
        assert max(detection_lags(actives, 80, 6)) <= MECH_LAG_3RULE

    def test_inactive_slot_models_exactly_frozen(self) -> None:
        """An inactive slot's model is bit-identical across a foreign phase.

        Exclusive gating is the memory mechanism: once slot 0 hands over to
        slot 1 (during phase 1), slot 0's reward model receives exactly zero
        updates, so its snapshot at the end of phase 1 equals its snapshot at
        the end of phase 2 bit for bit (same for slot 1 across phase 3).
        """
        config = ContextInferenceConfig(n_actions=2, observation_dim=3, max_contexts=4)
        _, _, snapshots = run_scripted(
            [RULE_R0, RULE_R1, RULE_R2, RULE_R0], 80, config, 3
        )
        assert np.array_equal(snapshots[1][0], snapshots[2][0]), "slot 0 changed while inactive"
        assert np.array_equal(snapshots[2][1], snapshots[3][1]), "slot 1 changed while inactive"

    def test_slot_bound_holds_via_lru_eviction(self) -> None:
        """Three rules with K=2 slots: allocation evicts, the bound holds.

        Measured: the bank cycles through the rules with never more than 2
        slots allocated and every flip still detected (lag 9).
        """
        config = ContextInferenceConfig(n_actions=2, observation_dim=3, max_contexts=2)
        actives, n_in_use, _ = run_scripted(
            [RULE_R0, RULE_R1, RULE_R2, RULE_R0, RULE_R1], 80, config, 3
        )
        assert n_in_use.max() <= 2, "slot bound violated"
        assert max(detection_lags(actives, 80, 5)) <= MECH_LAG_3RULE

    def test_update_is_scan_and_vmap_safe(self) -> None:
        """The pure-functional update composes with lax.scan and vmap."""
        module = ContextInference(_context_config())

        def one(key: jax.Array) -> tuple:
            obs_key, act_key, rew_key = jr.split(key, 3)
            observations = jax.nn.one_hot(
                jr.randint(obs_key, (50,), 0, 2), 2, dtype=jnp.float32
            )
            actions = jr.randint(act_key, (50,), 0, 2)
            rewards = jr.bernoulli(rew_key, 0.5, (50,)).astype(jnp.float32)

            def step(state, inputs):
                obs, action, reward = inputs
                new_state, onehot = module.update(state, obs, action, reward)
                return new_state, onehot

            return jax.lax.scan(step, module.init(), (observations, actions, rewards))

        final_states, onehots = jax.vmap(one)(jr.split(jr.key(0), 3))
        assert onehots.shape == (3, 50, K)
        assert bool(jnp.all(jnp.isfinite(onehots)))
        assert bool(jnp.all(jnp.sum(onehots, axis=-1) == 1.0))
        assert bool(jnp.all(final_states.active_context >= 0))
        assert bool(jnp.all(final_states.active_context < K))
        assert bool(jnp.all(jnp.isfinite(final_states.reward_weights)))


# ---------------------------------------------------------------------------
# L3 life runners (vmapped over paired seeds)
# ---------------------------------------------------------------------------


def _perturbed(obs2: jax.Array, env_state, key: jax.Array, stress: bool) -> jax.Array:
    """The mid-life input stressor on the (2-dim, context-free) percept."""
    if not stress:
        return obs2
    in_window = (
        (env_state.step_count >= STRESS_START) & (env_state.step_count < STRESS_END)
    ).astype(jnp.float32)
    return obs2 + in_window * STRESS_STD * jr.normal(key, (2,), dtype=jnp.float32)


def run_inferred_batch(keys: jax.Array, stress: bool) -> dict[str, np.ndarray]:
    """Inferred-context-gated SARSA lives: features = kron(ctx_onehot, x)."""
    module = ContextInference(_context_config())
    agent = DifferentialSARSAAgent(AGENT_CONFIG)
    feature_dim = 2 * K

    def one(seed_key: jax.Array) -> tuple:
        env_key, agent_key, scan_key = jr.split(seed_key, 3)
        e_state = ENV.init(env_key)
        obs0 = ENV.observe(e_state)
        c_state = module.init()
        phi0 = (module.context_onehot(c_state)[:, None] * obs0[None, :]).reshape(-1)
        a_state = agent.init(feature_dim, agent_key)
        a_state, _ = agent.start(a_state, phi0)

        def scan_fn(carry, step_key):
            a_st, e_st, c_st, prev_obs = carry
            k_env, k_noise = jr.split(step_key)
            phase = ENV.phase_id(e_st)  # hidden truth, recorded for eval only
            _, reward, e2 = ENV.step(e_st, a_st.last_action, k_env)
            obs_next = _perturbed(ENV.observe(e2), e2, k_noise, stress)
            c2, onehot = module.update(c_st, prev_obs, a_st.last_action, reward)
            phi_next = (onehot[:, None] * obs_next[None, :]).reshape(-1)
            result = agent.update(a_st, reward, phi_next)
            outputs = (reward, c2.active_context, jnp.sum(c2.in_use), phase)
            return (result.state, e2, c2, obs_next), outputs

        (a_fin, _, _, _), (rewards, actives, n_in_use, phases) = jax.lax.scan(
            scan_fn, (a_state, e_state, c_state, obs0), jr.split(scan_key, LIFE_STEPS)
        )
        return (
            rewards,
            actives,
            n_in_use,
            phases,
            a_fin.q_weights,
            a_fin.q_bias,
            a_fin.average_reward,
        )

    out = jax.vmap(one)(keys)
    names = ("rewards", "actives", "n_in_use", "phases", "q_weights", "q_bias", "rbar")
    return {name: np.asarray(value) for name, value in zip(names, out)}


def run_ablation_batch(keys: jax.Array) -> dict[str, np.ndarray]:
    """The no-context twin: same agent, same life, features are just x."""
    agent = DifferentialSARSAAgent(AGENT_CONFIG)

    def one(seed_key: jax.Array) -> tuple:
        env_key, agent_key, scan_key = jr.split(seed_key, 3)
        e_state = ENV.init(env_key)
        a_state = agent.init(2, agent_key)
        a_state, _ = agent.start(a_state, ENV.observe(e_state))

        def scan_fn(carry, step_key):
            a_st, e_st = carry
            k_env, k_noise = jr.split(step_key)
            _, reward, e2 = ENV.step(e_st, a_st.last_action, k_env)
            obs_next = _perturbed(ENV.observe(e2), e2, k_noise, True)
            result = agent.update(a_st, reward, obs_next)
            return (result.state, e2), reward

        (a_fin, _), rewards = jax.lax.scan(
            scan_fn, (a_state, e_state), jr.split(scan_key, LIFE_STEPS)
        )
        return rewards, a_fin.q_weights, a_fin.q_bias, a_fin.average_reward

    out = jax.vmap(one)(keys)
    names = ("rewards", "q_weights", "q_bias", "rbar")
    return {name: np.asarray(value) for name, value in zip(names, out)}


def early_per_phase(rewards: np.ndarray) -> np.ndarray:
    """Per-seed early-window (first 50 steps) mean reward per phase occurrence."""
    return rewards.reshape(-1, N_PHASES, PHASE)[:, :, :EARLY_W].mean(axis=2)


def era_agreement(actives: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Per-seed agreement of inferred slot with the hidden phase.

    Settled clean phases only, first LAG_EXCLUDE steps of every phase
    excluded (the inference lag).  Slot identities are arbitrary names and
    the stressor may permute them, so the slot->phase majority mapping is
    fitted separately for the pre-stress and post-stress eras and the
    agreement is pooled over both.
    """
    step_phase = np.arange(LIFE_STEPS) // PHASE
    within = np.arange(LIFE_STEPS) % PHASE
    base = _SETTLED[step_phase] & (within >= LAG_EXCLUDE)
    eras = [
        base & (np.arange(LIFE_STEPS) < STRESS_START),
        base & (np.arange(LIFE_STEPS) >= STRESS_END),
    ]
    out = []
    for s in range(actives.shape[0]):
        agree = total = 0
        for include in eras:
            slot_ids, phase_ids = actives[s][include], phases[s][include]
            for slot in np.unique(slot_ids):
                mask = slot_ids == slot
                agree += np.bincount(phase_ids[mask], minlength=2).max()
            total += len(slot_ids)
        out.append(agree / total)
    return np.array(out)


@pytest.fixture(scope="module")
def l3_keys() -> jax.Array:
    return jnp.stack([jr.key(seed) for seed in range(NUM_SEEDS)])


@pytest.fixture(scope="module")
def inferred_run(l3_keys) -> dict[str, np.ndarray]:
    run = run_inferred_batch(l3_keys, stress=True)
    assert np.isfinite(run["rewards"]).all()
    return run


@pytest.fixture(scope="module")
def clean_run(l3_keys) -> dict[str, np.ndarray]:
    run = run_inferred_batch(l3_keys, stress=False)
    assert np.isfinite(run["rewards"]).all()
    return run


@pytest.fixture(scope="module")
def ablation_run(l3_keys) -> dict[str, np.ndarray]:
    run = run_ablation_batch(l3_keys)
    assert np.isfinite(run["rewards"]).all()
    return run


class TestHiddenContextLife:
    """The L3 life protocol with the context inferred, not observed."""

    def test_control_learning_above_random_toward_optimum(self, inferred_run) -> None:
        """Lifetime reward far above random; late life approaches the optimum.

        Measured: life per-seed min 0.9542 [worst batch] against random 0.5;
        final-8000 per-seed min 0.9528 against the exact optimum 1.0.
        """
        life = inferred_run["rewards"].mean(axis=1)
        late = inferred_run["rewards"][:, -LATE_WINDOW:].mean(axis=1)
        assert life.min() >= LIFE_FLOOR, f"lifetime reward {life.min():.4f} < {LIFE_FLOOR}"
        assert life.mean() >= RANDOM + LIFE_VS_RANDOM, (
            f"lifetime reward {life.mean():.4f} does not clear random "
            f"{RANDOM} by {LIFE_VS_RANDOM}"
        )
        assert late.min() >= OPTIMAL - NEAR_OPT_SLACK, (
            f"late-life reward {late.min():.4f} not within {NEAR_OPT_SLACK} "
            f"of the analytic optimum {OPTIMAL}"
        )

    def test_inferred_memory_beats_no_context_ablation_paired(
        self, inferred_run, ablation_run
    ) -> None:
        """(a) Inferred-context gating recovers most of the oracle's memory gap.

        Same agent, same seeds, same life; only the feature gating differs
        (inferred slot one-hot vs nothing).  Measured paired settled
        early-window gap: mean +0.519 / per-seed min +0.493 [worst batch]
        against the oracle-gated +0.598 from test_integrated_life — ~87% of
        the oracle's advantage with the context inferred from rewards alone.
        Late-life re-coordination stays at its best (measured late-third mean
        0.8962, per-seed min 0.8855).
        """
        inferred_settled = early_per_phase(inferred_run["rewards"])[:, _SETTLED].mean(axis=1)
        ablation_settled = early_per_phase(ablation_run["rewards"])[:, _SETTLED].mean(axis=1)
        gap = inferred_settled - ablation_settled
        assert gap.min() >= MEM_GAP, (
            f"paired memory gap collapsed on a seed: min {gap.min():.4f} < {MEM_GAP}"
        )
        assert gap.mean() >= MEM_GAP, f"mean paired memory gap {gap.mean():.4f} < {MEM_GAP}"
        assert inferred_settled.min() >= MEM_SETTLED_MEAN, (
            f"settled early-window {inferred_settled.min():.4f} < {MEM_SETTLED_MEAN}"
        )
        late_phases = _SETTLED & (np.arange(N_PHASES) >= 80)
        late_early = early_per_phase(inferred_run["rewards"])[:, late_phases].mean(axis=1)
        assert late_early.min() >= MEM_LATE_MEAN, (
            f"late-life re-coordination degraded: {late_early.min():.4f} < {MEM_LATE_MEAN}"
        )

    def test_ablation_twin_is_plastic_but_memoryless(self, ablation_run) -> None:
        """The twin learns fine within phases — it just cannot remember.

        Measured: ablation lifetime per-seed min 0.8895 (strong control);
        its settled early-window mean never exceeds 0.4002 per seed (every
        recurrence pays the full relearning cost, all 120 switches).
        """
        life = ablation_run["rewards"].mean(axis=1)
        settled = early_per_phase(ablation_run["rewards"])[:, _SETTLED].mean(axis=1)
        assert life.min() >= ABLATION_LIFE_FLOOR, (
            f"ablation twin failed to learn: lifetime {life.min():.4f} < {ABLATION_LIFE_FLOOR}"
        )
        assert settled.max() <= ABLATION_MEMORYLESS_CEIL, (
            f"ablation twin unexpectedly retained conventions: "
            f"{settled.max():.4f} > {ABLATION_MEMORYLESS_CEIL}"
        )

    def test_inferred_context_agrees_with_hidden_phase(
        self, inferred_run, clean_run
    ) -> None:
        """(b) The inferred context IS the hidden phase, up to slot naming.

        Settled clean steps, first 20 steps of each phase excluded (inference
        lag; scripted lag is 4 steps), slot->phase mapping fitted per stress
        era (the stressor may permute slot identities — a label remap, not
        misclassification: measured whole-life fixed-mapping agreement drops
        to 0.546 on one seed while both eras measure 1.0 under their own
        mappings).  Measured per-seed per-era agreement min 0.9999 across all
        24 calibration seeds; no-stress arm min 1.0000.
        """
        stressed = era_agreement(inferred_run["actives"], inferred_run["phases"])
        clean = era_agreement(clean_run["actives"], clean_run["phases"])
        assert stressed.min() >= AGREEMENT_FLOOR, (
            f"inferred context diverged from the hidden phase: "
            f"{stressed.min():.4f} < {AGREEMENT_FLOOR}"
        )
        assert clean.min() >= AGREEMENT_FLOOR, (
            f"clean-life agreement {clean.min():.4f} < {AGREEMENT_FLOOR}"
        )

    def test_bounded_context_allocation(self, inferred_run, clean_run) -> None:
        """(c) Two rules occupy two slots — no unbounded allocation in 120 switches.

        Measured: the no-stress life allocates exactly 2 slots and switches
        exactly 119 times (= the phase boundaries) on every seed of all three
        calibration batches; the stressed life still holds exactly 2 slots at
        step 22k (55 phases in) and only the 2000-step noise stressor churns
        the bank up to its hard K=4 bound.
        """
        clean_in_use = clean_run["n_in_use"]
        assert clean_in_use.max() == 2, (
            f"clean life allocated {clean_in_use.max()} slots for 2 rules"
        )
        switches = (clean_run["actives"][:, 1:] != clean_run["actives"][:, :-1]).sum(axis=1)
        assert switches.max() <= CLEAN_SWITCH_CEIL, (
            f"clean life switched {switches.max()} times (> {CLEAN_SWITCH_CEIL})"
        )
        pre_stress = inferred_run["n_in_use"][:, STRESS_START - 1]
        assert np.all(pre_stress == 2), f"pre-stress slot counts {pre_stress} != 2"
        assert inferred_run["n_in_use"].max() <= K

    def test_stability_and_stressor_survived(self, inferred_run, ablation_run) -> None:
        """No NaN; weights bounded; bias channel closed; stressor survived.

        Measured: |q_weights| max 1.706 (inferred) / 2.323 (ablation); rbar
        in [0.942, 0.992]; q_bias identically 0; during-stress reward per-seed
        min 0.7630; post-vs-pre worst -0.0155 (the slot remap costs 1-2
        phases of rebuild, then the memory re-forms).
        """
        for run, bound in ((inferred_run, Q_BOUND_INFERRED), (ablation_run, Q_BOUND_ABLATION)):
            assert np.isfinite(run["q_weights"]).all()
            assert np.abs(run["q_weights"]).max() <= bound, (
                f"|q_weights| {np.abs(run['q_weights']).max():.3f} > {bound}"
            )
            assert np.all(run["q_bias"] == 0.0), "the bias forgetting channel opened"
            assert np.all((run["rbar"] >= RBAR_LO) & (run["rbar"] <= RBAR_HI))
        rewards = inferred_run["rewards"]
        during = rewards[:, STRESS_START:STRESS_END].mean(axis=1)
        pre = rewards[:, STRESS_START - 4_000 : STRESS_START].mean(axis=1)
        post = rewards[:, STRESS_END + PHASE : STRESS_END + PHASE + 4_000].mean(axis=1)
        assert during.min() >= STRESS_DURING_FLOOR, (
            f"collapsed during the stressor: {during.min():.4f} < {STRESS_DURING_FLOOR}"
        )
        assert (pre - post).max() <= STRESS_DROP, (
            f"post-stressor reward did not recover: worst drop "
            f"{(pre - post).max():.4f} > {STRESS_DROP}"
        )
