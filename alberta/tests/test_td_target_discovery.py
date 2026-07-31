"""TD-target feature discovery: candidates scored by the control agent's own TD target.

``tests/test_discovery_control_life.py`` closed the discovery-control loop with
an *auxiliary* utility signal: the candidate bank was trained to predict the
immediate reward of the taken action.  That leaves the Alberta Plan Step 3
research boundary named in the ROADMAP open — discovery driven by the control
agent's OWN learning signal, not a stand-in supervised task.  This module
closes that gap: the ONLY change from the reward-target reference file is the
regression target handed to :class:`FixedBudgetInteractionLearner`.

Per step with taken action ``a`` in state ``s``: after the control update, the
discovery target for task ``a`` is the differential SARSA target the agent
itself just regressed ``Q(s, a)`` toward,

    ``y = r - rbar + Q(s', a')``  (equivalently ``td_error + Q(s, a)``),

with ``rbar``/``Q`` read *before* the update and ``a'`` the action the agent
actually selected (all recovered from ``DifferentialSARSAUpdateResult``:
``y = r - rbar_pre + result.q_values[result.action]``; the identity with
``result.td_error`` is unit-tested below).  Other actions are NaN-masked.
A candidate feature that predicts ``y`` well is exactly a feature that would
reduce the Bellman residual of the control Q-function — utility grounded in
the agent's own value-learning error rather than in an auxiliary task.

Architecture is otherwise the reference file's, replicated here so the three
target modes share one code path (``target_mode`` in {"td", "reward",
"random"}): :class:`ContextObservableSwitchingEnv` (SwitchingTwoStateMDP +
observable rule context + 2 Rademacher distractors), the frozen scale-robust
discovery configuration (8 slots, full 15-pair archive), DifferentialSARSA
control with ``use_bias=False``, and two-timescale coupling with Q-weight
carry-over by pair identity.  One deliberate ordering delta: the bank refresh
reads the *previous* step's discovery state so the control update (whose
result defines the TD target) can run before the discovery update.  The bank
only changes at 50-step replacement events, so this one-step lag is
immaterial — the reward mode below reproduces the reference file's numbers.

The "random" mode is the grounding ablation: the target is an information-free
Bernoulli(0.5) draw with the reward's support, so any structure it finds is
chance.  With 8 slots over 15 pairs (4 relevant), chance alone parks ~2
relevant products in the bank — the ablation contrast is therefore against
that chance floor, not against zero.

Calibration (2026-07-30, CPU, this file's exact protocol; development batch
``jr.key(0)`` plus robustness batches ``jr.key(100)/jr.key(7)/jr.key(42)``,
ranges are across the four batches):

    arm            recurrence-early mean (worst seed)  final distinct relevant
    raw twin       0.7517-0.7669 mean                  --
    td-target      0.9190-0.9300 (0.8883)              min 3, mean 3.25-3.625
    reward-target  0.9329-0.9417 (0.9067)              min 3, all-4 in 29/32 seeds
    random-target  0.7729-0.8090 mean                  mean 2.25-2.375, min 1-2

    paired diffs (phases 2..11)     dev mean/min       worst batch mean/min
    td - raw                        +0.1654 / +0.0900  +0.1585 / +0.0833
    td - reward                     -0.0110 / -0.0483  -0.0177 / -0.0483
    td - random (rec-early)         +0.1129 / -0.0017  +0.1129 / -0.0233
    td distinct - random distinct   +0.875 / 0         +0.875 / 0
    td lifetime - raw lifetime      +0.0074 / -0.0244  +0.0050 / -0.0297

    td promotions per seed: 5-10 in every batch.  Random-target promotions
    still fire (6-10 per seed) but churn toward chance-level banks: final
    distinct relevant mean 2.25-2.375 with seeds down at 1, vs a hard
    per-seed min of 3 for the td bank in every batch.

Head-to-head finding: the privileged immediate-reward target keeps a small,
consistent edge over the agent's own Bellman target — paired mean diff
-0.0110..-0.0177 (reward wins every batch; per-seed extremes [-0.0483,
+0.0167]), and it pins all four products more often (29/32 seeds vs 11/32;
both hold a per-seed min of 3).  The TD target pays for bootstrapping: early
in learning its target mixes reward structure with a still-wrong Q, diluting
the candidate-scoring signal.  The result under test is that this cost is
small and bounded — TD-target discovery is non-inferior within a calibrated
band while removing the auxiliary-task crutch entirely.  Every frozen
threshold keeps a >= 2x margin against the worst measured batch value.

Honest scope limits: the rule context is supplied as an observable channel
(context inference stays a separate problem); the pair space is closed and
tiny; the TD target is still bootstrapped from a linear Q over the same
discovered bank, so target and representation co-evolve — stability here is
demonstrated empirically, not guaranteed.

Runtime: ~15-25s on CPU (four vmapped scan arms, module-scoped fixtures).
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
)
from alberta_framework.core.interaction_features import FixedBudgetInteractionLearner
from alberta_framework.streams.closed_loop import (
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

N_SEEDS = 8
PHASE_LEN = 300
NUM_PHASES = 12
NUM_STEPS = PHASE_LEN * NUM_PHASES
N_DISTRACTORS = 2
OBS_DIM = 2 + 2 + N_DISTRACTORS
N_SLOTS = 8
N_CANDIDATES = 15  # C(6, 2): the full pair space of the 6-dim observation
REFRESH_EVERY = 50
EARLY_WINDOW = 60
FIRST_RECURRENCE_PHASE = 2  # phases 0/1 are the first exposure to each rule

# The four context-binding products: s_i * c_k with state channels {0, 1} and
# context channels {2, 3} (canonical left < right).
ORACLE_PAIRS = {(0, 2), (1, 2), (0, 3), (1, 3)}


# ---------------------------------------------------------------------------
# Adapters (replicated from test_discovery_control_life.py so all target
# modes share one code path; that file remains the reward-target reference)
# ---------------------------------------------------------------------------


class ContextObservableSwitchingEnv:
    """SwitchingTwoStateMDP with observable rule context and distractor channels.

    Observation: ``[one_hot(state), one_hot(active rule), rademacher(2)]``.
    The context channel makes the recurring rule *identifiable* without making
    it linearly *actionable*: expressing either rule's optimal policy requires
    a state-conditional action preference, which raw channels cannot bind.
    """

    def __init__(self, phase_length: int = PHASE_LEN, n_distractors: int = N_DISTRACTORS):
        self._env = SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=phase_length))
        self._n_distractors = n_distractors

    @property
    def feature_dim(self) -> int:
        """Wrapped observation dimension."""
        return self._env.feature_dim + 2 + self._n_distractors

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return self._env.n_actions

    def init(self, key: Array) -> Any:
        """Delegate to the wrapped MDP."""
        return self._env.init(key)

    def observe(self, env_state: Any, key: Array) -> Array:
        """Wrapped observation of ``env_state`` with fresh distractor draws."""
        state = self._env.observe(env_state)
        context = jax.nn.one_hot(self._env.phase_id(env_state), 2, dtype=jnp.float32)
        distractors = jr.rademacher(key, (self._n_distractors,), dtype=jnp.float32)
        return jnp.concatenate([state, context, distractors])

    def step(self, env_state: Any, action: Array, key: Array) -> tuple[Array, Array, Any]:
        """One transition; returns (wrapped next obs, reward, next env state)."""
        k_env, k_obs = jr.split(key)
        _, reward, next_state = self._env.step(env_state, action, k_env)
        return self.observe(next_state, k_obs), reward, next_state


def pair_products(obs: Array, left: Array, right: Array) -> Array:
    """Constructed pair-product features ``obs[left] * obs[right]``."""
    return obs[left] * obs[right]


def carry_q_weights_by_pair_identity(
    old_left: Array,
    old_right: Array,
    old_q: Array,
    new_left: Array,
    new_right: Array,
) -> Array:
    """Carry Q-weight columns from an old bank to a new bank by pair identity.

    For each new slot: if its ``(left, right)`` descriptor exists anywhere in
    the old bank, copy that pair's Q column (first match); otherwise start at
    zero.  Duplicate descriptors within the new bank keep only their first
    copy.  This function IS the coupled agent's task memory across refreshes.
    """
    match = (new_left[:, None] == old_left[None, :]) & (new_right[:, None] == old_right[None, :])
    has_match = match.any(axis=1).astype(jnp.float32)
    source = jnp.argmax(match, axis=1)
    carried = old_q[:, source] * has_match[None, :]
    slots = jnp.arange(new_left.shape[0])
    earlier_duplicate = (
        (new_left[:, None] == new_left[None, :])
        & (new_right[:, None] == new_right[None, :])
        & (slots[None, :] < slots[:, None])
    ).any(axis=1)
    return carried * (~earlier_duplicate).astype(jnp.float32)[None, :]


def distinct_relevant_in_bank(left: Array, right: Array) -> Array:
    """Count DISTINCT oracle context-binding products in a bank (traceable)."""
    is_relevant = (left < 2) & (right >= 2) & (right < 4)
    slots = jnp.arange(left.shape[0])
    earlier_duplicate = (
        (left[:, None] == left[None, :])
        & (right[:, None] == right[None, :])
        & (slots[None, :] < slots[:, None])
    ).any(axis=1)
    return jnp.sum(is_relevant & ~earlier_duplicate)


@chex.dataclass(frozen=True)
class CoupledState:
    """Carry for the discovery-driven control loop."""

    env_state: Any
    raw_obs: Array
    disc_state: Any
    bank_left: Array
    bank_right: Array
    control_state: DifferentialSARSAState


def make_control_agent(n_actions: int) -> DifferentialSARSAAgent:
    """The repo-proven flat-control configuration (bias disabled)."""
    return DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=n_actions,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=0.1,
            epsilon_end=0.1,
            epsilon_decay_steps=0,
            use_bias=False,
        )
    )


class TargetModeDiscoveryControlAgent:
    """Discovery-driven control where the discovery target is switchable.

    ``target_mode``:
      * ``"td"`` — the control agent's own differential SARSA target for the
        taken action, ``y = r - rbar_pre + Q_pre(s', a')``, read off the
        control update result (the quantity under test in this module);
      * ``"reward"`` — the immediate reward (the reference file's target);
      * ``"random"`` — an information-free Bernoulli(0.5) draw (grounding
        ablation: same support as the reward, zero mutual information with
        the transition).

    Everything else — env, discovery configuration, control agent, refresh
    cadence, pair-identity weight carry-over — is identical across modes.
    """

    def __init__(
        self,
        env: ContextObservableSwitchingEnv,
        target_mode: str = "td",
        refresh_every: int = REFRESH_EVERY,
    ):
        if target_mode not in {"td", "reward", "random"}:
            raise ValueError("target_mode must be 'td', 'reward', or 'random'")
        self._env = env
        self._target_mode = target_mode
        self._refresh_every = refresh_every
        self._control = make_control_agent(env.n_actions)
        self._disc = FixedBudgetInteractionLearner(
            n_features=N_SLOTS,
            n_tasks=env.n_actions,
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

    def init(self, key: Array) -> CoupledState:
        """Initialize env, discovery bank, and control state; select first action."""
        k_env, k_obs, k_disc, k_ctrl = jr.split(key, 4)
        env_state = self._env.init(k_env)
        raw_obs = self._env.observe(env_state, k_obs)
        disc_state = self._disc.init(self._env.feature_dim, k_disc)
        bank_left = disc_state.feature_left
        bank_right = disc_state.feature_right
        control_state = self._control.init(N_SLOTS, k_ctrl)
        control_state, _ = self._control.start(
            control_state, pair_products(raw_obs, bank_left, bank_right)
        )
        return CoupledState(
            env_state=env_state,
            raw_obs=raw_obs,
            disc_state=disc_state,
            bank_left=bank_left,
            bank_right=bank_right,
            control_state=control_state,
        )

    def step(self, state: CoupledState, t: Array, key: Array) -> tuple[CoupledState, tuple]:
        """One closed-loop step: act, control-update, discover on the TD target."""
        k_env, k_noise = jr.split(key)
        action = state.control_state.last_action
        next_obs, reward, next_env_state = self._env.step(state.env_state, action, k_env)

        # Refresh from the PREVIOUS step's discovery state so the control
        # update (whose result defines the TD target) runs first.  The bank
        # only changes at 50-step replacement events, so the one-step lag is
        # immaterial (the reward mode reproduces the reference file's numbers).
        do_refresh = (t % self._refresh_every == 0) & (t > 0)
        new_left = state.disc_state.feature_left
        new_right = state.disc_state.feature_right
        carried = carry_q_weights_by_pair_identity(
            state.bank_left,
            state.bank_right,
            state.control_state.q_weights,
            new_left,
            new_right,
        )
        q_weights = jnp.where(do_refresh, carried, state.control_state.q_weights)
        q_traces = jnp.where(
            do_refresh,
            jnp.zeros_like(state.control_state.q_trace_weights),
            state.control_state.q_trace_weights,
        )
        bank_left = jnp.where(do_refresh, new_left, state.bank_left)
        bank_right = jnp.where(do_refresh, new_right, state.bank_right)

        phi_prev = pair_products(state.raw_obs, bank_left, bank_right)
        phi_next = pair_products(next_obs, bank_left, bank_right)
        control_state = state.control_state.replace(
            q_weights=q_weights,
            q_trace_weights=q_traces,
            last_observation=phi_prev,
        )
        rbar_pre = control_state.average_reward
        control_result = self._control.update(control_state, reward, phi_next)

        # The agent's own regression target for Q(s, a): pre-update rbar and
        # pre-update Q at (s', a') with the actually selected a'.  Identity
        # with td_error + Q_pre(s, a) is unit-tested in TestTDTargetMechanism.
        td_target = reward - rbar_pre + control_result.q_values[control_result.action]
        if self._target_mode == "td":
            target_value = td_target
        elif self._target_mode == "reward":
            target_value = reward
        else:
            target_value = jr.bernoulli(k_noise).astype(jnp.float32)

        # Discovery: the taken action's head sees the pre-action observation.
        targets = jnp.full((self._env.n_actions,), jnp.nan, dtype=jnp.float32)
        targets = targets.at[action].set(target_value)
        disc_result = self._disc.update(state.disc_state, state.raw_obs, targets)

        new_state = CoupledState(
            env_state=next_env_state,
            raw_obs=next_obs,
            disc_state=disc_result.state,
            bank_left=bank_left,
            bank_right=bank_right,
            control_state=control_result.state,
        )
        outputs = (
            reward,
            distinct_relevant_in_bank(bank_left, bank_right),
            (disc_result.promoted_candidate >= 0).astype(jnp.int32),
        )
        return new_state, outputs


# ---------------------------------------------------------------------------
# Runners (vmapped over paired seeds)
# ---------------------------------------------------------------------------


def run_coupled_batch(keys: Array, target_mode: str) -> tuple[Array, Array, Array, Array, Array]:
    """Vmapped coupled runs; returns (rewards, distinct trace, promotions, bank)."""
    env = ContextObservableSwitchingEnv()
    agent = TargetModeDiscoveryControlAgent(env, target_mode=target_mode)

    def one_seed(key: Array) -> tuple[Array, Array, Array, Array, Array]:
        k_init, k_scan = jr.split(key)
        state = agent.init(k_init)
        final, (rewards, distinct, promoted) = jax.lax.scan(
            lambda s, inp: agent.step(s, inp[0], inp[1]),
            state,
            (jnp.arange(NUM_STEPS), jr.split(k_scan, NUM_STEPS)),
        )
        return rewards, distinct, promoted, final.bank_left, final.bank_right

    return jax.vmap(one_seed)(keys)


def run_raw_batch(keys: Array) -> Array:
    """Vmapped raw-observation SARSA twin (the interference baseline)."""
    env = ContextObservableSwitchingEnv()
    agent = make_control_agent(env.n_actions)

    def one_seed(key: Array) -> Array:
        k_env, k_obs, k_agent, k_scan = jr.split(key, 4)
        env_state = env.init(k_env)
        obs = env.observe(env_state, k_obs)
        a_state = agent.init(env.feature_dim, k_agent)
        a_state, _ = agent.start(a_state, obs)

        def step(carry: tuple, key: Array) -> tuple[tuple, Array]:
            a_st, e_st = carry
            k_env_step, _ = jr.split(key)
            next_obs, reward, e_next = env.step(e_st, a_st.last_action, k_env_step)
            result = agent.update(a_st, reward, next_obs)
            return (result.state, e_next), reward

        _, rewards = jax.lax.scan(step, (a_state, env_state), jr.split(k_scan, NUM_STEPS))
        return rewards

    return jax.vmap(one_seed)(keys)


def recurrence_early(rewards: Array) -> np.ndarray:
    """Per-seed mean reward in the first EARLY_WINDOW steps of phases 2..11."""
    per_phase = np.asarray(rewards).reshape(N_SEEDS, NUM_PHASES, PHASE_LEN)
    early = per_phase[:, :, :EARLY_WINDOW].mean(axis=2)
    return early[:, FIRST_RECURRENCE_PHASE:].mean(axis=1)


def final_distinct_counts(run: tuple[Array, Array, Array, Array, Array]) -> np.ndarray:
    """Per-seed distinct oracle products in the final control bank."""
    return np.asarray(run[1])[:, -1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keys() -> Array:
    return jr.split(jr.key(0), N_SEEDS)


@pytest.fixture(scope="module")
def raw_rewards(keys: Array) -> Array:
    rewards = run_raw_batch(keys)
    assert bool(jnp.all(jnp.isfinite(rewards)))
    return rewards


@pytest.fixture(scope="module")
def td_run(keys: Array) -> tuple[Array, Array, Array, Array, Array]:
    run = run_coupled_batch(keys, "td")
    assert bool(jnp.all(jnp.isfinite(run[0])))
    return run


@pytest.fixture(scope="module")
def reward_run(keys: Array) -> tuple[Array, Array, Array, Array, Array]:
    run = run_coupled_batch(keys, "reward")
    assert bool(jnp.all(jnp.isfinite(run[0])))
    return run


@pytest.fixture(scope="module")
def random_run(keys: Array) -> tuple[Array, Array, Array, Array, Array]:
    run = run_coupled_batch(keys, "random")
    assert bool(jnp.all(jnp.isfinite(run[0])))
    return run


# ---------------------------------------------------------------------------
# Mechanism: the discovery target IS the agent's own regression target
# ---------------------------------------------------------------------------


class TestTDTargetMechanism:
    def test_td_target_equals_td_error_plus_q_prev(self) -> None:
        """``r - rbar_pre + Q_pre(s', a')`` == ``td_error + Q_pre(s, a)``.

        The adapter computes the discovery target from the update result as
        ``reward - rbar_pre + result.q_values[result.action]``; differential
        SARSA defines ``td_error = y - Q_pre(s, a)``.  This pins the two
        formulations together on a warm, asymmetric state so the adapter's
        target is provably the quantity the agent regressed toward.
        """
        agent = make_control_agent(2)
        k_init, k_w, k_phi, k_phi2 = jr.split(jr.key(3), 4)
        state = agent.init(N_SLOTS, k_init)
        phi_prev = jr.normal(k_phi, (N_SLOTS,))
        state = state.replace(
            q_weights=jr.normal(k_w, (2, N_SLOTS)),
            average_reward=jnp.array(0.37, dtype=jnp.float32),
            last_observation=phi_prev,
            last_action=jnp.array(1, dtype=jnp.int32),
        )
        phi_next = jr.normal(k_phi2, (N_SLOTS,))
        reward = jnp.array(0.6, dtype=jnp.float32)

        q_prev = agent.q_values(state, phi_prev)[1]
        result = agent.update(state, reward, phi_next)
        target_from_result = reward - state.average_reward + result.q_values[result.action]

        assert jnp.allclose(target_from_result, result.td_error + q_prev, atol=1e-6)
        # And q_values really is the PRE-update Q at s' (not post-update).
        assert jnp.allclose(result.q_values, state.q_weights @ phi_next, atol=1e-6)

    def test_carry_matches_by_identity_zeroes_new_and_deduplicates(self) -> None:
        """Guard the replicated carry-over helper (the coupled agent's memory)."""
        old_left = jnp.array([0, 1, 4], dtype=jnp.int32)
        old_right = jnp.array([2, 3, 5], dtype=jnp.int32)
        old_q = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
        new_left = jnp.array([1, 0, 0, 2], dtype=jnp.int32)
        new_right = jnp.array([3, 2, 2, 4], dtype=jnp.int32)

        carried = carry_q_weights_by_pair_identity(old_left, old_right, old_q, new_left, new_right)

        expected = jnp.array([[2.0, 1.0, 0.0, 0.0], [5.0, 4.0, 0.0, 0.0]], dtype=jnp.float32)
        assert jnp.array_equal(carried, expected)


# ---------------------------------------------------------------------------
# (1) The TD-target-driven bank finds the context-binding products
# ---------------------------------------------------------------------------


class TestTDTargetDiscoveredRepresentation:
    def test_context_binding_products_found_from_td_target(self, td_run) -> None:
        """The TD-target bank holds distinct oracle products, early and finally.

        Measured (dev batch): distinct relevant at end of phase 3
        [3 3 4 3 4 3 3 3], unchanged at the end; across all four calibration
        batches the per-seed minimum was 3 both mid-run and finally, and the
        final mean 3.25-3.625 (chance floor for a random 8-slot bank over
        this 15-pair space is ~2.1-2.4, see the ablation test).  Frozen:
        at least 2 per seed at both time points, final mean >= 2.75 (the
        half-effect point between the measured worst mean and chance).
        """
        _, distinct_trace, promoted, bank_left, bank_right = td_run

        distinct = np.asarray(distinct_trace)
        end_phase_3 = 4 * PHASE_LEN - 1
        assert int(distinct[:, end_phase_3].min()) >= 2
        assert int(distinct[:, -1].min()) >= 2
        assert float(distinct[:, -1].mean()) >= 2.75

        # Cross-check the trace against the returned final descriptors.
        left_np, right_np = np.asarray(bank_left), np.asarray(bank_right)
        final_counts = [
            len({(int(a), int(b)) for a, b in zip(left_np[s], right_np[s])} & ORACLE_PAIRS)
            for s in range(N_SEEDS)
        ]
        assert final_counts == [int(c) for c in distinct[:, -1]]

        # Built by online promotions, not init luck (measured 5-10 per seed).
        assert int(np.asarray(promoted).sum(axis=1).min()) >= 2


# ---------------------------------------------------------------------------
# (2) Control over TD-target-discovered features retains on recurrence
# ---------------------------------------------------------------------------


class TestTDTargetRetention:
    def test_td_discovered_features_beat_raw_twin(self, td_run, raw_rewards) -> None:
        """Paired early-recurrence advantage of TD-discovered features over raw.

        Measured paired diff (phases 2..11, 60-step early windows): dev batch
        +0.1654 mean / +0.0900 min; worst calibration batch +0.1585 mean /
        +0.0833 min.  Frozen: mean >= 0.07, per-seed min >= 0.04 (>= 2x
        margins).  The absolute floor prevents a pass via a degraded twin.
        """
        td_rec = recurrence_early(td_run[0])
        raw_rec = recurrence_early(raw_rewards)
        paired = td_rec - raw_rec

        assert float(paired.mean()) >= 0.07
        assert float(paired.min()) >= 0.04
        assert float(td_rec.mean()) >= 0.85  # measured 0.9219 (worst batch 0.9190)

    def test_lifetime_reward_is_not_sacrificed(self, td_run, raw_rewards) -> None:
        """Retention is not bought with lifetime average reward.

        Measured paired lifetime diff (td - raw): positive mean in all four
        batches (dev +0.0074, range +0.0050..+0.0181) but with per-seed
        excursions to -0.0297, so only a no-worse claim is frozen: mean
        >= -0.01, per-seed min >= -0.06 (>= 2x margins against the worst
        measured deficits).
        """
        paired = np.asarray(td_run[0]).mean(axis=1) - np.asarray(raw_rewards).mean(axis=1)
        assert float(paired.mean()) >= -0.01
        assert float(paired.min()) >= -0.06


# ---------------------------------------------------------------------------
# (3) Head-to-head: TD target vs reward target on the same seeds
# ---------------------------------------------------------------------------


class TestHeadToHead:
    def test_td_target_non_inferior_to_reward_target(self, td_run, reward_run) -> None:
        """The agent's own Bellman target nearly matches the reward oracle.

        Measured paired early-recurrence diff (td - reward): dev batch
        -0.0110 mean / -0.0483 min / +0.0167 max; across all four batches the
        mean was -0.0110..-0.0177 — the privileged immediate-reward target
        wins every batch, but by under 2% reward against a ~17% shared
        advantage over raw.  The finding frozen here is non-inferiority of
        the self-grounded target: mean >= -0.04, per-seed min >= -0.10
        (>= 2x margins against the worst measured deficits -0.0177/-0.0483).
        """
        paired = recurrence_early(td_run[0]) - recurrence_early(reward_run[0])
        assert float(paired.mean()) >= -0.04
        assert float(paired.min()) >= -0.10

    def test_reward_target_reference_reproduces(self, reward_run, raw_rewards) -> None:
        """The reward-target arm reproduces the reference file's result.

        Guards the shared code path: with the target switched back to the
        immediate reward, this file's adapter must recover the reference
        file's coupled-vs-raw advantage (its frozen margins: mean >= 0.06,
        min >= 0.03; measured here +0.1765 mean / +0.1050 min on dev, worst
        batch +0.1733 / +0.0950, final distinct products min 3).
        """
        paired = recurrence_early(reward_run[0]) - recurrence_early(raw_rewards)
        assert float(paired.mean()) >= 0.06
        assert float(paired.min()) >= 0.03
        assert int(final_distinct_counts(reward_run).min()) >= 2


# ---------------------------------------------------------------------------
# (4) Grounding ablation: an information-free target fails to find the products
# ---------------------------------------------------------------------------


class TestRandomTargetAblation:
    def test_random_target_fails_to_find_binding_products(self, td_run, random_run) -> None:
        """Utility comes from the TD signal, not from churn or init luck.

        With 8 slots over 15 pairs (4 relevant), chance parks ~2 relevant
        products in any bank, so the contrast is against that floor, not
        zero.  Measured final distinct relevant, random target: dev batch
        [3 2 2 2 3 3 2 2] (mean 2.375, median 2); across batches mean
        2.25-2.375 with per-seed values down to 1 — despite 6-10 promotions
        per seed, i.e. the machinery churns but cannot select.  TD target:
        per-seed min 3 and mean 3.25-3.625 in every batch.  Frozen: random
        mean <= 3.0 (td's worst mean is 3.25) and paired td-minus-random
        mean count advantage >= 0.4 (measured +0.875 to +1.25 across
        batches; the td arm's own floors live in the representation test).
        """
        td_counts = final_distinct_counts(td_run).astype(np.float64)
        random_counts = final_distinct_counts(random_run).astype(np.float64)

        assert float(random_counts.mean()) <= 3.0
        assert float((td_counts - random_counts).mean()) >= 0.4

    def test_random_target_control_retains_less(self, td_run, random_run) -> None:
        """The chance-level bank also costs control retention.

        Measured paired early-recurrence diff (td - random): dev batch
        +0.1129 mean / -0.0017 min; across batches +0.1129..+0.1460 mean
        (per-seed values touch zero when a random bank luckily keeps 3
        products, so only the mean is frozen).  Frozen: mean >= 0.05
        (>= 2.2x margin against the worst measured mean).
        """
        paired = recurrence_early(td_run[0]) - recurrence_early(random_run[0])
        assert float(paired.mean()) >= 0.05
