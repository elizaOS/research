"""Discovery-driven control: an online-constructed feature bank drives a control policy.

The gauntlet discovery suite (``tests/test_gauntlet_discovery.py``) showed that
:class:`FixedBudgetInteractionLearner` can select context-interaction products
``ctx * x`` from raw observations — but only for open-loop *prediction*.  The
control-side results (``test_prototype_nonsaturating.py``,
``streams/matrix_game.py``) used either raw observations or oracle-supplied
gated features.  This module closes that loop: the control agent's feature map
is constructed *online by the discovery learner itself* (no oracle gating), and
control retention on rule recurrence is measured against an interference twin.

Architecture (adapter classes below; no library changes)
--------------------------------------------------------
* **Environment** — :class:`ContextObservableSwitchingEnv`: a
  :class:`SwitchingTwoStateMDP` (payoff matrix inverts every ``PHASE_LEN``
  steps) wrapped with an observation of ``[state one-hot (2), rule-context
  one-hot (2), Rademacher distractors (2)]``.  The rule identity is observable
  but only *useful* through multiplicative ``state x context`` binding: a
  linear Q over the raw channels gets an additive per-context action offset
  that cannot express either phase's state-dependent optimal policy, so its
  state-block weights must invert at every switch (the interference channel).
  The four products ``s_i * c_k`` make the phases' Q-tables disjoint.
* **Discovery** — :class:`FixedBudgetInteractionLearner` (the frozen
  scale-robust configuration from ``test_gauntlet_discovery.py``, shrunk to
  this 6-dim observation: 8 active slots, full 15-pair candidate archive)
  trained online on **action-conditioned next-reward prediction**: 2 tasks,
  the taken action's task receives the observed reward, the other is
  NaN-masked.  Rewards here are exactly bilinear in ``state x context``, so
  the oracle products are points in the learner's closed pair space — but it
  must still promote them against 11 distractor/one-hot-degenerate pairs from
  behavior-policy data it partially controls.
* **Control** — :class:`DifferentialSARSAAgent` with ``use_bias=False`` (the
  ``matrix_game`` lesson: any always-on shared parameter is a forgetting
  channel) whose feature map is the discovery learner's current active bank
  ``obs[left] * obs[right]``.  Coupling is two-timescale: discovery updates
  every step; every ``REFRESH_EVERY`` steps the control feature map is
  refreshed from the current bank, and Q-weight columns are **carried over by
  pair identity** (surviving pairs keep their weights wherever they sit; new
  pairs start at 0; duplicate descriptors keep only their first copy).  The
  carry-over is the task memory and is unit-tested and ablated below.

Protocol: 8 paired seeds (``jr.split(jr.key(0), 8)``), 12 phases of 300 steps
(6 A/B rule alternations), constant epsilon 0.1, ``q_step_size=0.1`` (the
repo-proven flat-control configuration).  The retention metric is the mean
reward in the first 60 steps of each phase occurrence, averaged over phases
2..11 — every occurrence after the first exposure to each rule.  Arms: raw
twin (identical SARSA on raw observations), oracle (SARSA on the four true
products), coupled (discovered bank, carry), no-carry ablation, and the
fully-simultaneous variant (refresh every step).

Calibration (2026-07-30, CPU, this file's exact protocol; development batch
``jr.key(0)`` plus robustness batches ``jr.key(100)/jr.key(7)/jr.key(42)``,
worst value across the four batches in brackets):

    arm                recurrence-early mean/min      lifetime mean
    raw twin           0.7513 / 0.7217 [0.775 mean]   0.869-0.889
    oracle gated       0.9512 / 0.9383 [0.948 mean]   0.944-0.954
    coupled (carry)    0.9390 / 0.9267 [0.929 mean]   0.896-0.928
    no-carry ablation  0.8442 / 0.7933 [0.848 mean]   0.820-0.877
    simultaneous K=1   0.9340 / 0.9050 [0.932 mean]   —

    paired diffs (phases 2..11)   dev-batch mean/min   worst batch mean/min
    coupled - raw                 +0.1877 / +0.1567    +0.1540 / +0.0883
    oracle - coupled              +0.0123 / max 0.025  +0.0223 / max 0.0483
    coupled - no-carry            +0.0948 / +0.0733    +0.0812 / +0.0517
    K=1     - raw                 +0.1827 / +0.1483    +0.1567 / +0.0883
    lifetime coupled - raw        +0.0298 / +0.0069    +0.0187 / -0.0092

    distinct oracle products in the final control bank: min 3, median 3.5-4
    in every batch (of 4 possible); banks stabilize by the end of phase 3.
    Promotions per seed: 5-7.  Refresh interval is insensitive across
    K in {1, 25, 50}.

Every frozen threshold below keeps a >= 2x margin against the worst measured
batch value.  Honest scope limits: the rule context is *supplied* as an
observable channel (context inference is a separate open problem, see
``test_step2_context_inference.py``); the pair space is closed and tiny (15
pairs, degree two); the environment is deterministic with binary rewards; the
discovery target is the immediate reward, not the control agent's own TD
target — TD/GVF-target feature discovery proper remains open; and the
development seeds were inspected during calibration with no held-out
confidence interval.  What this file establishes is narrower and new in-tree:
a control policy over autonomously constructed features, with recurrence
retention attributable to weight carry-over on persistent discovered features.

Runtime: ~35-50s on CPU (five vmapped scan arms, module-scoped fixtures).
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

# The four context-binding products the oracle arm uses: s_i * c_k with
# state channels {0, 1} and context channels {2, 3} (canonical left < right).
ORACLE_LEFT = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
ORACLE_RIGHT = jnp.array([2, 2, 3, 3], dtype=jnp.int32)
ORACLE_PAIRS = {(0, 2), (1, 2), (0, 3), (1, 3)}


# ---------------------------------------------------------------------------
# Adapters
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
    copy — carrying both would double that product's contribution to Q.
    This function IS the coupled agent's task memory across bank refreshes.
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


class DiscoveryDrivenControlAgent:
    """Two-timescale coupling of interaction-feature discovery and control.

    Discovery (action-conditioned reward prediction over raw observations)
    updates every step.  The control agent acts on pair-product features of a
    bank *snapshot*; every ``refresh_every`` steps the snapshot is replaced by
    the discovery learner's current active bank with Q-weights carried over by
    pair identity (``carry_weights=False`` ablates the carry to zero-init).
    ``refresh_every=1`` is the fully-simultaneous coupling variant.
    """

    def __init__(
        self,
        env: ContextObservableSwitchingEnv,
        refresh_every: int = REFRESH_EVERY,
        carry_weights: bool = True,
    ):
        self._env = env
        self._refresh_every = refresh_every
        self._carry_weights = carry_weights
        self._control = DifferentialSARSAAgent(
            DifferentialSARSAConfig(
                n_actions=env.n_actions,
                q_step_size=0.1,
                average_reward_step_size=0.01,
                epsilon_start=0.1,
                epsilon_end=0.1,
                epsilon_decay_steps=0,
                use_bias=False,
            )
        )
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
        """One closed-loop step: act, discover on the reward, maybe refresh."""
        action = state.control_state.last_action
        next_obs, reward, next_env_state = self._env.step(state.env_state, action, key)

        # Discovery: the taken action's reward head sees the pre-action obs.
        targets = jnp.full((self._env.n_actions,), jnp.nan, dtype=jnp.float32)
        targets = targets.at[action].set(reward)
        disc_result = self._disc.update(state.disc_state, state.raw_obs, targets)
        disc_state = disc_result.state

        # Two-timescale refresh with pair-identity weight carry-over.
        do_refresh = (t % self._refresh_every == 0) & (t > 0)
        new_left = disc_state.feature_left
        new_right = disc_state.feature_right
        carried = carry_q_weights_by_pair_identity(
            state.bank_left,
            state.bank_right,
            state.control_state.q_weights,
            new_left,
            new_right,
        )
        if not self._carry_weights:
            carried = jnp.zeros_like(carried)
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
        control_result = self._control.update(control_state, reward, phi_next)

        new_state = CoupledState(
            env_state=next_env_state,
            raw_obs=next_obs,
            disc_state=disc_state,
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


def run_coupled_batch(
    keys: Array,
    refresh_every: int = REFRESH_EVERY,
    carry_weights: bool = True,
) -> tuple[Array, Array, Array, Array, Array]:
    """Vmapped coupled runs; returns (rewards, distinct trace, promotions, bank)."""
    env = ContextObservableSwitchingEnv()
    agent = DiscoveryDrivenControlAgent(
        env, refresh_every=refresh_every, carry_weights=carry_weights
    )

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


def run_fixed_map_batch(keys: Array, mode: str) -> Array:
    """Vmapped fixed-feature-map SARSA twins: ``mode`` in {"raw", "oracle"}."""
    env = ContextObservableSwitchingEnv()
    if mode == "raw":
        feat_dim = env.feature_dim

        def feature_fn(obs: Array) -> Array:
            return obs
    else:
        feat_dim = 4

        def feature_fn(obs: Array) -> Array:
            return pair_products(obs, ORACLE_LEFT, ORACLE_RIGHT)

    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=env.n_actions,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=0.1,
            epsilon_end=0.1,
            epsilon_decay_steps=0,
            use_bias=False,
        )
    )

    def one_seed(key: Array) -> Array:
        k_env, k_obs, k_agent, k_scan = jr.split(key, 4)
        env_state = env.init(k_env)
        obs = env.observe(env_state, k_obs)
        a_state = agent.init(feat_dim, k_agent)
        a_state, _ = agent.start(a_state, feature_fn(obs))

        def step(carry: tuple, key: Array) -> tuple[tuple, Array]:
            a_st, e_st = carry
            next_obs, reward, e_next = env.step(e_st, a_st.last_action, key)
            result = agent.update(a_st, reward, feature_fn(next_obs))
            return (result.state, e_next), reward

        _, rewards = jax.lax.scan(step, (a_state, env_state), jr.split(k_scan, NUM_STEPS))
        return rewards

    return jax.vmap(one_seed)(keys)


def recurrence_early(rewards: Array) -> np.ndarray:
    """Per-seed mean reward in the first EARLY_WINDOW steps of phases 2..11."""
    per_phase = np.asarray(rewards).reshape(N_SEEDS, NUM_PHASES, PHASE_LEN)
    early = per_phase[:, :, :EARLY_WINDOW].mean(axis=2)
    return early[:, FIRST_RECURRENCE_PHASE:].mean(axis=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keys() -> Array:
    return jr.split(jr.key(0), N_SEEDS)


@pytest.fixture(scope="module")
def raw_rewards(keys: Array) -> Array:
    rewards = run_fixed_map_batch(keys, "raw")
    assert bool(jnp.all(jnp.isfinite(rewards)))
    return rewards


@pytest.fixture(scope="module")
def oracle_rewards(keys: Array) -> Array:
    rewards = run_fixed_map_batch(keys, "oracle")
    assert bool(jnp.all(jnp.isfinite(rewards)))
    return rewards


@pytest.fixture(scope="module")
def coupled_run(keys: Array) -> tuple[Array, Array, Array, Array, Array]:
    run = run_coupled_batch(keys)
    assert bool(jnp.all(jnp.isfinite(run[0])))
    return run


@pytest.fixture(scope="module")
def nocarry_rewards(keys: Array) -> Array:
    rewards, _, _, _, _ = run_coupled_batch(keys, carry_weights=False)
    assert bool(jnp.all(jnp.isfinite(rewards)))
    return rewards


@pytest.fixture(scope="module")
def simultaneous_run(keys: Array) -> tuple[Array, Array, Array, Array, Array]:
    run = run_coupled_batch(keys, refresh_every=1)
    assert bool(jnp.all(jnp.isfinite(run[0])))
    return run


# ---------------------------------------------------------------------------
# Mechanism: the carry-over function is the memory, so it is unit-tested
# ---------------------------------------------------------------------------


class TestCarryOverMechanism:
    def test_carry_matches_by_identity_zeroes_new_and_deduplicates(self) -> None:
        """Pairs keep weights wherever they move; new and duplicate slots get 0."""
        old_left = jnp.array([0, 1, 4], dtype=jnp.int32)
        old_right = jnp.array([2, 3, 5], dtype=jnp.int32)
        old_q = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
        # New bank: (1,3) moved slots, (0,2) present twice, (2,4) is new.
        new_left = jnp.array([1, 0, 0, 2], dtype=jnp.int32)
        new_right = jnp.array([3, 2, 2, 4], dtype=jnp.int32)

        carried = carry_q_weights_by_pair_identity(old_left, old_right, old_q, new_left, new_right)

        expected = jnp.array([[2.0, 1.0, 0.0, 0.0], [5.0, 4.0, 0.0, 0.0]], dtype=jnp.float32)
        assert jnp.array_equal(carried, expected)

    def test_identical_banks_carry_everything_unchanged(self) -> None:
        left = jnp.array([0, 1], dtype=jnp.int32)
        right = jnp.array([2, 3], dtype=jnp.int32)
        q = jnp.array([[1.5, -2.5], [0.0, 4.0]], dtype=jnp.float32)
        assert jnp.array_equal(carry_q_weights_by_pair_identity(left, right, q, left, right), q)

    def test_distinct_relevant_count_ignores_duplicates_and_distractor_pairs(self) -> None:
        left = jnp.array([0, 0, 1, 0, 2, 4], dtype=jnp.int32)
        right = jnp.array([2, 2, 3, 4, 4, 5], dtype=jnp.int32)
        # Relevant: (0,2) once (duplicate dropped), (1,3).  (0,4)/(2,4)/(4,5) are not.
        assert int(distinct_relevant_in_bank(left, right)) == 2


# ---------------------------------------------------------------------------
# (a) The discovery module finds context-interaction features during control
# ---------------------------------------------------------------------------


class TestDiscoveredRepresentation:
    def test_context_binding_products_found_during_control(self, coupled_run) -> None:
        """The control bank holds distinct oracle products, early and at the end.

        Measured on the development batch: distinct relevant products in the
        control bank at the end of phase 3 were [3 4 4 4 4 4 3 4] and final
        [3 3 3 4 4 4 3 4]; across all four calibration batches the final
        minimum was 3 of 4 and the median 3.5-4.  Frozen floors: at least 2
        per seed both mid-run and finally, median at least 3 finally.
        """
        _, distinct_trace, promoted, bank_left, bank_right = coupled_run

        distinct = np.asarray(distinct_trace)
        end_phase_3 = 4 * PHASE_LEN - 1
        assert int(distinct[:, end_phase_3].min()) >= 2
        assert int(distinct[:, -1].min()) >= 2
        assert float(np.median(distinct[:, -1])) >= 3.0

        # Cross-check the trace against the returned final descriptors.
        left_np, right_np = np.asarray(bank_left), np.asarray(bank_right)
        final_counts = [
            len({(int(a), int(b)) for a, b in zip(left_np[s], right_np[s])} & ORACLE_PAIRS)
            for s in range(N_SEEDS)
        ]
        assert final_counts == [int(c) for c in distinct[:, -1]]

        # The bank was built by online promotions, not init luck: every seed
        # promoted at least twice (measured 5-7 promotions per seed).
        assert int(np.asarray(promoted).sum(axis=1).min()) >= 2


# ---------------------------------------------------------------------------
# (b) Discovered features beat the raw-observation interference twin
# ---------------------------------------------------------------------------


class TestRecurrenceRetention:
    def test_discovered_features_beat_raw_twin_on_recurrence_early_window(
        self, coupled_run, raw_rewards
    ) -> None:
        """Paired early-recurrence advantage of discovered features over raw.

        Measured paired diff (phases 2..11, 60-step early windows): dev batch
        +0.1877 mean / +0.1567 min; worst calibration batch +0.1540 mean /
        +0.0883 min.  Frozen: mean >= 0.06, per-seed min >= 0.03 (>= 2.5x
        margins).  The absolute floor prevents a pass via a degraded twin.
        """
        coupled_rec = recurrence_early(coupled_run[0])
        raw_rec = recurrence_early(raw_rewards)
        paired = coupled_rec - raw_rec

        assert float(paired.mean()) >= 0.06
        assert float(paired.min()) >= 0.03
        assert float(coupled_rec.mean()) >= 0.85  # measured 0.9390 (worst batch 0.9287)

    def test_raw_twin_is_competent_within_phases(self, raw_rewards) -> None:
        """The twin's deficit is interference, not incompetence.

        The raw twin re-learns each rule within its phase — measured lifetime
        mean reward 0.867-0.889 against the 1.0 optimum and 0.5 uniform-random
        baseline — while its recurrence early-window stays interference-bound
        (measured 0.7513, i.e. it must unlearn the previous rule every time).
        """
        lifetime = np.asarray(raw_rewards).mean(axis=1)
        assert float(lifetime.min()) >= 0.80
        assert float(recurrence_early(raw_rewards).mean()) <= 0.85

    def test_lifetime_reward_is_not_sacrificed_for_retention(
        self, coupled_run, raw_rewards
    ) -> None:
        """Discovery-driven control also wins on lifetime average reward.

        Measured paired lifetime diff: dev batch +0.0298 mean / +0.0069 min;
        worst batch +0.0187 mean / -0.0092 min.  Frozen: mean >= 0.005 with a
        per-seed no-worse bound of 0.03.
        """
        paired = np.asarray(coupled_run[0]).mean(axis=1) - np.asarray(raw_rewards).mean(axis=1)
        assert float(paired.mean()) >= 0.005
        assert float(paired.min()) >= -0.03


# ---------------------------------------------------------------------------
# (c) Gap to the oracle-gated agent
# ---------------------------------------------------------------------------


class TestOracleGap:
    def test_gap_to_oracle_gated_control_is_small_and_bounded(
        self, coupled_run, oracle_rewards, raw_rewards
    ) -> None:
        """The discovered bank recovers most of the oracle's retention.

        Measured oracle-minus-coupled early-recurrence gap: dev batch +0.0123
        mean / 0.0250 per-seed max; worst batch +0.0223 mean / 0.0483 max —
        against a 0.19-0.20 oracle-minus-raw gap, i.e. the discovered bank
        recovers ~90% of the oracle's advantage over raw.  Frozen bounds:
        mean gap <= 0.06, per-seed gap <= 0.10, and the coupled agent must
        recover at least half of the oracle-vs-raw advantage.
        """
        oracle_rec = recurrence_early(oracle_rewards)
        coupled_rec = recurrence_early(coupled_run[0])
        raw_rec = recurrence_early(raw_rewards)

        gap = oracle_rec - coupled_rec
        assert float(oracle_rec.mean()) >= 0.90  # oracle sanity (measured 0.9512)
        assert float(gap.mean()) <= 0.06
        assert float(gap.max()) <= 0.10
        recovered = (coupled_rec - raw_rec).mean() / (oracle_rec - raw_rec).mean()
        assert float(recovered) >= 0.5  # measured 0.92-0.94


# ---------------------------------------------------------------------------
# Coupling variants: carry-over ablation and fully-simultaneous refresh
# ---------------------------------------------------------------------------


class TestCouplingVariants:
    def test_weight_carry_over_is_the_retention_mechanism(
        self, coupled_run, nocarry_rewards
    ) -> None:
        """Zeroing Q at every refresh removes most of the retention benefit.

        The no-carry ablation keeps the identical discovered bank, so its
        deficit isolates the carried weights as the memory (gated features
        alone still beat raw — zero-init is neutral where raw must unlearn —
        but carried weights add the instant policy recall).  Measured paired
        carry-minus-no-carry: dev batch +0.0948 mean / +0.0733 min; worst
        batch +0.0812 mean / +0.0517 min.  Frozen: mean >= 0.03, min >= 0.015.
        """
        paired = recurrence_early(coupled_run[0]) - recurrence_early(nocarry_rewards)
        assert float(paired.mean()) >= 0.03
        assert float(paired.min()) >= 0.015

    def test_fully_simultaneous_coupling_also_retains(
        self, simultaneous_run, raw_rewards
    ) -> None:
        """Variant 2 (refresh every step) matches the two-timescale result.

        Measured (refresh_every=1): early-recurrence 0.9340 mean / 0.9050 min;
        paired vs raw +0.1827 mean / +0.1483 min on the dev batch (worst batch
        +0.1567 / +0.0883); final distinct products min 3.  Same frozen
        margins as the two-timescale variant.
        """
        rewards, distinct_trace, _, _, _ = simultaneous_run
        paired = recurrence_early(rewards) - recurrence_early(raw_rewards)
        assert float(paired.mean()) >= 0.06
        assert float(paired.min()) >= 0.03
        assert int(np.asarray(distinct_trace)[:, -1].min()) >= 2
