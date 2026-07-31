# mypy: disable-error-code="call-arg"
"""Two learning agents forming, forgetting, and recalling joint conventions.

The minimal multi-agent continual-learning simulation: two independent
:class:`~alberta_framework.core.average_reward.DifferentialSARSAAgent`s play a
common-payoff continuing *convention game*.  Each step both agents pick one of
``n_actions``; the team is rewarded iff the joint action satisfies the active
rule ``(a0 - a1) mod n_actions == offset``.  The required offset alternates on
a fixed schedule (rule A, rule B, rule A, ...), so every rule *recurs*.

Why this is the right minimal testbed:

- **The partner is the non-stationarity.**  Nothing about the environment
  drifts except the other learning agent (and the scheduled rule flip): each
  agent faces a moving target created by co-adaptation — the Alberta Plan's
  multi-agent distinguishing feature in its smallest form.
- **Coordination must be discovered.**  A rule admits ``n_actions`` valid
  conventions; two independent epsilon-greedy learners must break symmetry
  jointly, which takes real search — so *re*-coordination speed on a rule's
  recurrence is a measurable memory signal (savings).
- **Memory is a representation property.**  With ``feature_mode="context"``
  each agent observes only which rule is active (a one-hot context); the two
  rules then occupy disjoint Q-weight blocks and an inactive rule's
  convention is untouched while the other rule is in force — recurrence is
  met with immediate re-coordination.  With ``feature_mode="plain"`` (a bare
  constant observation) both rules compete for the same weights and every
  flip forces relearning: perfect plasticity, zero memory.

One caveat this testbed exposed (and which motivates
``DifferentialSARSAConfig.use_bias=False`` here): the agent's per-action
*bias* is an always-on shared parameter, so with the bias enabled the rule-B
convention seeps into the biases and overrides intact rule-A weights at
re-entry.  Any always-on shared parameter is a forgetting channel; the
experiments below disable the bias so the context blocks are truly exclusive.

Everything is pure JAX; :func:`run_matrix_game` is a single ``jax.lax.scan``.

This stream is a mechanism testbed rather than a complete Alberta Plan
evaluation.  In context mode the active rule identity is supplied directly,
so results isolate representational retention but do not establish autonomous
state construction, feature discovery, or causal intelligence amplification.
"""

from dataclasses import dataclass
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAState,
)

FeatureMode = Literal["plain", "context"]


@dataclass(frozen=True)
class ConventionGameConfig:
    """Static configuration for the recurring convention game.

    Attributes:
        n_actions: Actions per agent; also the number of valid conventions
            per rule (larger = harder joint symmetry breaking).
        phase_length: Steps per rule phase.
        offsets: Required ``(a0 - a1) mod n_actions`` per rule; the schedule
            cycles through these (each rule recurs every ``len(offsets)``
            phases).
        feature_mode: ``"context"`` observes the active rule as a one-hot
            (memory possible); ``"plain"`` observes a constant (memory
            impossible in principle).
    """

    n_actions: int = 12
    phase_length: int = 2000
    offsets: tuple[int, ...] = (0, 3)
    feature_mode: FeatureMode = "context"

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        if self.phase_length < 1:
            raise ValueError("phase_length must be positive")
        if len(self.offsets) < 1:
            raise ValueError("offsets must be non-empty")
        if any(not 0 <= o < self.n_actions for o in self.offsets):
            raise ValueError("every offset must lie in [0, n_actions)")
        if self.feature_mode not in ("plain", "context"):
            raise ValueError("feature_mode must be plain or context")

    @property
    def n_rules(self) -> int:
        """Number of distinct rules in the schedule."""
        return len(self.offsets)

    @property
    def observation_dim(self) -> int:
        """Per-agent observation dimension for the configured feature mode."""
        return 1 if self.feature_mode == "plain" else self.n_rules


@chex.dataclass(frozen=True)
class ConventionGameState:
    """Dynamic state of the convention game.

    Attributes:
        key: Game RNG key (kept for stochastic variants).
        step_count: Global step counter (drives the rule schedule).
        last_actions: The previous joint action, shape ``(2,)``.
    """

    key: PRNGKeyArray
    step_count: Int[Array, ""]
    last_actions: Int[Array, " 2"]


class RecurringConventionGame:
    """The recurring convention game (see module docstring)."""

    def __init__(self, config: ConventionGameConfig | None = None):
        """Initialize the game."""
        self._config = config or ConventionGameConfig()
        self._offsets = jnp.asarray(self._config.offsets, dtype=jnp.int32)

    @property
    def config(self) -> ConventionGameConfig:
        """The static game configuration."""
        return self._config

    @property
    def observation_dim(self) -> int:
        """Per-agent observation dimension."""
        return self._config.observation_dim

    def init(self, key: Array) -> ConventionGameState:
        """Initialize with an arbitrary previous joint action."""
        return ConventionGameState(
            key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
            last_actions=jnp.zeros((2,), dtype=jnp.int32),
        )

    def rule_of(self, step: Array) -> Array:
        """Index into ``offsets`` of the rule active at *step*."""
        return ((step // self._config.phase_length) % self._config.n_rules).astype(jnp.int32)

    def phase_index_of(self, step: Array) -> Array:
        """Ordinal of the phase at *step* (0, 1, 2, ... across the run)."""
        return (step // self._config.phase_length).astype(jnp.int32)

    def observe(self, state: ConventionGameState) -> Array:
        """Build the (shared) observation from the current game state.

        Both agents receive the same observation: the active rule as a
        one-hot context (``"context"``), or a bare constant (``"plain"``).
        """
        if self._config.feature_mode == "plain":
            return jnp.ones((1,), dtype=jnp.float32)
        rule = self.rule_of(state.step_count)
        return jax.nn.one_hot(rule, self._config.n_rules, dtype=jnp.float32)

    def step(
        self, state: ConventionGameState, action_0: Array, action_1: Array
    ) -> tuple[Array, ConventionGameState]:
        """Apply the joint action; return the common reward and next state."""
        rule = self.rule_of(state.step_count)
        offset = self._offsets[rule]
        hit = ((action_0 - action_1) % self._config.n_actions) == offset
        reward = hit.astype(jnp.float32)
        new_state = ConventionGameState(
            key=state.key,
            step_count=state.step_count + 1,
            last_actions=jnp.stack([action_0, action_1]).astype(jnp.int32),
        )
        return reward, new_state


@chex.dataclass(frozen=True)
class ConventionGameRunResult:
    """Result of :func:`run_matrix_game`.

    Attributes:
        rewards: Common reward per step, shape ``(num_steps,)``.
        actions: Joint actions per step, shape ``(num_steps, 2)``.
        agent_states: Final per-agent learner states.
    """

    rewards: Float[Array, " num_steps"]
    actions: Int[Array, "num_steps 2"]
    agent_states: tuple[DifferentialSARSAState, DifferentialSARSAState]


def run_matrix_game(
    agent: DifferentialSARSAAgent,
    game: RecurringConventionGame,
    num_steps: int,
    key: Array,
) -> ConventionGameRunResult:
    """Run two independent copies of *agent* through the recurring game.

    Both agents share the same architecture/config but hold independent
    states, seeds, and experience — self-play without weight sharing.

    Args:
        agent: The differential SARSA agent template (for continual-memory
            experiments configure it with ``use_bias=False``; see module
            docstring).
        game: The recurring convention game.
        num_steps: Number of joint steps.
        key: RNG key (split between the game and both agents).

    Returns:
        :class:`ConventionGameRunResult` with per-step rewards and actions.
    """
    k_game, k_a0, k_a1 = jr.split(key, 3)
    g_state = game.init(k_game)
    obs = game.observe(g_state)
    st_0, _ = agent.start(agent.init(game.observation_dim, k_a0), obs)
    st_1, _ = agent.start(agent.init(game.observation_dim, k_a1), obs)

    def step_fn(
        carry: tuple[
            ConventionGameState,
            DifferentialSARSAState,
            DifferentialSARSAState,
        ],
        _: Array,
    ) -> tuple[
        tuple[
            ConventionGameState,
            DifferentialSARSAState,
            DifferentialSARSAState,
        ],
        tuple[Array, Array],
    ]:
        g_st, s0, s1 = carry
        a0, a1 = s0.last_action, s1.last_action
        reward, g_st = game.step(g_st, a0, a1)
        next_obs = game.observe(g_st)
        s0 = agent.update(s0, reward, next_obs).state
        s1 = agent.update(s1, reward, next_obs).state
        return (g_st, s0, s1), (reward, jnp.stack([a0, a1]))

    (g_state, st_0, st_1), (rewards, actions) = jax.lax.scan(
        step_fn, (g_state, st_0, st_1), jnp.arange(num_steps)
    )
    return ConventionGameRunResult(
        rewards=rewards,
        actions=actions.astype(jnp.int32),
        agent_states=(st_0, st_1),
    )


def phase_reward_profile(
    rewards: Array, phase_length: int, window: int = 200
) -> tuple[Array, Array]:
    """Per-phase-occurrence early-window and tail-window mean rewards.

    Args:
        rewards: Per-step rewards, shape ``(num_steps,)`` or
            ``(n_seeds, num_steps)``.
        phase_length: The game's phase length.
        window: Width of the early/tail windows.

    Returns:
        ``(early, tail)`` arrays of shape ``(..., n_phases)``: mean reward in
        the first and last *window* steps of each phase occurrence.  High
        early-window reward on a *recurring* phase is the control-side
        savings signal — the agents re-coordinate immediately instead of
        searching again.
    """
    num_steps = rewards.shape[-1]
    n_phases = num_steps // phase_length
    trimmed = rewards[..., : n_phases * phase_length]
    phases = trimmed.reshape(*trimmed.shape[:-1], n_phases, phase_length)
    early = jnp.mean(phases[..., :window], axis=-1)
    tail = jnp.mean(phases[..., -window:], axis=-1)
    return early, tail


def time_to_coordination(phase_rewards: Array, threshold: float = 0.7, window: int = 20) -> Array:
    """Steps until the trailing-*window* mean reward first reaches *threshold*.

    Args:
        phase_rewards: Rewards within one phase, shape ``(phase_len,)`` or
            ``(n_seeds, phase_len)``.
        threshold: Coordination criterion on the trailing-window mean.
        window: Trailing window width (also the metric's floor).

    Returns:
        Steps-to-coordination (capped at the phase length when never
        reached), scalar or ``(n_seeds,)``.
    """
    r = jnp.atleast_2d(phase_rewards)
    kernel = jnp.ones((window,), dtype=jnp.float32) / window
    trailing = jax.vmap(lambda row: jnp.convolve(row, kernel, mode="valid"))(r)
    reached = trailing >= threshold
    phase_len = phase_rewards.shape[-1]
    first = jnp.where(
        jnp.any(reached, axis=-1),
        jnp.argmax(reached, axis=-1) + window,
        phase_len,
    )
    return first if phase_rewards.ndim > 1 else jnp.squeeze(first)
