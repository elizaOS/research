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

import dataclasses
import functools
from dataclasses import dataclass
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray, UInt

from alberta_framework.core.average_reward import (
    DIFFERENTIAL_SARSA_LIFETIME_COUNTER_DELTA_NBYTES,
    DifferentialSARSAAgent,
    DifferentialSARSAState,
)

FeatureMode = Literal["plain", "context"]

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

# ``step_count`` remains a four-byte compatibility/diagnostic field.  These
# eight bytes are the exact schedule authority added to the dynamic state.
CONVENTION_GAME_EXACT_CLOCK_NBYTES = 8
CONVENTION_GAME_EXACT_CLOCK_DELTA_NBYTES = 8

CONVENTION_GAME_RUNNER_STATE_SCHEMA = "alberta.convention-game-runner-state.v1"
CONVENTION_GAME_RUNNER_EXACT_IDENTITY_NBYTES = (
    CONVENTION_GAME_EXACT_CLOCK_NBYTES
    + 2 * DIFFERENTIAL_SARSA_LIFETIME_COUNTER_DELTA_NBYTES
)


def _checked_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose one exact uint64-word increment without all-ones wrap."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("convention-game step_words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("convention-game step_words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(array == maximum)
    low = array[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((array[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity_available, proposed, array), capacity_available


def _words_to_int32_telemetry(words: Array) -> Int[Array, ""]:
    """Project an exact identity to saturating non-negative telemetry."""

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("convention-game step_words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("convention-game step_words must have dtype uint32")
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    below_saturation = (array[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        array[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(below_saturation, array[1].astype(jnp.int32), maximum)


def _clock_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    """Authenticate exact schedule words against compatibility telemetry."""

    projected = _words_to_int32_telemetry(words)
    count = jnp.asarray(telemetry)
    if count.shape != ():
        raise ValueError("convention-game step_count must be scalar")
    if count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("convention-game step_count must have dtype int32")
    return (count >= 0) & (count == projected)


def _divmod_words_by_positive_int32(
    words: Array,
    divisor: int,
) -> tuple[UInt[Array, " 2"], UInt[Array, ""]]:
    """Divide a two-word unsigned identity by a positive signed-int32 value.

    JAX commonly runs with x64 disabled, so casting the clock to ``uint64`` is
    not a portable exact implementation.  This fixed 64-round long division
    uses only uint32 operations.  Because ``divisor <= INT32_MAX``, doubling a
    remainder cannot overflow uint32.
    """

    array = jnp.asarray(words)
    if array.shape != (2,):
        raise ValueError("convention-game schedule words must have shape (2,)")
    if array.dtype != jnp.dtype(jnp.uint32):
        raise TypeError("convention-game schedule words must have dtype uint32")
    if type(divisor) is not int or not 1 <= divisor <= _INT32_MAX:
        raise ValueError("schedule divisor must be a positive signed-int32 integer")

    divisor_u = jnp.asarray(divisor, dtype=jnp.uint32)
    zero = jnp.asarray(0, dtype=jnp.uint32)
    one = jnp.asarray(1, dtype=jnp.uint32)

    def divide_bit(
        index: int,
        carry: tuple[Array, Array, Array],
    ) -> tuple[Array, Array, Array]:
        quotient_high, quotient_low, remainder = carry
        bit_index = jnp.asarray(63 - index, dtype=jnp.int32)
        from_high = bit_index >= 32
        shift = jnp.where(from_high, bit_index - 32, bit_index)
        source = jnp.where(from_high, array[0], array[1])
        bit = (source >> shift.astype(jnp.uint32)) & one

        doubled = remainder + remainder + bit
        quotient_bit = doubled >= divisor_u
        next_remainder = jnp.where(quotient_bit, doubled - divisor_u, doubled)

        next_high = (quotient_high << one) | (quotient_low >> jnp.uint32(31))
        next_low = (quotient_low << one) | quotient_bit.astype(jnp.uint32)
        return next_high, next_low, next_remainder

    high, low, remainder = jax.lax.fori_loop(
        0,
        64,
        divide_bit,
        (zero, zero, zero),
    )
    return jnp.stack((high, low)).astype(jnp.uint32), remainder.astype(jnp.uint32)


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
        if type(self.n_actions) is not int or not 2 <= self.n_actions <= _INT32_MAX:
            raise ValueError("n_actions must be at least 2")
        if (
            type(self.phase_length) is not int
            or not 1 <= self.phase_length <= _INT32_MAX
        ):
            raise ValueError("phase_length must be a positive signed-int32 integer")
        if not isinstance(self.offsets, tuple) or not 1 <= len(self.offsets) <= _INT32_MAX:
            raise ValueError("offsets must be a non-empty tuple")
        if any(type(offset) is not int for offset in self.offsets):
            raise ValueError("every offset must be a non-boolean integer")
        if any(not 0 <= offset < self.n_actions for offset in self.offsets):
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
        step_count: Saturating signed-int32 compatibility telemetry.
        step_words: Exact big-endian uint32 schedule identity.  This is the
            only authority used to select phases and rules.
        last_actions: The previous joint action, shape ``(2,)``.
    """

    key: PRNGKeyArray
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    last_actions: Int[Array, " 2"]


@chex.dataclass(frozen=True)
class ConventionGameStepResult:
    """Fail-closed result of one convention-game transition attempt."""

    reward: Float[Array, ""]
    state: ConventionGameState
    rule_index: Int[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


def measure_convention_game_state_nbytes(state: ConventionGameState) -> int:
    """Measure persistent JAX-array bytes in one convention-game state."""

    total = 0
    for leaf in jax.tree.leaves(state):
        if isinstance(leaf, Array):
            total += int(leaf.size) * int(leaf.dtype.itemsize)
    return total


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
        self._require_key_contract(key)
        return ConventionGameState(
            key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            last_actions=jnp.zeros((2,), dtype=jnp.int32),
        )

    @staticmethod
    def _require_key_contract(key: Array) -> None:
        """Reject malformed scalar PRNG keys before any state is created."""

        try:
            key_words = jnp.asarray(jr.key_data(key))
        except (TypeError, ValueError) as error:
            raise TypeError("convention-game key must be a scalar PRNG key") from error
        if key_words.shape != (2,) or key_words.dtype != jnp.dtype(jnp.uint32):
            raise TypeError("convention-game key must be a scalar PRNG key")

    def _require_state_contract(self, state: ConventionGameState) -> None:
        """Validate fixed state shapes and dtypes (safe during JAX tracing)."""

        self._require_key_contract(state.key)
        _checked_words_increment(state.step_words)
        count = jnp.asarray(state.step_count)
        if count.shape != ():
            raise ValueError("convention-game step_count must be scalar")
        if count.dtype != jnp.dtype(jnp.int32):
            raise TypeError("convention-game step_count must have dtype int32")
        actions = jnp.asarray(state.last_actions)
        if actions.shape != (2,):
            raise ValueError("convention-game last_actions must have shape (2,)")
        if actions.dtype != jnp.dtype(jnp.int32):
            raise TypeError("convention-game last_actions must have dtype int32")

    def _state_values_valid(self, state: ConventionGameState) -> Bool[Array, ""]:
        """Authenticate dynamic state values without reading telemetry as time."""

        return _clock_valid(state.step_words, state.step_count) & jnp.all(
            (state.last_actions >= 0) & (state.last_actions < self._config.n_actions)
        )

    @staticmethod
    def _require_action_contract(action: Array, *, name: str) -> Int[Array, ""]:
        raw = jnp.asarray(action)
        if raw.shape != ():
            raise ValueError(f"convention-game {name} must be scalar")
        if raw.dtype != jnp.dtype(jnp.int32):
            raise TypeError(f"convention-game {name} must have dtype int32")
        return raw

    @staticmethod
    def _legacy_scalar_to_words(step: Array) -> tuple[Array, Array]:
        """Convert only a non-negative unsaturated scalar compatibility input."""

        raw = jnp.asarray(step)
        if raw.shape != ():
            raise ValueError("legacy convention-game step must be scalar")
        if raw.dtype not in (jnp.dtype(jnp.int32), jnp.dtype(jnp.uint32)):
            raise TypeError("legacy convention-game step must have dtype int32 or uint32")
        valid = jnp.asarray(True, dtype=jnp.bool_)
        if raw.dtype == jnp.dtype(jnp.int32):
            # INT32_MAX is the compatibility field's saturation sentinel: it
            # could denote this exact step or any later one, so the scalar
            # surface must not invent a history for it.
            valid = (raw >= 0) & (raw < _INT32_MAX)
            low = jnp.maximum(raw, jnp.asarray(0, dtype=jnp.int32)).astype(jnp.uint32)
        else:
            valid = raw < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
            low = raw
        return jnp.stack((jnp.asarray(0, dtype=jnp.uint32), low)), valid

    def phase_words_of(self, step_words: Array) -> UInt[Array, " 2"]:
        """Return the exact phase ordinal for an exact two-word step identity."""

        quotient, _ = _divmod_words_by_positive_int32(
            step_words,
            self._config.phase_length,
        )
        return quotient

    def _rule_of_words(self, step_words: Array) -> Int[Array, ""]:
        """Select the active rule using only the exact schedule identity."""

        # The common case needs one 64-bit modular reduction.  The fallback
        # avoids constraining otherwise valid configs whose full cycle is
        # larger than signed int32.
        cycle_length = self._config.phase_length * self._config.n_rules
        if cycle_length <= _INT32_MAX:
            _, within_cycle = _divmod_words_by_positive_int32(step_words, cycle_length)
            return (within_cycle // self._config.phase_length).astype(jnp.int32)
        phase_words = self.phase_words_of(step_words)
        _, rule = _divmod_words_by_positive_int32(phase_words, self._config.n_rules)
        return rule.astype(jnp.int32)

    def rule_of(self, step: Array) -> Array:
        """Index into ``offsets`` of the rule active at an exact step.

        A uint32 array of shape ``(2,)`` is the exact surface.  A scalar
        int32/uint32 remains accepted for backward-compatible short-horizon
        diagnostics; negative signed values fail closed to ``-1`` and a
        saturated scalar is never interpreted as later history.
        """

        raw = jnp.asarray(step)
        if raw.shape == (2,):
            if raw.dtype != jnp.dtype(jnp.uint32):
                raise TypeError("exact convention-game step must have dtype uint32")
            words = raw
            valid = jnp.asarray(True, dtype=jnp.bool_)
        else:
            words, valid = self._legacy_scalar_to_words(raw)
        rule = self._rule_of_words(words)
        return jnp.where(valid, rule, jnp.asarray(-1, dtype=jnp.int32))

    def phase_index_of(self, step: Array) -> Array:
        """Saturating telemetry for the exact phase ordinal at *step*."""

        raw = jnp.asarray(step)
        if raw.shape == (2,):
            if raw.dtype != jnp.dtype(jnp.uint32):
                raise TypeError("exact convention-game step must have dtype uint32")
            words = raw
            valid = jnp.asarray(True, dtype=jnp.bool_)
        else:
            words, valid = self._legacy_scalar_to_words(raw)
        telemetry = _words_to_int32_telemetry(self.phase_words_of(words))
        return jnp.where(valid, telemetry, jnp.asarray(-1, dtype=jnp.int32))

    def observe(self, state: ConventionGameState) -> Array:
        """Build the (shared) observation from the current game state.

        Both agents receive the same observation: the active rule as a
        one-hot context (``"context"``), or a bare constant (``"plain"``).
        """
        self._require_state_contract(state)
        state_valid = self._state_values_valid(state)
        if self._config.feature_mode == "plain":
            candidate = jnp.ones((1,), dtype=jnp.float32)
        else:
            rule = self._rule_of_words(state.step_words)
            candidate = jax.nn.one_hot(rule, self._config.n_rules, dtype=jnp.float32)
        return jnp.where(state_valid, candidate, jnp.zeros_like(candidate))

    def step_result(
        self,
        state: ConventionGameState,
        action_0: Array,
        action_1: Array,
    ) -> ConventionGameStepResult:
        """Attempt one transition, rolling back every leaf on any rejection."""

        self._require_state_contract(state)
        action_0_i = self._require_action_contract(action_0, name="action_0")
        action_1_i = self._require_action_contract(action_1, name="action_1")
        proposed_words, capacity_available = _checked_words_increment(state.step_words)
        counter_valid = _clock_valid(state.step_words, state.step_count)
        state_valid = self._state_values_valid(state)
        input_valid = (
            (action_0_i >= 0)
            & (action_0_i < self._config.n_actions)
            & (action_1_i >= 0)
            & (action_1_i < self._config.n_actions)
        )
        update_applied = state_valid & input_valid & capacity_available

        rule = self._rule_of_words(state.step_words)
        offset = self._offsets[rule]
        hit = ((action_0_i - action_1_i) % self._config.n_actions) == offset
        candidate_reward = hit.astype(jnp.float32)
        proposed_actions = jnp.stack((action_0_i, action_1_i)).astype(jnp.int32)
        next_state = ConventionGameState(
            key=state.key,
            step_count=jnp.where(
                update_applied,
                _words_to_int32_telemetry(proposed_words),
                state.step_count,
            ),
            step_words=jnp.where(update_applied, proposed_words, state.step_words),
            last_actions=jnp.where(update_applied, proposed_actions, state.last_actions),
        )
        return ConventionGameStepResult(
            reward=jnp.where(
                update_applied,
                candidate_reward,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            state=next_state,
            rule_index=jnp.where(
                update_applied,
                rule,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            pre_step_words=state.step_words,
            post_step_words=next_state.step_words,
            lifetime_counter_valid=counter_valid,
            lifetime_capacity_available=capacity_available,
            state_valid=state_valid,
            input_valid=input_valid,
            update_applied=update_applied,
        )

    def step(
        self, state: ConventionGameState, action_0: Array, action_1: Array
    ) -> tuple[Array, ConventionGameState]:
        """Apply the joint action; return the common reward and next state."""
        result = self.step_result(state, action_0, action_1)
        return result.reward, result.state


@chex.dataclass(frozen=True)
class ConventionGameRunnerState:
    """Complete resumable state for the game and both learning agents.

    The environment's ``step_words`` is the schedule authority.  Both learner
    identities must equal it before and after every committed joint update.
    The runner owns no fourth clock and therefore cannot drift independently
    of the three states it composes.
    """

    environment_state: ConventionGameState
    agent_0_state: DifferentialSARSAState
    agent_1_state: DifferentialSARSAState


@dataclass(frozen=True)
class ConventionGameRunnerResourceBudget:
    """Exact persistent-state accounting for one concrete runner state."""

    state_schema: str
    environment_state_nbytes: int
    agent_0_state_nbytes: int
    agent_1_state_nbytes: int
    environment_exact_identity_nbytes: int
    learner_exact_identity_nbytes: int
    exact_identity_nbytes: int
    state_nbytes: int


@chex.dataclass(frozen=True)
class ConventionGameRunnerStepResult:
    """Result of one staged environment/two-learner transaction attempt."""

    state: ConventionGameRunnerState
    reward: Float[Array, ""]
    actions: Int[Array, " 2"]
    rule_index: Int[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    environment_update_applied: Bool[Array, ""]
    learner_updates_applied: Bool[Array, " 2"]
    runner_state_valid: Bool[Array, ""]
    child_counters_aligned: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ConventionGameRunResult:
    """Resumable result of :func:`run_matrix_game`.

    Attributes:
        rewards: Common reward per step, shape ``(num_steps,)``.
        actions: Joint actions per step, shape ``(num_steps, 2)``.
        state: Complete final environment/two-agent runner state.
        updates_applied: Whether each attempted joint transaction committed.
        pre_step_words: Exact environment identity before every attempt.
        post_step_words: Exact committed environment identity after every
            attempt; equal to the pre-identity when the attempt is refused.
    """

    rewards: Float[Array, " num_steps"]
    actions: Int[Array, "num_steps 2"]
    state: ConventionGameRunnerState
    updates_applied: Bool[Array, " num_steps"]
    environment_updates_applied: Bool[Array, " num_steps"]
    learner_updates_applied: Bool[Array, "num_steps 2"]
    runner_states_valid: Bool[Array, " num_steps"]
    child_counters_aligned: Bool[Array, " num_steps"]
    candidate_states_finite: Bool[Array, " num_steps"]
    pre_step_words: UInt[Array, "num_steps 2"]
    post_step_words: UInt[Array, "num_steps 2"]

    @property
    def environment_state(self) -> ConventionGameState:
        """Final environment state (compatibility/readability view)."""

        return self.state.environment_state

    @property
    def agent_states(
        self,
    ) -> tuple[DifferentialSARSAState, DifferentialSARSAState]:
        """Final learner states in the historical tuple surface."""

        return self.state.agent_0_state, self.state.agent_1_state


def _persistent_state_nbytes(value: object) -> int:
    """Count persistent array bytes, excluding host-only timing metadata."""

    if isinstance(value, Array):
        return int(value.size) * int(value.dtype.itemsize)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(
            _persistent_state_nbytes(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name not in {"birth_timestamp", "uptime_s"}
        )
    if isinstance(value, (tuple, list)):
        return sum(_persistent_state_nbytes(item) for item in value)
    return 0


def measure_convention_game_runner_state_nbytes(
    state: ConventionGameRunnerState,
) -> int:
    """Measure persistent learning-state bytes in a full runner state."""

    return _persistent_state_nbytes(state)


def convention_game_runner_resource_budget(
    state: ConventionGameRunnerState,
) -> ConventionGameRunnerResourceBudget:
    """Return exact component and joint bytes for one runner state."""

    environment_nbytes = _persistent_state_nbytes(state.environment_state)
    agent_0_nbytes = _persistent_state_nbytes(state.agent_0_state)
    agent_1_nbytes = _persistent_state_nbytes(state.agent_1_state)
    return ConventionGameRunnerResourceBudget(
        state_schema=CONVENTION_GAME_RUNNER_STATE_SCHEMA,
        environment_state_nbytes=environment_nbytes,
        agent_0_state_nbytes=agent_0_nbytes,
        agent_1_state_nbytes=agent_1_nbytes,
        environment_exact_identity_nbytes=CONVENTION_GAME_EXACT_CLOCK_NBYTES,
        learner_exact_identity_nbytes=(
            2 * DIFFERENTIAL_SARSA_LIFETIME_COUNTER_DELTA_NBYTES
        ),
        exact_identity_nbytes=CONVENTION_GAME_RUNNER_EXACT_IDENTITY_NBYTES,
        state_nbytes=environment_nbytes + agent_0_nbytes + agent_1_nbytes,
    )


def _floating_tree_finite(tree: object) -> Bool[Array, ""]:
    """Require every floating/complex state leaf to be finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def init_matrix_game_runner(
    agent: DifferentialSARSAAgent,
    game: RecurringConventionGame,
    key: Array,
) -> ConventionGameRunnerState:
    """Initialize and prime both learners at exact environment identity zero."""

    if agent.config.n_actions != game.config.n_actions:
        raise ValueError("agent and convention game must have the same n_actions")
    game_key, agent_0_key, agent_1_key = jr.split(key, 3)
    environment_state = game.init(game_key)
    observation = game.observe(environment_state)
    agent_0_state, _ = agent.start(
        agent.init(game.observation_dim, agent_0_key),
        observation,
    )
    agent_1_state, _ = agent.start(
        agent.init(game.observation_dim, agent_1_key),
        observation,
    )
    return ConventionGameRunnerState(
        environment_state=environment_state,
        agent_0_state=agent_0_state,
        agent_1_state=agent_1_state,
    )


@functools.partial(jax.jit, static_argnums=(0, 1))
def step_matrix_game_runner(
    agent: DifferentialSARSAAgent,
    game: RecurringConventionGame,
    state: ConventionGameRunnerState,
) -> ConventionGameRunnerStepResult:
    """Stage and atomically commit one environment/two-learner transition."""

    environment = state.environment_state
    agent_0_state = state.agent_0_state
    agent_1_state = state.agent_1_state
    actions = jnp.stack(
        (agent_0_state.last_action, agent_1_state.last_action)
    ).astype(jnp.int32)
    child_counters_aligned = (
        jnp.all(environment.step_words == agent_0_state.step_words)
        & jnp.all(environment.step_words == agent_1_state.step_words)
    )
    state_finite = _floating_tree_finite(state)

    environment_result = game.step_result(
        environment,
        actions[0],
        actions[1],
    )
    next_observation = game.observe(environment_result.state)
    agent_0_result = agent.update(
        agent_0_state,
        environment_result.reward,
        next_observation,
    )
    agent_1_result = agent.update(
        agent_1_state,
        environment_result.reward,
        next_observation,
    )
    learner_updates_applied = jnp.stack(
        (agent_0_result.update_applied, agent_1_result.update_applied)
    ).astype(jnp.bool_)

    candidate_state = ConventionGameRunnerState(
        environment_state=environment_result.state,
        agent_0_state=agent_0_result.state,
        agent_1_state=agent_1_result.state,
    )
    candidate_counters_aligned = (
        jnp.all(
            environment_result.state.step_words
            == agent_0_result.state.step_words
        )
        & jnp.all(
            environment_result.state.step_words
            == agent_1_result.state.step_words
        )
    )
    runner_state_valid = (
        child_counters_aligned
        & state_finite
        & environment_result.state_valid
        & agent_0_result.state_valid
        & agent_1_result.state_valid
    )
    lifetime_capacity_available = (
        environment_result.lifetime_capacity_available
        & agent_0_result.lifetime_capacity_available
        & agent_1_result.lifetime_capacity_available
    )
    input_valid = (
        environment_result.input_valid
        & agent_0_result.input_valid
        & agent_1_result.input_valid
    )
    candidate_state_finite = (
        _floating_tree_finite(candidate_state)
        & agent_0_result.candidate_state_finite
        & agent_1_result.candidate_state_finite
    )
    update_applied = (
        runner_state_valid
        & lifetime_capacity_available
        & input_valid
        & environment_result.update_applied
        & jnp.all(learner_updates_applied)
        & candidate_counters_aligned
        & candidate_state_finite
    )
    committed_state = jax.lax.cond(
        update_applied,
        lambda _: candidate_state,
        lambda _: state,
        operand=None,
    )
    return ConventionGameRunnerStepResult(
        state=committed_state,
        reward=jnp.where(
            update_applied,
            environment_result.reward,
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        actions=actions,
        rule_index=jnp.where(
            update_applied,
            environment_result.rule_index,
            jnp.asarray(-1, dtype=jnp.int32),
        ),
        pre_step_words=environment.step_words,
        post_step_words=committed_state.environment_state.step_words,
        environment_update_applied=environment_result.update_applied,
        learner_updates_applied=learner_updates_applied,
        runner_state_valid=runner_state_valid,
        child_counters_aligned=child_counters_aligned,
        lifetime_capacity_available=lifetime_capacity_available,
        input_valid=input_valid,
        candidate_state_finite=candidate_state_finite,
        update_applied=update_applied,
    )


def run_matrix_game(
    agent: DifferentialSARSAAgent,
    game: RecurringConventionGame,
    num_steps: int,
    key: Array | None = None,
    *,
    initial_state: ConventionGameRunnerState | None = None,
) -> ConventionGameRunResult:
    """Run or resume two independent learners in the recurring game.

    Both agents share the same architecture/config but hold independent
    states, seeds, and experience.  Exactly one initialization authority is
    required: ``key`` starts a fresh run, while ``initial_state`` resumes a
    prior result without reinitializing or replaying history.

    Args:
        agent: The differential SARSA agent template (for continual-memory
            experiments configure it with ``use_bias=False``; see module
            docstring).
        game: The recurring convention game.
        num_steps: Number of joint steps.
        key: RNG key for a fresh run.
        initial_state: Complete state returned by an earlier run.

    Returns:
        :class:`ConventionGameRunResult` with per-step rewards and actions.
    """
    if type(num_steps) is not int or not 0 <= num_steps <= _INT32_MAX:
        raise ValueError("num_steps must be a non-negative signed-int32 integer")
    if (key is None) == (initial_state is None):
        raise ValueError("provide exactly one of key or initial_state")
    state = (
        init_matrix_game_runner(agent, game, key)
        if key is not None
        else initial_state
    )
    if state is None:  # Narrowing guard for static type checkers.
        raise ValueError("matrix-game runner initialization state is unavailable")

    def step_fn(
        carry: ConventionGameRunnerState,
        _: Array,
    ) -> tuple[
        ConventionGameRunnerState,
        tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ]:
        result = step_matrix_game_runner(agent, game, carry)
        return result.state, (
            result.reward,
            result.actions,
            result.update_applied,
            result.environment_update_applied,
            result.learner_updates_applied,
            result.runner_state_valid,
            result.child_counters_aligned,
            result.candidate_state_finite,
            result.pre_step_words,
            result.post_step_words,
        )

    final_state, outputs = jax.lax.scan(
        step_fn,
        state,
        jnp.arange(num_steps, dtype=jnp.int32),
    )
    (
        rewards,
        actions,
        updates_applied,
        environment_updates_applied,
        learner_updates_applied,
        runner_states_valid,
        child_counters_aligned,
        candidate_states_finite,
        pre_step_words,
        post_step_words,
    ) = outputs
    return ConventionGameRunResult(
        rewards=rewards,
        actions=actions.astype(jnp.int32),
        state=final_state,
        updates_applied=updates_applied,
        environment_updates_applied=environment_updates_applied,
        learner_updates_applied=learner_updates_applied,
        runner_states_valid=runner_states_valid,
        child_counters_aligned=child_counters_aligned,
        candidate_states_finite=candidate_states_finite,
        pre_step_words=pre_step_words,
        post_step_words=post_step_words,
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
