"""A tiny recurring two-agent world for continual-control experiments.

The world is deliberately small enough to support fast, exhaustive seeded
experiments while retaining the properties needed by a continual-control
testbed:

* operation is continuing -- transitions never terminate or reset;
* two agents act jointly in a bounded one-dimensional world;
* damped velocity dynamics remain stable under an infinite action stream;
* the visible context recurs ``meet -> avoid -> meet -> ...``;
* a scripted partner changes behaviour with that visible context, and can be
  replaced by any pure JAX-compatible policy hook;
* independent nuisance channels are replaced on every transition; and
* evaluator-only oracle data is returned separately from agent observations.

Transition convention
---------------------
At step ``t``, ``transition.observation`` and the reward objective use the
context visible in ``state``.  The action is then applied, ``step_count`` is
incremented, and ``transition.next_observation`` shows the next context.  Thus a
transition at a context boundary has ``oracle.context_switched == True`` while
its reward still matches the context that was visible when the action was
chosen.

The state and transition records are immutable chex dataclasses.  ``init``,
``observe``, ``step``, and ``step_with_partner`` are pure functions suitable
for ``jax.jit`` and ``jax.lax.scan``.
"""

from __future__ import annotations

from collections.abc import Callable

import chex
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray

MEET_CONTEXT = 0
AVOID_CONTEXT = 1
N_AGENTS = 2

# The first six channels have stable meanings.  Nuisance channels, if any,
# follow them.
OWN_POSITION_INDEX = 0
RELATIVE_POSITION_INDEX = 1
OWN_VELOCITY_INDEX = 2
OTHER_VELOCITY_INDEX = 3
MEET_CONTEXT_INDEX = 4
AVOID_CONTEXT_INDEX = 5
BASE_OBSERVATION_DIM = 6

type PartnerPolicy = Callable[[Array, Array], Array]


@chex.dataclass(frozen=True)
class RecurringTwoAgentState:
    """Immutable dynamic state of :class:`RecurringTwoAgentWorld`.

    ``nuisance`` is stored in the state so repeated calls to :meth:`observe`
    are referentially transparent.  It affects neither physics nor reward.
    """

    key: PRNGKeyArray
    positions: Float[Array, " 2"]
    velocities: Float[Array, " 2"]
    nuisance: Float[Array, "2 nuisance_dim"]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class RecurringTwoAgentOracle:
    """Evaluator-only diagnostics for one transition.

    These fields are intentionally separate from ``observation`` and
    ``next_observation``.  In particular, agents are not given the absolute
    step, segment/cycle indices, imminent-boundary flag, counterfactual reward,
    or unclipped requested action.
    """

    step_count: Int[Array, ""]
    segment_index: Int[Array, ""]
    cycle_index: Int[Array, ""]
    context_id: Int[Array, ""]
    next_context_id: Int[Array, ""]
    context_switched: Bool[Array, ""]
    positions_before: Float[Array, " 2"]
    positions_after: Float[Array, " 2"]
    velocities_before: Float[Array, " 2"]
    velocities_after: Float[Array, " 2"]
    distance_before: Float[Array, ""]
    distance_after: Float[Array, ""]
    meet_reward: Float[Array, ""]
    avoid_reward: Float[Array, ""]
    requested_actions: Float[Array, " 2"]
    applied_actions: Float[Array, " 2"]
    hit_boundary: Bool[Array, " 2"]


@chex.dataclass(frozen=True)
class RecurringTwoAgentTransition:
    """One continuing joint-action transition."""

    observation: Float[Array, "2 observation_dim"]
    action: Float[Array, " 2"]
    reward: Float[Array, " 2"]
    next_observation: Float[Array, "2 observation_dim"]
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]
    oracle: RecurringTwoAgentOracle


def scripted_meet_avoid_partner_policy(
    partner_observation: Array,
    key: Array,
) -> Array:
    """Move the partner toward agent 0 in ``meet`` and away in ``avoid``.

    The hook consumes only the partner's ordinary observation and a PRNG key;
    it has no access to evaluator-only oracle data.  The key is accepted to
    match stochastic custom-policy hooks but is unused by this deterministic
    script.
    """

    del key
    relative_position = partner_observation[RELATIVE_POSITION_INDEX]
    objective_direction = (
        partner_observation[MEET_CONTEXT_INDEX]
        - partner_observation[AVOID_CONTEXT_INDEX]
    )
    return jnp.sign(relative_position) * objective_direction


class RecurringTwoAgentWorld:
    """Bounded recurring-context continual-control environment.

    Each agent observes, in order:

    ``[own_position, relative_position, own_velocity, other_velocity,
    meet_visible, avoid_visible, nuisance...]``.

    Positions are divided by ``world_limit``, relative position by
    ``2 * world_limit``, and velocities by ``max_speed``.  The context
    channels are a one-hot visible cue.  No segment, cycle, boundary, or
    counterfactual-reward oracle is present in the agent input.

    Context A (id ``MEET_CONTEXT``) rewards proximity:

    ``meet_reward = 1 - distance / (2 * world_limit)``.

    Context B (id ``AVOID_CONTEXT``) rewards separation:

    ``avoid_reward = distance / (2 * world_limit)``.

    Both agents receive the same cooperative reward in ``[0, 1]``.

    Args:
        context_length: Number of transitions per A or B segment.
        nuisance_dim: Ephemeral nuisance features per agent.
        nuisance_scale: Standard deviation of nuisance features.
        world_limit: Symmetric position bound; positions lie in
            ``[-world_limit, world_limit]``.
        damping: Velocity retention in ``[0, 1)``.
        acceleration: Action-to-velocity gain.
        time_delta: Position integration interval.
        max_speed: Absolute velocity bound.
        initial_positions: Fixed two-agent initial positions.
        partner_policy: Optional pure hook ``policy(observation, key) -> scalar``.
            The default is :func:`scripted_meet_avoid_partner_policy`.
    """

    def __init__(
        self,
        *,
        context_length: int = 64,
        nuisance_dim: int = 4,
        nuisance_scale: float = 1.0,
        world_limit: float = 1.0,
        damping: float = 0.75,
        acceleration: float = 0.15,
        time_delta: float = 1.0,
        max_speed: float = 0.25,
        initial_positions: tuple[float, float] = (-0.5, 0.5),
        partner_policy: PartnerPolicy | None = None,
    ) -> None:
        if context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {context_length}"
            )
        if nuisance_dim < 0:
            raise ValueError(f"nuisance_dim must be non-negative, got {nuisance_dim}")
        if nuisance_scale < 0.0:
            raise ValueError(
                f"nuisance_scale must be non-negative, got {nuisance_scale}"
            )
        if world_limit <= 0.0:
            raise ValueError(f"world_limit must be positive, got {world_limit}")
        if not 0.0 <= damping < 1.0:
            raise ValueError(f"damping must lie in [0, 1), got {damping}")
        if acceleration <= 0.0:
            raise ValueError(f"acceleration must be positive, got {acceleration}")
        if time_delta <= 0.0:
            raise ValueError(f"time_delta must be positive, got {time_delta}")
        if max_speed <= 0.0:
            raise ValueError(f"max_speed must be positive, got {max_speed}")
        if len(initial_positions) != N_AGENTS:
            raise ValueError(
                f"initial_positions must contain {N_AGENTS} values, "
                f"got {len(initial_positions)}"
            )
        if any(abs(position) > world_limit for position in initial_positions):
            raise ValueError(
                "initial_positions must lie within [-world_limit, world_limit]"
            )

        self._context_length = int(context_length)
        self._nuisance_dim = int(nuisance_dim)
        self._nuisance_scale = float(nuisance_scale)
        self._world_limit = float(world_limit)
        self._damping = float(damping)
        self._acceleration = float(acceleration)
        self._time_delta = float(time_delta)
        self._max_speed = float(max_speed)
        self._initial_positions = jnp.asarray(initial_positions, dtype=jnp.float32)
        self._partner_policy = (
            scripted_meet_avoid_partner_policy
            if partner_policy is None
            else partner_policy
        )

    @property
    def n_agents(self) -> int:
        """Number of jointly acting agents (always two)."""

        return N_AGENTS

    @property
    def action_dim(self) -> int:
        """Scalar action dimensions per agent."""

        return 1

    @property
    def observation_dim(self) -> int:
        """Number of observation channels per agent."""

        return BASE_OBSERVATION_DIM + self._nuisance_dim

    @property
    def feature_dim(self) -> int:
        """Alias for the per-agent observation dimension."""

        return self.observation_dim

    @property
    def nuisance_dim(self) -> int:
        """Number of ephemeral channels per agent."""

        return self._nuisance_dim

    @property
    def context_length(self) -> int:
        """Number of transitions in each recurring context segment."""

        return self._context_length

    @property
    def world_limit(self) -> float:
        """Absolute position bound."""

        return self._world_limit

    @property
    def max_speed(self) -> float:
        """Absolute velocity bound."""

        return self._max_speed

    def context_id(self, state: RecurringTwoAgentState) -> Array:
        """Return the visible recurring context id for ``state``."""

        segment_index = jnp.floor_divide(state.step_count, self._context_length)
        return jnp.mod(segment_index, 2).astype(jnp.int32)

    def init(self, key: Array) -> RecurringTwoAgentState:
        """Create a deterministic initial state from ``key``."""

        next_key, nuisance_key = jr.split(key)
        nuisance = self._sample_nuisance(nuisance_key)
        return RecurringTwoAgentState(  # type: ignore[call-arg]
            key=next_key,
            positions=self._initial_positions,
            velocities=jnp.zeros((N_AGENTS,), dtype=jnp.float32),
            nuisance=nuisance,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def observe(self, state: RecurringTwoAgentState) -> Array:
        """Construct the ordinary, non-oracle observation for both agents."""

        other = jnp.array([1, 0], dtype=jnp.int32)
        own_positions = state.positions / self._world_limit
        relative_positions = (
            state.positions[other] - state.positions
        ) / (2.0 * self._world_limit)
        own_velocities = state.velocities / self._max_speed
        other_velocities = state.velocities[other] / self._max_speed

        context = self.context_id(state)
        context_features = jnp.stack(
            (
                (context == MEET_CONTEXT).astype(jnp.float32),
                (context == AVOID_CONTEXT).astype(jnp.float32),
            )
        )
        shared_context = jnp.broadcast_to(context_features, (N_AGENTS, 2))
        physical = jnp.stack(
            (
                own_positions,
                relative_positions,
                own_velocities,
                other_velocities,
            ),
            axis=1,
        )
        return jnp.concatenate((physical, shared_context, state.nuisance), axis=1)

    def step(
        self,
        state: RecurringTwoAgentState,
        joint_action: Array,
    ) -> tuple[RecurringTwoAgentTransition, RecurringTwoAgentState]:
        """Apply an externally supplied joint continuous action.

        ``joint_action`` has shape ``(2,)`` and is clipped componentwise to
        ``[-1, 1]``.  The input state is never mutated.
        """

        next_key, nuisance_key = jr.split(state.key)
        return self._advance(state, joint_action, next_key, nuisance_key)

    def step_with_partner(
        self,
        state: RecurringTwoAgentState,
        learner_action: Array,
    ) -> tuple[RecurringTwoAgentTransition, RecurringTwoAgentState]:
        """Apply agent 0's action and obtain agent 1's action from the hook.

        The policy hook sees only agent 1's ordinary observation and an
        independent PRNG key.  A scalar or one-element learner action is
        accepted for convenience.
        """

        next_key, nuisance_key, policy_key = jr.split(state.key, 3)
        observation = self.observe(state)
        partner_action = jnp.asarray(
            self._partner_policy(observation[1], policy_key),
            dtype=jnp.float32,
        )
        learner_action_array = jnp.asarray(learner_action, dtype=jnp.float32)
        if partner_action.size != 1:
            raise ValueError(
                "partner policy must return a scalar or one-element action"
            )
        if learner_action_array.size != 1:
            raise ValueError("learner_action must be scalar or one-element")
        joint_action = jnp.stack(
            (
                jnp.reshape(learner_action_array, ()),
                jnp.reshape(partner_action, ()),
            )
        )
        return self._advance(state, joint_action, next_key, nuisance_key)

    def _sample_nuisance(self, key: Array) -> Array:
        return self._nuisance_scale * jr.normal(
            key,
            (N_AGENTS, self._nuisance_dim),
            dtype=jnp.float32,
        )

    def _advance(
        self,
        state: RecurringTwoAgentState,
        joint_action: Array,
        next_key: Array,
        nuisance_key: Array,
    ) -> tuple[RecurringTwoAgentTransition, RecurringTwoAgentState]:
        requested_actions = jnp.asarray(joint_action, dtype=jnp.float32)
        if requested_actions.shape != (N_AGENTS,):
            raise ValueError(
                f"joint_action must have shape ({N_AGENTS},), "
                f"got {requested_actions.shape}"
            )
        applied_actions = jnp.clip(requested_actions, -1.0, 1.0)

        accelerated = (
            self._damping * state.velocities
            + self._acceleration * applied_actions
        )
        candidate_velocities = jnp.clip(
            accelerated,
            -self._max_speed,
            self._max_speed,
        )
        candidate_positions = (
            state.positions + self._time_delta * candidate_velocities
        )
        positions = jnp.clip(
            candidate_positions,
            -self._world_limit,
            self._world_limit,
        )
        hit_boundary = candidate_positions != positions
        velocities = jnp.where(hit_boundary, 0.0, candidate_velocities)

        distance_before = jnp.abs(state.positions[0] - state.positions[1])
        distance_after = jnp.abs(positions[0] - positions[1])
        normalized_distance = jnp.clip(
            distance_after / (2.0 * self._world_limit),
            0.0,
            1.0,
        )
        meet_reward = 1.0 - normalized_distance
        avoid_reward = normalized_distance

        context = self.context_id(state)
        reward_scalar = jnp.where(
            context == MEET_CONTEXT,
            meet_reward,
            avoid_reward,
        ).astype(jnp.float32)
        reward = jnp.full((N_AGENTS,), reward_scalar, dtype=jnp.float32)

        new_state = RecurringTwoAgentState(  # type: ignore[call-arg]
            key=next_key,
            positions=positions,
            velocities=velocities,
            nuisance=self._sample_nuisance(nuisance_key),
            step_count=state.step_count + jnp.array(1, dtype=jnp.int32),
        )
        next_context = self.context_id(new_state)
        segment_index = jnp.floor_divide(
            state.step_count,
            self._context_length,
        ).astype(jnp.int32)
        oracle = RecurringTwoAgentOracle(  # type: ignore[call-arg]
            step_count=state.step_count,
            segment_index=segment_index,
            cycle_index=jnp.floor_divide(segment_index, 2).astype(jnp.int32),
            context_id=context,
            next_context_id=next_context,
            context_switched=context != next_context,
            positions_before=state.positions,
            positions_after=positions,
            velocities_before=state.velocities,
            velocities_after=velocities,
            distance_before=distance_before,
            distance_after=distance_after,
            meet_reward=meet_reward,
            avoid_reward=avoid_reward,
            requested_actions=requested_actions,
            applied_actions=applied_actions,
            hit_boundary=hit_boundary,
        )
        transition = RecurringTwoAgentTransition(  # type: ignore[call-arg]
            observation=self.observe(state),
            action=applied_actions,
            reward=reward,
            next_observation=self.observe(new_state),
            terminated=jnp.array(False),
            discount=jnp.array(1.0, dtype=jnp.float32),
            oracle=oracle,
        )
        return transition, new_state
