# mypy: disable-error-code="call-arg"
"""Minimal recurring Lewis game for genuine learning-partner experiments.

The world separates the two roles asymmetrically.  A helper privately observes
a fair binary cue ``x`` and emits one binary message.  A beneficiary observes
only that delivered message and a public recurring context, then emits one
binary action.  The common reward is one exactly when the beneficiary action
matches ``x XOR context``.

The target is evaluator-only oracle data.  It is never part of either role's
ordinary observation.  Cue and channel randomness use named independent keys,
so a shuffled-channel control cannot accidentally consume or perturb the cue
stream.  There are no episode resets: contexts alternate on a fixed schedule
and the step counter increases forever.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import chex
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray

DIRECT_CHANNEL: Literal["direct"] = "direct"
CONSTANT_ZERO_CHANNEL: Literal["constant_0"] = "constant_0"
CONSTANT_ONE_CHANNEL: Literal["constant_1"] = "constant_1"
SHUFFLED_CHANNEL: Literal["shuffled"] = "shuffled"

type LearningPartnerChannel = Literal[
    "direct",
    "constant_0",
    "constant_1",
    "shuffled",
]

LEARNING_PARTNER_CHANNELS: tuple[LearningPartnerChannel, ...] = (
    DIRECT_CHANNEL,
    CONSTANT_ZERO_CHANNEL,
    CONSTANT_ONE_CHANNEL,
    SHUFFLED_CHANNEL,
)

# Stable tags are part of the development-world contract.  Adding a random
# consumer must use a new tag rather than splitting an existing substream.
_CUE_RNG_TAG = 0x435545  # ASCII "CUE"
_CHANNEL_RNG_TAG = 0x43484E  # ASCII "CHN"


@dataclasses.dataclass(frozen=True)
class LearningPartnerWorldConfig:
    """Static schedule for the continuing binary signaling game."""

    phase_length: int = 512

    def __post_init__(self) -> None:
        if type(self.phase_length) is not int or self.phase_length < 1:
            raise ValueError("phase_length must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {"phase_length": self.phase_length}


@chex.dataclass(frozen=True)
class LearningPartnerWorldKeys:
    """Named, independent random streams owned by the world."""

    cue: PRNGKeyArray
    channel: PRNGKeyArray


def learning_partner_world_keys(root_key: Array) -> LearningPartnerWorldKeys:
    """Derive stable named world keys without positional split coupling."""

    return LearningPartnerWorldKeys(
        cue=jr.fold_in(root_key, _CUE_RNG_TAG),
        channel=jr.fold_in(root_key, _CHANNEL_RNG_TAG),
    )


@chex.dataclass(frozen=True)
class LearningPartnerObservation:
    """Ordinary observations available before a message is sent.

    ``public_context`` is available to both roles.  ``helper_cue`` is private
    to the helper; an evaluator must not pass it to the beneficiary.
    """

    public_context: Int[Array, ""]
    helper_cue: Int[Array, ""]


@chex.dataclass(frozen=True)
class LearningPartnerWorldState:
    """Fixed-shape state of the continuing world."""

    cue_key: PRNGKeyArray
    channel_key: PRNGKeyArray
    cue: Int[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class LearningPartnerOracle:
    """Evaluator-only facts that must never enter either learner policy."""

    step_count: Int[Array, ""]
    phase_index: Int[Array, ""]
    context: Int[Array, ""]
    target: Int[Array, ""]


@chex.dataclass(frozen=True)
class LearningPartnerTransition:
    """One continuing signaling transition."""

    observation: LearningPartnerObservation
    helper_message: Int[Array, ""]
    delivered_message: Int[Array, ""]
    beneficiary_action: Int[Array, ""]
    reward: Float[Array, ""]
    next_observation: LearningPartnerObservation
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]
    oracle: LearningPartnerOracle


class LearningPartnerWorld:
    """Pure-JAX recurring binary Lewis signaling world."""

    def __init__(self, config: LearningPartnerWorldConfig | None = None) -> None:
        self._config = config or LearningPartnerWorldConfig()

    @property
    def config(self) -> LearningPartnerWorldConfig:
        """Static world configuration."""

        return self._config

    def context_of(self, step_count: Array) -> Array:
        """Return the recurring public context at a global step."""

        return ((step_count // self._config.phase_length) % 2).astype(jnp.int32)

    def phase_index_of(self, step_count: Array) -> Array:
        """Return the zero-based phase index at a global step."""

        return (step_count // self._config.phase_length).astype(jnp.int32)

    def init(self, keys: LearningPartnerWorldKeys) -> LearningPartnerWorldState:
        """Initialize the continuing state and sample its first fair cue."""

        cue_draw_key, next_cue_key = jr.split(keys.cue)
        cue = jr.randint(cue_draw_key, (), 0, 2, dtype=jnp.int32)
        return LearningPartnerWorldState(
            cue_key=next_cue_key,
            channel_key=keys.channel,
            cue=cue,
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def observe(self, state: LearningPartnerWorldState) -> LearningPartnerObservation:
        """Build the ordinary pre-message observation."""

        return LearningPartnerObservation(
            public_context=self.context_of(state.step_count),
            helper_cue=state.cue,
        )

    def deliver(
        self,
        state: LearningPartnerWorldState,
        helper_message: Array,
        channel: LearningPartnerChannel,
    ) -> Array:
        """Apply a causal channel without advancing the world.

        The shuffled arm is a fair bit drawn only from ``state.channel_key``;
        it is therefore independent of both the cue key and helper message.
        """

        if channel == DIRECT_CHANNEL:
            return jnp.asarray(helper_message, dtype=jnp.int32)
        if channel == CONSTANT_ZERO_CHANNEL:
            return jnp.asarray(0, dtype=jnp.int32)
        if channel == CONSTANT_ONE_CHANNEL:
            return jnp.asarray(1, dtype=jnp.int32)
        if channel == SHUFFLED_CHANNEL:
            draw_key, _ = jr.split(state.channel_key)
            return jr.randint(draw_key, (), 0, 2, dtype=jnp.int32)
        raise ValueError(f"unknown learning-partner channel: {channel!r}")

    def step(
        self,
        state: LearningPartnerWorldState,
        helper_message: Array,
        beneficiary_action: Array,
        channel: LearningPartnerChannel = DIRECT_CHANNEL,
    ) -> tuple[LearningPartnerTransition, LearningPartnerWorldState]:
        """Apply one message/action pair and advance without termination."""

        delivered_message = self.deliver(state, helper_message, channel)
        return self.step_with_delivery(
            state,
            helper_message,
            delivered_message,
            beneficiary_action,
        )

    def step_with_delivery(
        self,
        state: LearningPartnerWorldState,
        helper_message: Array,
        delivered_message: Array,
        beneficiary_action: Array,
    ) -> tuple[LearningPartnerTransition, LearningPartnerWorldState]:
        """Advance using a channel output already resolved at decision time.

        This keeps two-stage helper/channel/beneficiary causality explicit and
        lets a compiled evaluator select among channel interventions with JAX
        arrays.  Callers remain responsible for obtaining ``delivered_message``
        from :meth:`deliver` or an explicitly declared evaluator intervention.
        """

        observation = self.observe(state)
        delivered = jnp.asarray(delivered_message, dtype=jnp.int32)
        target = jnp.bitwise_xor(observation.helper_cue, observation.public_context)
        action = jnp.asarray(beneficiary_action, dtype=jnp.int32)
        reward = (action == target).astype(jnp.float32)

        cue_draw_key, next_cue_key = jr.split(state.cue_key)
        next_cue = jr.randint(cue_draw_key, (), 0, 2, dtype=jnp.int32)
        _, next_channel_key = jr.split(state.channel_key)
        next_state = LearningPartnerWorldState(
            cue_key=next_cue_key,
            channel_key=next_channel_key,
            cue=next_cue,
            step_count=state.step_count + jnp.asarray(1, dtype=jnp.int32),
        )
        transition = LearningPartnerTransition(
            observation=observation,
            helper_message=jnp.asarray(helper_message, dtype=jnp.int32),
            delivered_message=delivered,
            beneficiary_action=action,
            reward=reward,
            next_observation=self.observe(next_state),
            terminated=jnp.asarray(False),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            oracle=LearningPartnerOracle(
                step_count=state.step_count,
                phase_index=self.phase_index_of(state.step_count),
                context=observation.public_context,
                target=target,
            ),
        )
        return transition, next_state
