"""Unit tests for the recurring binary learning-partner world."""

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.learning_partner import (
    CONSTANT_ONE_CHANNEL,
    CONSTANT_ZERO_CHANNEL,
    DIRECT_CHANNEL,
    SHUFFLED_CHANNEL,
    LearningPartnerWorld,
    LearningPartnerWorldConfig,
    LearningPartnerWorldKeys,
    learning_partner_world_keys,
)

pytestmark = pytest.mark.unit


def test_world_config_rejects_nonpositive_and_boolean_phase_lengths() -> None:
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="phase_length"):
            LearningPartnerWorldConfig(phase_length=value)  # type: ignore[arg-type]


def test_context_recurs_without_reset_and_target_is_oracle_only() -> None:
    world = LearningPartnerWorld(LearningPartnerWorldConfig(phase_length=2))
    state = world.init(learning_partner_world_keys(jr.key(3)))
    contexts: list[int] = []
    targets: list[int] = []
    for _ in range(8):
        observation = world.observe(state)
        assert "target" not in {field.name for field in dataclasses.fields(observation)}
        transition, state = world.step(state, jnp.int32(0), jnp.int32(0))
        contexts.append(int(transition.oracle.context))
        targets.append(int(transition.oracle.target))
        assert int(transition.oracle.target) == (
            int(transition.observation.helper_cue) ^ int(transition.oracle.context)
        )
        assert not bool(transition.terminated)
        assert float(transition.discount) == 1.0
    assert contexts == [0, 0, 1, 1, 0, 0, 1, 1]
    assert int(state.step_count) == 8
    assert set(targets) <= {0, 1}


def test_channel_interventions_have_exact_causal_semantics() -> None:
    world = LearningPartnerWorld()
    state = world.init(learning_partner_world_keys(jr.key(11)))
    assert int(world.deliver(state, jnp.int32(0), DIRECT_CHANNEL)) == 0
    assert int(world.deliver(state, jnp.int32(1), DIRECT_CHANNEL)) == 1
    for message in (0, 1):
        assert int(world.deliver(state, jnp.int32(message), CONSTANT_ZERO_CHANNEL)) == 0
        assert int(world.deliver(state, jnp.int32(message), CONSTANT_ONE_CHANNEL)) == 1
    # A shuffled draw is a function of the named channel key, not the message.
    shuffled_zero = world.deliver(state, jnp.int32(0), SHUFFLED_CHANNEL)
    shuffled_one = world.deliver(state, jnp.int32(1), SHUFFLED_CHANNEL)
    np.testing.assert_array_equal(shuffled_zero, shuffled_one)


def test_named_channel_stream_is_deterministic_and_independent_of_cue_key() -> None:
    world = LearningPartnerWorld(LearningPartnerWorldConfig(phase_length=3))
    common_channel_key = jr.key(919)
    state_a = world.init(LearningPartnerWorldKeys(cue=jr.key(1), channel=common_channel_key))
    state_b = world.init(LearningPartnerWorldKeys(cue=jr.key(2), channel=common_channel_key))
    deliveries_a: list[int] = []
    deliveries_b: list[int] = []
    cues_a: list[int] = []
    cues_b: list[int] = []
    for _ in range(32):
        expected_draw_key, _ = jr.split(state_a.channel_key)
        expected = jr.randint(expected_draw_key, (), 0, 2, dtype=jnp.int32)
        delivered_a = world.deliver(state_a, jnp.int32(0), SHUFFLED_CHANNEL)
        delivered_b = world.deliver(state_b, jnp.int32(1), SHUFFLED_CHANNEL)
        np.testing.assert_array_equal(delivered_a, expected)
        np.testing.assert_array_equal(delivered_b, expected)
        deliveries_a.append(int(delivered_a))
        deliveries_b.append(int(delivered_b))
        cues_a.append(int(state_a.cue))
        cues_b.append(int(state_b.cue))
        _, state_a = world.step(state_a, jnp.int32(0), jnp.int32(0), SHUFFLED_CHANNEL)
        _, state_b = world.step(state_b, jnp.int32(1), jnp.int32(0), SHUFFLED_CHANNEL)
    assert deliveries_a == deliveries_b
    assert cues_a != cues_b
    assert set(deliveries_a) == {0, 1}


def test_direct_and_shuffled_channels_do_not_perturb_future_cues() -> None:
    world = LearningPartnerWorld()
    keys = learning_partner_world_keys(jr.key(27))
    direct_state = world.init(keys)
    shuffled_state = world.init(keys)
    for _ in range(24):
        np.testing.assert_array_equal(direct_state.cue, shuffled_state.cue)
        _, direct_state = world.step(
            direct_state,
            jnp.int32(0),
            jnp.int32(0),
            DIRECT_CHANNEL,
        )
        _, shuffled_state = world.step(
            shuffled_state,
            jnp.int32(1),
            jnp.int32(0),
            SHUFFLED_CHANNEL,
        )


def test_world_is_jittable_scannable_fixed_shape_and_finite() -> None:
    world = LearningPartnerWorld(LearningPartnerWorldConfig(phase_length=4))
    state = world.init(learning_partner_world_keys(jr.key(42)))

    @jax.jit
    def run(initial_state):
        def body(carry, action):
            transition, next_state = world.step(carry, action, action)
            output = jnp.stack(
                [
                    transition.reward,
                    transition.discount,
                    transition.oracle.target.astype(jnp.float32),
                ]
            )
            return next_state, output

        return jax.lax.scan(body, initial_state, jnp.arange(16, dtype=jnp.int32) % 2)

    final_state, outputs = run(state)
    assert outputs.shape == (16, 3)
    assert bool(jnp.all(jnp.isfinite(outputs)))
    assert final_state.cue.shape == ()
    assert final_state.step_count.shape == ()
    assert jr.key_data(final_state.cue_key).shape == (2,)
    assert jr.key_data(final_state.channel_key).shape == (2,)
