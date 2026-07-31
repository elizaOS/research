"""Integration contracts for the hidden-regime world and slot dyad."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.slot_signaling_agent import (
    SLOT_VALUE_SHAPE,
    SlotSignalingAgent,
    SlotSignalingConfig,
    slot_signaling_keys,
    slot_signaling_resource_budget,
)
from alberta_framework.streams.hidden_regime_signaling import (
    DIRECT_TERNARY_CHANNEL,
    HiddenRegimeSignalingWorld,
    HiddenRegimeWorldConfig,
    hidden_regime_world_keys,
)

pytestmark = pytest.mark.integration


def _world() -> HiddenRegimeSignalingWorld:
    return HiddenRegimeSignalingWorld(
        HiddenRegimeWorldConfig(
            segment_lengths=(5, 4, 6),
            segment_regimes=(0, 1, 0),
            regime_permutations=((0, 1, 2), (1, 2, 0)),
            repeat_schedule=True,
        )
    )


def test_joint_continuing_scan_uses_only_ordinary_role_inputs() -> None:
    world = _world()
    learner = SlotSignalingAgent(
        SlotSignalingConfig(
            learning_rate=0.2,
            epsilon=0.1,
            relevance_rate=0.1,
            lease_length=3,
            confirmation_steps=2,
        )
    )
    world_state = world.init(hidden_regime_world_keys(jr.key(101)))
    learner_state = learner.init(slot_signaling_keys(jr.key(202)))
    resource_before = slot_signaling_resource_budget(learner_state)

    @jax.jit
    def run(initial_world, initial_learner):
        def body(carry, _):
            old_world, old_learner = carry
            observation = world.observe(old_world)
            helper = learner.select_helper(
                old_learner.helper,
                observation.helper_cue,
            )
            delivered = world.deliver(
                old_world,
                helper.action,
                DIRECT_TERNARY_CHANNEL,
            )
            beneficiary = learner.select_beneficiary(
                old_learner.beneficiary,
                delivered,
            )
            transition, next_world = world.step_with_delivery(
                old_world,
                helper.action,
                delivered,
                beneficiary.action,
            )
            update = learner.update(
                old_learner,
                helper,
                beneficiary,
                transition.reward,
            )
            primitive = jnp.stack(
                (
                    transition.observation.helper_cue,
                    transition.helper_message,
                    transition.delivered_message,
                    transition.beneficiary_action,
                    transition.reward,
                    transition.oracle.segment_index,
                    transition.oracle.regime_id,
                    transition.oracle.target,
                    helper.slot,
                    beneficiary.slot,
                    update.helper.value_write,
                    update.beneficiary.value_write,
                    update.lifecycle_synchronized,
                )
            ).astype(jnp.float32)
            return (next_world, update.state), primitive

        return jax.lax.scan(
            body,
            (initial_world, initial_learner),
            xs=None,
            length=90,
        )

    (final_world, final_learner), trace = run(world_state, learner_state)
    assert trace.shape == (90, 13)
    assert bool(jnp.all(jnp.isfinite(trace)))
    np.testing.assert_array_equal(trace[:, 1], trace[:, 2])
    np.testing.assert_array_equal(
        trace[:, 4],
        (trace[:, 3] == trace[:, 7]).astype(jnp.float32),
    )
    np.testing.assert_array_equal(trace[:, 8], trace[:, 9])
    np.testing.assert_array_equal(trace[:, 12], np.ones((90,), dtype=np.float32))
    assert set(np.asarray(trace[:, 6], dtype=np.int32)) == {0, 1}
    assert int(final_world.step_count) == 90
    assert final_learner.helper.values.shape == SLOT_VALUE_SHAPE
    assert final_learner.beneficiary.values.shape == SLOT_VALUE_SHAPE
    assert slot_signaling_resource_budget(final_learner) == resource_before


def test_frozen_role_still_acts_and_advances_but_never_changes_value_bits() -> None:
    world = _world()
    learner = SlotSignalingAgent(
        SlotSignalingConfig(
            learning_rate=1.0,
            epsilon=0.2,
            lease_length=2,
            confirmation_steps=1,
        )
    )
    world_state = world.init(hidden_regime_world_keys(jr.key(303)))
    learner_state = learner.init(slot_signaling_keys(jr.key(404)))
    helper_values_before = np.asarray(learner_state.helper.values).view(np.uint32).copy()
    helper_status_before = np.asarray(learner_state.helper.status).copy()
    helper_generation_before = np.asarray(learner_state.helper.generation).copy()
    helper_key_before = np.asarray(jr.key_data(learner_state.helper.key)).copy()
    saw_nonzero_candidate = False
    for _ in range(20):
        observation = world.observe(world_state)
        helper = learner.select_helper(learner_state.helper, observation.helper_cue)
        beneficiary = learner.select_beneficiary(
            learner_state.beneficiary,
            helper.action,
        )
        transition, world_state = world.step(
            world_state,
            helper.action,
            beneficiary.action,
        )
        update = learner.update(
            learner_state,
            helper,
            beneficiary,
            transition.reward,
            helper_write=False,
            beneficiary_write=True,
        )
        saw_nonzero_candidate |= float(update.helper.candidate_value) != 0.0
        learner_state = update.state
    np.testing.assert_array_equal(
        np.asarray(learner_state.helper.values).view(np.uint32),
        helper_values_before,
    )
    np.testing.assert_array_equal(learner_state.helper.status, helper_status_before)
    np.testing.assert_array_equal(learner_state.helper.generation, helper_generation_before)
    np.testing.assert_array_equal(learner_state.beneficiary.status, helper_status_before)
    np.testing.assert_array_equal(learner_state.beneficiary.generation, helper_generation_before)
    assert not np.array_equal(jr.key_data(learner_state.helper.key), helper_key_before)
    assert saw_nonzero_candidate
    assert np.count_nonzero(np.asarray(learner_state.beneficiary.values)) > 0
