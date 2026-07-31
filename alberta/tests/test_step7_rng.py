"""Tests for RNG threading through Step 7 planning when the gate rejects output.

Regression coverage for a defect where `_maybe_accept_planning_state` restored
the OLD `rng_key` into the carried control state whenever planning was rejected
(pre-warmup), freezing the planning RNG so every rejected planning iteration
re-sampled identical anchors and actions.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step7 import (
    Step7DynaConfig,
    Step7DynaState,
    init_step7_state,
    make_step7_components,
    step7_update,
)
from alberta_framework.steps.step8 import Step8WorldModelConfig

OBS_DIM = 4
N_ACTIONS = 6


def _cfg(planning_steps: int, warmup: int) -> Step7DynaConfig:
    return Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
        world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
        planning_steps=planning_steps,
        planning_warmup_steps=warmup,
        planning_memory_size=16,
        planning_strategy="random",
    )


def _init(cfg: Step7DynaConfig, seed: int = 0) -> tuple[object, object, Step7DynaState]:
    agent, model = make_step7_components(cfg)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(seed),
        initial_observation=jnp.zeros(OBS_DIM),
        memory_size=cfg.planning_memory_size,
    )
    return agent, model, state


class TestStep7RejectedPlanningRng:
    def test_rejected_planning_still_advances_rng_key(self) -> None:
        """Pre-warmup planning must consume RNG even though its output is dropped."""
        cfg = _cfg(planning_steps=2, warmup=100)
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        assert not bool(jnp.all(result.planning_accepted))
        key_after_real = jr.key_data(result.real_control_result.state.rng_key)
        key_after_planning = jr.key_data(result.state.control_state.rng_key)
        assert not jnp.array_equal(key_after_real, key_after_planning), (
            "planning scan carried the pre-planning rng_key unchanged: rejected "
            "planning steps froze the RNG stream"
        )

    def test_rejected_planning_iterations_sample_distinct_actions(self) -> None:
        """Each rejected planning iteration must draw from a fresh key."""
        cfg = _cfg(planning_steps=8, warmup=100)
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        assert not bool(jnp.all(result.planning_accepted))
        # With a frozen key every iteration re-samples the identical random
        # action; with threaded keys 8 uniform draws over 6 actions collide
        # all-equal only with probability 6**-7 (and the seed is fixed).
        assert int(jnp.unique(result.planning_actions).shape[0]) > 1, (
            "all planning iterations sampled the same action from a frozen rng_key"
        )

    def test_accepted_planning_advances_rng_key(self) -> None:
        """Accepted planning already threads the rollout key; keep it that way."""
        cfg = _cfg(planning_steps=2, warmup=0)
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        assert bool(jnp.all(result.planning_accepted))
        key_after_real = jr.key_data(result.real_control_result.state.rng_key)
        key_after_planning = jr.key_data(result.state.control_state.rng_key)
        assert not jnp.array_equal(key_after_real, key_after_planning)
