"""Control learning gates on the closed-loop switching two-state MDP.

These gates demonstrate that the repo's control agents actually *learn* in an
environment where their actions determine the next observation. Each agent is
run from scratch on multiple seeds; the average reward over the final window
must clearly exceed both the average over the first window and the exact
uniform-random-policy baseline, and must approach the analytic optimum.

The environment is held in phase A throughout (the phase length exceeds the
rollout), so the gates measure single-task control learning; recurring-phase
experiments build on ``test_closed_loop_env.py``.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)
from alberta_framework.core.sarsa import SARSAAgent, SARSAConfig
from alberta_framework.streams import (
    PHASE_A,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

NUM_SEEDS = 5
NUM_STEPS = 4000
WINDOW = 500


def _make_env() -> SwitchingTwoStateMDP:
    """Environment fixed in phase A for the whole rollout."""
    return SwitchingTwoStateMDP(SwitchingTwoStateConfig(phase_length=1_000_000))


def _run_differential_sarsa(env: SwitchingTwoStateMDP, seed: int) -> jnp.ndarray:
    """Closed-loop rollout of a fresh DifferentialSARSAAgent; returns rewards."""
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=env.n_actions,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=0.5,
            epsilon_end=0.02,
            epsilon_decay_steps=2500,
        )
    )
    env_key, agent_key, scan_key = jr.split(jr.key(seed), 3)
    env_state = env.init(env_key)
    agent_state = agent.init(env.feature_dim, agent_key)
    agent_state, _ = agent.start(agent_state, env.observe(env_state))

    def scan_fn(carry, step_key):
        a_state, e_state = carry
        obs, reward, new_e_state = env.step(e_state, a_state.last_action, step_key)
        result = agent.update(a_state, reward, obs)
        return (result.state, new_e_state), reward

    _final, rewards = jax.lax.scan(
        scan_fn, (agent_state, env_state), jr.split(scan_key, NUM_STEPS)
    )
    return rewards


def _run_sarsa(env: SwitchingTwoStateMDP, seed: int) -> jnp.ndarray:
    """Closed-loop rollout of a fresh SARSAAgent; returns rewards."""
    agent = SARSAAgent(
        sarsa_config=SARSAConfig(
            n_actions=env.n_actions,
            gamma=0.9,
            epsilon_start=0.5,
            epsilon_end=0.02,
            epsilon_decay_steps=2500,
        ),
        hidden_sizes=(16,),
        step_size=0.05,
        sparsity=0.0,
    )
    env_key, agent_key, scan_key = jr.split(jr.key(seed), 3)
    env_state = env.init(env_key)
    obs = env.observe(env_state)
    agent_state = agent.init(env.feature_dim, agent_key)
    action, new_key = agent.select_action(agent_state, obs)
    agent_state = agent_state.replace(  # type: ignore[attr-defined]
        last_action=action, last_observation=obs, rng_key=new_key
    )

    def scan_fn(carry, step_key):
        a_state, e_state = carry
        obs, reward, new_e_state = env.step(e_state, a_state.last_action, step_key)
        next_action, new_key = agent.select_action(a_state, obs)
        a_state = a_state.replace(rng_key=new_key)  # type: ignore[attr-defined]
        result = agent.update(a_state, reward, obs, jnp.array(0.0), next_action)
        return (result.state, new_e_state), reward

    _final, rewards = jax.lax.scan(
        scan_fn, (agent_state, env_state), jr.split(scan_key, NUM_STEPS)
    )
    return rewards


def _assert_learning_gate(env: SwitchingTwoStateMDP, run_fn) -> None:
    """Multi-seed gate: final-window reward beats first window and baseline."""
    baseline = env.uniform_random_average_reward(PHASE_A)
    optimal = env.optimal_average_reward(PHASE_A)

    first_windows = []
    final_windows = []
    for seed in range(NUM_SEEDS):
        rewards = run_fn(env, seed)
        first = float(rewards[:WINDOW].mean())
        final = float(rewards[-WINDOW:].mean())
        first_windows.append(first)
        final_windows.append(final)
        # Every seed must improve on its own start and beat random play.
        assert final > first + 0.05, (
            f"seed {seed}: final window {final:.3f} does not exceed "
            f"first window {first:.3f}"
        )
        assert final > baseline + 0.2, (
            f"seed {seed}: final window {final:.3f} does not clearly beat "
            f"uniform-random baseline {baseline:.3f}"
        )

    mean_first = sum(first_windows) / NUM_SEEDS
    mean_final = sum(final_windows) / NUM_SEEDS
    assert mean_final > mean_first + 0.1, (
        f"mean final window {mean_final:.3f} does not clearly exceed "
        f"mean first window {mean_first:.3f}"
    )
    # Learned control should approach the analytic optimum (1.0), not just
    # improve: epsilon_end=0.02 alone costs about 0.01 of average reward.
    assert mean_final > optimal - 0.1, (
        f"mean final window {mean_final:.3f} is far from the optimal "
        f"average reward {optimal:.3f}"
    )


def test_differential_sarsa_learns_closed_loop_control() -> None:
    """DifferentialSARSAAgent learns state-dependent control across 5 seeds."""
    _assert_learning_gate(_make_env(), _run_differential_sarsa)


def test_sarsa_learns_closed_loop_control() -> None:
    """SARSAAgent (Horde MLP) learns state-dependent control across 5 seeds."""
    _assert_learning_gate(_make_env(), _run_sarsa)
