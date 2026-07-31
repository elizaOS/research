"""Focused integration tests for the trainable RTU/RTRL Forager variant."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.forager import (
    FORAGAX_INSTALL_TREE_SHA256,
    FORAGER_ENVIRONMENT_RNG_SCHEDULE,
    ForagerBenchmarkConfig,
    ForagerEnvConfig,
    ForagerFeatureConfig,
    RTUForagerAgent,
    RTUForagerConfig,
    RTURTRLForagerAgent,
    RTURTRLForagerConfig,
    _agent_key,
    _jax_encode_forager,
    _make_rtu_rtrl_scan_chunk,
    _recurrent_key,
    _rtu_rtrl_key,
    foragax_install_tree_sha256,
    run_forager,
    run_rtu_rtrl_forager_seeds,
)
from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticConfig,
)

pytestmark = pytest.mark.integration

_OPEN_DEVELOPMENT_SEEDS = (2_000_001, 2_000_002)


def _small_config(**changes: Any) -> RTURTRLForagerConfig:
    core = RecurrentTraceActorCriticConfig(
        n_actions=4,
        hidden_size=2,
        encoder_width=2,
        output_width=2,
        gamma=0.9,
        actor_lamda=0.5,
        critic_lamda=0.5,
        actor_alpha=0.01,
        critic_alpha=0.01,
        actor_kappa=1.0,
        critic_kappa=1.0,
        entropy_coefficient=0.01,
        sparsity=0.5,
        r_min=0.1,
        r_max=0.9,
    )
    return RTURTRLForagerConfig(
        core=core,
        features=ForagerFeatureConfig(reward_trace_decays=(0.5,)),
        **changes,
    )


def _rng_marker(key: jax.Array) -> jax.Array:
    return jr.uniform(key, (), dtype=jnp.float32)


class _KeyedFakeForagax:
    """Continuing action-sensitive stream with an auditable key marker."""

    default_params = None

    def reset(self, key: Any, params: Any) -> tuple[Any, Any]:
        del params
        observation = jr.uniform(key, (2, 2, 1), dtype=jnp.float32)
        return observation, jnp.asarray(0, dtype=jnp.int32)

    def step(
        self,
        key: Any,
        state: Any,
        action: Any,
        params: Any,
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        del params
        next_state = state + jnp.asarray(1, dtype=jnp.int32)
        reward = jnp.where(
            action == state % 4,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(-0.25, dtype=jnp.float32),
        )
        random_plane = jr.uniform(key, (2, 2), dtype=jnp.float32)
        observation = jnp.expand_dims(
            random_plane
            + jnp.asarray(action, dtype=jnp.float32) / jnp.asarray(8.0),
            axis=-1,
        )
        return (
            observation,
            next_state,
            reward,
            jnp.asarray(False),
            {
                "biome_regret": _rng_marker(key),
                # Deliberately non-finite evaluator-only state.  A runner that
                # leaks the whole info mapping into features/finiteness checks
                # cannot complete this test.
                "hidden_task_label": jnp.asarray(jnp.nan, dtype=jnp.float32),
            },
        )


def _fake_make(self: ForagerEnvConfig) -> tuple[_KeyedFakeForagax, None]:
    del self
    return _KeyedFakeForagax(), None


class _MemoryTraceSink:
    def __init__(self) -> None:
        self.rewards: list[np.ndarray] = []
        self.regrets: list[np.ndarray] = []

    def append(self, rewards: np.ndarray, biome_regrets: np.ndarray) -> None:
        self.rewards.append(np.array(rewards, copy=True))
        self.regrets.append(np.array(biome_regrets, copy=True))

    def finalize(self) -> Mapping[str, Any]:
        return {"kind": "in_memory_test_trace"}

    def abort(self) -> None:
        self.rewards.clear()
        self.regrets.clear()

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.rewards), np.concatenate(self.regrets)


class _ForbiddenContext:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"ordinary RTU policy read evaluator context field {name}")


def _expected_key_markers(seed: int, steps: int) -> np.ndarray:
    key = jr.key(seed)
    key, _ = jr.split(key)
    markers: list[float] = []
    for _ in range(steps):
        key, step_key = jr.split(key)
        markers.append(float(_rng_marker(step_key)))
    return np.asarray(markers, dtype=np.float32)


def _expected_adjusted_ewm(rewards: np.ndarray, decay: float) -> np.ndarray:
    numerator = 0.0
    denominator = 0.0
    values: list[float] = []
    for reward in rewards.astype(np.float64):
        numerator = float(reward) + decay * numerator
        denominator = 1.0 + decay * denominator
        values.append(numerator / denominator)
    return np.asarray(values, dtype=np.float64)


def _structured_primitive_names(traced: Any) -> tuple[str, ...]:
    """Return primitive names by traversing structured nested JAXPR values."""
    names: list[str] = []
    visited: set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                names.append(equation.primitive.name)
                visit(equation.params)
        elif hasattr(value, "jaxpr"):
            visit(value.jaxpr)
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                visit(nested)

    visit(traced)
    return tuple(names)


def _trace_rtu_scan(
    freeze_after_steps: int | None,
) -> tuple[Any, Any, Any, jax.Array]:
    """Trace a complete learning/frozen Forager chunk and return its inputs."""
    config = _small_config(freeze_after_steps=freeze_after_steps)
    benchmark = ForagerBenchmarkConfig(
        steps=2,
        seed=_OPEN_DEVELOPMENT_SEEDS[0],
        jax_chunk_size=2,
    )
    env = _KeyedFakeForagax()
    policy = RTURTRLForagerAgent(config, seed=benchmark.seed)
    core = policy._build_core()
    frozen_core = policy._build_frozen_core()
    env_key = jr.key(benchmark.seed)
    env_key, reset_key = jr.split(env_key)
    observation, env_state = env.reset(reset_key, None)
    reward_traces = jnp.zeros(
        (len(config.features.reward_trace_decays),),
        dtype=jnp.float32,
    )
    features = _jax_encode_forager(
        observation,
        config.features,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        reward_traces,
    )
    core_state = core.init(features.shape[0], _rtu_rtrl_key(benchmark.seed))
    core_state, action, _ = core.start(core_state, features)
    carry = (
        env_state,
        env_key,
        core_state,
        action,
        reward_traces,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    scan = _make_rtu_rtrl_scan_chunk(
        env,
        None,
        core,
        frozen_core,
        config,
        benchmark,
    )
    traced = jax.make_jaxpr(scan)(
        carry,
        jnp.asarray(benchmark.steps, dtype=jnp.int32),
    )
    return traced, core, core_state, features


def test_rtu_variant_identity_is_explicit_and_not_the_fixed_gru() -> None:
    config = _small_config()
    agent = RTURTRLForagerAgent(config, seed=_OPEN_DEVELOPMENT_SEEDS[0])
    metadata = agent.metadata()

    assert RTUForagerConfig is RTURTRLForagerConfig
    assert RTUForagerAgent is RTURTRLForagerAgent
    assert agent.name == "alberta_rtu_rtrl_ac"
    assert metadata["recurrent_core"] == {
        "kind": "diagonal_complex_rtu",
        "trainable": True,
        "gradient_estimator": "compressed_rtrl",
        "sensitivity_memory": "linear_in_rtu_parameter_count",
        "fixed_weight_echo_state_gru": False,
        "exactness_qualification": (
            "compressed sensitivities contain every structural RTU derivative "
            "for fixed parameters; retained sensitivities become stale after "
            "online recurrent-parameter changes"
        ),
    }
    assert "no protected-seed" in metadata["claim_scope"]
    assert config.to_dict()["core"]["n_actions"] == 4

    with pytest.raises(ValueError, match="exactly four actions"):
        RTURTRLForagerConfig(
            core=dataclasses.replace(config.core, n_actions=3),
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        RTURTRLForagerConfig(core=config.core, freeze_after_steps=True)


def test_rtu_rng_namespace_is_deterministic_and_disjoint() -> None:
    for seed in _OPEN_DEVELOPMENT_SEEDS:
        first = _rtu_rtrl_key(seed)
        second = _rtu_rtrl_key(seed)
        environment_key = jr.key(seed)
        environment_key, reset_key = jr.split(environment_key)
        _, step_key = jr.split(environment_key)

        np.testing.assert_array_equal(jr.key_data(first), jr.key_data(second))
        for other in (
            reset_key,
            step_key,
            _agent_key(seed),
            _recurrent_key(seed),
        ):
            assert not np.array_equal(jr.key_data(first), jr.key_data(other))

    assert not np.array_equal(
        jr.key_data(_rtu_rtrl_key(_OPEN_DEVELOPMENT_SEEDS[0])),
        jr.key_data(_rtu_rtrl_key(_OPEN_DEVELOPMENT_SEEDS[1])),
    )


def test_full_rtu_scan_learning_and_frozen_paths_are_callback_free() -> None:
    learning_jaxpr, core, started_state, features = _trace_rtu_scan(None)
    frozen_jaxpr, _, _, _ = _trace_rtu_scan(0)

    for traced in (learning_jaxpr, frozen_jaxpr):
        assert not traced.effects
        assert not any(
            "callback" in name
            for name in _structured_primitive_names(traced)
        )

    checked_jaxpr = jax.make_jaxpr(core.update)(
        started_state,
        jnp.asarray(0.25, dtype=jnp.float32),
        features,
    )
    assert checked_jaxpr.effects
    assert any(
        "callback" in name
        for name in _structured_primitive_names(checked_jaxpr)
    )


def test_host_lifecycle_freezes_parameters_but_advances_recurrence() -> None:
    config = _small_config(freeze_after_steps=2)
    agent = RTURTRLForagerAgent(config, seed=_OPEN_DEVELOPMENT_SEEDS[0])
    context = _ForbiddenContext()
    initial_observation = jnp.zeros((2, 2, 1), dtype=jnp.float32)
    agent.start(initial_observation, context)  # type: ignore[arg-type]
    for index in range(2):
        agent.step(
            0.25,
            jnp.full((2, 2, 1), index + 1, dtype=jnp.float32),
            context,  # type: ignore[arg-type]
        )

    assert agent._state is not None
    frozen_parameters = (agent._state.actor_params, agent._state.critic_params)
    frozen_recurrence = (agent._state.actor_rtu_state, agent._state.critic_rtu_state)
    for index in range(2, 5):
        agent.step(
            -0.5,
            jnp.full((2, 2, 1), index + 1, dtype=jnp.float32),
            context,  # type: ignore[arg-type]
        )

    assert agent._state is not None
    chex.assert_trees_all_equal(
        frozen_parameters,
        (agent._state.actor_params, agent._state.critic_params),
    )
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(frozen_recurrence),
            jax.tree_util.tree_leaves(
                (agent._state.actor_rtu_state, agent._state.critic_rtu_state)
            ),
            strict=True,
        )
    )
    assert agent._updates == 2
    assert int(agent._state.step_count) == 5
    # The core starts Welford moments with one zero-valued pseudocount, then
    # consumes the initial observation and all five continuing transitions.
    assert int(agent._state.observation_statistics.sample_count) == 7


def test_single_vmap_strict_rng_metrics_and_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    config = _small_config()
    benchmark = ForagerBenchmarkConfig(
        steps=7,
        seed=_OPEN_DEVELOPMENT_SEEDS[0],
        ewm_decay=0.75,
        record_every=3,
        final_window=4,
        jax_chunk_size=3,
    )
    single_agent = RTURTRLForagerAgent(config, seed=benchmark.seed)
    single = run_forager(single_agent, benchmark)

    mode_results: dict[str, tuple[Any, ...]] = {}
    mode_traces: dict[str, dict[int, _MemoryTraceSink]] = {}
    for mode in ("strict", "vmap"):
        sinks: dict[int, _MemoryTraceSink] = {}

        def factory(seed: int, steps: int) -> _MemoryTraceSink:
            assert steps == benchmark.steps
            sink = _MemoryTraceSink()
            sinks[seed] = sink
            return sink

        mode_results[mode] = run_rtu_rtrl_forager_seeds(
            config,
            benchmark,
            _OPEN_DEVELOPMENT_SEEDS,
            mode=mode,
            reward_trace_sink_factory=factory,
        )
        mode_traces[mode] = sinks

    for mode in ("strict", "vmap"):
        first = mode_results[mode][0]
        assert first.total_reward == single.total_reward
        assert first.curve_steps == single.curve_steps == (1, 3, 6, 7)
        np.testing.assert_allclose(first.curve_ewm_reward, single.curve_ewm_reward)
        np.testing.assert_allclose(
            first.curve_window_reward,
            single.curve_window_reward,
        )
        assert first.agent_metadata["runner"]["batch_mode"] == mode
        assert first.agent_metadata["runner"]["batch_size"] == 2
        assert first.agent_metadata["environment_rng_schedule"] == (
            FORAGER_ENVIRONMENT_RNG_SCHEDULE
        )

    for lane, seed in enumerate(_OPEN_DEVELOPMENT_SEEDS):
        strict = mode_results["strict"][lane]
        vmapped = mode_results["vmap"][lane]
        assert strict.total_reward == vmapped.total_reward
        assert strict.curve_steps == vmapped.curve_steps
        np.testing.assert_allclose(strict.curve_ewm_reward, vmapped.curve_ewm_reward)
        np.testing.assert_allclose(strict.curve_window_reward, vmapped.curve_window_reward)
        strict_rewards, strict_regrets = mode_traces["strict"][seed].arrays()
        vmap_rewards, vmap_regrets = mode_traces["vmap"][seed].arrays()
        np.testing.assert_array_equal(strict_rewards, vmap_rewards)
        np.testing.assert_array_equal(strict_regrets, vmap_regrets)
        np.testing.assert_array_equal(
            strict_regrets,
            _expected_key_markers(seed, benchmark.steps),
        )

        expected_ewm = _expected_adjusted_ewm(strict_rewards, benchmark.ewm_decay)
        assert strict.total_reward == pytest.approx(
            float(np.sum(strict_rewards, dtype=np.float64))
        )
        assert strict.final_ewm_reward == pytest.approx(float(expected_ewm[-1]))
        assert strict.mean_ewm_reward == pytest.approx(float(np.mean(expected_ewm)))
        assert strict.final_window_mean_reward == pytest.approx(
            float(np.mean(strict_rewards[-benchmark.final_window :]))
        )
        assert strict.mean_biome_regret == pytest.approx(
            float(np.mean(strict_regrets, dtype=np.float64))
        )
        assert strict.final_biome_regret == pytest.approx(float(strict_regrets[-1]))

    assert single.agent_metadata["runner"]["kind"] == "jax_scan"
    assert single.agent_metadata["runner"]["bounded_reward_buffer_steps"] == 4
    assert single_agent._updates == benchmark.steps
    assert single_agent._state is not None
    assert int(single_agent._state.step_count) == benchmark.steps
    assert all(
        bool(np.all(np.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(single_agent._state)
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )

    long_lived = ForagerBenchmarkConfig(
        steps=500_003,
        seed=_OPEN_DEVELOPMENT_SEEDS[0],
        final_window=100_000,
        jax_chunk_size=4_096,
    )
    assert long_lived.jax_chunk_size == 4_096
    assert long_lived.final_window < long_lived.steps


def test_official_foragax_rtu_scan_is_finite_on_open_development_seed() -> None:
    pytest.importorskip("foragax.registry")
    if foragax_install_tree_sha256() != FORAGAX_INSTALL_TREE_SHA256:
        pytest.skip("installed Foragax payload is not the audited release")
    config = _small_config()
    seed = _OPEN_DEVELOPMENT_SEEDS[0]
    agent = RTURTRLForagerAgent(config, seed=seed)
    result = run_forager(
        agent,
        ForagerBenchmarkConfig(
            environment=ForagerEnvConfig.paper_relearning(),
            steps=2,
            seed=seed,
            record_every=1,
            final_window=2,
            jax_chunk_size=2,
        ),
    )

    assert result.steps == 2
    assert result.agent == "alberta_rtu_rtrl_ac"
    assert not result.privileged
    assert math.isfinite(result.mean_reward)
    assert math.isfinite(result.mean_ewm_reward)
    assert agent._state is not None
    assert all(
        bool(np.all(np.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(agent._state)
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
