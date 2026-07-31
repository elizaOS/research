"""Focused tests for fixed recurrent features in Alberta Forager."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.forager import (
    FORAGAX_INSTALL_TREE_SHA256,
    AlbertaForagerAgent,
    AlbertaForagerConfig,
    ForagerBenchmarkConfig,
    ForagerEnvConfig,
    ForagerFeatureConfig,
    _augment_with_recurrent_features,
    _init_forager_recurrent_state,
    _recurrent_key,
    foragax_install_tree_sha256,
    run_alberta_forager_seeds,
    run_forager,
)

pytestmark = pytest.mark.integration


class _FakeForagax:
    """Small continuing environment compatible with JAX scans."""

    default_params = None

    def reset(self, key: Any, params: Any) -> tuple[Any, Any]:
        del key, params
        observation = jnp.zeros((2, 2, 2), dtype=jnp.float32)
        return observation, jnp.asarray(0, dtype=jnp.int32)

    def step(
        self,
        key: Any,
        state: Any,
        action: Any,
        params: Any,
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        del key, params
        next_state = state + jnp.asarray(1, dtype=jnp.int32)
        reward = jnp.where(
            action == state % 4,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(-0.25, dtype=jnp.float32),
        )
        observation = jnp.stack(
            (
                jnp.full((2, 2), next_state, dtype=jnp.float32) / 10.0,
                jnp.full((2, 2), action, dtype=jnp.float32) / 3.0,
            ),
            axis=-1,
        )
        return (
            observation,
            next_state,
            reward,
            jnp.asarray(False),
            {"biome_regret": jnp.abs(reward)},
        )


class _HostAlberta(AlbertaForagerAgent):
    """Subclass selects the public host loop instead of the exact-type scan."""


def _fake_make(self: ForagerEnvConfig) -> tuple[_FakeForagax, None]:
    del self
    return _FakeForagax(), None


def _small_recurrent_config(**kwargs: Any) -> AlbertaForagerConfig:
    return AlbertaForagerConfig(
        actor_hidden_sizes=(4,),
        critic_hidden_sizes=(4,),
        recurrent_hidden_size=3,
        features=ForagerFeatureConfig(reward_trace_decays=(0.5,)),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("recurrent_hidden_size", True, "non-negative integer"),
        ("recurrent_hidden_size", -1, "non-negative integer"),
        ("recurrent_hidden_size", 1.5, "non-negative integer"),
        ("recurrent_input_scale", 0.0, "finite and positive"),
        ("recurrent_input_scale", math.inf, "finite and positive"),
        ("recurrent_scale", -0.1, r"\[0, 1\)"),
        ("recurrent_scale", 1.0, r"\[0, 1\)"),
        ("recurrent_scale", math.nan, r"\[0, 1\)"),
        ("recurrent_update_bias", math.inf, "must be finite"),
    ],
)
def test_recurrent_config_validation(
    field_name: str,
    value: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AlbertaForagerConfig(**{field_name: value})


def test_recurrent_features_are_seeded_fixed_causal_and_stop_gradient() -> None:
    config = _small_recurrent_config()
    first_input = jnp.linspace(-1.0, 1.0, 6, dtype=jnp.float32)
    common_input = jnp.linspace(0.2, 0.8, 6, dtype=jnp.float32)
    initial = _init_forager_recurrent_state(
        first_input.shape[0],
        config,
        _recurrent_key(7),
    )
    same_seed = _init_forager_recurrent_state(
        first_input.shape[0],
        config,
        _recurrent_key(7),
    )
    other_seed = _init_forager_recurrent_state(
        first_input.shape[0],
        config,
        _recurrent_key(8),
    )

    chex.assert_trees_all_equal(initial, same_seed)
    assert not np.array_equal(initial.input_kernel, other_seed.input_kernel)

    first_state, first_augmented = _augment_with_recurrent_features(
        first_input,
        initial,
        config,
    )
    alternate_state, _ = _augment_with_recurrent_features(
        -first_input,
        initial,
        config,
    )
    next_state, next_augmented = _augment_with_recurrent_features(
        common_input,
        first_state,
        config,
    )
    alternate_next_state, _ = _augment_with_recurrent_features(
        common_input,
        alternate_state,
        config,
    )

    assert first_augmented.shape == (first_input.shape[0] + 3,)
    np.testing.assert_array_equal(first_augmented[: first_input.shape[0]], first_input)
    np.testing.assert_array_equal(first_augmented[-3:], first_state.hidden)
    assert not np.allclose(next_state.hidden, alternate_next_state.hidden)
    assert np.all(np.isfinite(next_augmented))
    np.testing.assert_array_equal(next_state.input_kernel, initial.input_kernel)
    np.testing.assert_array_equal(next_state.recurrent_kernel, initial.recurrent_kernel)
    np.testing.assert_array_equal(next_state.bias, initial.bias)

    def augmented_hidden_sum(input_kernel: jax.Array) -> jax.Array:
        varied = initial._replace(input_kernel=input_kernel)
        _, augmented = _augment_with_recurrent_features(first_input, varied, config)
        return jnp.sum(augmented)

    kernel_gradient = jax.grad(augmented_hidden_sum)(initial.input_kernel)
    np.testing.assert_array_equal(kernel_gradient, jnp.zeros_like(kernel_gradient))


def test_zero_width_recurrence_is_an_exact_feature_noop() -> None:
    config = AlbertaForagerConfig(recurrent_hidden_size=0)
    features = jnp.arange(7, dtype=jnp.float32)
    state = _init_forager_recurrent_state(
        features.shape[0],
        config,
        _recurrent_key(3),
    )
    next_state, augmented = _augment_with_recurrent_features(
        features,
        state,
        config,
    )

    assert augmented is features
    chex.assert_trees_all_equal(next_state, state)
    assert next_state.hidden.shape == (0,)
    assert next_state.input_kernel.shape == (3, 0, features.shape[0])


def test_recurrent_host_scan_and_chunk_boundaries_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    agent_config = _small_recurrent_config()
    benchmark = ForagerBenchmarkConfig(
        steps=7,
        seed=5,
        record_every=2,
        final_window=4,
        jax_chunk_size=3,
    )
    host_agent = _HostAlberta(agent_config, seed=5)
    scan_agent = AlbertaForagerAgent(agent_config, seed=5)
    other_chunk_agent = AlbertaForagerAgent(agent_config, seed=5)

    host_result = run_forager(host_agent, benchmark)
    scan_result = run_forager(scan_agent, benchmark)
    other_chunk_result = run_forager(
        other_chunk_agent,
        dataclasses.replace(benchmark, jax_chunk_size=5),
    )

    assert host_result.total_reward == scan_result.total_reward
    assert scan_result.total_reward == other_chunk_result.total_reward
    np.testing.assert_allclose(
        host_result.curve_ewm_reward,
        scan_result.curve_ewm_reward,
        rtol=1e-6,
        atol=1e-7,
    )
    chex.assert_trees_all_close(
        host_agent._recurrent_state,
        scan_agent._recurrent_state,
        rtol=1e-6,
        atol=1e-7,
    )
    chex.assert_trees_all_equal(
        scan_agent._recurrent_state,
        other_chunk_agent._recurrent_state,
    )
    assert scan_agent._state.last_observation.shape[-1] == (
        scan_agent.encoder.feature_dim(jnp.zeros((2, 2, 2), dtype=jnp.float32))
        + agent_config.recurrent_hidden_size
    )
    assert scan_agent._updates == benchmark.steps
    assert np.all(np.isfinite(scan_agent.recurrent_hidden))
    assert math.isfinite(scan_result.mean_ewm_reward)


def test_recurrent_batched_modes_match_independent_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ForagerEnvConfig, "make", _fake_make)
    agent_config = _small_recurrent_config()
    benchmark = ForagerBenchmarkConfig(
        steps=5,
        record_every=2,
        final_window=3,
        jax_chunk_size=3,
    )
    seeds = (2, 9)

    independent = tuple(
        run_forager(
            AlbertaForagerAgent(agent_config, seed=seed),
            benchmark.with_seed(seed),
        )
        for seed in seeds
    )

    for mode in ("vmap", "strict"):
        batched = run_alberta_forager_seeds(
            agent_config,
            benchmark,
            seeds,
            mode=mode,
        )
        for batch_run, independent_run in zip(batched, independent, strict=True):
            assert batch_run.total_reward == independent_run.total_reward
            assert batch_run.curve_steps == independent_run.curve_steps
            np.testing.assert_allclose(
                batch_run.curve_ewm_reward,
                independent_run.curve_ewm_reward,
            )
            np.testing.assert_allclose(
                batch_run.curve_window_reward,
                independent_run.curve_window_reward,
            )
            assert batch_run.agent_metadata["runner"]["batch_mode"] == mode


@pytest.mark.integration
def test_recurrent_official_foragax_scan_is_finite() -> None:
    pytest.importorskip("foragax.registry")
    if foragax_install_tree_sha256() != FORAGAX_INSTALL_TREE_SHA256:
        pytest.skip("installed Foragax payload is not the audited release")
    agent = AlbertaForagerAgent(
        _small_recurrent_config(),
        seed=0,
    )
    result = run_forager(
        agent,
        ForagerBenchmarkConfig(
            environment=ForagerEnvConfig.paper_relearning(),
            steps=2,
            seed=0,
            record_every=1,
            final_window=2,
            jax_chunk_size=2,
        ),
    )

    assert result.steps == 2
    assert math.isfinite(result.mean_reward)
    assert math.isfinite(result.mean_ewm_reward)
    assert np.all(np.isfinite(agent.recurrent_hidden))
