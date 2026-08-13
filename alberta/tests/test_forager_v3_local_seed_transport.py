"""Focused tests for explicit matched-v3 local agent-seed transport."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import sys
import textwrap
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# A concurrent unrelated Prototype change may temporarily leave this source
# module absent while imports in prototype_agent.py are already present.  Keep
# this focused runner test collectible without repairing or touching that work.
_ROOT = Path(__file__).resolve().parents[1]
_UTILITY_PATH = _ROOT / "alberta_framework/core/prototype_feature_utility.py"
if not _UTILITY_PATH.is_file():
    _UTILITY_MODULE = "alberta_framework.core.prototype_feature_utility"
    placeholder = types.ModuleType(_UTILITY_MODULE)
    for symbol in (
        "PrototypeFeatureUtilityAuditor",
        "PrototypeFeatureUtilityConfig",
        "PrototypeFeatureUtilityDiagnostics",
        "PrototypeFeatureUtilityEvent",
        "PrototypeFeatureUtilityResourceBudget",
        "PrototypeFeatureUtilityState",
    ):
        setattr(placeholder, symbol, type(symbol, (), {}))
    sys.modules[_UTILITY_MODULE] = placeholder

from alberta_framework.benchmarks import causal_map_forager as causal  # noqa: E402
from alberta_framework.benchmarks import forager  # noqa: E402


class _SeedTransportFakeForagax:
    """Small continuing stream for exact legacy/explicit-seed parity checks."""

    default_params = None

    def reset(self, key: Any, params: Any) -> tuple[Any, Any]:
        del key, params
        return jnp.zeros((3, 3, 2), dtype=jnp.float32), jnp.asarray(0, jnp.int32)

    def step(
        self,
        key: Any,
        state: Any,
        action: Any,
        params: Any,
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        del key, params
        reward = jnp.where(
            action == state % 4,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(-0.25, dtype=jnp.float32),
        )
        return (
            jnp.zeros((3, 3, 2), dtype=jnp.float32),
            state + jnp.asarray(1, jnp.int32),
            reward,
            jnp.asarray(False),
            {"biome_regret": jnp.abs(reward)},
        )


def _fake_make(
    self: forager.ForagerEnvConfig,
) -> tuple[_SeedTransportFakeForagax, None]:
    del self
    return _SeedTransportFakeForagax(), None


def _assert_metric_parity(
    legacy: forager.ForagerRunResult,
    explicit: forager.ForagerRunResult,
) -> None:
    assert explicit.seed == legacy.seed
    assert explicit.total_reward == legacy.total_reward
    assert explicit.curve_steps == legacy.curve_steps
    np.testing.assert_array_equal(
        explicit.curve_ewm_reward,
        legacy.curve_ewm_reward,
    )
    np.testing.assert_array_equal(
        explicit.curve_window_reward,
        legacy.curve_window_reward,
    )
    assert legacy.agent_metadata["seed"] == legacy.seed
    assert "seed" not in explicit.agent_metadata
    assert explicit.agent_metadata["environment_seed"] == explicit.seed
    assert explicit.agent_metadata["agent_seed"] == explicit.seed
    assert forager.summarize_forager_runs((explicit,)).seeds == (explicit.seed,)


def _key_data(key: Any) -> np.ndarray[Any, Any]:
    return np.asarray(jr.key_data(key), dtype=np.uint32)


@pytest.mark.unit
@pytest.mark.parametrize(
    "runner",
    [
        forager.run_alberta_forager_seeds,
        forager.run_rtu_rtrl_forager_seeds,
        causal.run_causal_map_forager_seeds,
    ],
)
def test_public_multi_seed_runners_expose_keyword_only_agent_seeds(runner: Any) -> None:
    parameter = inspect.signature(runner).parameters["agent_seeds"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        True,
        1.0,
        "1",
        -1,
        2**32,
    ],
)
def test_agent_seed_values_reject_aliases_and_uint32_overflow(value: object) -> None:
    with pytest.raises(ValueError, match=r"agent_seeds\[0\].*uint32"):
        forager._validated_explicit_agent_seeds(
            cast(Any, (value,)),
            lane_count=1,
        )


@pytest.mark.unit
def test_agent_seed_transport_requires_an_exact_length_sequence() -> None:
    with pytest.raises(ValueError, match="sequence"):
        forager._validated_explicit_agent_seeds(
            cast(Any, iter((1, 2))),
            lane_count=2,
        )
    with pytest.raises(ValueError, match="same length"):
        forager._validated_explicit_agent_seeds((1,), lane_count=2)
    assert forager._validated_explicit_agent_seeds(None, lane_count=2) is None


@pytest.mark.unit
def test_agent_seed_collisions_are_allowed_and_uint32_maximum_is_exact() -> None:
    assert forager._validated_explicit_agent_seeds(
        (2**32 - 1, 7, 7),
        lane_count=3,
    ) == (2**32 - 1, 7, 7)


@pytest.mark.unit
def test_alberta_roots_hold_environment_fixed_and_change_only_agent_roots() -> None:
    first = forager._alberta_lane_seed_roots(41, 101)
    second = forager._alberta_lane_seed_roots(41, 102)

    assert np.array_equal(_key_data(first.environment), _key_data(second.environment))
    assert not np.array_equal(_key_data(first.recurrent), _key_data(second.recurrent))
    assert not np.array_equal(_key_data(first.core), _key_data(second.core))


@pytest.mark.unit
def test_rtu_roots_hold_environment_fixed_and_change_only_agent_root() -> None:
    first = forager._rtu_rtrl_lane_seed_roots(41, 101)
    second = forager._rtu_rtrl_lane_seed_roots(41, 102)

    assert np.array_equal(_key_data(first.environment), _key_data(second.environment))
    assert not np.array_equal(_key_data(first.core), _key_data(second.core))


@pytest.mark.unit
def test_causal_roots_hold_environment_fixed_and_change_only_agent_root() -> None:
    first = causal._causal_map_lane_seed_roots(41, 101)
    second = causal._causal_map_lane_seed_roots(41, 102)

    assert np.array_equal(_key_data(first.environment), _key_data(second.environment))
    assert int(first.agent_seed) == 101
    assert int(second.agent_seed) == 102
    assert not np.array_equal(_key_data(first.agent), _key_data(second.agent))


@pytest.mark.unit
def test_explicit_metadata_removes_ambiguous_legacy_seed_and_binds_pairing() -> None:
    original = {"seed": 0, "name": "synthetic"}
    metadata = forager._with_explicit_agent_seed_metadata(
        original,
        environment_seed=17,
        agent_seed=23,
        lane_index=2,
        agent_root_uses=("horde_core", "recurrent_features"),
    )

    assert original == {"seed": 0, "name": "synthetic"}
    assert "seed" not in metadata
    assert metadata["environment_seed"] == 17
    assert metadata["agent_seed"] == 23
    assert metadata["seed_transport"] == {
        "schema_version": "alberta.forager_explicit_agent_seed_transport.v1",
        "transport": "explicit_lane_index_pairing",
        "lane_index": 2,
        "environment_seed": 17,
        "agent_seed": 23,
        "environment_root_uses": ["reset", "transition_split_chain"],
        "agent_root_uses": ["horde_core", "recurrent_features"],
        "agent_seed_collisions_allowed": True,
        "environment_agent_seed_equality_required": False,
    }


@pytest.mark.unit
def test_public_rtu_transport_forwards_colliding_agent_seeds_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(*args: Any, **kwargs: Any) -> tuple[forager.ForagerRunResult, ...]:
        captured["ordered_seeds"] = args[2]
        captured["agent_seeds"] = kwargs["agent_seeds"]
        return ()

    monkeypatch.setattr(forager, "_execute_rtu_rtrl_forager_seeds", fake_execute)
    result = forager.run_rtu_rtrl_forager_seeds(
        forager.RTURTRLForagerConfig(),
        forager.ForagerBenchmarkConfig(),
        (3, 4),
        agent_seeds=(9, 9),
    )

    assert result == ()
    assert captured == {"ordered_seeds": (3, 4), "agent_seeds": (9, 9)}


@pytest.mark.unit
def test_public_causal_transport_forwards_agent_seeds_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_lanes(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], None]:
        captured["seeds"] = args[2]
        captured["agent_seeds"] = kwargs["agent_seeds"]
        return (), None

    monkeypatch.setattr(causal, "_run_causal_map_lanes", fake_lanes)
    result = causal.run_causal_map_forager_seeds(
        causal.CausalMapForagerConfig(),
        forager.ForagerBenchmarkConfig(),
        (3, 4),
        agent_seeds=(2**32 - 1, 9),
    )

    assert result == ()
    assert captured == {
        "seeds": (3, 4),
        "agent_seeds": (2**32 - 1, 9),
    }


@pytest.mark.unit
def test_runner_sources_use_environment_and_agent_seeds_in_separate_roots() -> None:
    alberta_tree = ast.parse(
        textwrap.dedent(inspect.getsource(forager.run_alberta_forager_seeds))
    )
    rtu_tree = ast.parse(
        textwrap.dedent(inspect.getsource(forager._execute_rtu_rtrl_forager_seeds))
    )
    causal_tree = ast.parse(
        textwrap.dedent(inspect.getsource(causal._run_causal_map_lanes))
    )

    def names(tree: ast.AST) -> set[str]:
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert {"environment_seed", "agent_seed"} <= names(alberta_tree)
    assert {"environment_seed", "agent_seed"} <= names(rtu_tree)
    assert {"environment_seed", "agent_seed"} <= names(causal_tree)
    causal_source = inspect.getsource(causal._run_causal_map_lanes)
    assert "lane_state.initial_seed" in causal_source
    assert "effective_agent_seeds[lane]" in causal_source


@pytest.mark.unit
def test_single_policy_legacy_sources_do_not_accept_agent_seed_transport() -> None:
    assert "agent_seeds" not in inspect.signature(forager._run_rtu_rtrl_forager_scan).parameters
    assert "agent_seeds" not in inspect.signature(causal.run_causal_map_forager).parameters


def _parity_benchmark() -> forager.ForagerBenchmarkConfig:
    return forager.ForagerBenchmarkConfig(
        environment=forager.ForagerEnvConfig.paper_field_of_view(aperture_size=3),
        steps=3,
        record_every=1,
        final_window=2,
        jax_chunk_size=2,
    )


@pytest.mark.integration
def test_alberta_equal_explicit_roots_preserve_legacy_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forager.ForagerEnvConfig, "make", _fake_make)
    config = forager.AlbertaForagerConfig(
        actor_hidden_sizes=(2,),
        critic_hidden_sizes=(2,),
        features=forager.ForagerFeatureConfig(reward_trace_decays=(0.5,)),
    )
    legacy = forager.run_alberta_forager_seeds(
        config,
        _parity_benchmark(),
        (17,),
        mode="strict",
    )[0]
    explicit = forager.run_alberta_forager_seeds(
        config,
        _parity_benchmark(),
        (17,),
        agent_seeds=(17,),
        mode="strict",
    )[0]
    _assert_metric_parity(legacy, explicit)


@pytest.mark.integration
def test_rtu_equal_explicit_roots_preserve_legacy_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forager.ForagerEnvConfig, "make", _fake_make)
    config = forager.RTURTRLForagerConfig(
        features=forager.ForagerFeatureConfig(reward_trace_decays=(0.5,)),
    )
    legacy = forager.run_rtu_rtrl_forager_seeds(
        config,
        _parity_benchmark(),
        (17,),
        mode="strict",
    )[0]
    explicit = forager.run_rtu_rtrl_forager_seeds(
        config,
        _parity_benchmark(),
        (17,),
        agent_seeds=(17,),
        mode="strict",
    )[0]
    _assert_metric_parity(legacy, explicit)


@pytest.mark.integration
def test_causal_equal_explicit_roots_preserve_legacy_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forager.ForagerEnvConfig, "make", _fake_make)
    config = causal.CausalMapForagerConfig()
    legacy = causal.run_causal_map_forager_seeds(
        config,
        _parity_benchmark(),
        (17,),
        mode="strict",
    )[0]
    explicit = causal.run_causal_map_forager_seeds(
        config,
        _parity_benchmark(),
        (17,),
        agent_seeds=(17,),
        mode="strict",
    )[0]
    _assert_metric_parity(legacy, explicit)


def _explicit_summary_result(
    *,
    lane_index: int,
    environment_seed: int,
    agent_seed: int,
) -> forager.ForagerRunResult:
    batch_environment_seeds = [17, 19]
    batch_agent_seeds = [23, 29]
    metadata = forager._with_explicit_agent_seed_metadata(
        {"name": "synthetic", "privileged": False, "config": {"id": "same"}},
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        lane_index=lane_index,
        agent_root_uses=("horde_core",),
    )
    metadata["runner"] = {
        "batch_seeds": batch_environment_seeds,
        "batch_agent_seeds": batch_agent_seeds,
        "seed_pairing": "lane_index",
    }
    return forager.ForagerRunResult(
        agent="synthetic",
        privileged=False,
        seed=environment_seed,
        steps=2,
        total_reward=1.0,
        mean_reward=0.5,
        final_window_mean_reward=0.5,
        final_ewm_reward=0.5,
        mean_ewm_reward=0.5,
        fov_last_10pct_ema_auc=0.5,
        mean_biome_regret=0.0,
        final_biome_regret=0.0,
        curve_steps=(1, 2),
        curve_ewm_reward=(0.5, 0.5),
        curve_window_reward=(0.5, 0.5),
        duration_s=1.0,
        frames_per_second=2.0,
        environment={"env_id": "synthetic"},
        metric_contract=forager.forager_metric_contract(
            ewm_decay=0.9,
            final_window=2,
            record_every=1,
            steps=2,
        ),
        agent_metadata=metadata,
    )


@pytest.mark.unit
def test_summary_accepts_exact_explicit_pairs_and_rejects_forged_metadata() -> None:
    first = _explicit_summary_result(lane_index=0, environment_seed=17, agent_seed=23)
    second = _explicit_summary_result(lane_index=1, environment_seed=19, agent_seed=29)
    assert forager.summarize_forager_runs((first, second)).seeds == (17, 19)

    forged_cases = (
        {"environment_seed": 999},
        {"agent_seed": 2**40},
        {"seed_transport.schema_version": "forged"},
        {"seed_transport.environment_seed": 999},
        {"seed_transport.agent_seed": 999},
        {
            "agent_seed": 1,
            "seed_transport.agent_seed": True,
            "runner.batch_agent_seeds": [1, 29],
        },
        {"seed_transport.lane_index": 1},
        {"runner.batch_seeds": [17, 17]},
        {"runner.batch_agent_seeds": [29, 23]},
        {"runner.seed_pairing": "by_value"},
    )
    for mutation in forged_cases:
        metadata = dict(first.agent_metadata)
        metadata["seed_transport"] = dict(metadata["seed_transport"])
        metadata["runner"] = dict(metadata["runner"])
        for dotted, value in mutation.items():
            if "." in dotted:
                parent, key = dotted.split(".", 1)
                metadata[parent][key] = value
            else:
                metadata[dotted] = value
        forged = dataclasses.replace(first, agent_metadata=metadata)
        with pytest.raises(ValueError, match="explicit agent seed transport"):
            forager.summarize_forager_runs((forged,))


@pytest.mark.unit
def test_explicit_agent_root_descriptors_match_actual_consumption() -> None:
    assert forager._alberta_agent_root_uses(forager.AlbertaForagerConfig()) == (
        "horde_core",
    )
    assert forager._alberta_agent_root_uses(
        forager.AlbertaForagerConfig(recurrent_hidden_size=4)
    ) == ("horde_core", "recurrent_features")

    metadata = forager._explicit_rtu_agent_metadata(
        forager.RTURTRLForagerAgent(forager.RTURTRLForagerConfig(), seed=0).metadata()
    )
    assert metadata["agent_rng"]["root"] == (
        "jax.random.fold_in(jax.random.key(agent_seed), namespace)"
    )
    assert metadata["agent_rng"]["seed_field"] == "agent_seed"
