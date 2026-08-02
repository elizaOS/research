"""Contract tests for
:mod:`alberta_framework.benchmarks._forager_matched_alberta_worker`.

The worker is the single-seed Alberta workload that the matched-current
executor snapshots and runs inside the qualified networkless OCI image; it
must emit exactly one deterministic reward-trace NPZ per seed for the frozen
in-container scorer and never compute a score itself.  Tests cover the
strict configuration loader (duplicate keys, non-finite JSON, and partially
specified configs are rejected — every field must be explicit), the atomic
reward-trace sink (single ``rewards.npy`` member with a zeroed timestamp for
byte-reproducible archives; incomplete or non-float32 chunks fail closed and
abort leaves nothing behind), dispatch across the three Alberta families
(causal-map, Horde actor-critic, RTU/RTRL), and seed-domain validation
before any output is created.  All family runners are monkeypatched — no
environment steps execute.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from alberta_framework.benchmarks import _forager_matched_alberta_worker as worker
from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerConfig
from alberta_framework.benchmarks.forager import (
    AlbertaForagerConfig,
    RTURTRLForagerConfig,
)
from alberta_framework.core.recurrent_trace_actor_critic import (
    RecurrentTraceActorCriticConfig,
)


def _write_configuration(
    path: Path,
    config: worker.MatchedAlbertaAgentConfig | None = None,
    *,
    implementation_kind: worker.MatchedAlbertaImplementationKind = "alberta_causal_map",
) -> None:
    payload = worker.MatchedAlbertaWorkerConfiguration(
        implementation_kind=implementation_kind,
        configuration=config or CausalMapForagerConfig(),
    ).to_dict()
    path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def test_configuration_loader_requires_strict_fully_explicit_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    expected = CausalMapForagerConfig(
        exploration_probability=0.1,
        respawn_safety_quantile=0.9,
    )
    _write_configuration(config_path, expected)
    loaded = worker._load_configuration(config_path)
    assert loaded.implementation_kind == "alberta_causal_map"
    assert loaded.configuration == expected

    config_path.write_text(
        '{"schema_version":"x","schema_version":"y"}', encoding="utf-8"
    )
    with pytest.raises(worker.MatchedAlbertaWorkerError, match="duplicate"):
        worker._load_configuration(config_path)

    config_path.write_text(
        '{"schema_version":"alberta.forager_matched_worker_configuration.v1",'
        '"implementation_kind":"alberta_causal_map","configuration":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(worker.MatchedAlbertaWorkerError, match="non-finite"):
        worker._load_configuration(config_path)

    config_path.write_text(
        '{"schema_version":"alberta.forager_matched_worker_configuration.v1",'
        '"implementation_kind":"alberta_causal_map",'
        '"configuration":{"exploration_probability":0.05}}',
        encoding="utf-8",
    )
    with pytest.raises(worker.MatchedAlbertaWorkerError, match="fully explicit"):
        worker._load_configuration(config_path)


@pytest.mark.parametrize(
    ("implementation_kind", "config"),
    (
        (
            "alberta_horde_actor_critic",
            AlbertaForagerConfig(
                actor_initial_step_size=0.003,
                critic_initial_step_size=0.003,
            ),
        ),
        (
            "alberta_rtu_rtrl",
            RTURTRLForagerConfig(
                core=RecurrentTraceActorCriticConfig(
                    n_actions=4,
                    hidden_size=8,
                    encoder_width=8,
                    output_width=8,
                    rtrl_taylor_correction=True,
                )
            ),
        ),
    ),
)
def test_configuration_loader_round_trips_distinct_alberta_families(
    tmp_path: Path,
    implementation_kind: worker.MatchedAlbertaImplementationKind,
    config: worker.MatchedAlbertaAgentConfig,
) -> None:
    path = tmp_path / "configuration.json"
    _write_configuration(
        path,
        config,
        implementation_kind=implementation_kind,
    )
    assert worker._load_configuration(path) == worker.MatchedAlbertaWorkerConfiguration(
        implementation_kind=implementation_kind,
        configuration=config,
    )


def test_reward_sink_writes_one_atomic_scorer_compatible_member(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    sink = worker._MatchedRewardTraceSink(data_root, seed=17, steps=5)
    sink.append(
        np.asarray([1.0, -1.0], dtype=np.float32),
        np.asarray([0.0, 0.5], dtype=np.float32),
    )
    sink.append(
        np.asarray([2.0, 0.0, 3.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    metadata = sink.finalize()

    archive_path = data_root / "17.npz"
    assert metadata["relative_path"] == "data/17.npz"
    assert sorted(path.name for path in data_root.iterdir()) == ["17.npz"]
    with zipfile.ZipFile(archive_path, "r") as archive:
        assert archive.namelist() == ["rewards.npy"]
        assert archive.getinfo("rewards.npy").date_time == (1980, 1, 1, 0, 0, 0)
    with np.load(archive_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["rewards"],
            np.asarray([1.0, -1.0, 2.0, 0.0, 3.0], dtype=np.float32),
        )

    sink.abort()
    assert not archive_path.exists()


def test_reward_sink_fails_closed_on_incomplete_or_invalid_chunks(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    sink = worker._MatchedRewardTraceSink(data_root, seed=19, steps=2)
    with pytest.raises(worker.MatchedAlbertaWorkerError, match="float32"):
        sink.append(np.asarray([1.0]), np.asarray([0.0]))
    with pytest.raises(worker.MatchedAlbertaWorkerError, match="cover"):
        sink.finalize()
    sink.abort()
    assert tuple(data_root.iterdir()) == ()


def test_main_freezes_task_and_seals_exact_single_seed_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = CausalMapForagerConfig(
        exploration_probability=0.025,
        respawn_safety_quantile=0.5,
    )
    _write_configuration(config_path, config)

    def fake_run(
        agent_config: CausalMapForagerConfig,
        benchmark: Any,
        seeds: tuple[int, ...],
        *,
        mode: str,
        reward_trace_sink_factory: Any,
    ) -> tuple[SimpleNamespace]:
        assert agent_config == config
        assert benchmark.steps == worker.MATCHED_CURRENT_HORIZON
        assert benchmark.environment.preset == "field_of_view"
        assert benchmark.environment.resolved_env_id == "ForagaxTwoBiomeLarge-v1"
        assert benchmark.environment.resolved_observation_type == "color"
        assert benchmark.environment.aperture_size == 9
        assert benchmark.environment.require_exact_version is True
        assert seeds == (2_300_001,)
        assert mode == "strict"
        sink = reward_trace_sink_factory(seeds[0], benchmark.steps)
        split = 123_456
        sink.append(
            np.zeros((split,), dtype=np.float32),
            np.zeros((split,), dtype=np.float32),
        )
        remaining = benchmark.steps - split
        sink.append(
            np.ones((remaining,), dtype=np.float32),
            np.zeros((remaining,), dtype=np.float32),
        )
        sink.finalize()
        return (SimpleNamespace(seed=seeds[0], steps=benchmark.steps),)

    monkeypatch.setattr(worker, "run_causal_map_forager_seeds", fake_run)
    assert (
        worker.main(
            (
                "--configuration",
                str(config_path),
                "--seed",
                "2300001",
                "--horizon",
                str(worker.MATCHED_CURRENT_HORIZON),
                "--output-root",
                str(output_root),
            )
        )
        == 0
    )
    archive_path = output_root / "data" / "2300001.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        rewards = archive["rewards"]
        assert rewards.dtype == np.dtype(np.float32)
        assert rewards.shape == (worker.MATCHED_CURRENT_HORIZON,)
        assert np.all(rewards[:123_456] == 0.0)
        assert np.all(rewards[123_456:] == 1.0)


@pytest.mark.parametrize(
    ("implementation_kind", "config", "runner_name"),
    (
        (
            "alberta_horde_actor_critic",
            AlbertaForagerConfig(
                actor_initial_step_size=0.003,
                critic_initial_step_size=0.003,
            ),
            "run_alberta_forager_seeds",
        ),
        (
            "alberta_rtu_rtrl",
            RTURTRLForagerConfig(
                core=RecurrentTraceActorCriticConfig(
                    n_actions=4,
                    hidden_size=8,
                    encoder_width=8,
                    output_width=8,
                    rtrl_taylor_correction=True,
                )
            ),
            "run_rtu_rtrl_forager_seeds",
        ),
    ),
)
def test_main_dispatches_each_noncausal_alberta_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    implementation_kind: worker.MatchedAlbertaImplementationKind,
    config: worker.MatchedAlbertaAgentConfig,
    runner_name: str,
) -> None:
    config_path = tmp_path / "config.json"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _write_configuration(
        config_path,
        config,
        implementation_kind=implementation_kind,
    )

    def fake_run(
        agent_config: worker.MatchedAlbertaAgentConfig,
        benchmark: Any,
        seeds: tuple[int, ...],
        *,
        mode: str,
        reward_trace_sink_factory: Any,
    ) -> tuple[SimpleNamespace]:
        assert agent_config == config
        assert seeds == (2_300_001,)
        assert mode == "strict"
        sink = reward_trace_sink_factory(seeds[0], benchmark.steps)
        sink.append(
            np.zeros((benchmark.steps,), dtype=np.float32),
            np.zeros((benchmark.steps,), dtype=np.float32),
        )
        sink.finalize()
        return (SimpleNamespace(seed=seeds[0], steps=benchmark.steps),)

    monkeypatch.setattr(worker, runner_name, fake_run)
    assert worker.main(
        (
            "--configuration",
            str(config_path),
            "--seed",
            "2300001",
            "--horizon",
            str(worker.MATCHED_CURRENT_HORIZON),
            "--output-root",
            str(output_root),
        )
    ) == 0
    assert (output_root / "data" / "2300001.npz").is_file()


@pytest.mark.parametrize("seed", ("-1", str(2**32)))
def test_main_rejects_out_of_domain_seed_before_output_creation(
    tmp_path: Path,
    seed: str,
) -> None:
    config_path = tmp_path / "config.json"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _write_configuration(config_path)
    with pytest.raises(worker.MatchedAlbertaWorkerError, match="uint32"):
        worker.main(
            (
                "--configuration",
                str(config_path),
                "--seed",
                seed,
                "--horizon",
                str(worker.MATCHED_CURRENT_HORIZON),
                "--output-root",
                str(output_root),
            )
        )
    assert tuple(output_root.iterdir()) == ()
