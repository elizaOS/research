"""Tests for strict resumable heterogeneous Forager matrices."""

from __future__ import annotations

import copy
import dataclasses
import fcntl
import hashlib
import io
import json
import os
import struct
import sys
import tarfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from alberta_framework.benchmarks import forager_matrix as matrix
from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerConfig
from alberta_framework.benchmarks.forager import (
    FORAGER_ENVIRONMENT_RNG_SCHEDULE,
    FORAGER_FOV_EMA_DECAY,
    FORAGER_FOV_EMA_SUBSAMPLE,
    FORAGER_FOV_TAIL_FRACTION,
    AlbertaForagerConfig,
    ForagerBatchMode,
    ForagerBenchmarkConfig,
    ForagerRunResult,
    RTURTRLForagerAgent,
    RTURTRLForagerConfig,
    _adjusted_ewm_chunk,
    _finalize_reward_trace_sinks,
    _unadjusted_ema_chunk,
    environment_rng_schedule_sha256,
    forager_metric_contract,
)

pytestmark = pytest.mark.integration

VariantConfig = (
    AlbertaForagerConfig | CausalMapForagerConfig | RTURTRLForagerConfig
)
ValueFunction = Callable[[str, VariantConfig, int], float]


@pytest.fixture(autouse=True)
def _isolated_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "source-root"
    package = root / "alberta_framework"
    benchmarks = package / "benchmarks"
    benchmarks.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='matrix-test'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "FORAGER_BENCHMARK.md").write_text("# Matrix test\n", encoding="utf-8")
    (package / "__init__.py").write_text('"""test package"""\n', encoding="utf-8")
    (package / "py.typed").write_bytes(b"")
    (benchmarks / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(matrix, "REPO_ROOT", root)
    return root


def _selection_rule(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metric": "fov_last_10pct_ema_auc",
        "direction": "maximize",
        "statistic": "mean",
        "confidence": 0.9,
        "bootstrap_resamples": 128,
        "bootstrap_seed": 7,
        "tie_break": "variant_id_ascending",
    }
    payload.update(updates)
    return payload


def _horde(
    config: Mapping[str, Any] | None = None,
    *,
    group: str = "policy",
) -> dict[str, Any]:
    return {
        "kind": "alberta_horde_ac",
        "selection_group": group,
        "config": dict(config or {}),
    }


def _causal(
    config: Mapping[str, Any] | None = None,
    *,
    group: str = "policy",
) -> dict[str, Any]:
    return {
        "kind": "alberta_causal_map",
        "selection_group": group,
        "config": dict(config or {}),
    }


def _rtu(
    config: Mapping[str, Any] | None = None,
    *,
    group: str = "policy",
) -> dict[str, Any]:
    return {
        "kind": matrix.RTU_RTRL_VARIANT_KIND,
        "selection_group": group,
        "config": dict(config or {}),
    }


def _manifest_payload(**updates: Any) -> dict[str, Any]:
    stage = updates.get("stage", "tuning")
    seeds = list(updates.get("seeds", [5, 1, 9]))
    counterpart = [matrix._MAX_JAX_SEED - index for index in range(len(seeds))]
    payload: dict[str, Any] = {
        "schema_version": "2.2",
        "preset": "field_of_view",
        "stage": stage,
        "steps": 12,
        "seeds": seeds,
        "jax_chunk_size": 4,
        "seed_batch_size": 2,
        "mode": "strict",
        "source_execution_mode": "live_tree_unsealed",
        "metric_evidence_mode": "scalar_summary_unsealed",
        "selection_rule": _selection_rule(),
        "variants": {"base": _horde()},
        "tuning_seeds": seeds if stage == "tuning" else counterpart,
        "evaluation_seeds": counterpart if stage == "tuning" else seeds,
    }
    payload.update(updates)
    return payload


def _fake_result(
    kind: str,
    config: VariantConfig,
    benchmark: ForagerBenchmarkConfig,
    seed: int,
    *,
    batch_seeds: Sequence[int],
    mode: ForagerBatchMode,
    value: float,
) -> ForagerRunResult:
    return ForagerRunResult(
        agent=kind,
        privileged=False,
        seed=seed,
        steps=benchmark.steps,
        total_reward=value * benchmark.steps,
        mean_reward=value,
        final_window_mean_reward=value,
        final_ewm_reward=value,
        mean_ewm_reward=value,
        fov_last_10pct_ema_auc=value,
        mean_biome_regret=0.0,
        final_biome_regret=0.0,
        curve_steps=(1, benchmark.steps),
        curve_ewm_reward=(value, value),
        curve_window_reward=(value, value),
        duration_s=1e-12,
        frames_per_second=float(benchmark.steps / 1e-12),
        environment=benchmark.environment.to_dict(),
        metric_contract=forager_metric_contract(
            ewm_decay=benchmark.ewm_decay,
            final_window=benchmark.final_window,
            record_every=benchmark.record_every,
            steps=benchmark.steps,
        ),
        agent_metadata={
            "name": kind,
            "privileged": False,
            "seed": seed,
            "config": config.to_dict(),
            "environment_rng_schedule": FORAGER_ENVIRONMENT_RNG_SCHEDULE,
            "environment_rng_schedule_sha256": (
                environment_rng_schedule_sha256()
            ),
            "runner": {
                "kind": (
                    "jax_scan"
                    if kind == "alberta_causal_map" and len(batch_seeds) == 1
                    else "jax_batched_scan"
                ),
                "batch_mode": mode,
                "batch_seeds": list(batch_seeds),
                "batch_size": len(batch_seeds),
                "chunk_size": benchmark.jax_chunk_size,
                "overall_duration_s": 1e-12,
                "setup_duration_s": 0.0,
                "compile_duration_s": 0.0,
                "execution_duration_s": 1e-12,
                "aggregate_transitions_per_second": (
                    len(batch_seeds) * benchmark.steps / 1e-12
                ),
                "per_seed_effective_frames_per_second": (
                    benchmark.steps / 1e-12
                ),
            },
        },
    )


def _trace_result(
    config: VariantConfig,
    benchmark: ForagerBenchmarkConfig,
    seed: int,
    *,
    batch_seeds: Sequence[int],
    mode: ForagerBatchMode,
    rewards: np.ndarray,
    regrets: np.ndarray,
    raw_metric_trace: Mapping[str, Any],
) -> ForagerRunResult:
    rewards64 = rewards.astype(np.float64)
    regrets64 = regrets.astype(np.float64)
    ewm_state = np.zeros((1,), dtype=np.float64)
    fov_state = np.zeros((1,), dtype=np.float64)
    ewm_chunks: list[np.ndarray] = []
    fov_samples: list[float] = []
    total_reward = 0.0
    ewm_total = 0.0
    regret_total = 0.0
    for offset in range(0, benchmark.steps, benchmark.jax_chunk_size):
        chunk = rewards64[offset : offset + benchmark.jax_chunk_size]
        regret_chunk = regrets64[offset : offset + benchmark.jax_chunk_size]
        ewm, ewm_state = _adjusted_ewm_chunk(
            chunk,
            decay=benchmark.ewm_decay,
            completed_steps=offset,
            filter_state=ewm_state,
        )
        fov, fov_state = _unadjusted_ema_chunk(
            chunk,
            decay=FORAGER_FOV_EMA_DECAY,
            filter_state=fov_state,
        )
        mask = (
            np.arange(offset, offset + chunk.size)
            % FORAGER_FOV_EMA_SUBSAMPLE
            == 0
        )
        fov_samples.extend(float(value) for value in fov[mask])
        ewm_chunks.append(ewm)
        total_reward += float(np.sum(chunk, dtype=np.float64))
        ewm_total += float(np.sum(ewm, dtype=np.float64))
        regret_total += float(np.sum(regret_chunk, dtype=np.float64))
    ewm_values = np.concatenate(ewm_chunks)
    target_steps = sorted(
        {
            1,
            benchmark.steps,
            *range(
                benchmark.record_every,
                benchmark.steps + 1,
                benchmark.record_every,
            ),
        }
    )
    curve_window = tuple(
        float(
            np.mean(
                rewards64[
                    max(0, step - benchmark.final_window) : step
                ]
            )
        )
        for step in target_steps
    )
    tail_start = int(
        (1.0 - FORAGER_FOV_TAIL_FRACTION) * len(fov_samples)
    )
    base = _fake_result(
        "alberta_horde_ac",
        config,
        benchmark,
        seed,
        batch_seeds=batch_seeds,
        mode=mode,
        value=0.0,
    )
    metadata = dict(base.agent_metadata)
    metadata["raw_metric_trace"] = dict(raw_metric_trace)
    return dataclasses.replace(
        base,
        total_reward=total_reward,
        mean_reward=total_reward / benchmark.steps,
        final_window_mean_reward=curve_window[-1],
        final_ewm_reward=float(ewm_values[-1]),
        mean_ewm_reward=ewm_total / benchmark.steps,
        fov_last_10pct_ema_auc=float(
            np.mean(np.asarray(fov_samples[tail_start:], dtype=np.float64))
        ),
        mean_biome_regret=regret_total / benchmark.steps,
        final_biome_regret=float(regrets64[-1]),
        curve_steps=tuple(target_steps),
        curve_ewm_reward=tuple(float(ewm_values[step - 1]) for step in target_steps),
        curve_window_reward=curve_window,
        agent_metadata=metadata,
    )


def _rtu_fake_result(
    config: RTURTRLForagerConfig,
    benchmark: ForagerBenchmarkConfig,
    seed: int,
    *,
    batch_seeds: Sequence[int],
    mode: ForagerBatchMode,
    value: float,
) -> ForagerRunResult:
    base = _fake_result(
        matrix.RTU_RTRL_RESULT_AGENT,
        config,
        benchmark,
        seed,
        batch_seeds=batch_seeds,
        mode=mode,
        value=value,
    )
    metadata = dict(RTURTRLForagerAgent(config, seed=seed).metadata())
    metadata.update(
        {
            "environment_rng_schedule": FORAGER_ENVIRONMENT_RNG_SCHEDULE,
            "environment_rng_schedule_sha256": environment_rng_schedule_sha256(),
            "runner": base.agent_metadata["runner"],
        }
    )
    return dataclasses.replace(base, agent_metadata=metadata)


def _rtu_trace_result(
    config: RTURTRLForagerConfig,
    benchmark: ForagerBenchmarkConfig,
    seed: int,
    *,
    batch_seeds: Sequence[int],
    mode: ForagerBatchMode,
    rewards: np.ndarray,
    regrets: np.ndarray,
    raw_metric_trace: Mapping[str, Any],
) -> ForagerRunResult:
    base = _trace_result(
        config,
        benchmark,
        seed,
        batch_seeds=batch_seeds,
        mode=mode,
        rewards=rewards,
        regrets=regrets,
        raw_metric_trace=raw_metric_trace,
    )
    metadata = dict(RTURTRLForagerAgent(config, seed=seed).metadata())
    metadata.update(
        {
            "environment_rng_schedule": FORAGER_ENVIRONMENT_RNG_SCHEDULE,
            "environment_rng_schedule_sha256": environment_rng_schedule_sha256(),
            "runner": base.agent_metadata["runner"],
            "raw_metric_trace": dict(raw_metric_trace),
        }
    )
    return dataclasses.replace(
        base,
        agent=matrix.RTU_RTRL_RESULT_AGENT,
        agent_metadata=metadata,
    )


def _install_raw_trace_runner(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[int, ...]],
    *,
    random_trace: bool,
) -> None:
    def horde(
        config: AlbertaForagerConfig,
        benchmark: ForagerBenchmarkConfig,
        seeds: Sequence[int],
        *,
        mode: ForagerBatchMode,
        reward_trace_sink_factory: Any = None,
    ) -> tuple[ForagerRunResult, ...]:
        assert reward_trace_sink_factory is not None
        ordered = tuple(seeds)
        calls.append(ordered)
        results: list[ForagerRunResult] = []
        for seed in ordered:
            if random_trace:
                generator = np.random.default_rng(seed + 4_321)
                rewards = generator.normal(
                    loc=0.25,
                    scale=0.75,
                    size=benchmark.steps,
                ).astype(np.float32)
                regrets = generator.uniform(
                    0.0,
                    2.0,
                    size=benchmark.steps,
                ).astype(np.float32)
            else:
                rewards = np.full(
                    (benchmark.steps,),
                    2.0,
                    dtype=np.float32,
                )
                regrets = np.full(
                    (benchmark.steps,),
                    0.5,
                    dtype=np.float32,
                )
            sink = reward_trace_sink_factory(seed, benchmark.steps)
            for offset in range(0, benchmark.steps, benchmark.jax_chunk_size):
                end = min(
                    benchmark.steps,
                    offset + benchmark.jax_chunk_size,
                )
                sink.append(rewards[offset:end], regrets[offset:end])
            trace = sink.finalize()
            results.append(
                _trace_result(
                    config,
                    benchmark,
                    seed,
                    batch_seeds=ordered,
                    mode=mode,
                    rewards=rewards,
                    regrets=regrets,
                    raw_metric_trace=trace,
                )
            )
        return tuple(results)

    monkeypatch.setattr(matrix, "run_alberta_forager_seeds", horde)


def _install_fake_runners(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]],
    *,
    value: ValueFunction | None = None,
) -> None:
    value_function = value or (lambda _kind, _config, seed: float(seed + 1))

    def horde(
        config: AlbertaForagerConfig,
        benchmark: ForagerBenchmarkConfig,
        seeds: Sequence[int],
        *,
        mode: ForagerBatchMode,
        reward_trace_sink_factory: Any = None,
    ) -> tuple[ForagerRunResult, ...]:
        assert reward_trace_sink_factory is None
        ordered = tuple(seeds)
        calls.append(("alberta_horde_ac", ordered, config.to_dict()))
        return tuple(
            _fake_result(
                "alberta_horde_ac",
                config,
                benchmark,
                seed,
                batch_seeds=ordered,
                mode=mode,
                value=value_function("alberta_horde_ac", config, seed),
            )
            for seed in ordered
        )

    def causal(
        config: CausalMapForagerConfig,
        benchmark: ForagerBenchmarkConfig,
        seeds: Sequence[int],
        *,
        mode: ForagerBatchMode,
        reward_trace_sink_factory: Any = None,
    ) -> tuple[ForagerRunResult, ...]:
        assert reward_trace_sink_factory is None
        ordered = tuple(seeds)
        calls.append(("alberta_causal_map", ordered, config.to_dict()))
        return tuple(
            _fake_result(
                "alberta_causal_map",
                config,
                benchmark,
                seed,
                batch_seeds=ordered,
                mode=mode,
                value=value_function("alberta_causal_map", config, seed),
            )
            for seed in ordered
        )

    monkeypatch.setattr(matrix, "run_alberta_forager_seeds", horde)
    monkeypatch.setattr(matrix, "run_causal_map_forager_seeds", causal)


def _install_rtu_fake_runner(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[tuple[int, ...], ForagerBatchMode]],
    *,
    raw_trace: bool = False,
) -> None:
    def rtu(
        config: RTURTRLForagerConfig,
        benchmark: ForagerBenchmarkConfig,
        seeds: Sequence[int],
        *,
        mode: ForagerBatchMode,
        reward_trace_sink_factory: Any = None,
    ) -> tuple[ForagerRunResult, ...]:
        ordered = tuple(seeds)
        calls.append((ordered, mode))
        if not raw_trace:
            assert reward_trace_sink_factory is None
            return tuple(
                _rtu_fake_result(
                    config,
                    benchmark,
                    seed,
                    batch_seeds=ordered,
                    mode=mode,
                    value=float(seed + 1),
                )
                for seed in ordered
            )
        assert reward_trace_sink_factory is not None
        results: list[ForagerRunResult] = []
        for seed in ordered:
            rewards = np.linspace(
                0.0,
                1.0,
                benchmark.steps,
                dtype=np.float32,
            ) + np.float32(seed)
            regrets = np.linspace(
                1.0,
                0.0,
                benchmark.steps,
                dtype=np.float32,
            )
            sink = reward_trace_sink_factory(seed, benchmark.steps)
            for offset in range(0, benchmark.steps, benchmark.jax_chunk_size):
                end = min(offset + benchmark.jax_chunk_size, benchmark.steps)
                sink.append(rewards[offset:end], regrets[offset:end])
            trace = sink.finalize()
            results.append(
                _rtu_trace_result(
                    config,
                    benchmark,
                    seed,
                    batch_seeds=ordered,
                    mode=mode,
                    rewards=rewards,
                    regrets=regrets,
                    raw_metric_trace=trace,
                )
            )
        return tuple(results)

    monkeypatch.setattr(matrix, "run_rtu_rtrl_forager_seeds", rtu)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    unhashed = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rewrite_hashed_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["payload_sha256"] = _canonical_hash(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_manifest_normalizes_explicit_mixed_kind_variants_and_hashes() -> None:
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            variants={
                "horde": _horde(
                    {
                        "actor_hidden_sizes": [8],
                        "features": {
                            "include_hint": False,
                            "reward_trace_decays": [0.5, 0.9],
                        },
                    }
                ),
                "map": _causal({"world_shape": [15, 15], "retry_penalty": 0.4}),
            }
        )
    )

    horde = manifest.variants["horde"]
    causal = manifest.variants["map"]
    assert horde.kind == "alberta_horde_ac"
    assert isinstance(horde.config, AlbertaForagerConfig)
    assert horde.config.actor_hidden_sizes == (8,)
    assert horde.config.features.include_hint is False
    assert causal.kind == "alberta_causal_map"
    assert isinstance(causal.config, CausalMapForagerConfig)
    assert causal.config.retry_penalty == pytest.approx(0.4)
    normalized = manifest.to_dict()
    assert normalized["variants"]["horde"] == {
        "kind": "alberta_horde_ac",
        "selection_group": "policy",
        "config": horde.config.to_dict(),
    }
    assert normalized["variants"]["map"]["config"] == causal.config.to_dict()
    reparsed = matrix.parse_forager_matrix_manifest(normalized)
    assert reparsed.to_dict() == normalized
    assert reparsed.config_sha256 == manifest.config_sha256
    assert len(horde.config_sha256) == len(horde.descriptor_sha256) == 64
    numeric_equivalent = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            variants={
                "map": _causal({"retry_penalty": 0}),
            }
        )
    )
    float_equivalent = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            variants={
                "map": _causal({"retry_penalty": 0.0}),
            }
        )
    )
    assert numeric_equivalent.config_sha256 == float_equivalent.config_sha256


def test_schema_22_contract_and_canonical_hash_remain_exactly_unchanged() -> None:
    payload = _manifest_payload(
        variants={
            "base": _horde(),
            "map": _causal(),
        }
    )
    manifest = matrix.parse_forager_matrix_manifest(payload)
    assert {variant.kind for variant in manifest.variants.values()} == {
        "alberta_horde_ac",
        "alberta_causal_map",
    }
    with pytest.raises(matrix.ForagerMatrixManifestError, match="kind is unknown"):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(variants={"rtu": _rtu()})
        )

    canonical_default = matrix.parse_forager_matrix_manifest(_manifest_payload())
    assert canonical_default.config_sha256 == (
        "41a267a4b3baf2f5cdd2b68e0d6136443cffd27f9d76b249af74d853051ef2de"
    )
    assert matrix._EXPECTED_MATRIX_RNG_CONTRACT_SHA256 == (
        "bbb6cf9a3cccd123ffa0f138cba37f85113eefd494d9148b89a796b371dda053"
    )
    assert matrix._json_sha256(matrix._matrix_rng_contract("2.2")) == (
        matrix._EXPECTED_MATRIX_RNG_CONTRACT_SHA256
    )
    assert "rtu_rtrl" not in matrix._matrix_rng_contract("2.2")["agent_isolation"]


def test_schema_23_normalizes_rtu_config_and_binds_disjoint_rng_identity() -> None:
    payload = _manifest_payload(
        schema_version="2.3",
        variants={
            "rtu": _rtu(
                {
                    "core": {
                        "hidden_size": 8,
                        "encoder_width": 4,
                        "output_width": 6,
                        "actor_alpha": 0,
                        "rtrl_taylor_correction": True,
                    },
                    "freeze_after_steps": 7,
                    "features": {
                        "include_hint": False,
                        "reward_trace_decays": [0.5, 0.9],
                    },
                }
            )
        },
    )
    manifest = matrix.parse_forager_matrix_manifest(payload)
    variant = manifest.variants["rtu"]
    assert variant.kind == matrix.RTU_RTRL_VARIANT_KIND
    assert isinstance(variant.config, RTURTRLForagerConfig)
    assert variant.config.core.n_actions == 4
    assert variant.config.core.hidden_size == 8
    assert variant.config.core.actor_alpha == 0.0
    assert variant.config.core.rtrl_taylor_correction is True
    assert variant.config.features.include_hint is False
    assert manifest.config_sha256 == (
        "fb658d2fc465fb06bae2181f101586e3a637433ee062382df9ec0d7c3f275aee"
    )
    assert variant.config_sha256 == (
        "1e70b0cf2fcd9285d56967924d539f90bc82b9b281750dd5220501e698047fd9"
    )
    assert variant.descriptor_sha256 == (
        "5d8282543b8fd229f711fd4a5503219e5b26f35c489a88ab7e39e31b6c2fa6a7"
    )
    normalized = manifest.to_dict()
    assert normalized["schema_version"] == "2.3"
    assert normalized["variants"]["rtu"]["config"] == variant.config.to_dict()
    reparsed = matrix.parse_forager_matrix_manifest(normalized)
    assert reparsed.to_dict() == normalized
    assert reparsed.config_sha256 == manifest.config_sha256

    contract = matrix._matrix_rng_contract("2.3")
    expected_rng = RTURTRLForagerAgent(
        RTURTRLForagerConfig(),
        seed=0,
    ).metadata()["agent_rng"]
    assert matrix._json_sha256(RTURTRLForagerConfig().to_dict()) == (
        "da49caa4a1fa6a10b1673955cb3c1a57a45b42294a413461a2bfe58c32cd2d31"
    )
    assert contract["agent_isolation"]["rtu_rtrl"] == expected_rng
    assert contract["agent_isolation"]["rtu_rtrl"]["environment_key_shared"] is False
    assert matrix._EXPECTED_MATRIX_RNG_CONTRACT_SHA256_2_3 == (
        "5e748169e2aad9cd4abf012293d6996392950341d8240d5c58f00e4268834ad7"
    )
    assert matrix._json_sha256(contract) == (
        matrix._EXPECTED_MATRIX_RNG_CONTRACT_SHA256_2_3
    )


def test_schema_24_is_the_fail_closed_adaptive_obgd_boundary() -> None:
    assert matrix.FORAGER_MATRIX_LATEST_SCHEMA_VERSION == "2.4"
    for adaptive_field, value in (
        ("adaptive_obgd", False),
        ("beta2", 0.999),
        ("epsilon", 1e-8),
    ):
        with pytest.raises(
            matrix.ForagerMatrixManifestError,
            match="require matrix schema '2.4'",
        ):
            matrix.parse_forager_matrix_manifest(
                _manifest_payload(
                    schema_version="2.3",
                    variants={
                        "rtu": _rtu({"core": {adaptive_field: value}})
                    },
                )
            )

    payload = _manifest_payload(
        schema_version="2.4",
        variants={
            "rtu": _rtu(
                {
                    "core": {
                        "hidden_size": 8,
                        "encoder_width": 4,
                        "output_width": 6,
                        "adaptive_obgd": True,
                        "beta2": 0.95,
                        "epsilon": 1e-6,
                    }
                }
            )
        },
    )
    manifest = matrix.parse_forager_matrix_manifest(payload)
    variant = manifest.variants["rtu"]
    assert isinstance(variant.config, RTURTRLForagerConfig)
    assert variant.config.core.adaptive_obgd is True
    assert variant.config.core.beta2 == 0.95
    assert variant.config.core.epsilon == 1e-6
    normalized = manifest.to_dict()
    normalized_core = normalized["variants"]["rtu"]["config"]["core"]
    assert normalized["schema_version"] == "2.4"
    assert normalized_core["adaptive_obgd"] is True
    assert normalized_core["beta2"] == 0.95
    assert normalized_core["epsilon"] == 1e-6
    assert matrix.parse_forager_matrix_manifest(normalized).to_dict() == normalized

    exact_defaults = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            schema_version="2.4",
            variants={
                "rtu": _rtu(
                    {
                        "core": {
                            "adaptive_obgd": True,
                            "beta2": 0.999,
                            "epsilon": 1e-8,
                        }
                    }
                )
            },
        )
    )
    exact_core = exact_defaults.to_dict()["variants"]["rtu"]["config"]["core"]
    assert exact_core["adaptive_obgd"] is True
    assert "beta2" not in exact_core
    assert "epsilon" not in exact_core

    contract = matrix._matrix_rng_contract("2.4")
    assert matrix._json_sha256(contract) == matrix._json_sha256(
        matrix._matrix_rng_contract("2.3")
    )
    assert matrix._json_sha256(contract) == (
        matrix._EXPECTED_MATRIX_RNG_CONTRACT_SHA256_2_4
    )
    assert matrix._EXPECTED_MATRIX_RNG_CONTRACT_SHA256_2_4 == (
        matrix._EXPECTED_MATRIX_RNG_CONTRACT_SHA256_2_3
    )

    bypassed = dataclasses.replace(manifest, schema_version="2.3")
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="adaptive ObGD.*schema '2.4'",
    ):
        matrix._preflight_manifest(
            bypassed,
            matrix._build_benchmark_config(bypassed),
        )


def test_schema_23_rtu_parser_fails_closed_on_core_and_resource_tamper() -> None:
    with pytest.raises(matrix.ForagerMatrixManifestError, match="exactly four actions"):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                schema_version="2.3",
                variants={"rtu": _rtu({"core": {"n_actions": 3}})},
            )
        )
    with pytest.raises(matrix.ForagerMatrixManifestError, match="core widths"):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                schema_version="2.3",
                variants={
                    "rtu": _rtu(
                        {"core": {"hidden_size": matrix._MAX_HIDDEN_WIDTH + 1}}
                    )
                },
            )
        )
    with pytest.raises(matrix.ForagerMatrixManifestError, match="unknown keys"):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                schema_version="2.3",
                variants={"rtu": _rtu({"core": {"unknown": 1}})},
            )
        )

    exact_core = RTURTRLForagerConfig().core
    uncorrected = dataclasses.replace(
        exact_core,
        hidden_size=3,
        encoder_width=5,
        output_width=7,
        rtrl_taylor_correction=False,
    )
    corrected = dataclasses.replace(uncorrected, rtrl_taylor_correction=True)
    adaptive = dataclasses.replace(uncorrected, adaptive_obgd=True)
    adaptive_corrected = dataclasses.replace(
        corrected,
        adaptive_obgd=True,
    )
    # For each actor/critic network: parameters and eligibility traces each
    # contain 2*E*H + 2*H*O elements, RTRL contains 4*E*H, and Taylor adds
    # another 4*E*H. This pins the factor-of-two output projection.
    assert matrix._rtu_persistent_product_elements(uncorrected) == 408
    assert matrix._rtu_persistent_product_elements(corrected) == 528
    assert matrix._rtu_persistent_product_elements(adaptive) == 552
    assert matrix._rtu_persistent_product_elements(adaptive_corrected) == 672

    accepted_without_taylor = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            schema_version="2.3",
            seed_batch_size=1,
            variants={
                "rtu": _rtu(
                    {
                        "core": {
                            "hidden_size": 2_048,
                            "encoder_width": 2_048,
                            "output_width": 2,
                            "rtrl_taylor_correction": False,
                        }
                    }
                )
            },
        )
    )
    assert isinstance(
        accepted_without_taylor.variants["rtu"].config,
        RTURTRLForagerConfig,
    )
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="RTU persistent product-element limit",
    ):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                schema_version="2.3",
                seed_batch_size=2,
                variants={
                    "rtu": _rtu(
                        {
                            "core": {
                                "hidden_size": 2_048,
                                "encoder_width": 2_048,
                                "output_width": 2,
                                "rtrl_taylor_correction": False,
                            }
                        }
                    )
                },
            )
        )
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="RTU persistent product-element limit",
    ):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                schema_version="2.3",
                seed_batch_size=1,
                variants={
                    "rtu": _rtu(
                        {
                            "core": {
                                "hidden_size": 2_048,
                                "encoder_width": 2_048,
                                "output_width": 2,
                                "rtrl_taylor_correction": True,
                            }
                        }
                    )
                },
            )
        )
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="RTU persistent product-element limit",
    ):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                schema_version="2.4",
                seed_batch_size=1,
                variants={
                    "rtu": _rtu(
                        {
                            "core": {
                                "hidden_size": 2_300,
                                "encoder_width": 2_300,
                                "output_width": 2,
                                "adaptive_obgd": True,
                            }
                        }
                    )
                },
            )
        )

    parsed_23 = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            schema_version="2.3",
            variants={"rtu": _rtu()},
        )
    )
    bypassed_22 = dataclasses.replace(parsed_23, schema_version="2.2")
    with pytest.raises(matrix.ForagerMatrixManifestError, match="requires matrix schema"):
        matrix._preflight_manifest(
            bypassed_22,
            matrix._build_benchmark_config(bypassed_22),
        )


def test_programmatic_manifest_is_canonically_reparsed_before_any_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            schema_version="2.3",
            seeds=[0],
            variants={"rtu": _rtu()},
        )
    )
    rtu_variant = parsed.variants["rtu"]
    assert isinstance(rtu_variant.config, RTURTRLForagerConfig)
    oversized_config = dataclasses.replace(
        rtu_variant.config,
        core=dataclasses.replace(
            rtu_variant.config.core,
            hidden_size=matrix._MAX_HIDDEN_WIDTH + 1,
        ),
    )
    oversized_variant = dataclasses.replace(
        rtu_variant,
        config=oversized_config,
    )
    invalid_manifests = (
        (
            "core widths",
            dataclasses.replace(parsed, variants={"rtu": oversized_variant}),
        ),
        (
            "duplicate seed",
            dataclasses.replace(parsed, seeds=(0, 0), tuning_seeds=(0, 0)),
        ),
        (
            "seed sets overlap",
            dataclasses.replace(parsed, evaluation_seeds=(0,)),
        ),
        (
            "mode must be",
            dataclasses.replace(parsed, mode="invalid"),  # type: ignore[arg-type]
        ),
        (
            "metric_evidence_mode must be",
            dataclasses.replace(
                parsed,
                metric_evidence_mode="invalid",  # type: ignore[arg-type]
            ),
        ),
    )
    for message, invalid in invalid_manifests:
        with pytest.raises(matrix.ForagerMatrixManifestError, match=message):
            matrix.run_forager_matrix(
                invalid,
                tmp_path / "must-not-start",
                dry_run=True,
            )

    source_path = tmp_path / "programmatic-manifest.json"
    with_source = dataclasses.replace(parsed, source_path=source_path)
    captured: list[matrix.ForagerMatrixManifest] = []

    def stop_after_reparse(
        canonical: matrix.ForagerMatrixManifest,
    ) -> ForagerBenchmarkConfig:
        captured.append(canonical)
        raise RuntimeError("stop after canonical reparse")

    monkeypatch.setattr(matrix, "_build_benchmark_config", stop_after_reparse)
    with pytest.raises(RuntimeError, match="stop after canonical reparse"):
        matrix.run_forager_matrix(
            with_source,
            tmp_path / "must-not-start-valid",
            dry_run=True,
        )
    assert len(captured) == 1
    assert captured[0].source_path == source_path
    assert captured[0].to_dict() == parsed.to_dict()


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"unknown": 1}, "unknown keys"),
        ({"steps": True}, "steps must be an integer"),
        ({"seeds": [0, True]}, r"seeds\[1\] must be an integer"),
        ({"seeds": [0, 0]}, "duplicate seed"),
        ({"mode": "sequential"}, "mode must be"),
        ({"variants": {"../escape": _horde()}}, "path-safe slug"),
        ({"variants": {"base": _horde(group="../bad")}}, "path-safe slug"),
        (
            {"variants": {"base": {"kind": "unknown", "selection_group": "x", "config": {}}}},
            "kind is unknown",
        ),
        (
            {"variants": {"base": {**_horde(), "extra": 1}}},
            "unknown keys",
        ),
        (
            {"variants": {"base": _horde({"bogus": 1})}},
            "unknown keys",
        ),
        (
            {"variants": {"base": _causal({"bogus": 1})}},
            "unknown keys",
        ),
        (
            {"selection_rule": _selection_rule(direction="largest")},
            "direction must be",
        ),
        (
            {"selection_rule": _selection_rule(statistic="median")},
            "statistic must be",
        ),
        (
            {"selection_rule": _selection_rule(confidence=1.0)},
            "confidence must",
        ),
        (
            {"selection_rule": _selection_rule(tie_break="config_hash")},
            "tie_break must",
        ),
        (
            {
                "stage": "evaluation",
                "seeds": [2],
                "tuning_seeds": [1, 2],
                "evaluation_seeds": [2, 3],
            },
            "seed sets overlap",
        ),
    ],
)
def test_manifest_rejects_unknown_keys_types_kinds_configs_and_rules(
    update: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(matrix.ForagerMatrixManifestError, match=message):
        matrix.parse_forager_matrix_manifest(_manifest_payload(**update))


@pytest.mark.parametrize("schema_version", ["1.0", "2.0", "2.1"])
def test_legacy_schema_is_rejected_with_unambiguous_migration_message(
    schema_version: str,
) -> None:
    legacy = _manifest_payload(schema_version=schema_version)
    with pytest.raises(matrix.ForagerMatrixManifestError, match="migrate to 2.2"):
        matrix.parse_forager_matrix_manifest(legacy)


def test_manifest_resource_bounds_and_signed_zero_are_canonical() -> None:
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="jax_chunk_size",
    ):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(jax_chunk_size=13)
        )
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="at most",
    ):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                seeds=list(range(matrix._MAX_SEED_COUNT + 1)),
            )
        )
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="confidence",
    ):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                selection_rule=_selection_rule(confidence=10**10_000),
            )
        )

    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            variants={"base": _horde({"actor_epsilon": -0.0})},
        )
    )
    epsilon = manifest.to_dict()["variants"]["base"]["config"][
        "actor_epsilon"
    ]
    assert epsilon == 0.0
    assert not bool(np.signbit(epsilon))


def test_strict_loader_rejects_duplicate_ids_nonfinite_and_unsafe_reference(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate_payload = _manifest_payload()
    variant = json.dumps(duplicate_payload["variants"]["base"])
    variants_object = json.dumps(duplicate_payload["variants"])
    duplicate.write_text(
        json.dumps(duplicate_payload).replace(
            variants_object,
            '{"base": ' + variant + ', "base": ' + variant + "}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(matrix.ForagerMatrixManifestError, match="duplicate JSON object key"):
        matrix.load_forager_matrix_manifest(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        json.dumps(_manifest_payload()).replace('"steps": 12', '"steps": NaN'),
        encoding="utf-8",
    )
    with pytest.raises(matrix.ForagerMatrixManifestError, match="non-finite"):
        matrix.load_forager_matrix_manifest(nonfinite)

    unsafe = _manifest_payload(
        stage="evaluation",
        seeds=[0],
        tuning_selection={
            "report_path": "../tuning/report.json",
            "file_sha256": "a" * 64,
            "selected_variants": {"base": "base"},
        },
    )
    with pytest.raises(matrix.ForagerMatrixManifestError, match="may not contain"):
        matrix.parse_forager_matrix_manifest(unsafe)


def test_source_snapshot_is_reproducible_canonical_and_path_independent(
    _isolated_source_tree: Path,
) -> None:
    first = matrix._build_source_snapshot()
    second = matrix._build_source_snapshot()
    assert first.archive_bytes == second.archive_bytes
    assert first.archive_sha256 == second.archive_sha256
    assert first.tree_sha256 == second.tree_sha256
    assert first.inventory_sha256 == second.inventory_sha256

    with tarfile.open(fileobj=io.BytesIO(first.archive_bytes), mode="r:") as archive:
        members = archive.getmembers()
        assert all(member.isfile() for member in members)
        assert all(
            not member.name.startswith("/") and ".." not in Path(member.name).parts
            for member in members
        )
        assert members[0].name == matrix.SOURCE_INVENTORY_MEMBER
        inventory_handle = archive.extractfile(members[0])
        assert inventory_handle is not None
        inventory = json.loads(inventory_handle.read())
        assert inventory == first.inventory
        archived_names = [member.name for member in members[1:]]
        inventory_names = [item["path"] for item in inventory["files"]]
        assert archived_names == inventory_names
        for member, item in zip(members[1:], inventory["files"], strict=True):
            member_handle = archive.extractfile(member)
            assert member_handle is not None
            contents = member_handle.read()
            assert len(contents) == item["size"]
            assert hashlib.sha256(contents).hexdigest() == item["sha256"]
            assert member.uid == member.gid == member.mtime == 0

    source = _isolated_source_tree / "alberta_framework/benchmarks/runner.py"
    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = matrix._build_source_snapshot()
    assert changed.tree_sha256 != first.tree_sha256
    assert changed.archive_sha256 != first.archive_sha256


def test_source_snapshot_rejects_links_and_special_entries(
    _isolated_source_tree: Path,
) -> None:
    target = _isolated_source_tree / "alberta_framework/benchmarks/runner.py"
    link = _isolated_source_tree / "alberta_framework/benchmarks/linked.py"
    link.symlink_to(target)
    with pytest.raises(matrix.ForagerMatrixStateError, match="contains symlink"):
        matrix._build_source_snapshot()
    link.unlink()
    fifo = _isolated_source_tree / "alberta_framework/benchmarks/special"
    os.mkfifo(fifo)
    with pytest.raises(matrix.ForagerMatrixStateError, match="special entry"):
        matrix._build_source_snapshot()


def test_dry_run_is_hash_bearing_and_does_not_create_output(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    output = tmp_path / "does-not-exist"

    plan = matrix.run_forager_matrix(
        matrix.parse_forager_matrix_manifest(_manifest_payload()),
        output,
        dry_run=True,
    )

    assert plan["dry_run"] is True
    assert plan["payload_sha256"] == _canonical_hash(plan)
    assert [item["seeds"] for item in plan["batch_plan"]] == [[5, 1], [9]]
    assert (
        plan["source_snapshot"]["archive_sha256"]
        == plan["execution_identity"]["source_archive_sha256"]
    )
    assert calls == []
    assert not output.exists()


def test_matrix_dispatches_mixed_kinds_and_hashes_every_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls)
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            variants={
                "zeta": _causal({"retry_penalty": 0.4}),
                "alpha": _horde({"actor_hidden_sizes": [8]}),
            }
        )
    )
    output = tmp_path / "matrix"

    report = matrix.run_forager_matrix(manifest, output)

    assert [(kind, seeds) for kind, seeds, _ in calls] == [
        ("alberta_horde_ac", (5, 1)),
        ("alberta_horde_ac", (9,)),
        ("alberta_causal_map", (5, 1)),
        ("alberta_causal_map", (9,)),
    ]
    assert report["variants"]["alpha"]["kind"] == "alberta_horde_ac"
    assert report["variants"]["zeta"]["kind"] == "alberta_causal_map"
    assert report["variants"]["zeta"]["summary"]["agent"] == "alberta_causal_map"
    assert report["selection_results"]["groups"]["policy"]["ranked_variants"]
    assert (output / matrix.SOURCE_SNAPSHOT_FILENAME).read_bytes()

    artifact_paths = [
        output / matrix.EXECUTION_MANIFEST_FILENAME,
        *(output / item["path"] for item in report["batch_artifacts"]),
        output / matrix.FINAL_REPORT_FILENAME,
    ]
    for artifact_path in artifact_paths:
        raw = artifact_path.read_bytes()
        payload = json.loads(raw)
        assert (
            raw
            == (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode()
        )
        assert payload["payload_sha256"] == _canonical_hash(payload)


@pytest.mark.parametrize("mode", ["strict", "vmap"])
def test_schema_23_rtu_dispatches_batches_summarizes_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: ForagerBatchMode,
) -> None:
    calls: list[tuple[tuple[int, ...], ForagerBatchMode]] = []
    _install_rtu_fake_runner(monkeypatch, calls)
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            schema_version="2.3",
            seeds=[2, 4, 6],
            seed_batch_size=2,
            mode=mode,
            variants={
                "rtu": _rtu(
                    {
                        "core": {
                            "hidden_size": 4,
                            "encoder_width": 2,
                            "output_width": 2,
                        }
                    }
                )
            },
        )
    )
    output = tmp_path / f"rtu-{mode}"
    report = matrix.run_forager_matrix(manifest, output)

    assert calls == [((2, 4), mode), ((6,), mode)]
    assert report["schema_version"] == "2.3"
    assert report["variants"]["rtu"]["kind"] == matrix.RTU_RTRL_VARIANT_KIND
    assert report["variants"]["rtu"]["summary"]["agent"] == (
        matrix.RTU_RTRL_RESULT_AGENT
    )
    assert report["variants"]["rtu"]["summary"]["seeds"] == [2, 4, 6]
    assert report["selection_results"]["groups"]["policy"][
        "selected_variant_id"
    ] == "rtu"
    assert json.loads(
        (output / matrix.EXECUTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )["schema_version"] == "2.3"
    assert json.loads(
        (output / "batches/rtu/batch-00000.json").read_text(encoding="utf-8")
    )["schema_version"] == "2.3"

    calls.clear()
    assert matrix.run_forager_matrix(manifest, output) == report
    assert calls == []


def test_schema_23_rtu_metadata_tamper_fails_closed_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], ForagerBatchMode]] = []
    _install_rtu_fake_runner(monkeypatch, calls)
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            schema_version="2.3",
            seeds=[0],
            variants={"rtu": _rtu()},
        )
    )
    output = tmp_path / "rtu-metadata-tamper"
    matrix.run_forager_matrix(manifest, output)
    batch = output / "batches/rtu/batch-00000.json"

    def alter_rng(payload: dict[str, Any]) -> None:
        payload["runs"][0]["agent_metadata"]["agent_rng"]["namespace"] += 1

    _rewrite_hashed_json(batch, alter_rng)
    with pytest.raises(
        matrix.ForagerMatrixStateError,
        match="RTU/RTRL metadata agent_rng",
    ):
        matrix.run_forager_matrix(manifest, output)


def test_schema_23_tuning_report_validator_binds_distinct_result_identity() -> None:
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            schema_version="2.3",
            seeds=[0],
            variants={"rtu": _rtu()},
        )
    )
    variant = manifest.variants["rtu"]
    entry = {
        "kind": variant.kind,
        "selection_group": variant.selection_group,
        "config": variant.config.to_dict(),
        "config_sha256": variant.config_sha256,
        "variant_sha256": variant.descriptor_sha256,
        "seeds": list(manifest.seeds),
        "seed_batches": [],
        "summary": {
            "agent": matrix.RTU_RTRL_RESULT_AGENT,
            "privileged": False,
            "seeds": sorted(manifest.seeds),
            "metric": manifest.selection_rule.metric,
            "mean": 1.0,
            "ci_low": 1.0,
            "ci_high": 1.0,
            "confidence": manifest.selection_rule.confidence,
            "bootstrap_resamples": manifest.selection_rule.bootstrap_resamples,
            "bootstrap_seed": manifest.selection_rule.bootstrap_seed,
        },
    }
    validated = matrix._validate_report_variants_for_selection(
        manifest,
        {"rtu": entry},
    )
    assert validated["rtu"]["summary"]["agent"] == matrix.RTU_RTRL_RESULT_AGENT

    relabelled = copy.deepcopy(entry)
    relabelled["summary"]["agent"] = matrix.RTU_RTRL_VARIANT_KIND
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="summary is not bound",
    ):
        matrix._validate_report_variants_for_selection(
            manifest,
            {"rtu": relabelled},
        )


def test_schema_23_snapshot_worker_dispatches_rtu_with_raw_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], ForagerBatchMode]] = []
    _install_rtu_fake_runner(monkeypatch, calls, raw_trace=True)
    monkeypatch.setattr(matrix, "_assert_framework_modules_from_repo_root", lambda: None)
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            schema_version="2.3",
            steps=5,
            seeds=[3],
            jax_chunk_size=2,
            seed_batch_size=1,
            source_execution_mode=matrix.SNAPSHOT_SOURCE_EXECUTION_MODE,
            metric_evidence_mode="raw_reward_npz_v2",
            variants={"rtu": _rtu()},
        )
    )
    exchange = tmp_path / "exchange"
    exchange.mkdir()
    request = {
        "schema_version": matrix._IMMUTABLE_WORKER_SCHEMA,
        "source_tree_sha256": matrix._source_tree_sha256(),
        "matrix_config": manifest.to_dict(),
        "variant_id": "rtu",
        "seeds": [3],
    }
    stdout = io.BytesIO()
    with monkeypatch.context() as isolated:
        isolated.setattr(
            sys,
            "argv",
            ["worker", str(tmp_path), str(exchange)],
        )
        isolated.setattr(
            sys,
            "stdin",
            SimpleNamespace(buffer=io.BytesIO(matrix._canonical_json_bytes(request))),
        )
        isolated.setattr(
            sys,
            "stdout",
            SimpleNamespace(buffer=stdout),
        )
        status = matrix._immutable_worker_main()

    assert status == 0
    assert calls == [((3,), "strict")]
    response = json.loads(stdout.getvalue())
    assert response["payload_sha256"] == _canonical_hash(response)
    assert response["variant_id"] == "rtu"
    assert response["runs"][0]["agent"] == matrix.RTU_RTRL_RESULT_AGENT
    assert response["runs"][0]["agent_metadata"]["raw_metric_trace"][
        "exchange_file"
    ] == "seed-3.npz"
    assert (exchange / "seed-3.npz").is_file()


@pytest.mark.parametrize(
    "tamper",
    ["kind", "config", "environment", "metric", "rng", "seed"],
)
def test_new_result_binding_rejects_relabel_and_metadata_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    config = AlbertaForagerConfig()

    def bad_runner(
        _config: AlbertaForagerConfig,
        benchmark: ForagerBenchmarkConfig,
        seeds: Sequence[int],
        *,
        mode: ForagerBatchMode,
        reward_trace_sink_factory: Any = None,
    ) -> tuple[ForagerRunResult, ...]:
        assert reward_trace_sink_factory is None
        result = _fake_result(
            "alberta_horde_ac",
            config,
            benchmark,
            seeds[0],
            batch_seeds=seeds,
            mode=mode,
            value=1.0,
        )
        payload = result.to_dict()
        if tamper == "kind":
            payload["agent"] = "alberta_causal_map"
        elif tamper == "config":
            payload["agent_metadata"]["config"]["gamma"] = 0.5
        elif tamper == "environment":
            payload["environment"]["aperture_size"] = 3
        elif tamper == "metric":
            payload["metric_contract"]["ewm_decay"] = 0.5
        elif tamper == "rng":
            payload["agent_metadata"]["environment_rng_schedule"] = "legacy_shared_key"
        else:
            payload["agent_metadata"]["seed"] = seeds[0] + 1
        return (
            matrix._run_from_payload(
                payload,
                path="fixture",
                expected_seed=seeds[0],
                expected_steps=benchmark.steps,
                expected_kind="alberta_horde_ac",
                expected_config=config.to_dict(),
                expected_environment=benchmark.environment.to_dict(),
                expected_metric_contract=forager_metric_contract(
                    ewm_decay=benchmark.ewm_decay,
                    final_window=benchmark.final_window,
                    record_every=benchmark.record_every,
                    steps=benchmark.steps,
                ),
                expected_chunk_size=benchmark.jax_chunk_size,
                expected_mode=mode,
                expected_batch_seeds=seeds,
            ),
        )

    monkeypatch.setattr(matrix, "run_alberta_forager_seeds", bad_runner)
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0]))
    with pytest.raises(matrix.ForagerMatrixStateError):
        matrix.run_forager_matrix(manifest, tmp_path / tamper)


def test_interrupted_run_resumes_only_missing_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(seeds=[0, 1, 2], seed_batch_size=1)
    )
    output = tmp_path / "resume"
    first_calls: list[tuple[int, ...]] = []

    def interrupted(
        config: AlbertaForagerConfig,
        benchmark: ForagerBenchmarkConfig,
        seeds: Sequence[int],
        *,
        mode: ForagerBatchMode,
        reward_trace_sink_factory: Any = None,
    ) -> tuple[ForagerRunResult, ...]:
        assert reward_trace_sink_factory is None
        ordered = tuple(seeds)
        first_calls.append(ordered)
        if ordered == (1,):
            raise RuntimeError("simulated interruption")
        return tuple(
            _fake_result(
                "alberta_horde_ac",
                config,
                benchmark,
                seed,
                batch_seeds=ordered,
                mode=mode,
                value=float(seed + 1),
            )
            for seed in ordered
        )

    monkeypatch.setattr(matrix, "run_alberta_forager_seeds", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        matrix.run_forager_matrix(manifest, output)
    assert first_calls == [(0,), (1,)]
    assert (output / matrix.SOURCE_SNAPSHOT_FILENAME).is_file()
    assert (output / "batches/base/batch-00000.json").is_file()
    assert not (output / matrix.FINAL_REPORT_FILENAME).exists()

    resumed_calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, resumed_calls)
    report = matrix.run_forager_matrix(manifest, output)
    assert [call[1] for call in resumed_calls] == [(1,), (2,)]
    assert report["variants"]["base"]["summary"]["seeds"] == [0, 1, 2]

    resumed_calls.clear()
    repeated = matrix.run_forager_matrix(manifest, output)
    assert resumed_calls == []
    assert repeated == report


def test_snapshot_only_atomic_prefix_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls)
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0]))
    output = tmp_path / "snapshot-prefix"
    output.mkdir()
    snapshot = matrix._build_source_snapshot()
    matrix._atomic_create_bytes(
        output / matrix.SOURCE_SNAPSHOT_FILENAME,
        snapshot.archive_bytes,
    )

    report = matrix.run_forager_matrix(manifest, output)

    assert report["status"] == "complete"
    assert (output / matrix.EXECUTION_MANIFEST_FILENAME).is_file()
    assert calls and calls[0][1] == (0,)


def test_resumed_batch_rejects_rehashed_kind_and_config_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls)
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0]))

    for name, mutation, message in (
        (
            "descriptor-kind",
            lambda payload: payload["variant"].__setitem__("kind", "alberta_causal_map"),
            "mismatched variant",
        ),
        (
            "kind",
            lambda payload: payload["runs"][0].__setitem__("agent", "alberta_causal_map"),
            "agent kind",
        ),
        (
            "config",
            lambda payload: payload["runs"][0]["agent_metadata"]["config"].__setitem__(
                "gamma", 0.5
            ),
            "configuration",
        ),
    ):
        output = tmp_path / name
        matrix.run_forager_matrix(manifest, output)
        batch_path = output / "batches/base/batch-00000.json"
        _rewrite_hashed_json(batch_path, mutation)
        with pytest.raises(matrix.ForagerMatrixStateError, match=message):
            matrix.run_forager_matrix(manifest, output)


def test_ranking_mean_conservative_ci_tie_break_and_multiple_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        0.11: (5.0, 5.0, 5.0, 5.0),
        0.22: (0.0, 0.0, 0.0, 30.0),
        0.33: (2.0, 2.0, 2.0, 2.0),
        0.44: (4.0, 4.0, 4.0, 4.0),
    }

    def value(_kind: str, config: VariantConfig, seed: int) -> float:
        assert isinstance(config, AlbertaForagerConfig)
        return values[round(config.actor_epsilon, 2)][seed]

    variants = {
        "stable": _horde({"actor_epsilon": 0.11}, group="reward"),
        "risky": _horde({"actor_epsilon": 0.22}, group="reward"),
        "alpha": _horde({"actor_epsilon": 0.33}, group="tie"),
        "zeta": _horde({"actor_epsilon": 0.33}, group="tie"),
        "lower": _horde({"actor_epsilon": 0.33}, group="other"),
        "upper": _horde({"actor_epsilon": 0.44}, group="other"),
    }
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls, value=value)
    mean_manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            seeds=[0, 1, 2, 3],
            seed_batch_size=4,
            variants=variants,
            selection_rule=_selection_rule(
                statistic="mean",
                confidence=0.95,
                bootstrap_resamples=2_000,
            ),
        )
    )
    mean_report = matrix.run_forager_matrix(mean_manifest, tmp_path / "mean")
    mean_groups = mean_report["selection_results"]["groups"]
    assert mean_groups["reward"]["selected_variant_id"] == "risky"
    assert mean_groups["tie"]["selected_variant_id"] == "alpha"
    assert mean_groups["other"]["selected_variant_id"] == "upper"
    assert [row["rank"] for row in mean_groups["reward"]["ranked_variants"]] == [1, 2]

    conservative_manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            seeds=[0, 1, 2, 3],
            seed_batch_size=4,
            variants=variants,
            selection_rule=_selection_rule(
                statistic="conservative_ci_endpoint",
                confidence=0.95,
                bootstrap_resamples=2_000,
            ),
        )
    )
    conservative = matrix.run_forager_matrix(
        conservative_manifest,
        tmp_path / "conservative",
    )
    assert conservative["selection_results"]["groups"]["reward"]["selected_variant_id"] == "stable"

    minimize_manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            seeds=[0, 1, 2, 3],
            seed_batch_size=4,
            variants={
                "lower": variants["lower"],
                "upper": variants["upper"],
            },
            selection_rule=_selection_rule(direction="minimize"),
        )
    )
    minimized = matrix.run_forager_matrix(minimize_manifest, tmp_path / "minimize")
    assert minimized["selection_results"]["groups"]["other"]["selected_variant_id"] == "lower"


def test_report_validation_recomputes_complete_ranking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls)
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            seeds=[0],
            variants={"alpha": _horde(), "zeta": _horde()},
        )
    )
    output = tmp_path / "rank-tamper"
    matrix.run_forager_matrix(manifest, output)
    report_path = output / matrix.FINAL_REPORT_FILENAME

    def reverse_winner(payload: dict[str, Any]) -> None:
        payload["selection_results"]["groups"]["policy"]["selected_variant_id"] = "zeta"

    _rewrite_hashed_json(report_path, reverse_winner)
    with pytest.raises(matrix.ForagerMatrixStateError, match="selection_results"):
        matrix.run_forager_matrix(manifest, output)


def test_evaluation_requires_declared_top_ranked_matching_group_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def value(_kind: str, config: VariantConfig, _seed: int) -> float:
        assert isinstance(config, AlbertaForagerConfig)
        return config.actor_epsilon

    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls, value=value)
    tuning_payload = _manifest_payload(
        seeds=[10, 11],
        evaluation_seeds=[20],
        seed_batch_size=2,
        variants={
            "loser": _horde({"actor_epsilon": 0.1}, group="policy"),
            "winner": _horde({"actor_epsilon": 0.2}, group="policy"),
        },
    )
    tuning_input = tmp_path / "tuning-input.json"
    tuning_input.write_text(json.dumps(tuning_payload), encoding="utf-8")
    tuning_output = tmp_path / "tuning"
    tuning_report = matrix.run_forager_matrix(tuning_input, tuning_output)
    assert tuning_report["selection_results"]["groups"]["policy"]["selected_variant_id"] == "winner"
    report_path = tuning_output / matrix.FINAL_REPORT_FILENAME
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()

    loser_evaluation = _manifest_payload(
        stage="evaluation",
        seeds=[20],
        tuning_seeds=[10, 11],
        evaluation_seeds=[20],
        variants={
            "champion": _horde({"actor_epsilon": 0.1}, group="policy"),
        },
        tuning_selection={
            "report_path": "tuning/report.json",
            "file_sha256": report_sha,
            "selected_variants": {"champion": "loser"},
        },
    )
    loser_input = tmp_path / "loser-evaluation.json"
    loser_input.write_text(json.dumps(loser_evaluation), encoding="utf-8")
    with pytest.raises(matrix.ForagerMatrixManifestError, match="top-ranked winner"):
        matrix.run_forager_matrix(
            loser_input,
            tmp_path / "loser-output",
            dry_run=True,
        )

    winner_evaluation = copy.deepcopy(loser_evaluation)
    winner_evaluation["variants"] = {
        "champion": _horde({"actor_epsilon": 0.2}, group="policy"),
    }
    winner_evaluation["tuning_selection"]["selected_variants"] = {"champion": "winner"}
    winner_input = tmp_path / "winner-evaluation.json"
    winner_input.write_text(json.dumps(winner_evaluation), encoding="utf-8")
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="host/snapshot tuning evidence cannot authorize",
    ):
        matrix.run_forager_matrix(
            winner_input,
            tmp_path / "winner-output",
            dry_run=True,
        )


def test_evaluation_recomputes_referenced_ranking_instead_of_trusting_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls)
    tuning_input = tmp_path / "tuning-input.json"
    tuning_input.write_text(
        json.dumps(
            _manifest_payload(
                seeds=[0],
                evaluation_seeds=[20],
                variants={"alpha": _horde(), "zeta": _horde()},
            )
        ),
        encoding="utf-8",
    )
    tuning_output = tmp_path / "tuning"
    matrix.run_forager_matrix(tuning_input, tuning_output)
    report_path = tuning_output / matrix.FINAL_REPORT_FILENAME

    def forge(payload: dict[str, Any]) -> None:
        payload["selection_results"]["groups"]["policy"]["selected_variant_id"] = "zeta"

    _rewrite_hashed_json(report_path, forge)
    evaluation = _manifest_payload(
        stage="evaluation",
        seeds=[20],
        tuning_seeds=[0],
        evaluation_seeds=[20],
        variants={"candidate": _horde()},
        tuning_selection={
            "report_path": "tuning/report.json",
            "file_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "selected_variants": {"candidate": "zeta"},
        },
    )
    evaluation_input = tmp_path / "evaluation.json"
    evaluation_input.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="mismatched selection_results|does not recompute",
    ):
        matrix.run_forager_matrix(
            evaluation_input,
            tmp_path / "evaluation-output",
            dry_run=True,
        )


def test_snapshot_tamper_live_source_change_and_extra_directory_fail_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_source_tree: Path,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls)
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0]))

    tampered_output = tmp_path / "snapshot-tamper"
    matrix.run_forager_matrix(manifest, tampered_output)
    snapshot_path = tampered_output / matrix.SOURCE_SNAPSHOT_FILENAME
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b"tamper")
    with pytest.raises(matrix.ForagerMatrixStateError, match="source snapshot"):
        matrix.run_forager_matrix(manifest, tampered_output)

    source_output = tmp_path / "source-change"
    matrix.run_forager_matrix(manifest, source_output)
    source = _isolated_source_tree / "alberta_framework/benchmarks/runner.py"
    source.write_text("VALUE = 99\n", encoding="utf-8")
    with pytest.raises(matrix.ForagerMatrixStateError, match="source snapshot"):
        matrix.run_forager_matrix(manifest, source_output)

    source.write_text("VALUE = 1\n", encoding="utf-8")
    extra_output = tmp_path / "extra"
    matrix.run_forager_matrix(manifest, extra_output)
    (extra_output / "unexpected").mkdir()
    with pytest.raises(matrix.ForagerMatrixStateError, match="unexpected directory"):
        matrix.run_forager_matrix(manifest, extra_output)


def test_source_edit_during_batch_refuses_batch_and_final_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_source_tree: Path,
) -> None:
    source = _isolated_source_tree / "alberta_framework/benchmarks/runner.py"

    def edits_source(
        config: AlbertaForagerConfig,
        benchmark: ForagerBenchmarkConfig,
        seeds: Sequence[int],
        *,
        mode: ForagerBatchMode,
        reward_trace_sink_factory: Any = None,
    ) -> tuple[ForagerRunResult, ...]:
        assert reward_trace_sink_factory is None
        ordered = tuple(seeds)
        results = tuple(
            _fake_result(
                "alberta_horde_ac",
                config,
                benchmark,
                seed,
                batch_seeds=ordered,
                mode=mode,
                value=1.0,
            )
            for seed in ordered
        )
        source.write_text("VALUE = 8\n", encoding="utf-8")
        return results

    monkeypatch.setattr(matrix, "run_alberta_forager_seeds", edits_source)
    output = tmp_path / "source-race"
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0]))

    with pytest.raises(matrix.ForagerMatrixStateError, match="source tree changed"):
        matrix.run_forager_matrix(manifest, output)

    assert (output / matrix.SOURCE_SNAPSHOT_FILENAME).is_file()
    assert (output / matrix.EXECUTION_MANIFEST_FILENAME).is_file()
    assert not (output / "batches/base/batch-00000.json").exists()
    assert not (output / matrix.FINAL_REPORT_FILENAME).exists()


@pytest.mark.parametrize("random_trace", [False, True])
def test_raw_metric_sidecars_recompute_every_evaluator_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    random_trace: bool,
) -> None:
    calls: list[tuple[int, ...]] = []
    _install_raw_trace_runner(
        monkeypatch,
        calls,
        random_trace=random_trace,
    )
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            steps=257,
            seeds=[3, 8],
            jax_chunk_size=17,
            seed_batch_size=2,
            metric_evidence_mode="raw_reward_npz_v2",
        )
    )
    output = tmp_path / ("random-raw" if random_trace else "constant-raw")

    report = matrix.run_forager_matrix(manifest, output)

    assert calls == [(3, 8)]
    assert report["evidence_eligibility"] == {
        "schema_version": "alberta.forager_evidence_eligibility.v1",
        "source_immutable": False,
        "runtime_binding_mode": "host_runtime_inventory_advisory",
        "runtime_immutable": False,
        "metric_evidence_mode": "raw_reward_npz_v2",
        "raw_metric_evidence_complete": True,
        "sealed_eligible": False,
        "unsealed_reasons": [
            "source_executed_from_live_tree",
            "host_runtime_inventory_is_advisory",
        ],
    }
    batch = json.loads(
        (output / "batches/base/batch-00000.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = batch["metric_evidence"]
    assert evidence["all_reported_evaluator_metrics_recomputable"] is True
    assert evidence["runtime_immutable"] is False
    assert evidence["sealed_eligible"] is False
    assert len(evidence["raw_metric_sidecars"]) == 2
    for sidecar in evidence["raw_metric_sidecars"]:
        path = output / sidecar["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sidecar["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert archive.files == ["rewards", "biome_regrets"]
            assert archive["rewards"].dtype == np.dtype("<f4")
            assert archive["biome_regrets"].dtype == np.dtype("<f4")
            assert archive["rewards"].shape == (257,)
            assert np.all(np.isfinite(archive["biome_regrets"]))

    calls.clear()
    assert matrix.run_forager_matrix(manifest, output) == report
    assert calls == []


def test_raw_sidecar_tamper_and_lossy_sink_input_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = tmp_path / "lossy-exchange"
    exchange.mkdir()
    sink = matrix._NpzMetricTraceSink(exchange, 1, 1)
    try:
        with pytest.raises(
            matrix.ForagerMatrixStateError,
            match="finite float32",
        ):
            sink.append(
                np.asarray([1.0], dtype=np.float64),
                np.asarray([0.0], dtype=np.float32),
            )
    finally:
        sink.abort()

    calls: list[tuple[int, ...]] = []
    _install_raw_trace_runner(monkeypatch, calls, random_trace=True)
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            steps=33,
            seeds=[4],
            jax_chunk_size=8,
            metric_evidence_mode="raw_reward_npz_v2",
        )
    )
    output = tmp_path / "raw-tamper"
    matrix.run_forager_matrix(manifest, output)
    sidecar = output / "reward-traces/base/batch-00000/seed-4.npz"
    encoded = bytearray(sidecar.read_bytes())
    encoded[len(encoded) // 2] ^= 0x01
    sidecar.write_bytes(encoded)

    with pytest.raises(matrix.ForagerMatrixStateError, match="digest mismatch"):
        matrix.run_forager_matrix(manifest, output)


def test_npz_trace_close_attempts_mapping_close_after_flush_failure() -> None:
    events: list[str] = []

    class FailingFlush:
        class Mapping:
            def close(self) -> None:
                events.append("close")

        _mmap = Mapping()

        def flush(self) -> None:
            events.append("flush")
            raise OSError("injected flush failure")

    with pytest.raises(OSError, match="injected flush failure"):
        matrix._NpzMetricTraceSink._close_memmap(FailingFlush())  # type: ignore[arg-type]
    assert events == ["flush", "close"]


@pytest.mark.parametrize("failure_call", [1, 2])
def test_npz_trace_constructor_failure_removes_partial_exchange_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    exchange = tmp_path / f"constructor-failure-{failure_call}"
    exchange.mkdir()
    original_open_memmap = matrix.np.lib.format.open_memmap
    calls = 0

    def fail_selected_creation(
        filename: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> np.memmap:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            Path(filename).write_bytes(b"injected partial NPY")
            raise OSError(f"injected creation failure {failure_call}")
        return original_open_memmap(filename, *args, **kwargs)

    monkeypatch.setattr(matrix.np.lib.format, "open_memmap", fail_selected_creation)
    with pytest.raises(OSError, match=f"injected creation failure {failure_call}"):
        matrix._NpzMetricTraceSink(exchange, 13, 1)

    assert list(exchange.iterdir()) == []


def test_npz_trace_finalize_close_failure_removes_every_exchange_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = tmp_path / "close-failure-exchange"
    exchange.mkdir()
    sink = matrix._NpzMetricTraceSink(exchange, 17, 1)
    sink.append(
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
    )
    original_close = matrix._NpzMetricTraceSink._close_memmap
    calls = 0

    def fail_first_close(value: np.memmap | None) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected close failure")
        original_close(value)

    monkeypatch.setattr(
        matrix._NpzMetricTraceSink,
        "_close_memmap",
        staticmethod(fail_first_close),
    )
    with pytest.raises(OSError, match="injected close failure"):
        sink.finalize()

    assert calls == 3
    assert list(exchange.iterdir()) == []


def test_npz_trace_finalize_post_replace_failure_is_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = tmp_path / "replace-failure-exchange"
    exchange.mkdir()
    sink = matrix._NpzMetricTraceSink(exchange, 23, 2)
    sink.append(
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([0.0, -1.0], dtype=np.float32),
    )
    original_replace = os.replace

    def replace_then_fail(source: str | bytes | Path, target: str | bytes | Path) -> None:
        original_replace(source, target)
        raise OSError("injected post-replace failure")

    monkeypatch.setattr(matrix.os, "replace", replace_then_fail)
    with pytest.raises(OSError, match="injected post-replace failure"):
        sink.finalize()

    assert list(exchange.iterdir()) == []


def test_npz_trace_multi_sink_failure_removes_already_finalized_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = tmp_path / "multi-sink-failure-exchange"
    exchange.mkdir()
    first = matrix._NpzMetricTraceSink(exchange, 29, 1)
    second = matrix._NpzMetricTraceSink(exchange, 31, 1)
    for sink in (first, second):
        sink.append(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        )

    def fail_second_finalize() -> Mapping[str, Any]:
        raise OSError("injected second-sink failure")

    monkeypatch.setattr(second, "finalize", fail_second_finalize)
    with pytest.raises(OSError, match="injected second-sink failure"):
        _finalize_reward_trace_sinks((first, second))

    assert list(exchange.iterdir()) == []


def test_schema_23_real_tiny_rtu_run_captures_and_recomputes_raw_trace(
    tmp_path: Path,
) -> None:
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            schema_version="2.3",
            steps=4,
            seeds=[0],
            jax_chunk_size=2,
            seed_batch_size=1,
            mode="strict",
            metric_evidence_mode="raw_reward_npz_v2",
            variants={
                "rtu": _rtu(
                    {
                        "core": {
                            "hidden_size": 2,
                            "encoder_width": 2,
                            "output_width": 2,
                            "actor_lamda": 0.0,
                            "critic_lamda": 0.0,
                            "actor_alpha": 0.05,
                            "critic_alpha": 0.05,
                            "entropy_coefficient": 0.0,
                            "normalize_observations": False,
                            "normalize_rewards": False,
                        }
                    }
                )
            },
        )
    )
    output = tmp_path / "real-rtu"
    report = matrix.run_forager_matrix(manifest, output)

    assert report["status"] == "complete"
    assert report["evidence_eligibility"]["raw_metric_evidence_complete"] is True
    assert report["variants"]["rtu"]["summary"]["agent"] == (
        matrix.RTU_RTRL_RESULT_AGENT
    )
    sidecar = output / "reward-traces/rtu/batch-00000/seed-0.npz"
    with np.load(sidecar, allow_pickle=False) as archive:
        assert archive["rewards"].shape == (4,)
        assert archive["rewards"].dtype == np.dtype("<f4")
        assert archive["biome_regrets"].shape == (4,)

    encoded = bytearray(sidecar.read_bytes())
    encoded[len(encoded) // 2] ^= 0x01
    sidecar.write_bytes(encoded)
    with pytest.raises(matrix.ForagerMatrixStateError, match="digest mismatch"):
        matrix.run_forager_matrix(manifest, output)


def test_crash_orphaned_sidecar_is_adopted_only_by_exact_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, ...]] = []
    _install_raw_trace_runner(monkeypatch, calls, random_trace=True)
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            steps=65,
            seeds=[6],
            jax_chunk_size=16,
            metric_evidence_mode="raw_reward_npz_v2",
        )
    )
    output = tmp_path / "orphan-adoption"
    matrix.run_forager_matrix(manifest, output)
    sidecar = output / "reward-traces/base/batch-00000/seed-6.npz"
    sidecar_identity = (
        sidecar.stat().st_ino,
        hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    )
    (output / matrix.FINAL_REPORT_FILENAME).unlink()
    (output / "batches/base/batch-00000.json").unlink()
    calls.clear()

    report = matrix.run_forager_matrix(manifest, output)

    assert report["status"] == "complete"
    assert calls == [(6,)]
    assert (
        sidecar.stat().st_ino,
        hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    ) == sidecar_identity


def test_bound_output_detects_parent_and_ancestor_replacement(
    tmp_path: Path,
) -> None:
    ancestor = tmp_path / "advertised"
    output = ancestor / "nested" / "output"
    bound = matrix._open_bound_directory(output, create=True)
    moved = tmp_path / "moved"
    try:
        ancestor.rename(moved)
        output.mkdir(parents=True)
        with pytest.raises(
            matrix.ForagerMatrixStateError,
            match="ancestor",
        ):
            bound.assert_bound()
    finally:
        bound.close()


def test_bound_output_detects_lock_replacement(tmp_path: Path) -> None:
    output = tmp_path / "lock-replacement"
    with pytest.raises(
        matrix.ForagerMatrixStateError,
        match="output lock was replaced",
    ):
        with matrix._output_lock(output):
            lock_path = output / matrix.LOCK_FILENAME
            lock_path.rename(tmp_path / "moved-lock")
            lock_path.write_text("replacement\n", encoding="utf-8")


def test_protocol_conformance_binds_selection_evidence_rng_and_runtime() -> None:
    conforming_rule = _selection_rule(
        statistic="mean",
        confidence=0.95,
        bootstrap_resamples=10_000,
        bootstrap_seed=0,
    )
    manifest = matrix.parse_forager_matrix_manifest(
        _manifest_payload(
            source_execution_mode="content_verified_snapshot_subprocess_unsealed",
            metric_evidence_mode="raw_reward_npz_v2",
            selection_rule=conforming_rule,
        )
    )

    conformance = matrix._protocol_conformance(manifest, None)

    assert conformance["selection_statistic_conformant"] is True
    assert conformance["bootstrap_resamples_conformant"] is True
    assert conformance["bootstrap_seed_conformant"] is True
    assert conformance["tie_break_conformant"] is True
    assert conformance["metric_evidence_conformant"] is True
    assert conformance["rng_schedule_conformant"] is True
    assert conformance["seed_labels_alone_authorize_paired_inference"] is False
    assert conformance["immutable_source_execution"] is False
    assert conformance["runtime_immutable"] is False
    assert conformance["full_paper_protocol_conformant"] is False


def test_nonempty_invalid_state_symlinks_and_lock_are_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload())
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    user_file = occupied / "keep.txt"
    user_file.write_text("important\n", encoding="utf-8")
    with pytest.raises(matrix.ForagerMatrixStateError, match="unexpected file"):
        matrix.run_forager_matrix(manifest, occupied)
    assert user_file.read_text(encoding="utf-8") == "important\n"

    linked = tmp_path / "linked"
    target = tmp_path / "target"
    target.mkdir()
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(matrix.ForagerMatrixStateError, match="may not be a symlink"):
        matrix.run_forager_matrix(manifest, linked)

    locked = tmp_path / "locked"
    locked.mkdir()
    lock_path = locked / matrix.LOCK_FILENAME
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(matrix.ForagerMatrixLockedError):
            matrix.run_forager_matrix(manifest, locked)


def test_raw_trace_v2_rejects_zip_overlays_metadata_and_alternate_deflate(
    tmp_path: Path,
) -> None:
    steps = 2_048
    exchange = tmp_path / "exchange"
    exchange.mkdir()
    sink = matrix._NpzMetricTraceSink(exchange, 7, steps)
    values = np.resize(np.arange(16, dtype=np.float32), steps)
    sink.append(values, -values)
    descriptor = sink.finalize()
    canonical_path = exchange / descriptor["exchange_file"]
    canonical = canonical_path.read_bytes()

    def validate_layout(encoded: bytes) -> None:
        candidate = tmp_path / "candidate.npz"
        candidate.write_bytes(encoded)
        candidate.chmod(0o600)
        file_descriptor = os.open(candidate, os.O_RDONLY)
        try:
            with zipfile.ZipFile(candidate, mode="r") as archive:
                matrix._validate_canonical_zip_layout(
                    file_descriptor,
                    archive,
                    byte_size=len(encoded),
                    expected_steps=steps,
                )
        finally:
            os.close(file_descriptor)

    validate_layout(canonical)
    for overlaid in (
        canonical + b"ARBITRARY-TRAILING-PAYLOAD",
        b"SELF-EXTRACTING-STUB" + canonical,
        canonical[:-2] + struct.pack("<H", 3) + b"xyz",
    ):
        with pytest.raises(matrix.ForagerMatrixStateError):
            validate_layout(overlaid)

    with zipfile.ZipFile(io.BytesIO(canonical), mode="r") as archive:
        members = {name: archive.read(name) for name in matrix._RAW_TRACE_MEMBERS}

    def rebuilt(*, level: int, year: int = 1980) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(
            buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=level,
            allowZip64=True,
        ) as archive:
            for name in matrix._RAW_TRACE_MEMBERS:
                info = zipfile.ZipInfo(name, date_time=(year, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = matrix._ZIP_EXTERNAL_ATTR
                info._compresslevel = level
                with archive.open(info, mode="w", force_zip64=True) as handle:
                    handle.write(members[name])
        return buffer.getvalue()

    with pytest.raises(matrix.ForagerMatrixStateError, match="ZIP metadata|record"):
        validate_layout(rebuilt(level=9, year=2000))

    alternate_deflate = rebuilt(level=1)
    alternate_path = tmp_path / "alternate-deflate.npz"
    alternate_path.write_bytes(alternate_deflate)
    alternate_path.chmod(0o600)
    file_descriptor = os.open(alternate_path, os.O_RDONLY)
    try:
        with zipfile.ZipFile(alternate_path, mode="r") as archive:
            locations = matrix._validate_canonical_zip_layout(
                file_descriptor,
                archive,
                byte_size=len(alternate_deflate),
                expected_steps=steps,
            )
            with pytest.raises(
                matrix.ForagerMatrixStateError,
                match="non-canonical DEFLATE",
            ):
                member = archive.infolist()[0]
                offset, size = locations[member.filename]
                matrix._validate_canonical_deflate_stream(
                    file_descriptor,
                    archive,
                    member,
                    compressed_offset=offset,
                    compressed_size=size,
                )
    finally:
        os.close(file_descriptor)


def test_raw_trace_v2_rejects_valid_but_noncanonical_npy_header() -> None:
    steps = 3
    header_dict = (
        b"{'shape': (3,), 'fortran_order': False, 'descr': '<f4', }"
    )
    header_size = 184
    padded = header_dict + b" " * (header_size - len(header_dict) - 1) + b"\n"
    noncanonical = b"\x93NUMPY\x01\x00" + struct.pack("<H", header_size) + padded
    encoded = noncanonical + np.zeros(steps, dtype="<f4").tobytes()
    loaded = np.load(io.BytesIO(encoded), allow_pickle=False)
    assert loaded.shape == (steps,)
    with pytest.raises(matrix.ForagerMatrixStateError, match="non-canonical NPY"):
        matrix._validate_npy_member_header(
            io.BytesIO(encoded),
            member_name="rewards.npy",
            expected_steps=steps,
        )


def test_raw_trace_descriptor_rejects_boolean_integer_aliases(
    tmp_path: Path,
) -> None:
    exchange = tmp_path / "boolean-trace"
    exchange.mkdir()
    sink = matrix._NpzMetricTraceSink(exchange, 1, 1)
    sink.append(
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
    )
    descriptor = dict(sink.finalize())
    descriptor["path"] = "reward-traces/base/batch-00000/seed-1.npz"
    descriptor.pop("exchange_file")
    for field in ("seed", "steps"):
        invalid = copy.deepcopy(descriptor)
        invalid[field] = True
        with pytest.raises(matrix.ForagerMatrixStateError, match="identity"):
            matrix._validate_trace_descriptor(
                invalid,
                path="trace",
                expected_seed=1,
                expected_steps=1,
                expected_output_path=descriptor["path"],
                exchange=False,
            )
    invalid = copy.deepcopy(descriptor)
    invalid["arrays"]["rewards"]["shape"] = [True]
    with pytest.raises(matrix.ForagerMatrixStateError, match="arrays.rewards"):
        matrix._validate_trace_descriptor(
            invalid,
            path="trace",
            expected_seed=1,
            expected_steps=1,
            expected_output_path=descriptor["path"],
            exchange=False,
        )


def test_snapshot_extraction_is_read_only_and_post_verified(
    tmp_path: Path,
) -> None:
    snapshot = matrix._build_source_snapshot()
    destination = tmp_path / "snapshot"
    matrix._extract_source_snapshot(
        snapshot,
        destination,
        source_execution_mode=matrix.SNAPSHOT_SOURCE_EXECUTION_MODE,
    )
    target = destination / "alberta_framework/benchmarks/runner.py"
    assert target.stat().st_mode & 0o222 == 0
    assert destination.stat().st_mode & 0o222 == 0

    target.chmod(0o644)
    target.write_text("VALUE = 999\n", encoding="utf-8")
    target.chmod(0o444)
    with pytest.raises(matrix.ForagerMatrixStateError, match="changed"):
        matrix._verify_extracted_source_snapshot(snapshot, destination)


def test_execution_identity_binds_ld_library_path_advisory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = matrix._build_benchmark_config(
        matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0]))
    )
    snapshot = matrix._build_source_snapshot()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/runtime/a")
    first = matrix._execution_context(benchmark, snapshot)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/runtime/b")
    second = matrix._execution_context(benchmark, snapshot)
    assert (
        first["execution_identity"]["runtime_sha256"]
        != second["execution_identity"]["runtime_sha256"]
    )
    assert first["execution_identity"]["runtime_profile_id"] is None
    assert first["execution_identity"]["environment_runtime_profile_sha256"] is None


def test_manifest_rejects_extreme_allocation_products() -> None:
    with pytest.raises(matrix.ForagerMatrixManifestError, match="recurrent_hidden_size"):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                variants={
                    "base": _horde({"recurrent_hidden_size": 2_147_483_647})
                }
            )
        )
    with pytest.raises(matrix.ForagerMatrixManifestError, match="reward trace decays"):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                variants={
                    "base": _horde(
                        {
                            "features": {
                                "reward_trace_decays": [0.9]
                                * (matrix._MAX_REWARD_TRACE_COUNT + 1)
                            }
                        }
                    )
                }
            )
        )
    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match=r"world_shape.*at most 4096 total cells",
    ):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                variants={
                    "base": _causal({"world_shape": [4_096, 4_096]})
                }
            )
        )


def test_late_output_injection_is_detected_in_same_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls)
    original = matrix._execute_new_batch

    def inject(**kwargs: Any) -> Any:
        result = original(**kwargs)
        (kwargs["output_root"].path / "UNEXPECTED").write_text(
            "late injection\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(matrix, "_execute_new_batch", inject)
    with pytest.raises(matrix.ForagerMatrixStateError, match="unexpected file"):
        matrix.run_forager_matrix(
            matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0])),
            tmp_path / "late-extra",
        )


def test_recovery_never_deletes_unrelated_temp_pattern(tmp_path: Path) -> None:
    output = tmp_path / "recovery"
    bound = matrix._open_bound_directory(output, create=True)
    unrelated = output / ".unrelated.123.456.0.tmp"
    unrelated.write_text("user data\n", encoding="utf-8")
    unrelated.chmod(0o600)
    try:
        plan = matrix._batch_plan(
            matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0]))
        )
        with pytest.raises(matrix.ForagerMatrixStateError, match="unexpected file"):
            matrix._validate_output_inventory(
                bound,
                plan,
                metric_evidence_mode="scalar_summary_unsealed",
            )
        assert unrelated.read_text(encoding="utf-8") == "user data\n"
    finally:
        bound.close()


def test_resume_rejects_hardlinked_committed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    _install_fake_runners(monkeypatch, calls)
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0]))
    output = tmp_path / "hardlink"
    matrix.run_forager_matrix(manifest, output)
    os.link(output / matrix.FINAL_REPORT_FILENAME, tmp_path / "report-alias.json")
    with pytest.raises(matrix.ForagerMatrixStateError, match="singly linked"):
        matrix.run_forager_matrix(manifest, output)


def test_percentile_bootstrap_interval_need_not_contain_sample_mean() -> None:
    manifest = matrix.parse_forager_matrix_manifest(_manifest_payload(seeds=[0, 1]))
    variant = manifest.variants["base"]
    entry = {
        "kind": variant.kind,
        "selection_group": variant.selection_group,
        "config": variant.config.to_dict(),
        "config_sha256": variant.config_sha256,
        "variant_sha256": variant.descriptor_sha256,
        "seeds": list(manifest.seeds),
        "seed_batches": [],
        "summary": {
            "agent": variant.kind,
            "privileged": False,
            "seeds": sorted(manifest.seeds),
            "metric": manifest.selection_rule.metric,
            "mean": 50.0,
            "ci_low": 100.0,
            "ci_high": 100.0,
            "confidence": manifest.selection_rule.confidence,
            "bootstrap_resamples": manifest.selection_rule.bootstrap_resamples,
            "bootstrap_seed": manifest.selection_rule.bootstrap_seed,
        },
    }
    validated = matrix._validate_report_variants_for_selection(
        manifest,
        {"base": entry},
    )
    assert validated["base"]["summary"]["mean"] == 50.0


def test_json_integer_bomb_and_embedded_host_paths_fail_closed() -> None:
    with pytest.raises(matrix.ForagerMatrixManifestError, match="strict JSON"):
        matrix._decode_strict_json(
            '{"value":' + "9" * 5_000 + "}",
            description="integer bomb",
        )
    for value in (
        "prefix /tmp/secret",
        "see file:///etc/passwd",
        r"prefix \\server\share",
    ):
        with pytest.raises(matrix.ForagerMatrixStateError, match="host path"):
            matrix._assert_path_sanitized({"value": value})


def test_nested_directory_creation_fsyncs_each_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = matrix._open_bound_directory(tmp_path / "fsync", create=True)
    original_fsync = os.fsync
    synchronized: list[tuple[int, int]] = []

    def record(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        synchronized.append((metadata.st_dev, metadata.st_ino))
        original_fsync(descriptor)

    monkeypatch.setattr(matrix.os, "fsync", record)
    try:
        with matrix._open_beneath(bound, ("one", "two"), create=True):
            pass
        one = os.stat(tmp_path / "fsync/one")
        assert (bound.device, bound.inode) in synchronized
        assert (one.st_dev, one.st_ino) in synchronized
    finally:
        bound.close()


def test_trusted_envelope_requires_external_authority_and_exact_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_sha256 = "a" * 64
    schedule_sha256 = matrix._EXPECTED_ENVIRONMENT_RNG_SCHEDULE_SHA256
    expected_identity = matrix.EnvironmentRuntimeIdentity(
        runtime_profile_id="matched-gpu-runtime",
        environment_runtime_profile_sha256=profile_sha256,
        environment_rng_schedule="dedicated_environment_split_chain_v1",
        environment_rng_schedule_sha256=schedule_sha256,
    )
    monkeypatch.setattr(
        matrix,
        "validate_environment_runtime_identity",
        lambda **_kwargs: expected_identity,
    )
    digests = {
        "tuning_report_file_sha256": "1" * 64,
        "tuning_report_payload_sha256": "2" * 64,
        "raw_metric_evidence_sha256": "3" * 64,
        "source_tree_sha256": "4" * 64,
        "source_archive_sha256": "5" * 64,
    }
    envelope = {
        "schema_version": matrix.TRUSTED_EXECUTION_ENVELOPE_SCHEMA,
        "issuer": "release-verifier",
        "key_id": "release-key:1",
        "signature": "opaque-signature",
        "signed_evidence": {
            "executor_kind": "oci",
            "source_mount_mode": "read_only_content_addressed_mount",
            **digests,
            "runtime_profile_id": expected_identity.runtime_profile_id,
            "environment_runtime_profile": {"delegated": "profile"},
            "environment_runtime_profile_sha256": profile_sha256,
            "environment_rng_schedule": (
                expected_identity.environment_rng_schedule
            ),
            "environment_rng_schedule_sha256": schedule_sha256,
        },
    }
    observed: list[bytes] = []

    def accepts(
        signed: bytes,
        issuer: str,
        key_id: str,
        signature: str,
    ) -> bool:
        assert issuer == "release-verifier"
        assert key_id == "release-key:1"
        assert signature == "opaque-signature"
        observed.append(signed)
        return True

    validated = matrix.validate_verifier_issued_tuning_envelope(
        envelope,
        verifier=accepts,
        expected_runtime_identity=expected_identity,
        **{f"expected_{key}": value for key, value in digests.items()},
    )
    assert observed
    assert validated["runtime_identity"]["runtime_profile_id"] == (
        "matched-gpu-runtime"
    )

    with pytest.raises(
        matrix.ForagerMatrixManifestError,
        match="signature or authority",
    ):
        matrix.validate_verifier_issued_tuning_envelope(
            envelope,
            verifier=lambda *_args: 1,  # type: ignore[return-value]
            expected_runtime_identity=expected_identity,
            **{f"expected_{key}": value for key, value in digests.items()},
        )

    with pytest.raises(matrix.ForagerMatrixManifestError, match="unknown keys"):
        matrix.parse_forager_matrix_manifest(
            _manifest_payload(
                trusted_execution_envelope=envelope,
            )
        )
