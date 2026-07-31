"""Synthetic fail-closed tests for published-scale OPMNIST ingestion."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path

import pytest

from alberta_framework.evaluation import opmnist_development_ingest as ingest

pytestmark = pytest.mark.integration

_SYNTHETIC_DATASET_BYTES = gzip.compress(
    b"@relation mnist_784\n@data\n0,synthetic\n",
    mtime=0,
)


@pytest.fixture(autouse=True)
def _pin_synthetic_dataset_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests cheap while exercising the production pinned-identity gate."""
    monkeypatch.setattr(
        ingest,
        "EXPECTED_OPENML_DATASET_SHA256",
        _sha256(_SYNTHETIC_DATASET_BYTES),
    )
    monkeypatch.setattr(
        ingest,
        "EXPECTED_OPENML_DATASET_BYTE_SIZE",
        len(_SYNTHETIC_DATASET_BYTES),
    )


@dataclass(frozen=True)
class _FixturePaths:
    plan: Path
    runbook: Path
    results: tuple[Path, Path, Path]
    statuses: tuple[Path, Path, Path]
    merged_result: Path
    solution_gate: Path
    runner_source: Path
    published_stressors_source: Path
    merge_source: Path
    solution_gate_source: Path
    dataset: Path


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _seed_command(seed: int) -> list[str]:
    prefix = f"step2_opmnist_solution_800task_3seed_seed{seed}"
    output_dir = "outputs/step2_opmnist_solution_full/seed_splits"
    return [
        "python",
        "examples/The Alberta Plan/Step2/step2_upgd_memory_opmnist.py",
        "--mnist-published-scale",
        "--allow-openml-download",
        "--n-seeds",
        "1",
        "--seed",
        str(seed),
        "--final-window",
        "5000",
        "--chunk-size",
        "60000",
        "--include-sharpened-mlp",
        "--include-adaptive-primary-sharpened",
        "--evaluate-all-permutation-views",
        "--max-test-permutation-views",
        "800",
        "--only-methods",
        ",".join(ingest.EXPECTED_METHODS),
        "--output-dir",
        output_dir,
        "--result-prefix",
        prefix,
        "--note-path",
        f"{output_dir}/{prefix}.md",
        "--status-path",
        f"{output_dir}/{prefix}_status.json",
    ]


def _plan_payload() -> dict[str, object]:
    commands = [_seed_command(seed) for seed in ingest.EXPECTED_SEEDS]
    input_paths = [
        f"outputs/step2_opmnist_solution_full/seed_splits/"
        f"step2_opmnist_solution_800task_3seed_seed{seed}_results.json"
        for seed in ingest.EXPECTED_SEEDS
    ]
    merged_path = (
        "outputs/step2_opmnist_solution_full/"
        "step2_opmnist_solution_800task_3seed_results.json"
    )
    gate_path = (
        "outputs/step2_opmnist_solution_full/"
        "step2_opmnist_solution_800task_3seed_solution_gate.json"
    )
    runner = list(commands[0])
    runner[runner.index("--n-seeds") + 1] = "3"
    merge = [
        "python",
        "benchmarks/step2_opmnist_merge_seed_results.py",
        *input_paths,
        "--output",
        merged_path,
        "--write-summary",
        "outputs/step2_opmnist_solution_full/summary.md",
    ]
    audit = [
        "python",
        "benchmarks/step2_opmnist_solution_gate.py",
        merged_path,
        "--min-seeds",
        "3",
        "--write-status",
        gate_path,
    ]
    return {
        "schema": ingest.UPSTREAM_PLAN_SCHEMA,
        "claim": "Step 2 OPMNIST solution candidate run",
        "n_seeds": 3,
        "seed_start": 0,
        "methods": list(ingest.EXPECTED_METHODS),
        "protocol": {
            "mnist_source": "openml",
            "mnist_split": "canonical_60000_10000",
            "n_permutations": 800,
            "prediction_before_update": True,
            "task_block_size": 60_000,
            "task_id_provided_to_learner": False,
            "test_views": "all_800_permutation_views",
            "updates_per_seed": 48_000_000,
        },
        "runner_command": runner,
        "runner_command_shell": shlex.join(runner),
        "split_seed_runner_commands": commands,
        "split_seed_runner_command_shells": [shlex.join(command) for command in commands],
        "merge_command": merge,
        "merge_command_shell": shlex.join(merge),
        "audit_command": audit,
        "audit_command_shell": shlex.join(audit),
        "expected_result": merged_path,
        "expected_solution_status": gate_path,
        "promotion_rule": (
            "The audit command must exit 0 without --allow-unsolved and report "
            "status.solved_opmnist_step2=true."
        ),
    }


def _config(seed: int) -> dict[str, object]:
    prefix = f"step2_opmnist_solution_800task_3seed_seed{seed}"
    output_dir = "outputs/step2_opmnist_solution_full/seed_splits"
    config: dict[str, object] = {
        "allow_openml_download": True,
        "allow_torchvision_download": False,
        "chunk_size": 60_000,
        "evaluate_all_permutation_views": True,
        "final_window": 5_000,
        "force_restart": False,
        "include_adaptive_primary_sharpened": True,
        "include_brier_single_upgd": False,
        "include_centroid_candidates": False,
        "include_delight_candidates": False,
        "include_dreaming_candidates": False,
        "include_identity_permutation": False,
        "include_primary_sharpened": False,
        "include_prototype_memory": False,
        "include_rls_calibrated": False,
        "include_sharpened_mlp": True,
        "include_single_upgd": False,
        "include_smoothed_single_upgd": False,
        "include_temperature_single_upgd": False,
        "max_test_examples": None,
        "max_test_permutation_views": 800,
        "max_train_examples": None,
        "mnist_published_scale": True,
        "mnist_source": "openml",
        "mnist_split": "canonical",
        "n_permutations": 800,
        "n_seeds": 1,
        "note_path": f"{output_dir}/{prefix}.md",
        "only_methods": ",".join(ingest.EXPECTED_METHODS),
        "openml_data_home": None,
        "openml_n_retries": 2,
        "openml_retry_delay": 1.0,
        "opmnist_fraction": None,
        "output_dir": output_dir,
        "result_prefix": prefix,
        "resume": True,
        "resume_path": None,
        "sample_with_replacement": False,
        "seed": seed,
        "status_path": f"{output_dir}/{prefix}_status.json",
        "steps": 48_000_000,
        "stop_after_chunks": None,
        "task_block_size": 60_000,
        "task_sampling": "sequential_epoch",
        "torchvision_data_home": None,
        "train_fraction": 0.7,
        "created_at": f"2026-08-01T0{seed}:00:00+00:00",
        "runner": "step2_upgd_memory_opmnist",
        "methods": list(ingest.EXPECTED_METHODS),
    }
    assert set(config) == ingest._RESULT_CONFIG_FIELDS
    return config


def _dataset(seed: int, cache_root: Path) -> dict[str, object]:
    elapsed = 120_000.0 + seed
    dataset: dict[str, object] = {
        "all_experts_update_every_step": True,
        "benchmark": "permuted_mnist_like",
        "candidate_methods": list(ingest.EXPECTED_CANDIDATE_METHODS),
        "checkpoint_loaded": False,
        "completed_full_task_blocks": 800,
        "dataset": "sklearn.datasets.fetch_openml('mnist_784', version=1)",
        "description": "OpenML MNIST 28x28 handwritten digits.",
        "evaluate_all_permutation_views": True,
        "feature_dim": 784,
        "full_mnist_task_blocks": True,
        "heldout_deployment_objective": "post-run selected weights; no task id",
        "include_identity_permutation": False,
        "is_full_mnist_split": True,
        "is_true_mnist": True,
        "limitations": "Synthetic fixture preserving the exact result shape.",
        "matches_dohare_opmnist_core_protocol": True,
        "matches_dohare_opmnist_published_task_count": True,
        "max_test_examples": None,
        "max_test_permutation_views": 800,
        "max_train_examples": None,
        "methods": list(ingest.EXPECTED_METHODS),
        "mlp_methods": list(ingest.EXPECTED_MLP_METHODS),
        "n_classes": 10,
        "n_permutations": 800,
        "n_test": 10_000,
        "n_total": 70_000,
        "n_train": 60_000,
        "observed_task_blocks": 800,
        "openml_data_home": str(cache_root),
        "opmnist_completed_full_60000_task_blocks": 800,
        "opmnist_elapsed_s": elapsed,
        "opmnist_eta_to_800_tasks_s": 0.0,
        "opmnist_overall_steps_per_second": 48_000_000 / elapsed,
        "partial_task_steps": 0,
        "permutations_are_random_pixel_orders": True,
        "prediction_before_update_every_step": True,
        "protocol": "chunked_online_permuted_pixels",
        "published_protocol_delta": "Synthetic fixture for the audited protocol text.",
        "random_pixel_permutation_seed": seed + 10_000,
        "resumable_runner": True,
        "resume_checkpoint_path": f"outputs/checkpoints/seed-{seed}.pkl",
        "sample_with_replacement": False,
        "sequence_seed": seed + 10_000,
        "sequential_single_pass_task_epochs": True,
        "single_pass_examples_within_task": True,
        "source_kind": "openml_mnist_784",
        "split": "openml_canonical_60000_10000",
        "split_seed": seed,
        "steps": 48_000_000,
        "stream_chunk_size": 60_000,
        "streaming_runner": True,
        "task_block_size": 60_000,
        "task_id_provided_to_learner": False,
        "task_ids_observed": list(range(800)),
        "task_sampling": "sequential_epoch",
        "test_permutation_views": 800,
        "test_task_ids_evaluated": list(range(800)),
        "test_views_cover_all_permutations": True,
        "test_views_cover_observed_permutations": True,
        "train_fraction": 0.7,
    }
    assert set(dataset) == ingest._DATASET_FIELDS
    return dataset


def _methods(seed: int) -> dict[str, dict[str, float]]:
    methods: dict[str, dict[str, float]] = {}
    for index, method in enumerate(ingest.EXPECTED_METHODS):
        is_winner = method == ingest.EXPECTED_CANDIDATE_METHODS[0]
        accuracy = (0.95 if is_winner else 0.82 - index * 0.005) - seed * 0.001
        mse = (0.01 if is_winner else 0.08 + index * 0.005) + seed * 0.0001
        methods[method] = {
            "online_mean_mse": mse,
            "online_mean_accuracy": accuracy,
            "final_window_mse": mse + 0.001,
            "final_window_accuracy": accuracy - 0.01,
            "test_mse": mse + 0.002,
            "test_accuracy": accuracy - 0.02,
        }
    return methods


def _status(seed: int, dataset: dict[str, object]) -> dict[str, object]:
    elapsed_value = dataset["opmnist_elapsed_s"]
    assert isinstance(elapsed_value, (int, float)) and not isinstance(elapsed_value, bool)
    elapsed = float(elapsed_value)
    recent_rate = 700.0 + seed
    protocol = {
        key: value
        for key, value in dataset.items()
        if key
        not in {
            "opmnist_elapsed_s",
            "opmnist_eta_to_800_tasks_s",
            "opmnist_overall_steps_per_second",
        }
    }
    return {
        "schema": "alberta.upgd_memory_opmnist.status.v1",
        "updated_at_utc": f"2026-08-01T0{seed}:30:00+00:00",
        "seed": seed,
        "checkpoint_path": dataset["resume_checkpoint_path"],
        "requested_steps": 48_000_000,
        "dohare_target_steps": 48_000_000,
        "completed_steps": 48_000_000,
        "status": {
            "completed_full_task_blocks": 800,
            "completed_steps": 48_000_000,
            "elapsed_s": elapsed,
            "eta_human": "0s",
            "eta_seconds": 0.0,
            "overall_steps_per_second": 48_000_000 / elapsed,
            "progress_fraction": 1.0,
            "recent_steps_per_second": recent_rate,
            "remaining_full_task_blocks": 0,
            "remaining_steps": 0,
            "target_full_task_blocks": 800,
            "target_steps": 48_000_000,
        },
        "latest_progress": {
            "chunk_elapsed_s": 60_000 / recent_rate,
            "chunk_steps": 60_000,
            "chunks_run_this_invocation": 800,
            "completed_full_task_blocks": 800,
            "completed_steps": 48_000_000,
            "dohare_target_steps": 48_000_000,
            "elapsed_s": elapsed,
            "eta_to_dohare_800_s": 0.0,
            "requested_steps": 48_000_000,
            "seed": seed,
            "steps_per_second": recent_rate,
            "stop_after_chunks": None,
            "timestamp_utc": f"2026-08-01T0{seed}:29:59+00:00",
        },
        "protocol": protocol,
    }


def _write_fixture(tmp_path: Path) -> _FixturePaths:
    plan_payload = _plan_payload()
    plan = tmp_path / "plan.json"
    plan.write_bytes(_json_bytes(plan_payload))
    runbook = tmp_path / "RUNBOOK.md"
    runbook.write_text(
        "# 800-task 3-seed run\n\n"
        "48,000,000 updates. Seeds 0, 1, 2. Evaluate all 800 views. "
        "Audit solved_opmnist_step2 and present this NOT as a solved-Step-2 claim.\n",
        encoding="utf-8",
    )
    runner_source = tmp_path / "runner.py"
    published_stressors_source = tmp_path / "published.py"
    merge_source = tmp_path / "merge.py"
    solution_gate_source = tmp_path / "gate.py"
    runner_source.write_bytes(b"# synthetic runner\n")
    published_stressors_source.write_bytes(b"# synthetic published stressors\n")
    merge_source.write_bytes(b"# synthetic merge\n")
    solution_gate_source.write_bytes(b"# synthetic solution gate\n")
    source_hashes = {
        "runner": _sha256(runner_source.read_bytes()),
        "published_stressors": _sha256(published_stressors_source.read_bytes()),
    }

    cache_root = tmp_path / "openml-cache"
    dataset = cache_root / ingest._OPENML_CACHE_RELATIVE_PATH
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(_SYNTHETIC_DATASET_BYTES)

    result_paths: list[Path] = []
    status_paths: list[Path] = []
    payloads: list[dict[str, object]] = []
    for seed in ingest.EXPECTED_SEEDS:
        config = _config(seed)
        dataset_meta = _dataset(seed, cache_root)
        record: dict[str, object] = {
            "dataset_name": "permuted_mnist_like",
            "seed": seed,
            "dataset": dataset_meta,
            "methods": _methods(seed),
        }
        aggregate = ingest.recompute_opmnist_metrics([record])
        manifest = {
            "schema": ingest.UPSTREAM_RESULT_MANIFEST_SCHEMA,
            "created_at_utc": f"2026-08-01T0{seed}:31:00+00:00",
            "argv": _seed_command(seed)[2:],
            "config": {key: config[key] for key in ingest._ARG_CONFIG_FIELDS},
            "methods": list(ingest.EXPECTED_METHODS),
            "git": {
                "commit": "a" * 40,
                "branch": "main",
                "describe": "synthetic-dirty",
                "dirty": True,
                "status_porcelain": [" M synthetic.py"],
            },
            "environment": {
                "python": "3.12.3",
                "python_executable": "/opt/venv/bin/python",
                "platform": "test-platform",
                "jax": "0.4.test",
                "jaxlib": "0.4.test",
                "numpy": "2.test",
                "jax_default_backend": "cpu",
                "jax_devices": ["TFRT_CPU_0"],
            },
            "source_sha256": source_hashes,
        }
        payload: dict[str, object] = {
            "config": config,
            "datasets": {"permuted_mnist_like": dataset_meta},
            "records": [record],
            "primary_method": ingest.EXPECTED_CANDIDATE_METHODS[0],
            "mlp_methods": list(ingest.EXPECTED_MLP_METHODS),
            "candidate_methods": list(ingest.EXPECTED_CANDIDATE_METHODS),
            "aggregate": {"permuted_mnist_like": aggregate},
            "wall_clock_s": 130_000.0 + seed,
            "evidence_level": "single_upgd_memory_opmnist_resumable",
            "manifest": manifest,
            "solution_status": ingest._expected_upstream_seed_status(aggregate),
        }
        result_path = tmp_path / f"seed-{seed}-result.json"
        result_path.write_bytes(_json_bytes(payload))
        result_paths.append(result_path)
        status_path = tmp_path / f"seed-{seed}-status.json"
        status_path.write_bytes(_json_bytes(_status(seed, dataset_meta)))
        status_paths.append(status_path)
        payloads.append(payload)

    plan_merge = plan_payload["merge_command"]
    assert isinstance(plan_merge, list)
    input_paths = plan_merge[2 : plan_merge.index("--output")]
    records = [copy.deepcopy(payload["records"][0]) for payload in payloads]  # type: ignore[index]
    aggregate = ingest.recompute_opmnist_metrics(records)
    merged_config = copy.deepcopy(payloads[0]["config"])
    assert isinstance(merged_config, dict)
    merged_config.update(
        {
            "created_at": "2026-08-01T04:00:00+00:00",
            "merged_from_seed_splits": True,
            "split_result_paths": input_paths,
            "n_seeds": 3,
            "seeds": [0, 1, 2],
        }
    )
    final_dataset = copy.deepcopy(records[-1]["dataset"])
    assert isinstance(final_dataset, dict)
    final_dataset.update(
        {"merged_from_seed_splits": True, "split_result_paths": input_paths}
    )
    merged_manifest = {
        "schema": "alberta.step2.upgd_memory_opmnist.merge_manifest.v1",
        "created_at_utc": "2026-08-01T04:00:01+00:00",
        "merge_script": "/src/benchmarks/step2_opmnist_merge_seed_results.py",
        "merge_script_sha256": _sha256(merge_source.read_bytes()),
        "runner_path": "/src/examples/step2_upgd_memory_opmnist.py",
        "runner_sha256": _sha256(runner_source.read_bytes()),
        "methods": list(ingest.EXPECTED_METHODS),
        "seeds": [0, 1, 2],
        "split_results": [
            {
                "path": input_path,
                "sha256": _sha256(result_path.read_bytes()),
                "seeds": [seed],
                "manifest": payload["manifest"],
            }
            for seed, input_path, result_path, payload in zip(
                ingest.EXPECTED_SEEDS,
                input_paths,
                result_paths,
                payloads,
                strict=True,
            )
        ],
    }
    merged_status = ingest._expected_upstream_merged_status(aggregate)
    merged_payload = {
        "config": merged_config,
        "datasets": {"permuted_mnist_like": final_dataset},
        "records": records,
        "primary_method": ingest.EXPECTED_CANDIDATE_METHODS[0],
        "mlp_methods": list(ingest.EXPECTED_MLP_METHODS),
        "candidate_methods": list(ingest.EXPECTED_CANDIDATE_METHODS),
        "aggregate": {"permuted_mnist_like": aggregate},
        "split_results": input_paths,
        "manifest": merged_manifest,
        "evidence_level": "merged_upgd_memory_opmnist_seed_splits",
        "solution_status": merged_status,
    }
    merged_result = tmp_path / "merged-result.json"
    merged_result.write_bytes(_json_bytes(merged_payload))
    solution_gate = tmp_path / "solution-gate.json"
    solution_gate.write_bytes(
        _json_bytes(
            {
                "schema": ingest.UPSTREAM_SOLUTION_GATE_SCHEMA,
                "artifact_path": plan_payload["expected_result"],
                "status": merged_status,
            }
        )
    )
    return _FixturePaths(
        plan=plan,
        runbook=runbook,
        results=tuple(result_paths),  # type: ignore[arg-type]
        statuses=tuple(status_paths),  # type: ignore[arg-type]
        merged_result=merged_result,
        solution_gate=solution_gate,
        runner_source=runner_source,
        published_stressors_source=published_stressors_source,
        merge_source=merge_source,
        solution_gate_source=solution_gate_source,
        dataset=dataset,
    )


def _ingest(paths: _FixturePaths, output: Path) -> Path:
    return ingest.ingest_opmnist_development_bundle(
        plan_path=paths.plan,
        runbook_path=paths.runbook,
        result_paths=paths.results,
        status_paths=paths.statuses,
        merged_result_path=paths.merged_result,
        solution_gate_path=paths.solution_gate,
        runner_source_path=paths.runner_source,
        published_stressors_source_path=paths.published_stressors_source,
        merge_source_path=paths.merge_source,
        solution_gate_source_path=paths.solution_gate_source,
        dataset_path=paths.dataset,
        output_dir=output,
    )


def test_complete_bundle_recomputes_protocol_and_metrics_without_promoting(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "new-development-bundle"
    receipt_path = _ingest(paths, output)

    validation = ingest.validate_opmnist_development_bundle(output)
    assert validation.valid, validation.errors
    assert validation.protocol_complete
    assert validation.all_metric_mean_win
    assert not validation.solved_opmnist_step2
    assert validation.development_only
    assert not validation.scientific_promotion_allowed

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    claims = receipt["scientific_payload"]["claims"]
    coverage = receipt["scientific_payload"]["coverage"]
    provenance = receipt["scientific_payload"]["provenance"]
    assert claims["protocol_complete"] is True
    assert claims["all_metric_mean_win"] is True
    assert claims["solved_opmnist_step2"] is False
    assert claims["sota_claimed"] is False
    assert coverage["seeds"] == [0, 1, 2]
    assert [row["dynamic_completed_steps"] for row in coverage["per_seed"]] == [
        48_000_000,
    ] * 3
    assert provenance["plan_and_runbook_are_posthoc_locators"] is True
    assert provenance["command_execution_attested"] is False
    assert (output / "inputs/solution-gate.json").read_bytes() == paths.solution_gate.read_bytes()


@pytest.mark.parametrize("tamper", ["duplicate", "nonfinite", "extra"])
def test_strict_seed_json_rejects_duplicate_nonfinite_and_extra_fields(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = _write_fixture(tmp_path)
    target = paths.results[0]
    if tamper == "duplicate":
        raw = target.read_text(encoding="utf-8")
        target.write_text(raw.replace("{\n", '{\n  "config": {},\n', 1), encoding="utf-8")
    else:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if tamper == "nonfinite":
            payload["records"][0]["methods"][ingest.EXPECTED_METHODS[0]][
                "online_mean_mse"
            ] = float("nan")
            target.write_text(json.dumps(payload), encoding="utf-8")
        else:
            payload["unregistered"] = True
            target.write_bytes(_json_bytes(payload))

    with pytest.raises(ValueError, match="duplicate JSON key|non-finite|fields differ"):
        _ingest(paths, tmp_path / "rejected")


def test_static_protocol_metadata_cannot_replace_dynamic_completion(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    status = json.loads(paths.statuses[1].read_text(encoding="utf-8"))
    status["completed_steps"] = 47_940_000
    paths.statuses[1].write_bytes(_json_bytes(status))

    with pytest.raises(ValueError, match="completed_steps"):
        _ingest(paths, tmp_path / "rejected")


def test_exact_task_and_test_view_arrays_are_required(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    result = json.loads(paths.results[2].read_text(encoding="utf-8"))
    result["datasets"]["permuted_mnist_like"]["test_task_ids_evaluated"][-1] = 798
    result["records"][0]["dataset"]["test_task_ids_evaluated"][-1] = 798
    paths.results[2].write_bytes(_json_bytes(result))

    with pytest.raises(ValueError, match="test_task_ids_evaluated"):
        _ingest(paths, tmp_path / "rejected")


def test_aggregate_and_gate_claims_are_independently_recomputed(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    gate = json.loads(paths.solution_gate.read_text(encoding="utf-8"))
    gate["status"]["solved_opmnist_step2"] = False
    paths.solution_gate.write_bytes(_json_bytes(gate))

    with pytest.raises(ValueError, match="solution gate.status"):
        _ingest(paths, tmp_path / "rejected")


def test_source_hashes_must_match_every_final_result_manifest(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    paths.runner_source.write_bytes(paths.runner_source.read_bytes() + b"# drift\n")

    with pytest.raises(ValueError, match="source_sha256.runner"):
        _ingest(paths, tmp_path / "rejected")


def test_post_ingest_byte_tampering_invalidates_bundle(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "new-development-bundle"
    _ingest(paths, output)
    bundled_gate = output / "inputs/solution-gate.json"
    bundled_gate.chmod(0o640)
    bundled_gate.write_bytes(bundled_gate.read_bytes() + b"\n")

    validation = ingest.validate_opmnist_development_bundle(output)
    assert not validation.valid
    assert any("scientific_payload" in error for error in validation.errors)
    assert not validation.solved_opmnist_step2


def test_ingest_refuses_to_overwrite_an_existing_output(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "already-exists"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _ingest(paths, output)


def test_dataset_cache_entry_must_not_be_a_symlink(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    outside = tmp_path / "outside-cache-object.gz"
    outside.write_bytes(_SYNTHETIC_DATASET_BYTES)
    paths.dataset.unlink()
    paths.dataset.symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic link|symlink"):
        _ingest(paths, tmp_path / "rejected")


def test_source_inputs_must_not_be_symlinks(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    real_runner = tmp_path / "real-runner.py"
    paths.runner_source.rename(real_runner)
    paths.runner_source.symlink_to(real_runner)

    with pytest.raises(ValueError, match="symbolic link|symlink"):
        _ingest(paths, tmp_path / "rejected")


def test_dataset_identity_is_pinned_not_merely_nonempty(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    paths.dataset.write_bytes(gzip.compress(b"different dataset", mtime=0))

    with pytest.raises(ValueError, match="dataset.*(SHA-256|byte size|identity)"):
        _ingest(paths, tmp_path / "rejected")


def test_dataset_swap_between_path_check_and_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_fixture(tmp_path)
    rogue = tmp_path / "rogue-cache-object.gz"
    rogue.write_bytes(gzip.compress(b"rogue dataset", mtime=0))
    original = ingest._read_regular_file_no_symlinks
    swapped = False

    def swapping_read(path: Path, *, label: str) -> bytes:
        nonlocal swapped
        raw = original(path, label=label)
        if label == "solution-gate source" and not swapped:
            paths.dataset.unlink()
            paths.dataset.symlink_to(rogue)
            swapped = True
        return raw

    monkeypatch.setattr(ingest, "_read_regular_file_no_symlinks", swapping_read)
    with pytest.raises(ValueError, match="symbolic link|symlink"):
        _ingest(paths, tmp_path / "rejected")
    assert swapped


def test_atomic_publication_never_replaces_a_raced_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "raced-output"
    original = ingest._rename_noreplace_at

    def racing_rename(parent_fd: int, source_name: str, target_name: str) -> None:
        os.mkdir(target_name, dir_fd=parent_fd)
        original(parent_fd, source_name, target_name)

    monkeypatch.setattr(ingest, "_rename_noreplace_at", racing_rename)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _ingest(paths, output)
    assert output.is_dir()
    assert not any(output.iterdir())
    assert not list(tmp_path.glob(".raced-output.staging-*"))


def test_validation_rejects_a_symlink_bundle_root(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    real = tmp_path / "real-bundle"
    _ingest(paths, real)
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(real, target_is_directory=True)

    validation = ingest.validate_opmnist_development_bundle(alias)
    assert not validation.valid
    assert any("symbolic link" in error or "symlink" in error for error in validation.errors)


@pytest.mark.parametrize("node_kind", ["directory", "fifo"])
def test_validation_rejects_every_unregistered_filesystem_node(
    tmp_path: Path,
    node_kind: str,
) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "bundle"
    _ingest(paths, output)
    output.chmod(0o750)
    unexpected = output / "unregistered-node"
    if node_kind == "directory":
        unexpected.mkdir()
    else:
        os.mkfifo(unexpected)

    validation = ingest.validate_opmnist_development_bundle(output)
    assert not validation.valid
    assert any(
        "filesystem" in error or "file set" in error or "directory set" in error
        for error in validation.errors
    )


def test_missing_nested_metric_returns_invalid_instead_of_raising(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "bundle"
    _ingest(paths, output)
    result_path = output / "inputs/seed-results/seed-0.json"
    result_path.chmod(0o640)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    del payload["records"][0]["methods"][ingest.EXPECTED_METHODS[0]]["test_accuracy"]
    result_path.write_bytes(_json_bytes(payload))

    validation = ingest.validate_opmnist_development_bundle(output)
    assert not validation.valid
    assert any("test_accuracy" in error or "six metrics" in error for error in validation.errors)
    assert ingest.main(["validate", str(output)]) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eta_seconds", False),
        ("progress_fraction", True),
        ("remaining_full_task_blocks", False),
        ("remaining_steps", False),
    ],
)
def test_progress_status_rejects_boolean_numeric_aliases(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    paths = _write_fixture(tmp_path)
    payload = json.loads(paths.statuses[0].read_text(encoding="utf-8"))
    payload["status"][field] = value
    paths.statuses[0].write_bytes(_json_bytes(payload))

    with pytest.raises(ValueError, match=field):
        _ingest(paths, tmp_path / "rejected")


def test_latest_chunk_elapsed_and_rate_must_reconstruct(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    payload = json.loads(paths.statuses[0].read_text(encoding="utf-8"))
    payload["latest_progress"]["chunk_elapsed_s"] = 0.0
    paths.statuses[0].write_bytes(_json_bytes(payload))

    with pytest.raises(ValueError, match="chunk_elapsed_s|does not reconstruct"):
        _ingest(paths, tmp_path / "rejected")


@pytest.mark.parametrize("field", ["task_ids_observed", "test_task_ids_evaluated"])
def test_task_coverage_arrays_require_json_integers(tmp_path: Path, field: str) -> None:
    dataset = _dataset(0, tmp_path / "cache")
    values = dataset[field]
    assert isinstance(values, list)
    values[0] = False
    errors: list[str] = []

    ingest._validate_dataset(dataset, seed=0, errors=errors)
    assert any(field in error and "JSON integers" in error for error in errors)


def test_resumed_final_status_shape_is_accepted(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    result = json.loads(paths.results[1].read_text(encoding="utf-8"))
    result["datasets"]["permuted_mnist_like"]["checkpoint_loaded"] = True
    result["records"][0]["dataset"]["checkpoint_loaded"] = True
    paths.results[1].write_bytes(_json_bytes(result))

    status = json.loads(paths.statuses[1].read_text(encoding="utf-8"))
    assert status["protocol"]["checkpoint_loaded"] is False
    status["latest_progress"]["chunks_run_this_invocation"] = 17
    paths.statuses[1].write_bytes(_json_bytes(status))

    merged = json.loads(paths.merged_result.read_text(encoding="utf-8"))
    merged["records"][1]["dataset"]["checkpoint_loaded"] = True
    split_row = merged["manifest"]["split_results"][1]
    split_row["sha256"] = _sha256(paths.results[1].read_bytes())
    paths.merged_result.write_bytes(_json_bytes(merged))

    output = tmp_path / "bundle"
    _ingest(paths, output)
    assert ingest.validate_opmnist_development_bundle(output).valid


def test_published_bundle_modes_are_read_only_and_validator_enforces_them(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "bundle"
    _ingest(paths, output)

    nodes = [output, *output.rglob("*")]
    for node in nodes:
        assert stat.S_IMODE(node.stat().st_mode) & 0o222 == 0

    receipt = output / "receipt.v1.json"
    receipt.chmod(0o640)
    validation = ingest.validate_opmnist_development_bundle(output)
    assert not validation.valid
    assert any("mode" in error for error in validation.errors)


def test_failed_copy_removes_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "bundle"
    original = ingest._write_file_at
    calls = 0

    def failing_write(root_fd: int, relative_path: str, raw: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic copy failure")
        original(root_fd, relative_path, raw)

    monkeypatch.setattr(ingest, "_write_file_at", failing_write)
    with pytest.raises(OSError, match="synthetic copy failure"):
        _ingest(paths, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_metric_recomputation_matches_an_independent_hand_calculated_oracle() -> None:
    mse_rows = {
        ingest.EXPECTED_CANDIDATE_METHODS[0]: [1.0, 2.0, 3.0],
        ingest.EXPECTED_CANDIDATE_METHODS[1]: [3.0, 3.0, 3.0],
        "mlp_h64": [4.0, 5.0, 6.0],
        "mlp_h128": [2.0, 4.0, 6.0],
        "mlp_h64_sharp": [5.0, 5.0, 5.0],
        "mlp_h128_sharp": [6.0, 6.0, 6.0],
    }
    accuracy_rows = {
        ingest.EXPECTED_CANDIDATE_METHODS[0]: [0.8, 0.8, 0.8],
        ingest.EXPECTED_CANDIDATE_METHODS[1]: [0.7, 0.7, 0.7],
        "mlp_h64": [0.6, 0.6, 0.6],
        "mlp_h128": [0.75, 0.75, 0.75],
        "mlp_h64_sharp": [0.5, 0.5, 0.5],
        "mlp_h128_sharp": [0.4, 0.4, 0.4],
    }
    records: list[dict[str, object]] = []
    for seed in ingest.EXPECTED_SEEDS:
        methods = {}
        for method in ingest.EXPECTED_METHODS:
            methods[method] = {
                metric: (
                    accuracy_rows[method][seed]
                    if metric.endswith("accuracy")
                    else mse_rows[method][seed]
                )
                for metric in ingest.CORE_METRICS
            }
        records.append({"methods": methods})

    aggregate = ingest.recompute_opmnist_metrics(records)
    assert aggregate["mlp_h128"]["online_mean_mse"] == {
        "mean": 4.0,
        "stderr": 2.0 / math.sqrt(3.0),
        "per_seed": [2.0, 4.0, 6.0],
    }
    mse_comparison = aggregate["comparisons"]["online_mean_mse"]
    assert mse_comparison["best_mlp"] == "mlp_h128"
    primary_mse = mse_comparison["candidate_vs_best_mlp"][
        ingest.EXPECTED_CANDIDATE_METHODS[0]
    ]
    assert primary_mse == {
        "diff_mean_positive_favors_candidate": 2.0,
        "diff_stderr": 1.0 / math.sqrt(3.0),
        "wins_for_candidate": 3,
        "wins_for_baseline": 0,
        "ties": 0,
        "diffs": [1.0, 2.0, 3.0],
    }
    accuracy_comparison = aggregate["comparisons"]["test_accuracy"]
    assert accuracy_comparison["best_mlp"] == "mlp_h128"
    primary_accuracy = accuracy_comparison["candidate_vs_best_mlp"][
        ingest.EXPECTED_CANDIDATE_METHODS[0]
    ]
    assert math.isclose(primary_accuracy["diff_mean_positive_favors_candidate"], 0.05)
    assert primary_accuracy["wins_for_candidate"] == 3
