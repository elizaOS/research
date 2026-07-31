"""Integrity checks for explicitly nonpromoting Forager development receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks import (
    FORAGAX_AGENTS_URL,
    ForagerEnvConfig,
    OfficialForagaxRunSpec,
    import_official_foragax_npz,
)

_ROOT = Path(__file__).resolve().parents[1]
_RTU_RECEIPT = _ROOT / "outputs/forager/rtu_rtrl_500k_dev4/receipt.v1.json"
_RTU_RECEIPT_SHA256 = "d1c86f0fb4dd7e9e59797e99b39bd8e4bc9c8114e95cfa8b2232360a6b65926c"
_CAUSAL_MAP_RECEIPT = (
    _ROOT / "outputs/forager/causal_map_500k_dev4/receipt.v1.json"
)
_CAUSAL_MAP_RECEIPT_SHA256 = (
    "16f980299362fdc7d018e37f1988826dffb66f30aaf88dd42bf151d311c19e56"
)
_RTU_SCHEMA23_RECEIPT = (
    _ROOT / "outputs/forager/rtu_schema23_screening_v1_execution_receipt.json"
)
_RTU_SCHEMA23_RECEIPT_SHA256 = (
    "4a47bb47a2720e13455e170f0a9b539bd6ef7a798f7c982d0837d5af56af0bc7"
)
_RTU_CAPTURE_CORRECTION = (
    _ROOT / "outputs/forager/rtu_rtrl_500k_dev4/capture-correction.v1.json"
)
_RTU_CAPTURE_CORRECTION_SHA256 = (
    "9d05801f80eaece22eade3e329d097bf1c4da7351f6234c0a38f5d71d3dd47fd"
)
_SUPERSEDED_PAIRED_RECEIPT = (
    _ROOT / "outputs/forager/dqn_fov_500k_dev_seeds2000001_2000004"
    "/DEVELOPMENT_MANIFEST.json"
)
_SUPERSEDED_PAIRED_RECEIPT_SHA256 = (
    "634985cc8902ab957eaac5708296bf979b810290081144d782097f05ed51ff74"
)
_PAIRED_RECEIPT = (
    _ROOT
    / "outputs/forager/dqn_fov_500k_dev_seeds2000001_2000004_reconciled"
    "/receipt.v1.json"
)
_PAIRED_RECEIPT_SHA256 = (
    "52046a7ddfd304e4b1f87ec6a961d05620f6d7b8027494c212796915f67befa9"
)
_SCREENING_ROOT = _ROOT / "outputs/forager/fov_baseline_screening_v1"
_SCREENING_PROTOCOL = _SCREENING_ROOT / "PROTOCOL.json"
_SCREENING_PROTOCOL_SHA256 = (
    "b94c906f9d3c1f2abf226d7049bd9305700d7f5a81012b07b36cc10d458d3174"
)
_SCREENING_SMOKE_RECEIPT = (
    _SCREENING_ROOT / "CONFIGURATION_SMOKE_RECEIPT.json"
)
_SCREENING_SMOKE_RECEIPT_SHA256 = (
    "0513f141e8c186981957d3570f9da6fdfa6d4eafe1140c02615f51db437bbcf2"
)


@pytest.mark.unit
def test_rtu_rtrl_500k_receipt_is_exact_and_structurally_nonpromoting() -> None:
    raw = _RTU_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _RTU_RECEIPT_SHA256
    payload = json.loads(raw)

    assert payload["schema_version"] == (
        "alberta.forager_rtu_rtrl_development_receipt.v1"
    )
    assert payload["status"] == "complete"
    assert payload["evidence_class"] == "development"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["execution"]["exit_code"] == 0
    assert payload["runtime"]["backend"] == "gpu"
    assert payload["protocol"]["seeds"] == [2000001, 2000002, 2000003, 2000004]

    runs = payload["runs"]
    assert [run["seed"] for run in runs] == payload["protocol"]["seeds"]
    assert all(run["steps"] == payload["protocol"]["steps"] for run in runs)
    assert all(
        run["mean_reward"] == run["total_reward"] / run["steps"] for run in runs
    )
    assert all(
        run["final_window_mean_reward"] == run["curve_window_reward"][-1]
        for run in runs
    )
    assert all(
        len(run["curve_steps"])
        == len(run["curve_ewm_reward"])
        == len(run["curve_window_reward"])
        == 11
        for run in runs
    )

    values = np.asarray(
        [run["fov_last_10pct_ema_auc"] for run in runs],
        dtype=np.float64,
    )
    summary = payload["summary"]
    assert summary["metric"] == "fov_last_10pct_ema_auc"
    assert summary["n"] == values.size
    assert summary["mean"] == float(values.mean())
    assert summary["std_sample"] == float(values.std(ddof=1))
    assert summary["min"] == float(values.min())
    assert summary["max"] == float(values.max())

    capture = payload["execution"]["capture"]
    assert capture["canonical_seven_output_lines_bytes"] == 3262
    assert capture["canonical_seven_output_lines_sha256"] == (
        "0e9df9cb0724a70d23f5ea48e52e05a1304cf76c4fa91e353be88d9ce9cac71c"
    )
    assert any(
        "cannot support a scientific, SOTA" in item
        for item in payload["limitations"]
    )


@pytest.mark.unit
def test_causal_map_500k_receipt_recomputes_and_remains_nonpromoting() -> None:
    raw = _CAUSAL_MAP_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _CAUSAL_MAP_RECEIPT_SHA256
    payload = json.loads(raw)

    assert payload["schema_version"] == (
        "alberta.forager_causal_map_development_receipt.v1"
    )
    assert payload["status"] == "complete_unsealed"
    assert payload["evidence_class"] == "open_development"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["execution"]["exit_code"] == 0
    assert payload["execution"]["raw_reward_trace_persisted"] is False
    assert payload["runtime"]["backend"] == "cpu"
    assert payload["runtime"]["oci_image_id"] == (
        "sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768"
    )
    assert payload["protocol"]["seeds"] == [2000001, 2000002, 2000003, 2000004]

    runs = payload["runs"]
    assert [run["seed"] for run in runs] == payload["protocol"]["seeds"]
    for metric in (
        "fov_last_10pct_ema_auc",
        "mean_reward",
        "final_window_mean_reward",
    ):
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        summary = payload["summary"][metric]
        assert summary["mean"] == pytest.approx(
            float(values.mean()), rel=0.0, abs=1e-15
        )
        assert summary["sample_standard_deviation"] == pytest.approx(
            float(values.std(ddof=1)), rel=0.0, abs=1e-15
        )
        assert summary["minimum"] == float(values.min())
        assert summary["maximum"] == float(values.max())

    reference = payload["descriptive_only_comparison"]["reference"]
    assert (_CAUSAL_MAP_RECEIPT.parent / reference["receipt_path"]).resolve() == (
        _RTU_RECEIPT
    )
    assert reference["receipt_sha256"] == hashlib.sha256(
        _RTU_RECEIPT.read_bytes()
    ).hexdigest()
    rtu = json.loads(_RTU_RECEIPT.read_bytes())
    causal_values = np.asarray(
        [run["fov_last_10pct_ema_auc"] for run in runs],
        dtype=np.float64,
    )
    rtu_values = np.asarray(
        [run["fov_last_10pct_ema_auc"] for run in rtu["runs"]],
        dtype=np.float64,
    )
    differences = causal_values - rtu_values
    comparison = payload["descriptive_only_comparison"]
    np.testing.assert_array_equal(
        differences,
        np.asarray(comparison["paired_causal_map_minus_rtu_fov"]),
    )
    assert comparison["mean_difference"] == pytest.approx(
        float(differences.mean()), rel=0.0, abs=1e-15
    )
    assert comparison["sample_standard_deviation"] == pytest.approx(
        float(differences.std(ddof=1)), rel=0.0, abs=1e-15
    )
    assert comparison["positive_seed_count"] == int(
        np.count_nonzero(differences > 0.0)
    )
    assert comparison["inferential_claim_allowed"] is False
    assert any("SOTA claim" in item for item in payload["limitations"])


@pytest.mark.unit
def test_rtu_schema23_screening_receipt_binds_raw_trace_matrix() -> None:
    raw = _RTU_SCHEMA23_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _RTU_SCHEMA23_RECEIPT_SHA256
    receipt = json.loads(raw)
    unhashed = {
        key: value
        for key, value in receipt.items()
        if key != "payload_sha256"
    }
    canonical = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == receipt["payload_sha256"]

    assert receipt["status"] == "complete_open_development_unsealed"
    protocol = receipt["protocol"]
    assert set(protocol["seeds"]).isdisjoint(
        protocol["held_out_evaluation_seeds_untouched"]
    )
    assert protocol["steps_per_seed"] == 500_000
    execution = receipt["execution"]
    execution_root = _ROOT / execution["directory"]
    report = execution_root / "report.json"
    manifest = execution_root / "matrix-manifest.json"
    assert hashlib.sha256(report.read_bytes()).hexdigest() == (
        execution["report_file_sha256"]
    )
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        execution["execution_manifest_file_sha256"]
    )

    inventory_digest = hashlib.sha256()
    files = sorted(
        (path for path in execution_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(execution_root).as_posix(),
    )
    assert len(files) == execution["artifact_file_count"]
    assert sum(path.stat().st_size for path in files) == (
        execution["artifact_total_bytes"]
    )
    for path in files:
        relative = path.relative_to(execution_root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory_digest.update(
            f"{relative}\0{path.stat().st_size}\0{digest}\n".encode()
        )
        assert path.stat().st_mode & 0o777 == 0o444
        assert path.stat().st_nlink == 1
    for path in [execution_root, *execution_root.rglob("*")]:
        if path.is_dir():
            assert path.stat().st_mode & 0o777 == 0o555
            assert not path.is_symlink()
    assert inventory_digest.hexdigest() == execution["artifact_tree_sha256"]

    results = receipt["results"]
    summaries: dict[str, tuple[float, float]] = {}
    rule = protocol["selection_rule"]
    for variant_id, variant in results["variants"].items():
        values = np.asarray(list(variant["per_seed"].values()), dtype=np.float64)
        assert variant["mean"] == float(values.mean())
        rng = np.random.default_rng(rule["bootstrap_seed"])
        indices = rng.integers(
            0,
            values.size,
            size=(rule["bootstrap_resamples"], values.size),
        )
        means = values[indices].mean(axis=1)
        tail = (1.0 - rule["confidence"]) / 2.0
        low, high = np.quantile(means, [tail, 1.0 - tail])
        assert variant["ci_low"] == float(low)
        assert variant["ci_high"] == float(high)
        summaries[variant_id] = (float(low), float(values.mean()))
    expected_rank = sorted(
        summaries,
        key=lambda variant_id: (
            -summaries[variant_id][0],
            variant_id,
        ),
    )
    assert results["rank_order"] == expected_rank
    assert results["selected_variant_id"] == expected_rank[0]

    for discarded in receipt["discarded_attempts"]:
        root = _ROOT / discarded["directory"]
        trace_paths = sorted(root.glob("reward-traces/**/*.npz"))
        assert len(trace_paths) == discarded["uncommitted_trace_file_count"]
        assert sorted(path.stat().st_size for path in trace_paths) == (
            discarded["uncommitted_trace_sizes"]
        )
        assert not list(root.glob("batches/**/*.json"))
        assert not (root / "report.json").exists()
        assert discarded["reward_contents_inspected"] is False
        assert discarded["eligible_for_results"] is False
        for path in [root, *root.rglob("*")]:
            expected_mode = 0o444 if path.is_file() else 0o555
            assert path.stat().st_mode & 0o777 == expected_mode


@pytest.mark.unit
def test_rtu_capture_correction_binds_receipt_and_exact_seven_lines() -> None:
    raw = _RTU_CAPTURE_CORRECTION.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _RTU_CAPTURE_CORRECTION_SHA256
    correction = json.loads(raw)
    receipt_raw = _RTU_RECEIPT.read_bytes()
    receipt = json.loads(receipt_raw)

    assert correction["schema_version"] == (
        "alberta.forager_rtu_rtrl_capture_correction.v1"
    )
    assert correction["scientific_promotion_allowed"] is False
    binding = correction["receipt_binding"]
    assert _ROOT / binding["path"] == _RTU_RECEIPT
    assert binding["size_bytes"] == len(receipt_raw)
    assert binding["sha256"] == hashlib.sha256(receipt_raw).hexdigest()
    assert binding["schema_version"] == receipt["schema_version"]

    field = correction["corrected_field"]
    capture = receipt["execution"]["capture"]
    assert field["recorded_value"] == {
        "bytes": capture["canonical_seven_output_lines_bytes"],
        "sha256": capture["canonical_seven_output_lines_sha256"],
    }
    runtime_line = "RUNTIME " + json.dumps(
        {
            name: receipt["runtime"][name]
            for name in ("jax", "backend", "devices")
        }
    )
    result_lines = [
        "RESULT "
        + json.dumps(
            {**run, "variant": "h8_default_continuous"},
            sort_keys=True,
        )
        for run in receipt["runs"]
    ]
    summary_line = "SUMMARY " + json.dumps(
        {
            name: value
            for name, value in receipt["summary"].items()
            if name != "metric"
        },
        sort_keys=True,
    )
    execution = receipt["execution"]
    footer_line = (
        f"RTU_500K_DEV_WALL={execution['wall_seconds']:.2f} "
        f"RTU_500K_DEV_MAX_RSS_KB={execution['max_rss_kb']} "
        f"EXIT={execution['exit_code']}"
    )
    canonical = (
        "\n".join([runtime_line, *result_lines, summary_line, footer_line]) + "\n"
    ).encode()
    assert field["canonical_output_line_count"] == 7
    assert field["canonical_output_lines_bytes"] == len(canonical) == 3254
    assert (
        field["canonical_output_lines_sha256"]
        == hashlib.sha256(canonical).hexdigest()
        == "0468cc3e32b00b6d13bf2603402ea447b7b38acc26224aba012df659bff52e51"
    )
    assert field["excluded_serialization_artifacts"] == {
        "value": "null",
        "count": 2,
        "total_bytes": 8,
    }


@pytest.mark.unit
def test_reconciled_dqn_rtu_receipt_recomputes_and_remains_nonpromoting() -> None:
    raw = _PAIRED_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _PAIRED_RECEIPT_SHA256
    payload = json.loads(raw)

    assert payload["schema_version"] == (
        "alberta.forager_unsealed_development_comparison_receipt.v1"
    )
    assert payload["artifact_type"] == (
        "reconciled_unsealed_forager_development_comparison_receipt"
    )
    assert payload["status"] == "development_only_unsealed"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["admissible_as_official_evidence"] is False
    assert payload["reason_codes"] == [
        "post_output_comparator_binding",
        "development_runtime_not_independently_qualified",
        "runtime_envelope_mismatch",
        "representation_resource_mismatch",
        "rtu_raw_trace_absent",
        "open_unpreregistered_n4",
        "nonexact_paper_baseline",
    ]
    assert len(payload["reason_not_official"]) == len(payload["reason_codes"])
    assert payload["comparison_contract"]["seeds"] == [
        2000001,
        2000002,
        2000003,
        2000004,
    ]
    assert payload["comparison_contract"]["seed_and_metric_matched"] is True
    assert payload["comparison_contract"]["runtime_matched"] is False
    assert (
        payload["comparison_contract"]["representation_and_resource_matched"]
        is False
    )

    superseded = payload["superseded_receipt"]
    superseded_raw = _SUPERSEDED_PAIRED_RECEIPT.read_bytes()
    assert _ROOT / superseded["path"] == _SUPERSEDED_PAIRED_RECEIPT
    assert superseded["size_bytes"] == len(superseded_raw)
    assert superseded["sha256"] == _SUPERSEDED_PAIRED_RECEIPT_SHA256
    assert superseded["sha256"] == hashlib.sha256(superseded_raw).hexdigest()
    assert superseded["preserved_read_only"] is True

    rtu_raw = _RTU_RECEIPT.read_bytes()
    rtu_receipt = json.loads(rtu_raw)
    rtu_binding = payload["alberta"]["receipt_binding"]
    assert _ROOT / rtu_binding["path"] == _RTU_RECEIPT
    assert rtu_binding["size_bytes"] == len(rtu_raw)
    assert rtu_binding["sha256"] == hashlib.sha256(rtu_raw).hexdigest()
    assert rtu_binding["schema_version"] == rtu_receipt["schema_version"]

    correction_raw = _RTU_CAPTURE_CORRECTION.read_bytes()
    correction = json.loads(correction_raw)
    correction_binding = payload["alberta"]["capture_correction_binding"]
    assert _ROOT / correction_binding["path"] == _RTU_CAPTURE_CORRECTION
    assert correction_binding["size_bytes"] == len(correction_raw)
    assert correction_binding["sha256"] == hashlib.sha256(
        correction_raw
    ).hexdigest()
    assert correction_binding["schema_version"] == correction["schema_version"]

    expected_alberta_runs = [
        {
            "seed": run["seed"],
            "reward_sum": run["total_reward"],
            "mean_reward": run["mean_reward"],
            "fov_last_10pct_ema_auc": run["fov_last_10pct_ema_auc"],
            "final_window_mean_reward": run["final_window_mean_reward"],
        }
        for run in rtu_receipt["runs"]
    ]
    assert payload["alberta"]["runs"] == expected_alberta_runs
    direct_sources = rtu_receipt["execution"][
        "direct_source_sha256_pre_and_post_run"
    ]
    assert payload["alberta"]["source"]["forager_py_sha256"] == direct_sources[
        "alberta_framework/benchmarks/forager.py"
    ]
    assert payload["alberta"]["source"][
        "recurrent_trace_actor_critic_py_sha256"
    ] == direct_sources[
        "alberta_framework/core/recurrent_trace_actor_critic.py"
    ]
    assert payload["alberta"]["algorithm"] == "alberta_rtu_rtrl_ac"
    assert payload["alberta"]["variant"] == "h8_default_continuous"
    assert payload["alberta"]["resolved_execution_config"]["features"][
        "resolved_fov_feature_dimensions"
    ] == 254

    dqn_values: list[float] = []
    for run in payload["dqn"]["runs"]:
        path = _ROOT / run["npz_path"]
        assert path.stat().st_size == run["npz_size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run["npz_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            rewards = np.asarray(archive["rewards"])
        assert rewards.shape == (500_000,)
        assert rewards.dtype == np.dtype(np.float16)
        assert hashlib.sha256(rewards.tobytes(order="C")).hexdigest() == (
            run["reward_trace_sha256"]
        )
        rewards64 = rewards.astype(np.float64)
        assert float(rewards64.sum()) == run["reward_sum"]
        imported = import_official_foragax_npz(
            OfficialForagaxRunSpec(
                agent=payload["dqn"]["algorithm"],
                seed=run["seed"],
                path=path,
                environment=ForagerEnvConfig.paper_field_of_view(),
                source_repository=FORAGAX_AGENTS_URL,
                source_commit=payload["dqn"]["upstream_source_commit"],
                config_path=payload["dqn"]["config"]["path"],
                config_sha256=payload["dqn"]["config"]["sha256"],
                expected_archive_sha256=run["npz_sha256"],
                expected_steps=payload["comparison_contract"]["steps_per_seed"],
            ),
            ewm_decay=payload["comparison_contract"]["primary_metric"][
                "ema_decay"
            ],
            record_every=50_000,
            final_window=payload["comparison_contract"]["secondary_metric"][
                "window_steps"
            ],
        )
        assert imported.total_reward == run["reward_sum"]
        assert imported.fov_last_10pct_ema_auc == run["fov_last_10pct_ema_auc"]
        assert imported.final_window_mean_reward == run["final_window_mean_reward"]
        dqn_values.append(imported.fov_last_10pct_ema_auc)

    dqn = np.asarray(dqn_values, dtype=np.float64)
    alberta = np.asarray(
        [
            run["fov_last_10pct_ema_auc"]
            for run in payload["alberta"]["runs"]
        ],
        dtype=np.float64,
    )
    paired = alberta - dqn
    summary = payload["paired_development_summary"]
    np.testing.assert_array_equal(
        paired,
        np.asarray(
            summary["fov_last_10pct_ema_auc_differences"],
            dtype=np.float64,
        ),
    )
    assert float(dqn.mean()) == payload["dqn"]["summary"]["mean"]
    assert float(dqn.std(ddof=1)) == payload["dqn"]["summary"]["std_sample"]
    assert float(alberta.mean()) == payload["alberta"]["summary"]["mean"]
    assert float(alberta.std(ddof=1)) == payload["alberta"]["summary"]["std_sample"]
    assert float(paired.mean()) == summary["mean_difference"]
    assert float(paired.std(ddof=1)) == summary["sample_standard_deviation"]
    assert int(np.count_nonzero(paired > 0.0)) == summary["wins"] == 4
    assert summary["inferential_statistics_admissible"] is False
    assert summary["speed_comparison_admissible"] is False
    assert summary["causal_attribution_admissible"] is False
    assert "not an official, inferential, causal, speed, or SOTA claim" in summary[
        "interpretation"
    ]


@pytest.mark.unit
def test_fov_baseline_screening_protocol_freezes_common_control_configs() -> None:
    raw = _SCREENING_PROTOCOL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _SCREENING_PROTOCOL_SHA256
    protocol = json.loads(raw)

    assert protocol["status"] == "configuration_frozen_execution_pending"
    assert protocol["evidence_class"] == "open_development"
    assert protocol["scientific_promotion_allowed"] is False
    assert protocol["task"] == {
        "foragax_distribution": "continual-foragax",
        "foragax_version": "0.55.0",
        "env_id": "ForagaxTwoBiomeLarge-v1",
        "aperture_size": 9,
        "steps": 100_000,
        "seeds": [2_000_001, 2_000_002],
    }
    assert protocol["selection_rule"]["advance_count"] == 3
    assert protocol["runtime"]["qualified_production_image"] is False

    configurations = protocol["configurations"]
    assert len(configurations) == 11
    assert len({item["path"] for item in configurations}) == len(configurations)
    for item in configurations:
        path = _SCREENING_ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        config = json.loads(path.read_bytes())
        assert config["problem"] == "Foragax"
        assert config["total_steps"] == 100_000
        hypers = config["metaParameters"]
        assert hypers["environment"] == {
            "env_id": "ForagaxTwoBiomeLarge-v1",
            "aperture_size": 9,
        }
        assert hypers["experiment"] == {
            "seed_offset": 0,
            "ntk_freq": 0,
            "x_ref_steps": 0,
        }
        assert hypers["initial_epsilon"] == 1.0
        assert hypers["final_epsilon"] == 0.05
        assert hypers["epsilon_linear_decay"] == 400_000
        assert hypers["target_refresh"] == 128
        assert hypers["buffer_size"] == 10_000
        assert hypers["buffer_min_size"] == 32
        assert hypers["batch"] == 32
        assert hypers["gamma"] == 0.99
        assert hypers["update_freq"] == 4
        assert hypers["optimizer"] == {
            "name": "Adam",
            "alpha": 0.0003,
            "beta1": 0.9,
            "beta2": 0.9,
            "eps": 1e-08,
        }


@pytest.mark.unit
def test_fov_baseline_configuration_smoke_is_exact_and_nonpromoting() -> None:
    raw = _SCREENING_SMOKE_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        _SCREENING_SMOKE_RECEIPT_SHA256
    )
    receipt = json.loads(raw)
    protocol = json.loads(_SCREENING_PROTOCOL.read_bytes())

    assert receipt["status"] == "passed_development_only"
    assert receipt["scientific_promotion_allowed"] is False
    assert receipt["protocol"]["sha256"] == _SCREENING_PROTOCOL_SHA256
    assert receipt["scope"]["max_environment_steps"] == 1
    assert receipt["runtime"]["backend"] == "cpu"
    assert receipt["runtime"]["network"] == "none"
    assert receipt["runtime"]["root_filesystem"] == "read_only"
    assert receipt["runtime"]["environment_overrides"][
        "NVIDIA_VISIBLE_DEVICES"
    ] == "void"
    assert receipt["runtime"]["per_container_assertions"][
        "jax_default_backend"
    ] == "cpu"
    assert receipt["runtime"]["per_container_assertions"][
        "nvidia_device_glob"
    ] == []
    assert len(receipt["nonqualification_reasons"]) == 5

    expected = {
        item["path"]: item["sha256"]
        for item in protocol["configurations"]
    }
    actual = {
        item["config"]: item["sha256"]
        for item in receipt["results"]
    }
    assert actual == expected
    assert all(item["exit_status"] == 0 for item in receipt["results"])
    assert receipt["summary"] == {
        "configurations_expected": 11,
        "configurations_passed": 11,
        "configurations_failed": 0,
    }
