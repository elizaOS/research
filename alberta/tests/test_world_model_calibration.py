# mypy: disable-error-code="attr-defined,call-arg,index"
"""Focused contracts for frozen world-model calibration diagnostics."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import load_checkpoint_metadata, save_checkpoint
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
)
from alberta_framework.evaluation.world_model_calibration import (
    WORLD_MODEL_CALIBRATION_CHECKPOINT_SCHEMA,
    WORLD_MODEL_CALIBRATION_CONFIG_SCHEMA,
    WORLD_MODEL_CALIBRATION_REPORT_SCHEMA,
    WorldModelCalibrationCase,
    WorldModelCalibrationConfig,
    WorldModelCalibrationProbeSet,
    WorldModelOpenLoopProbe,
    build_world_model_calibration_report,
    canonical_world_model_calibration_report_bytes,
    frozen_world_model_state_sha256,
    load_world_model_calibration_report,
    load_world_model_calibration_snapshot_checkpoint,
    reconstruct_world_model_calibration_summary,
    save_world_model_calibration_report,
    save_world_model_calibration_snapshot_checkpoint,
    validate_world_model_calibration_report,
)

pytestmark = pytest.mark.unit


def _model_config() -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=0.95,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )


@lru_cache(maxsize=1)
def _ensemble_snapshot() -> tuple[WorldModelEnsemble, Any]:
    signals = LearningSignalEstimatorConfig(
        ensemble_size=2,
        target_dim=4,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        fast_loss_decay=0.5,
        slow_loss_decay=0.9,
        max_input_magnitude=100.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    ensemble = WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=_model_config(),
            signal_estimator=signals,
            ensemble_size=2,
            bootstrap_probability=0.5,
            residual_variance_decay=0.8,
            residual_variance_warmup_steps=1,
            residual_variance_floor=1.0e-6,
        )
    )
    state = ensemble.init(jr.key(17))
    for index in range(2):
        observation = jnp.asarray([0.1 + 0.1 * index, -0.2], dtype=jnp.float32)
        state = ensemble.update(
            state,
            observation,
            jnp.asarray(index % 2, dtype=jnp.int32),
            jnp.asarray(0.2, dtype=jnp.float32),
            jnp.asarray(0.9, dtype=jnp.float32),
            observation + jnp.asarray([0.05, 0.02], dtype=jnp.float32),
        ).state
    return ensemble, state


@lru_cache(maxsize=1)
def _single_snapshot() -> tuple[ActionConditionedWorldModel, Any]:
    model = ActionConditionedWorldModel(_model_config())
    return model, model.init(jr.key(23))


def _cases(count: int = 5) -> tuple[WorldModelCalibrationCase, ...]:
    return tuple(
        WorldModelCalibrationCase(
            case_id=f"heldout-{index}",
            observation=(0.05 * index, 0.15 + 0.02 * index),
            action=index % 2,
            next_observation_target=(0.05 * index + 0.03, 0.18 + 0.02 * index),
            reward_target=0.1 + 0.01 * index,
            continuation_target=0.9 if index != count - 1 else 0.0,
            partition="in_distribution" if index < max(1, count - 2) else "ood",
        )
        for index in range(count)
    )


def _probes(count: int = 5) -> WorldModelCalibrationProbeSet:
    return WorldModelCalibrationProbeSet(
        probe_set_id="heldout-action-probes-v1",
        cases=_cases(count),
    )


def _config(**overrides: Any) -> WorldModelCalibrationConfig:
    values: dict[str, Any] = {
        "epistemic_binning": "equal_count",
        "epistemic_bin_count": 3,
        "minimum_descriptive_bin_count": 2,
        "coverage_fractions": (0.5, 1.0),
        "state_norm_edges": (0.1, 0.3),
        "action_region_by_action": (0, 1),
        "max_one_step_cases": 16,
    }
    values.update(overrides)
    return WorldModelCalibrationConfig(**values)


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_structure = jax.tree.flatten(_materialize_keys(right))
    assert str(left_structure) == str(right_structure)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_ensemble_report_is_raw_reconstructable_nonmutating_and_not_assessed() -> None:
    ensemble, state = _ensemble_snapshot()
    config = _config()
    probes = _probes()
    before_hash = frozen_world_model_state_sha256(state)
    report = build_world_model_calibration_report(ensemble, state, config, probes)

    assert report["schema"] == WORLD_MODEL_CALIBRATION_REPORT_SCHEMA
    payload = report["payload"]
    assert payload["development_only"] is True
    assert payload["assessment_status"] == "not-assessed"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["calibration_claimed"] is False
    assert payload["performance_thresholds_applied"] is False
    assert frozen_world_model_state_sha256(state) == before_hash

    raw = payload["raw_trace"]
    cases = raw["cases"]
    assert len(cases) == 5
    first = cases[0]
    assert first["members"]["available"] is True
    assert first["members"]["count"] == 2
    assert len(first["members"]["next_observations"]) == 2
    assert first["targets"]["continuation"] == probes.cases[0].continuation_target
    assert "regime_id" not in json.dumps(raw)

    summary = payload["summary"]
    reconstructed = reconstruct_world_model_calibration_summary(
        raw,
        config,
        observation_dim=2,
        action_regions=(0, 1),
    )
    assert reconstructed == summary
    assert summary["epistemic_diagnostics"]["available"] is True
    assert len(summary["epistemic_diagnostics"]["coverage_risk_curve"]) == 2
    residual = summary["residual_variance_proxy_diagnostics"]
    assert residual["available"] is True
    assert residual["ready"] is True
    assert residual["probabilistic_calibration_available"] is False
    assert "non-probabilistic" in residual["interpretation"]

    resources = payload["resource_accounting"]
    budget = ensemble.resource_budget(state)
    assert resources["snapshot_state_bytes"] == budget.persistent_state_bytes
    assert resources["snapshot_state_logical_scalars"] == budget.persistent_state_scalars
    assert resources["one_step_predict_calls"] == 5
    assert resources["total_predict_api_calls"] == 5
    assert resources["underlying_member_predict_calls"] == 10
    assert resources["learner_update_calls"] == 0
    assert resources["model_update_calls"] == 0
    assert resources["regime_identifier_reads"] == 0

    validation = validate_world_model_calibration_report(
        report,
        model=ensemble,
        state=state,
        probes=probes,
    )
    assert validation.valid, validation.errors
    assert validation.assessment_status == "not-assessed"


def test_single_model_marks_member_epistemic_and_residual_fields_unavailable() -> None:
    model, state = _single_snapshot()
    report = build_world_model_calibration_report(model, state, _config(), _probes(3))
    payload = report["payload"]
    cases = payload["raw_trace"]["cases"]
    for case in cases:
        assert case["members"] == {
            "available": False,
            "count": None,
            "next_observations": None,
            "rewards": None,
            "continuations": None,
            "raw_predictions": None,
        }
        assert case["epistemic"]["available"] is False
        assert case["epistemic"]["decoded_mean_disagreement"] is None
        assert case["residual_variance_proxy"]["available"] is False
    summary = payload["summary"]
    assert summary["epistemic_diagnostics"] == {
        "available": False,
        "reason": "unavailable: snapshot has no member predictions",
        "realized_error_definition": "mean squared error across decoded heads",
        "binning": None,
        "bins": [],
        "correlations": [],
        "coverage_risk_curve": [],
    }
    assert summary["residual_variance_proxy_diagnostics"]["available"] is False
    assert payload["resource_accounting"]["underlying_member_predict_calls"] == 3
    validation = validate_world_model_calibration_report(
        report,
        model=model,
        state=state,
        probes=_probes(3),
    )
    assert validation.valid, validation.errors


def test_frozen_edge_bins_and_sparse_regions_remain_explicit_without_gates() -> None:
    ensemble, state = _ensemble_snapshot()
    config = _config(
        epistemic_binning="frozen_edges",
        epistemic_bin_count=99,
        epistemic_bin_edges=(1.0e-8, 1.0, 100.0),
        minimum_descriptive_bin_count=4,
        state_norm_edges=(0.01, 0.1, 1.0, 10.0),
    )
    report = build_world_model_calibration_report(ensemble, state, config, _probes(3))
    summary = report["payload"]["summary"]
    epistemic = summary["epistemic_diagnostics"]
    assert epistemic["binning"]["method"] == "frozen_edges"
    assert len(epistemic["bins"]) == 4
    assert any(entry["count"] == 0 for entry in epistemic["bins"])
    assert all(entry["descriptive_applicable"] is False for entry in epistemic["bins"])
    state_regions = summary["state_region_metrics"]
    assert len(state_regions) == 5
    empty = next(entry for entry in state_regions if entry["count"] == 0)
    assert empty["sparse"] is True
    assert empty["all_head_mean_squared_error"] is None
    assert summary["thresholds_applied"] is False
    assert validate_world_model_calibration_report(report).valid


def test_open_loop_runs_only_with_grounded_exact_bounded_reconstruction() -> None:
    ensemble, state = _ensemble_snapshot()
    config = _config(max_rollout_probes=1, max_rollout_horizon=2)
    grounded = WorldModelOpenLoopProbe(
        probe_id="grounded-rollout",
        initial_observation=(0.1, 0.2),
        actions=(0, 1),
        target_next_observations=((0.12, 0.21), (0.14, 0.23)),
        target_rewards=(0.1, 0.2),
        target_continuations=(0.9, 0.9),
        grounded_targets_available=True,
        exact_reconstruction_available=True,
    )
    probes = WorldModelCalibrationProbeSet(
        probe_set_id="heldout-with-rollout",
        cases=_cases(3),
        open_loop_probes=(grounded,),
    )
    report = build_world_model_calibration_report(ensemble, state, config, probes)
    open_loop = report["payload"]["summary"]["open_loop_diagnostics"]
    assert open_loop["available"] is True
    assert open_loop["probe_count"] == 1
    assert open_loop["prediction_call_count"] == 2
    resources = report["payload"]["resource_accounting"]
    assert resources["one_step_predict_calls"] == 3
    assert resources["open_loop_predict_calls"] == 2
    assert resources["total_predict_api_calls"] == 5

    unavailable = dataclasses.replace(
        grounded,
        target_next_observations=(),
        target_rewards=(),
        target_continuations=(),
        grounded_targets_available=False,
        exact_reconstruction_available=False,
    )
    unavailable_probes = dataclasses.replace(probes, open_loop_probes=(unavailable,))
    unavailable_report = build_world_model_calibration_report(
        ensemble,
        state,
        config,
        unavailable_probes,
    )
    unavailable_summary = unavailable_report["payload"]["summary"][
        "open_loop_diagnostics"
    ]
    assert unavailable_summary["available"] is False
    assert "grounded targets" in unavailable_summary["reason"]
    assert unavailable_report["payload"]["resource_accounting"][
        "open_loop_predict_calls"
    ] == 0


def test_tampering_noncanonical_inputs_and_regime_identifiers_fail_closed() -> None:
    ensemble, state = _ensemble_snapshot()
    config = _config()
    probes = _probes()
    report = build_world_model_calibration_report(ensemble, state, config, probes)
    tampered = copy.deepcopy(report)
    tampered["payload"]["raw_trace"]["cases"][0]["mean_predictions"]["reward"] += 1.0
    validation = validate_world_model_calibration_report(tampered)
    assert not validation.valid
    assert any("digest" in error or "reconstruct" in error for error in validation.errors)

    rehashed = copy.deepcopy(tampered)
    payload = rehashed["payload"]
    payload["hashes"]["raw_trace_sha256"] = _digest(payload["raw_trace"])
    payload["hashes"]["summary_sha256"] = _digest(payload["summary"])
    rehashed["payload_sha256"] = _digest(payload)
    replay_validation = validate_world_model_calibration_report(
        rehashed,
        model=ensemble,
        state=state,
        probes=probes,
    )
    assert not replay_validation.valid
    assert any("replay exactly" in error for error in replay_validation.errors)

    config_payload = config.to_config()
    assert config_payload["schema"] == WORLD_MODEL_CALIBRATION_CONFIG_SCHEMA
    config_payload["extra"] = 1
    with pytest.raises(ValueError, match="fields"):
        WorldModelCalibrationConfig.from_config(config_payload)
    with pytest.raises(ValueError, match="regime"):
        WorldModelCalibrationCase(
            case_id="regime-7",
            observation=(0.0, 0.0),
            action=0,
            next_observation_target=(0.0, 0.0),
            reward_target=0.0,
            continuation_target=0.0,
            partition="ood",
        )


def test_atomic_report_and_strict_snapshot_checkpoint_roundtrips(tmp_path: Path) -> None:
    ensemble, state = _ensemble_snapshot()
    report = build_world_model_calibration_report(ensemble, state, _config(), _probes())
    report_path = tmp_path / "report.json"
    save_world_model_calibration_report(report_path, report)
    assert report_path.read_bytes() == canonical_world_model_calibration_report_bytes(report)
    assert load_world_model_calibration_report(report_path) == report
    with pytest.raises(FileExistsError, match="overwrite"):
        save_world_model_calibration_report(report_path, report)

    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        load_world_model_calibration_report(noncanonical)

    checkpoint = tmp_path / "snapshot"
    save_world_model_calibration_snapshot_checkpoint(ensemble, state, checkpoint)
    metadata = load_checkpoint_metadata(checkpoint)
    assert metadata["schema"] == WORLD_MODEL_CALIBRATION_CHECKPOINT_SCHEMA
    restored_model, restored_state = load_world_model_calibration_snapshot_checkpoint(
        checkpoint
    )
    assert isinstance(restored_model, WorldModelEnsemble)
    _assert_tree_equal(restored_state, state)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_world_model_calibration_snapshot_checkpoint(ensemble, state, checkpoint)

    tampered_metadata = copy.deepcopy(metadata)
    tampered_metadata["snapshot_sha256"] = "0" * 64
    tampered_checkpoint = tmp_path / "tampered-snapshot"
    save_checkpoint(state, tampered_checkpoint, metadata=tampered_metadata)
    with pytest.raises(ValueError, match="digest"):
        load_world_model_calibration_snapshot_checkpoint(tampered_checkpoint)
