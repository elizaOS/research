# mypy: disable-error-code="attr-defined,call-arg,index"
"""Strict development-only recurrent world-model calibration contracts."""

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
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelEnsemble,
    RecurrentLatentWorldModelEnsembleConfig,
)
from alberta_framework.evaluation.world_model_calibration import (
    RECURRENT_WORLD_MODEL_CALIBRATION_CHECKPOINT_SCHEMA,
    RECURRENT_WORLD_MODEL_CALIBRATION_REPORT_SCHEMA,
    RecurrentWorldModelCalibrationEvent,
    RecurrentWorldModelCalibrationProbeSet,
    WorldModelCalibrationConfig,
    build_recurrent_world_model_calibration_report,
    canonical_recurrent_world_model_calibration_report_bytes,
    frozen_world_model_state_sha256,
    load_recurrent_world_model_calibration_report,
    load_recurrent_world_model_calibration_snapshot_checkpoint,
    reconstruct_recurrent_world_model_calibration_summary,
    save_recurrent_world_model_calibration_report,
    save_recurrent_world_model_calibration_snapshot_checkpoint,
    validate_recurrent_world_model_calibration_report,
)

pytestmark = pytest.mark.unit


def _model() -> RecurrentLatentWorldModelEnsemble:
    return RecurrentLatentWorldModelEnsemble(
        RecurrentLatentWorldModelEnsembleConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=3,
            ensemble_size=2,
            learning_rate=0.01,
            bootstrap_probability=0.75,
            uncertainty_warmup_steps=1,
            max_updates=8,
        )
    )


def _events() -> tuple[RecurrentWorldModelCalibrationEvent, ...]:
    return (
        RecurrentWorldModelCalibrationEvent(
            event_id="event-0",
            observation=(0.1, 0.2),
            action=0,
            bootstrap_observation_target=(0.2, 0.3),
            reward_target=0.1,
            continuation_target=0.9,
            terminated=False,
            truncated=False,
            next_decision_observation=(0.2, 0.3),
            partition="in_distribution",
        ),
        RecurrentWorldModelCalibrationEvent(
            event_id="event-1",
            observation=(0.2, 0.3),
            action=1,
            bootstrap_observation_target=(0.25, 0.35),
            reward_target=0.2,
            continuation_target=0.8,
            terminated=False,
            truncated=True,
            next_decision_observation=(-0.1, 0.0),
            partition="ood",
        ),
        RecurrentWorldModelCalibrationEvent(
            event_id="event-2",
            observation=(-0.1, 0.0),
            action=0,
            bootstrap_observation_target=(0.0, 0.1),
            reward_target=-0.1,
            continuation_target=0.0,
            terminated=True,
            truncated=False,
            next_decision_observation=(0.4, 0.4),
            partition="ood",
        ),
    )


def _probes() -> RecurrentWorldModelCalibrationProbeSet:
    return RecurrentWorldModelCalibrationProbeSet(
        probe_set_id="heldout-recurrent-v1",
        events=_events(),
    )


def _config() -> WorldModelCalibrationConfig:
    return WorldModelCalibrationConfig(
        epistemic_binning="equal_count",
        epistemic_bin_count=2,
        minimum_descriptive_bin_count=1,
        coverage_fractions=(0.5, 1.0),
        state_norm_edges=(0.15, 0.3),
        action_region_by_action=(0, 1),
        max_one_step_cases=8,
    )


@lru_cache(maxsize=1)
def _fixture() -> tuple[
    RecurrentLatentWorldModelEnsemble,
    Any,
    WorldModelCalibrationConfig,
    RecurrentWorldModelCalibrationProbeSet,
    dict[str, object],
]:
    model = _model()
    state = model.init(jr.key(7))
    config = _config()
    probes = _probes()
    report = build_recurrent_world_model_calibration_report(
        model,
        state,
        config,
        probes,
    )
    return model, state, config, probes, report


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert str(left_structure) == str(right_structure)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jnp.issubdtype(left_leaf.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jr.key_data(left_leaf), jr.key_data(right_leaf))
        else:
            np.testing.assert_array_equal(left_leaf, right_leaf)


def test_recurrent_probe_contract_is_ordered_strict_and_evaluator_owned() -> None:
    probes = _probes()
    assert RecurrentWorldModelCalibrationProbeSet.from_config(probes.to_config()) == probes
    assert probes.ownership == "evaluator-owned-held-out"
    assert probes.learner_use == "never"
    assert probes.regime_identifiers_available is False

    with pytest.raises(ValueError, match="regime"):
        dataclasses.replace(_events()[0], event_id="regime-0")
    with pytest.raises(ValueError, match="canonical finite JSON float"):
        dataclasses.replace(_events()[0], reward_target=1)
    with pytest.raises(ValueError, match="preceding next-decision"):
        RecurrentWorldModelCalibrationProbeSet(
            probe_set_id="stale-trace",
            events=(_events()[0], dataclasses.replace(_events()[1], observation=(9.0, 9.0))),
        )
    with pytest.raises(ValueError, match="zero exactly"):
        dataclasses.replace(_events()[0], continuation_target=0.0)


def test_report_scores_predict_before_update_and_advances_isolated_copy_once() -> None:
    model, state, _, probes, report = _fixture()
    before = frozen_world_model_state_sha256(state)
    assert report["schema"] == RECURRENT_WORLD_MODEL_CALIBRATION_REPORT_SCHEMA
    payload = report["payload"]
    assert payload["development_only"] is True
    assert payload["assessment_status"] == "not-assessed"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["calibration_claimed"] is False
    assert payload["performance_thresholds_applied"] is False
    assert frozen_world_model_state_sha256(state) == before

    first = payload["raw_trace"]["events"][0]
    decision = model.decide(
        state,
        model.start(state, jnp.asarray(probes.events[0].observation, dtype=jnp.float32)),
        jnp.asarray(probes.events[0].action, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        first["members"]["grounded_vectors"],
        decision.prediction.member_mean_predictions,
    )
    np.testing.assert_array_equal(
        first["aleatoric"]["member_variances"],
        decision.prediction.member_aleatoric_variances,
    )
    assert first["event_count_before"] == 0
    assert first["event_count_after"] == 1
    assert first["prequential_update"]["recurrent_advanced_once"] is True
    final = payload["final_isolated_state"]
    assert final["event_count"] == 3
    assert final["recurrent_advance_count"] == 3
    assert final["boundary_count"] == 2

    validation = validate_recurrent_world_model_calibration_report(
        report,
        model=model,
        state=state,
        probes=probes,
    )
    assert validation.valid, validation.errors


def test_raw_id_ood_region_and_warmup_summaries_reconstruct_without_claim() -> None:
    model, _, config, _, report = _fixture()
    payload = report["payload"]
    raw = payload["raw_trace"]
    summary = payload["summary"]
    reconstructed = reconstruct_recurrent_world_model_calibration_summary(
        raw,
        config,
        observation_dim=model.config.observation_dim,
        ensemble_size=model.config.ensemble_size,
        action_regions=(0, 1),
    )
    assert reconstructed == summary
    assert [item["partition"] for item in summary["partition_metrics"]] == [
        "in_distribution",
        "ood",
    ]
    assert [item["count"] for item in summary["partition_metrics"]] == [1, 2]
    assert len(summary["state_region_metrics"]) == 3
    assert len(summary["action_region_metrics"]) == 2
    assert summary["epistemic_diagnostics"]["warmup_excluded_count"] == 1
    assert summary["epistemic_diagnostics"]["applicable_event_count"] == 2
    aleatoric = summary["aleatoric_variance_head_diagnostics"]
    assert aleatoric["available"] is True
    assert aleatoric["calibrated_likelihood_claimed"] is False
    assert "does not establish calibrated likelihood" in aleatoric["interpretation"]
    assert len(aleatoric["grounded_head_diagnostics"]) == 4
    assert summary["applicability"]["probabilistic_calibration_established"] is False
    assert "state-of-the-art" not in json.dumps(report).lower()
    assert "sota" not in json.dumps(report).lower()


def test_tampering_rehashing_and_noncanonical_numeric_types_fail_closed() -> None:
    model, state, _, probes, report = _fixture()
    tampered = copy.deepcopy(report)
    tampered["payload"]["raw_trace"]["events"][0]["action"] = 1
    validation = validate_recurrent_world_model_calibration_report(tampered)
    assert not validation.valid

    rehashed = copy.deepcopy(tampered)
    payload = rehashed["payload"]
    payload["hashes"]["raw_trace_sha256"] = _digest(payload["raw_trace"])
    rehashed["payload_sha256"] = _digest(payload)
    replay = validate_recurrent_world_model_calibration_report(
        rehashed,
        model=model,
        state=state,
        probes=probes,
    )
    assert not replay.valid
    assert any("match" in error or "replay" in error for error in replay.errors)

    wrong_type = copy.deepcopy(report)
    payload = wrong_type["payload"]
    payload["config"]["coverage_fractions"][0] = 1
    payload["hashes"]["config_sha256"] = _digest(payload["config"])
    wrong_type["payload_sha256"] = _digest(payload)
    strict = validate_recurrent_world_model_calibration_report(wrong_type)
    assert not strict.valid
    assert any("canonical JSON floats" in error for error in strict.errors)


def test_nonfinite_out_of_range_and_corrupt_inputs_fail_atomically() -> None:
    model = _model()
    state = model.init(jr.key(11))
    original_hash = frozen_world_model_state_sha256(state)
    with pytest.raises(ValueError, match="canonical finite"):
        dataclasses.replace(_events()[0], reward_target=float("nan"))
    out_of_range = RecurrentWorldModelCalibrationProbeSet(
        probe_set_id="bad-action-trace",
        events=(dataclasses.replace(_events()[0], action=2),),
    )
    with pytest.raises(ValueError, match="outside model action space"):
        build_recurrent_world_model_calibration_report(
            model,
            state,
            _config(),
            out_of_range,
        )
    corrupt = state.replace(event_count=jnp.asarray(1, dtype=jnp.int32))
    with pytest.raises(ValueError, match="invalid"):
        build_recurrent_world_model_calibration_report(
            model,
            corrupt,
            _config(),
            RecurrentWorldModelCalibrationProbeSet(
                probe_set_id="one-event-trace",
                events=(_events()[0],),
            ),
        )
    assert frozen_world_model_state_sha256(state) == original_hash


def test_resource_accounting_is_bounded_exact_and_exposes_no_labels() -> None:
    model, _, config, _, report = _fixture()
    resources = report["payload"]["resource_accounting"]
    budget = model.resource_budget()
    assert resources["initial_snapshot_state_bytes"] == budget.persistent_state_bytes
    assert resources["bounded_event_capacity"] == config.max_one_step_cases
    assert resources["recorded_event_count"] == 3
    assert resources["predict_before_update_decide_calls"] == 3
    assert resources["isolated_copy_update_calls"] == 3
    assert resources["recurrent_advances"] == 3
    assert resources["member_prediction_records"] == 6
    assert resources["member_gradient_candidates"] == 6
    assert 0 <= resources["member_parameter_updates_applied"] <= 6
    assert resources["external_snapshot_mutations"] == 0
    assert resources["learner_visible_evaluator_label_reads"] == 0
    assert resources["regime_identifier_reads"] == 0
    assert resources["persistent_evaluator_state_bytes"] == 0


def test_canonical_report_and_source_bound_snapshot_roundtrip(tmp_path: Path) -> None:
    model, state, _, _, report = _fixture()
    report_path = tmp_path / "recurrent-report.json"
    save_recurrent_world_model_calibration_report(report_path, report)
    assert report_path.read_bytes() == canonical_recurrent_world_model_calibration_report_bytes(
        report
    )
    assert load_recurrent_world_model_calibration_report(report_path) == report
    with pytest.raises(FileExistsError, match="overwrite"):
        save_recurrent_world_model_calibration_report(report_path, report)

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        load_recurrent_world_model_calibration_report(pretty)

    checkpoint = tmp_path / "recurrent-snapshot"
    save_recurrent_world_model_calibration_snapshot_checkpoint(model, state, checkpoint)
    metadata = load_checkpoint_metadata(checkpoint)
    assert metadata["schema"] == RECURRENT_WORLD_MODEL_CALIBRATION_CHECKPOINT_SCHEMA
    restored_model, restored_state = (
        load_recurrent_world_model_calibration_snapshot_checkpoint(checkpoint)
    )
    assert restored_model.to_config() == model.to_config()
    _assert_tree_equal(restored_state, state)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_recurrent_world_model_calibration_snapshot_checkpoint(model, state, checkpoint)

    tampered_metadata = copy.deepcopy(metadata)
    tampered_metadata["snapshot_sha256"] = "0" * 64
    tampered_checkpoint = tmp_path / "tampered-recurrent-snapshot"
    save_checkpoint(state, tampered_checkpoint, metadata=tampered_metadata)
    with pytest.raises(ValueError, match="digest"):
        load_recurrent_world_model_calibration_snapshot_checkpoint(tampered_checkpoint)
