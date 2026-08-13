"""Strict development-only retention probes for the recurrent world model."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import load_checkpoint_metadata, save_checkpoint
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelEnsemble,
    RecurrentLatentWorldModelEnsembleConfig,
    RecurrentLatentWorldModelEnsembleState,
)
from alberta_framework.evaluation.recurrent_world_model_retention import (
    RECURRENT_WORLD_MODEL_RETENTION_CHECKPOINT_SCHEMA,
    RECURRENT_WORLD_MODEL_RETENTION_REPORT_SCHEMA,
    RecurrentWorldModelRetentionConfig,
    RecurrentWorldModelRetentionEvaluator,
    RecurrentWorldModelRetentionPhase,
    RecurrentWorldModelRetentionProtocol,
    build_recurrent_world_model_retention_report,
    canonical_recurrent_world_model_retention_report_bytes,
    load_recurrent_world_model_retention_report,
    load_recurrent_world_model_retention_snapshot_checkpoint,
    reconstruct_recurrent_world_model_retention_summary,
    recurrent_world_model_retention_source_snapshot,
    save_recurrent_world_model_retention_report,
    save_recurrent_world_model_retention_snapshot_checkpoint,
    validate_recurrent_world_model_retention_report,
)
from alberta_framework.evaluation.world_model_calibration import (
    RecurrentWorldModelCalibrationEvent,
    RecurrentWorldModelCalibrationProbeSet,
    WorldModelCalibrationConfig,
    frozen_world_model_state_sha256,
)

pytestmark = pytest.mark.development


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _event(
    event_id: str,
    *,
    observation: tuple[float, float],
    action: int,
    target: tuple[float, float],
    reward: float,
    continuation: float,
    next_observation: tuple[float, float],
    partition: Literal["in_distribution", "ood"],
) -> RecurrentWorldModelCalibrationEvent:
    return RecurrentWorldModelCalibrationEvent(
        event_id=event_id,
        observation=observation,
        action=action,
        bootstrap_observation_target=target,
        reward_target=reward,
        continuation_target=continuation,
        terminated=continuation == 0.0,
        truncated=False,
        next_decision_observation=next_observation,
        partition=partition,
    )


def _probes() -> RecurrentWorldModelCalibrationProbeSet:
    return RecurrentWorldModelCalibrationProbeSet(
        probe_set_id="retention-aba-v1",
        events=(
            _event(
                "a-one",
                observation=(0.0, 0.0),
                action=0,
                target=(0.25, -0.25),
                reward=0.5,
                continuation=0.9,
                next_observation=(0.25, -0.25),
                partition="in_distribution",
            ),
            _event(
                "a-two",
                observation=(0.25, -0.25),
                action=1,
                target=(0.5, -0.5),
                reward=1.0,
                continuation=0.0,
                next_observation=(2.0, 2.0),
                partition="in_distribution",
            ),
            _event(
                "b-one",
                observation=(2.0, 2.0),
                action=1,
                target=(2.25, 1.75),
                reward=-0.5,
                continuation=0.8,
                next_observation=(2.25, 1.75),
                partition="ood",
            ),
            _event(
                "b-two",
                observation=(2.25, 1.75),
                action=0,
                target=(2.5, 1.5),
                reward=-1.0,
                continuation=0.0,
                next_observation=(0.0, 0.0),
                partition="ood",
            ),
            _event(
                "a-return-one",
                observation=(0.0, 0.0),
                action=0,
                target=(0.25, -0.25),
                reward=0.5,
                continuation=0.9,
                next_observation=(0.25, -0.25),
                partition="in_distribution",
            ),
            _event(
                "a-return-two",
                observation=(0.25, -0.25),
                action=1,
                target=(0.5, -0.5),
                reward=1.0,
                continuation=0.0,
                next_observation=(0.0, 0.0),
                partition="in_distribution",
            ),
        ),
    )


def _protocol() -> RecurrentWorldModelRetentionProtocol:
    return RecurrentWorldModelRetentionProtocol(
        protocol_id="recurrent-retention-aba-v1",
        probes=_probes(),
        phases=(
            RecurrentWorldModelRetentionPhase(
                phase_id="first-a",
                evaluator_regime_id="context-a",
                event_count=2,
            ),
            RecurrentWorldModelRetentionPhase(
                phase_id="interference-b",
                evaluator_regime_id="context-b",
                event_count=2,
            ),
            RecurrentWorldModelRetentionPhase(
                phase_id="return-a",
                evaluator_regime_id="context-a",
                event_count=2,
            ),
        ),
    )


def _config() -> RecurrentWorldModelRetentionConfig:
    return RecurrentWorldModelRetentionConfig(
        diagnostic_config=WorldModelCalibrationConfig(
            epistemic_bin_count=2,
            minimum_descriptive_bin_count=1,
            coverage_fractions=(0.5, 1.0),
            state_norm_edges=(1.0, 3.0),
            action_region_by_action=(0, 1),
            max_one_step_cases=6,
        ),
        max_phases=3,
        max_initial_snapshot_bytes=1_000_000,
        max_report_bytes=5_000_000,
    )


def _model() -> RecurrentLatentWorldModelEnsemble:
    return RecurrentLatentWorldModelEnsemble(
        RecurrentLatentWorldModelEnsembleConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=3,
            ensemble_size=2,
            uncertainty_warmup_steps=1,
            bootstrap_probability=0.75,
            max_updates=100,
        )
    )


@pytest.fixture(scope="module")
def fixture() -> tuple[
    RecurrentLatentWorldModelEnsemble,
    RecurrentLatentWorldModelEnsembleState,
    RecurrentWorldModelRetentionConfig,
    RecurrentWorldModelRetentionProtocol,
    dict[str, object],
]:
    model = _model()
    state = model.init(jr.key(29))
    config = _config()
    protocol = _protocol()
    report = build_recurrent_world_model_retention_report(
        model,
        state,
        config,
        protocol,
    )
    return model, state, config, protocol, report


def _tree_equal(left: object, right: object) -> None:
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        left_dtype = getattr(left_leaf, "dtype", None)
        right_dtype = getattr(right_leaf, "dtype", None)
        left_value = (
            jr.key_data(left_leaf)
            if left_dtype is not None and jnp.issubdtype(left_dtype, jax.dtypes.prng_key)
            else left_leaf
        )
        right_value = (
            jr.key_data(right_leaf)
            if right_dtype is not None and jnp.issubdtype(right_dtype, jax.dtypes.prng_key)
            else right_leaf
        )
        np.testing.assert_array_equal(left_value, right_value)


def test_protocol_is_fixed_ordered_evaluator_owned_and_exactly_recurring() -> None:
    protocol = _protocol()
    assert RecurrentWorldModelRetentionProtocol.from_config(protocol.to_config()) == protocol
    assert protocol.probes.regime_identifiers_available is False
    assert protocol.probes.learner_use == "never"
    assert [phase.evaluator_regime_id for phase in protocol.phases] == [
        "context-a",
        "context-b",
        "context-a",
    ]

    mismatched_events = list(protocol.probes.events)
    mismatched_events[-1] = dataclasses.replace(mismatched_events[-1], reward_target=1.25)
    with pytest.raises(ValueError, match="exact ordered cases"):
        RecurrentWorldModelRetentionProtocol(
            protocol_id=protocol.protocol_id,
            probes=dataclasses.replace(protocol.probes, events=tuple(mismatched_events)),
            phases=protocol.phases,
        )

    adjacent_a_one = _event(
        "adjacent-a-one",
        observation=(0.0, 0.0),
        action=0,
        target=(0.25, -0.25),
        reward=0.5,
        continuation=0.0,
        next_observation=(0.0, 0.0),
        partition="in_distribution",
    )
    adjacent_a_two = dataclasses.replace(
        adjacent_a_one,
        event_id="adjacent-a-two",
        next_decision_observation=(2.0, 2.0),
    )
    final_b = _event(
        "final-b",
        observation=(2.0, 2.0),
        action=1,
        target=(2.25, 1.75),
        reward=-0.5,
        continuation=0.0,
        next_observation=(0.0, 0.0),
        partition="ood",
    )
    with pytest.raises(ValueError, match="intervening"):
        RecurrentWorldModelRetentionProtocol(
            protocol_id="bad-recurrence-v1",
            probes=RecurrentWorldModelCalibrationProbeSet(
                probe_set_id="bad-recurrence-v1",
                events=(adjacent_a_one, adjacent_a_two, final_b),
            ),
            phases=(
                RecurrentWorldModelRetentionPhase("first-a", "context-a", 1),
                RecurrentWorldModelRetentionPhase("return-a", "context-a", 1),
                RecurrentWorldModelRetentionPhase("final-b", "context-b", 1),
            ),
        )


def test_report_uses_exact_preupdate_predictions_and_never_mutates_snapshot(
    fixture: tuple[
        RecurrentLatentWorldModelEnsemble,
        RecurrentLatentWorldModelEnsembleState,
        RecurrentWorldModelRetentionConfig,
        RecurrentWorldModelRetentionProtocol,
        dict[str, object],
    ],
) -> None:
    model, state, _, protocol, report = fixture
    before = frozen_world_model_state_sha256(state)
    payload = cast(Mapping[str, object], report["payload"])
    assert report["schema"] == RECURRENT_WORLD_MODEL_RETENTION_REPORT_SCHEMA
    assert payload["development_only"] is True
    assert payload["assessment_status"] == "not-assessed"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["efficacy_claimed"] is False
    assert payload["calibration_claimed"] is False
    assert payload["performance_thresholds_applied"] is False

    base = cast(Mapping[str, object], payload["base_prequential_report"])
    base_payload = cast(Mapping[str, object], base["payload"])
    raw = cast(Mapping[str, object], base_payload["raw_trace"])
    first = cast(Mapping[str, object], cast(list[object], raw["events"])[0])
    first_probe = protocol.probes.events[0]
    decision = model.decide(
        state,
        model.start(
            state,
            jnp.asarray(first_probe.observation, dtype=jnp.float32),
        ),
        jnp.asarray(first_probe.action, dtype=jnp.int32),
    )
    members = cast(Mapping[str, object], first["members"])
    np.testing.assert_array_equal(
        members["grounded_vectors"],
        decision.prediction.member_mean_predictions,
    )
    update = cast(Mapping[str, object], first["prequential_update"])
    assert update["applied_to_isolated_copy"] is True
    assert update["recurrent_advanced_once"] is True
    final = cast(Mapping[str, object], base_payload["final_isolated_state"])
    assert final["event_count"] == 6
    assert final["recurrent_advance_count"] == 6
    assert final["boundary_count"] == 3
    assert frozen_world_model_state_sha256(state) == before

    annotations = cast(list[Mapping[str, object]], payload["evaluator_annotations"])
    assert [item["evaluator_regime_id"] for item in annotations] == [
        "context-a",
        "context-a",
        "context-b",
        "context-b",
        "context-a",
        "context-a",
    ]
    assert all(item["learner_visible"] is False for item in annotations)
    assert all(
        "evaluator_regime_id" not in event
        for event in cast(list[dict[str, object]], raw["events"])
    )


def test_id_ood_phase_and_recurrence_metrics_reconstruct_from_raw_trace(
    fixture: tuple[
        RecurrentLatentWorldModelEnsemble,
        RecurrentLatentWorldModelEnsembleState,
        RecurrentWorldModelRetentionConfig,
        RecurrentWorldModelRetentionProtocol,
        dict[str, object],
    ],
) -> None:
    _, _, _, protocol, report = fixture
    payload = cast(Mapping[str, object], report["payload"])
    summary = cast(Mapping[str, object], payload["summary"])
    reconstructed = reconstruct_recurrent_world_model_retention_summary(
        cast(Mapping[str, object], payload["base_prequential_report"]),
        protocol,
    )
    assert reconstructed == summary
    id_ood = cast(list[Mapping[str, object]], summary["id_ood_metrics"])
    assert [(item["partition"], item["event_count"]) for item in id_ood] == [
        ("in_distribution", 4),
        ("ood", 2),
    ]
    phases = cast(list[Mapping[str, object]], summary["phase_metrics"])
    assert [item["event_count"] for item in phases] == [2, 2, 2]
    retention = cast(list[Mapping[str, object]], summary["retention_by_regime"])
    a_retention = next(item for item in retention if item["evaluator_regime_id"] == "context-a")
    assert a_retention["recurrence_available"] is True
    assert a_retention["exact_ordered_case_reuse"] is True
    occurrences = cast(list[Mapping[str, object]], a_retention["occurrences"])
    assert len(occurrences) == 2
    assert a_retention["latest_entry_minus_first_entry_nll"] == pytest.approx(
        cast(float, occurrences[-1]["entry_preupdate_nll"])
        - cast(float, occurrences[0]["entry_preupdate_nll"]),
        abs=0.0,
    )
    recurrence = cast(list[Mapping[str, object]], a_retention["recurrence_measurements"])
    assert len(recurrence) == 1
    assert recurrence[0]["entry_nll_change_from_first_occurrence"] == (
        a_retention["latest_entry_minus_first_entry_nll"]
    )
    assert summary["claims"]["retention_established"] is False  # type: ignore[index]
    assert summary["claims"]["efficacy_established"] is False  # type: ignore[index]
    assert summary["claims"]["calibration_established"] is False  # type: ignore[index]


def test_validator_rejects_tampering_even_after_rehash_and_supports_live_replay(
    fixture: tuple[
        RecurrentLatentWorldModelEnsemble,
        RecurrentLatentWorldModelEnsembleState,
        RecurrentWorldModelRetentionConfig,
        RecurrentWorldModelRetentionProtocol,
        dict[str, object],
    ],
) -> None:
    model, state, _, protocol, report = fixture
    valid = validate_recurrent_world_model_retention_report(
        report,
        model=model,
        state=state,
        protocol=protocol,
    )
    assert valid.valid, valid.errors

    changed = copy.deepcopy(report)
    payload = cast(dict[str, object], changed["payload"])
    annotations = cast(list[dict[str, object]], payload["evaluator_annotations"])
    annotations[0]["evaluator_regime_id"] = "context-b"
    hashes = cast(dict[str, object], payload["hashes"])
    hashes["evaluator_annotations_sha256"] = _digest(
        annotations
    )
    changed["payload_sha256"] = _digest(payload)
    validation = validate_recurrent_world_model_retention_report(changed)
    assert not validation.valid
    assert any("annotation" in error for error in validation.errors)

    numeric = copy.deepcopy(report)
    numeric_payload = cast(dict[str, object], numeric["payload"])
    numeric_config = cast(dict[str, object], numeric_payload["config"])
    numeric_config["max_phases"] = 3.0
    numeric_hashes = cast(dict[str, object], numeric_payload["hashes"])
    numeric_hashes["config_sha256"] = _digest(numeric_config)
    numeric["payload_sha256"] = _digest(numeric_payload)
    validation = validate_recurrent_world_model_retention_report(numeric)
    assert not validation.valid
    assert any("config" in error for error in validation.errors)


def test_nonfinite_corrupt_or_unbounded_inputs_fail_without_snapshot_change() -> None:
    model = _model()
    state = model.init(jr.key(31))
    before = frozen_world_model_state_sha256(state)
    with pytest.raises(ValueError, match="canonical finite"):
        dataclasses.replace(_probes().events[0], reward_target=float("nan"))

    with pytest.raises(ValueError, match="outside model action space"):
        bad_events = list(_probes().events)
        bad_events[0] = dataclasses.replace(bad_events[0], action=2)
        bad_events[4] = dataclasses.replace(bad_events[4], action=2)
        build_recurrent_world_model_retention_report(
            model,
            state,
            _config(),
            dataclasses.replace(
                _protocol(),
                probes=dataclasses.replace(
                    _probes(),
                    events=tuple(bad_events),
                ),
            ),
        )

    corrupt = state.replace(  # type: ignore[attr-defined]
        event_count=jnp.asarray(1, dtype=jnp.int32)
    )
    with pytest.raises(ValueError, match="invalid"):
        build_recurrent_world_model_retention_report(
            model,
            corrupt,
            _config(),
            _protocol(),
        )

    small = dataclasses.replace(_config(), max_report_bytes=1_000)
    with pytest.raises(ValueError, match="report byte bound"):
        build_recurrent_world_model_retention_report(model, state, small, _protocol())
    assert frozen_world_model_state_sha256(state) == before


def test_resource_accounting_is_exact_bounded_and_discloses_label_isolation(
    fixture: tuple[
        RecurrentLatentWorldModelEnsemble,
        RecurrentLatentWorldModelEnsembleState,
        RecurrentWorldModelRetentionConfig,
        RecurrentWorldModelRetentionProtocol,
        dict[str, object],
    ],
) -> None:
    model, _, config, _, report = fixture
    payload = cast(Mapping[str, object], report["payload"])
    resources = cast(Mapping[str, object], payload["resource_accounting"])
    assert resources["initial_snapshot_state_bytes"] == (
        model.resource_budget().persistent_state_bytes
    )
    assert resources["initial_snapshot_state_byte_limit"] == config.max_initial_snapshot_bytes
    assert resources["bounded_event_capacity"] == 6
    assert resources["recorded_event_count"] == 6
    assert resources["bounded_phase_capacity"] == 3
    assert resources["recorded_phase_count"] == 3
    assert resources["predict_before_update_decide_calls"] == 6
    assert resources["isolated_copy_update_calls"] == 6
    assert resources["recurrent_advances"] == 6
    assert resources["member_prediction_records"] == 12
    assert resources["learner_visible_evaluator_label_reads"] == 0
    assert resources["regime_identifier_reads_by_model"] == 0
    assert resources["external_snapshot_mutations"] == 0
    assert resources["persistent_evaluator_state_bytes"] == 0
    assert resources["canonical_report_bytes"] == len(
        canonical_recurrent_world_model_retention_report_bytes(report)
    )
    assert resources["canonical_report_bytes"] <= config.max_report_bytes
    assert payload["source_sha256"] == recurrent_world_model_retention_source_snapshot()


def test_canonical_report_and_source_bound_snapshot_checkpoint_roundtrip(
    tmp_path: Path,
    fixture: tuple[
        RecurrentLatentWorldModelEnsemble,
        RecurrentLatentWorldModelEnsembleState,
        RecurrentWorldModelRetentionConfig,
        RecurrentWorldModelRetentionProtocol,
        dict[str, object],
    ],
) -> None:
    model, state, _, _, report = fixture
    report_path = tmp_path / "retention.json"
    save_recurrent_world_model_retention_report(report_path, report)
    assert report_path.read_bytes() == canonical_recurrent_world_model_retention_report_bytes(
        report
    )
    assert load_recurrent_world_model_retention_report(report_path) == report
    with pytest.raises(FileExistsError, match="overwrite"):
        save_recurrent_world_model_retention_report(report_path, report)

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        load_recurrent_world_model_retention_report(pretty)

    checkpoint = tmp_path / "retention-snapshot"
    save_recurrent_world_model_retention_snapshot_checkpoint(
        model,
        state,
        checkpoint,
    )
    metadata = load_checkpoint_metadata(checkpoint)
    assert metadata["schema"] == RECURRENT_WORLD_MODEL_RETENTION_CHECKPOINT_SCHEMA
    restored_model, restored_state = load_recurrent_world_model_retention_snapshot_checkpoint(
        checkpoint
    )
    assert restored_model.to_config() == model.to_config()
    _tree_equal(restored_state, state)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_recurrent_world_model_retention_snapshot_checkpoint(
            model,
            state,
            checkpoint,
        )

    tampered = copy.deepcopy(metadata)
    tampered["snapshot_sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered-snapshot"
    save_checkpoint(state, tampered_path, metadata=tampered)
    with pytest.raises(ValueError, match="digest"):
        load_recurrent_world_model_retention_snapshot_checkpoint(tampered_path)


def test_evaluator_object_binds_config_protocol_and_builds_same_report(
    fixture: tuple[
        RecurrentLatentWorldModelEnsemble,
        RecurrentLatentWorldModelEnsembleState,
        RecurrentWorldModelRetentionConfig,
        RecurrentWorldModelRetentionProtocol,
        dict[str, object],
    ],
) -> None:
    model, state, config, protocol, report = fixture
    evaluator = RecurrentWorldModelRetentionEvaluator(config, protocol)
    assert evaluator.to_config() == {
        "config": config.to_config(),
        "protocol": protocol.to_config(),
    }
    assert evaluator.build_report(model, state) == report
