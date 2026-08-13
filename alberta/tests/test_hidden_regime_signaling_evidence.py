"""Fail-closed tests for hidden-regime preregistration and compact evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import alberta_framework.evaluation.hidden_regime_signaling_evidence as evidence_module
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    BENEFICIARY_FROZEN,
    CONSTANT_CHANNEL,
    HELPER_FROZEN,
    SELECTIVE_FULL,
    WRITABLE_LRU,
)
from alberta_framework.evaluation.hidden_regime_signaling_evidence import (
    COMPACT_PRIMITIVES_SCHEMA,
    FROZEN_CONDITION_ORDER,
    FROZEN_NUM_STEPS,
    FROZEN_SEGMENT_BOUNDS,
    FROZEN_SEGMENT_LENGTHS,
    FROZEN_SEGMENT_REGIMES,
    FROZEN_THRESHOLDS,
    HELDOUT_SEED_COUNT,
    MANUAL_TEST_NAMESPACE_PREFIX,
    PROTECTED_EXECUTION_ENABLED,
    PROTOCOL_STATUS,
    RESERVED_HELDOUT_NAMESPACE,
    RESERVED_HELDOUT_NAMESPACE_EXECUTED,
    SOURCE_PATHS,
    build_evidence_artifact_from_records,
    build_preregistration_plan,
    execute_validated_plan,
    frozen_protocol_config,
    strict_json_load,
    strict_json_text,
    validate_evidence_artifact,
    validate_preregistration_plan,
)

pytestmark = pytest.mark.unit

MANUAL_NAMESPACE = f"{MANUAL_TEST_NAMESPACE_PREFIX}artifact-fixture-v1"
UTC_TIMESTAMP = "2026-07-31T00:00:00+00:00"


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _rehash_plan(plan: dict[str, object]) -> None:
    payload = plan["scientific_payload"]
    digest = cast(dict[str, object], plan["scientific_digest"])
    digest["sha256"] = _canonical_sha256(payload)


def _rehash_run(run: dict[str, object]) -> None:
    payload = {key: value for key, value in run.items() if key != "compact_digest_sha256"}
    run["compact_digest_sha256"] = _canonical_sha256(payload)


def _rehash_seed(record: dict[str, object]) -> None:
    payload = {key: value for key, value in record.items() if key != "seed_record_digest_sha256"}
    record["seed_record_digest_sha256"] = _canonical_sha256(payload)


def _rehash_artifact(artifact: dict[str, object]) -> None:
    payload = artifact["scientific_payload"]
    digest = cast(dict[str, object], artifact["scientific_digest"])
    digest["sha256"] = _canonical_sha256(payload)


def _plan(seed_count: int = HELDOUT_SEED_COUNT) -> dict[str, object]:
    return build_preregistration_plan(
        namespace=MANUAL_NAMESPACE,
        seed_count=seed_count,
        preregistered_at_utc=UTC_TIMESTAMP,
    )


def _segment_records(reward_rate: float, recurrence_rate: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[int] = set()
    for index, (steps, regime) in enumerate(
        zip(FROZEN_SEGMENT_LENGTHS, FROZEN_SEGMENT_REGIMES, strict=True)
    ):
        window = min(128, steps)
        early_rate = recurrence_rate if regime in seen and regime in (0, 1, 2, 3) else reward_rate
        records.append(
            {
                "segment_index": index,
                "regime_id": regime,
                "steps": steps,
                "reward_count": round(reward_rate * steps),
                "early_steps": window,
                "early_reward_count": round(early_rate * window),
                "late_steps": window,
                "late_reward_count": round(reward_rate * window),
            }
        )
        seen.add(regime)
    return records


def _lifecycle_events() -> list[dict[str, object]]:
    # Generation three is created under C-old (regime 2) and atomically
    # replaced by generation four under C-new (regime 3).
    commits = (
        (FROZEN_SEGMENT_BOUNDS[0][0] + 128, 0, 1, 1, -1, -1),
        (FROZEN_SEGMENT_BOUNDS[1][0] + 128, 1, 2, 2, -1, -1),
        (FROZEN_SEGMENT_BOUNDS[3][0] + 128, 2, 3, 3, -1, -1),
        (FROZEN_SEGMENT_BOUNDS[9][0] + 128, 3, 3, 4, 3, 3),
    )
    events: list[dict[str, object]] = []
    for step, regime, slot, generation, retired_slot, retired_generation in commits:
        segment_index = next(
            index for index, (start, end) in enumerate(FROZEN_SEGMENT_BOUNDS) if start <= step < end
        )
        for role in ("helper", "beneficiary"):
            events.append(
                {
                    "step": step,
                    "segment_index": segment_index,
                    "regime_id": regime,
                    "role": role,
                    "committed_slot": slot,
                    "committed_generation": generation,
                    "retired_slot": retired_slot,
                    "retired_generation": retired_generation,
                }
            )
    return events


def _d_checkpoints() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, regime in enumerate(FROZEN_SEGMENT_REGIMES):
        if regime != 4:
            continue
        record: dict[str, object] = {"segment_index": index}
        for role in ("helper", "beneficiary"):
            record[f"{role}_status_pre"] = [1, 2, 2, 2]
            record[f"{role}_status_post"] = [1, 2, 2, 2]
            record[f"{role}_generation_pre"] = [0, 1, 2, 4]
            record[f"{role}_generation_post"] = [0, 1, 2, 4]
        records.append(record)
    return records


def _run_record(condition: str) -> dict[str, object]:
    if condition == SELECTIVE_FULL:
        reward_rate, recurrence_rate = 0.84, 0.75
        events = _lifecycle_events()
        helper_writes = beneficiary_writes = 500
    elif condition == WRITABLE_LRU:
        reward_rate, recurrence_rate = 0.82, 0.67
        events = _lifecycle_events()
        helper_writes = beneficiary_writes = 600
    else:
        reward_rate, recurrence_rate = 1.0 / 3.0, 1.0 / 3.0
        events = []
        helper_writes = 0 if condition == HELPER_FROZEN else 400
        beneficiary_writes = 0 if condition == BENEFICIARY_FROZEN else 400
    diagnostics = {
        "helper_value_write_count": helper_writes,
        "beneficiary_value_write_count": beneficiary_writes,
        "helper_effective_learning_update_count": helper_writes,
        "beneficiary_effective_learning_update_count": beneficiary_writes,
        "helper_candidate_confirmation_steps": [],
        "beneficiary_candidate_confirmation_steps": [],
        "helper_scratch_retest_steps": [],
        "beneficiary_scratch_retest_steps": [],
        "helper_selective_mutation_coordinates": [],
        "beneficiary_selective_mutation_coordinates": [],
        "lifecycle_desynchronization_steps": [],
    }
    payload: dict[str, object] = {
        "schema_version": COMPACT_PRIMITIVES_SCHEMA,
        "condition": condition,
        "strict_run_validation_passed": True,
        "segments": _segment_records(reward_rate, recurrence_rate),
        "lifecycle_events": events,
        "diagnostics": diagnostics,
        "d_checkpoints": _d_checkpoints(),
        "resource": {
            "initial_state_scalars": 138,
            "final_state_scalars": 138,
            "initial_state_bytes": 552,
            "final_state_bytes": 552,
            "expected_state_bytes": 552,
            "resource_constant": True,
            "resource_matched": True,
        },
        "same_backend_trace_sha256": "1" * 64,
        "same_backend_final_state_sha256": "2" * 64,
    }
    payload["compact_digest_sha256"] = _canonical_sha256(payload)
    return payload


def _seed_records(plan: Mapping[str, object]) -> list[dict[str, object]]:
    payload = cast(Mapping[str, object], plan["scientific_payload"])
    seeds = cast(list[dict[str, object]], payload["seed_pairs"])
    records: list[dict[str, object]] = []
    for seed in seeds:
        record: dict[str, object] = {
            "schema_version": COMPACT_PRIMITIVES_SCHEMA,
            "seed_pair": copy.deepcopy(seed),
            "condition_order": list(FROZEN_CONDITION_ORDER),
            "runs": [_run_record(condition) for condition in FROZEN_CONDITION_ORDER],
        }
        record["seed_record_digest_sha256"] = _canonical_sha256(record)
        records.append(record)
    return records


def _operational_metadata() -> dict[str, object]:
    return {
        "generated_at_utc": UTC_TIMESTAMP,
        "wall_seconds": 1.0,
        "python_version": "3.12.test",
        "platform": "test-platform",
        "jax_version": "test-jax",
        "jaxlib_version": "test-jaxlib",
        "numpy_version": "test-numpy",
        "jax_backend": "test-backend",
        "jax_devices": ["test-device"],
        "argv": ["test-hidden-regime-evidence"],
    }


@pytest.fixture
def valid_bundle() -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    plan = _plan()
    records = _seed_records(plan)
    artifact = build_evidence_artifact_from_records(
        plan,
        records,
        operational_metadata=_operational_metadata(),
    )
    return plan, records, artifact


def test_frozen_protocol_declares_exact_candidate_and_unexecuted_sentinel() -> None:
    config = frozen_protocol_config()

    assert RESERVED_HELDOUT_NAMESPACE == "hidden-regime-signaling-v1-heldout-a-v1"
    assert RESERVED_HELDOUT_NAMESPACE_EXECUTED is False
    assert PROTECTED_EXECUTION_ENABLED is False
    assert PROTOCOL_STATUS.startswith("draft_execution_disabled")
    assert config["num_steps"] == FROZEN_NUM_STEPS
    assert cast(dict[str, object], config["learner"]) == {
        "learning_rate": 0.25,
        "epsilon": 0.1,
        "relevance_rate": 0.1,
        "lease_length": 16,
        "confirmation_steps": 8,
        "durable_retrieval_threshold": 0.5,
        "candidate_confirmation_threshold": 0.75,
        "candidate_confirmation_leases": 3,
        "scratch_training_leases_before_retest": 16,
        "writable_lru_ablation": False,
    }
    assert FROZEN_THRESHOLDS["required_seed_count"] == 30
    assert FROZEN_THRESHOLDS["paired_t_degrees_freedom"] == 29
    assert not hasattr(evidence_module, "derive_evidence_seed_pairs")


def test_source_snapshot_closes_over_protocol_algorithm_and_dependency_lock() -> None:
    paths = {path.as_posix() for path in SOURCE_PATHS}

    assert {
        "pyproject.toml",
        "uv.lock",
        "alberta_framework/core/slot_signaling_agent.py",
        "alberta_framework/evaluation/hidden_regime_signaling_development.py",
        "alberta_framework/evaluation/hidden_regime_signaling_evidence.py",
        "alberta_framework/streams/hidden_regime_signaling.py",
    } <= paths
    assert not any(path.startswith("tests/") for path in paths)
    assert "alberta_framework/evaluation/continual_ia_artifact.py" not in paths


def test_source_snapshot_is_independent_of_unrelated_loaded_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = evidence_module._algorithm_source_paths()
    unrelated_path = (
        evidence_module.REPO_ROOT
        / "alberta_framework/evaluation/continual_ia_artifact.py"
    )
    monkeypatch.setitem(
        sys.modules,
        "alberta_framework.evaluation.collection_order_probe",
        SimpleNamespace(__file__=str(unrelated_path)),
    )

    after = evidence_module._algorithm_source_paths()

    assert after == before
    assert unrelated_path.relative_to(evidence_module.REPO_ROOT) not in after


def test_manual_plan_is_sha_separated_deterministic_and_source_bound() -> None:
    first = _plan(seed_count=2)
    second = _plan(seed_count=2)
    validation = validate_preregistration_plan(first)
    payload = cast(dict[str, object], first["scientific_payload"])
    seeds = cast(list[dict[str, object]], payload["seed_pairs"])

    assert validation.valid, validation.errors
    assert first == second
    assert payload["executed"] is False
    assert payload["protocol_kind"] == "manual_nonpromoting"
    assert len({seed["world_seed"] for seed in seeds}) == 2
    assert len({seed["learner_seed"] for seed in seeds}) == 2
    assert {seed["world_seed"] for seed in seeds}.isdisjoint(
        {seed["learner_seed"] for seed in seeds}
    )


@pytest.mark.parametrize(
    "mutation",
    ("executed", "condition_order", "source", "seed", "threshold"),
)
def test_plan_rejects_hostile_semantic_mutation_even_after_redigest(mutation: str) -> None:
    plan = _plan(seed_count=2)
    payload = cast(dict[str, object], plan["scientific_payload"])
    if mutation == "executed":
        payload["executed"] = True
    elif mutation == "condition_order":
        cast(list[object], payload["condition_order"]).reverse()
    elif mutation == "source":
        source = cast(dict[str, object], payload["source_sha256"])
        source[next(iter(source))] = "0" * 64
    elif mutation == "seed":
        seeds = cast(list[dict[str, object]], payload["seed_pairs"])
        seeds[0]["learner_seed"] = seeds[0]["world_seed"]
    else:
        cast(dict[str, object], payload["thresholds"])["minimum_selective_mean_reward"] = 0.1
    _rehash_plan(plan)

    validation = validate_preregistration_plan(plan)

    assert not validation.valid
    assert validation.errors


@pytest.mark.parametrize(
    "document",
    (
        '{"schema_version":"a","schema_version":"b"}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":1e9999}',
    ),
)
def test_strict_json_rejects_duplicate_and_nonfinite_input(
    tmp_path: Path,
    document: str,
) -> None:
    path = tmp_path / "hostile.json"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError):
        strict_json_load(path)


def test_valid_manual_fixture_recomputes_gates_but_cannot_resemble_evidence(
    valid_bundle: tuple[dict[str, object], list[dict[str, object]], dict[str, object]],
) -> None:
    plan, _, artifact = valid_bundle
    validation = validate_evidence_artifact(artifact, plan)
    payload = cast(dict[str, object], artifact["scientific_payload"])
    aggregate = cast(dict[str, object], payload["aggregate_metrics"])
    inference = cast(dict[str, object], aggregate["paired_t_inference"])
    reward = cast(dict[str, object], inference["reward_delta"])

    assert validation.valid, validation.errors
    assert not validation.accepted
    assert not validation.same_backend_rerun_verified
    assert payload["frozen_gates_passed"] is True
    assert payload["eligible_for_scientific_review"] is False
    assert payload["outcome"] == "manual_nonpromoting"
    assert "only the explicit" in cast(str, payload["compact_record_trust_boundary"])
    assert aggregate["strict_run_validation_count"] == 180
    assert aggregate["selective_strict_lifecycle_pass_count"] == 30
    assert aggregate["selective_d_non_displacement_count"] == 30
    assert reward["degrees_freedom"] == 29
    assert reward["one_sided_confidence"] == 0.95
    assert "n - 1" in cast(str, reward["sample_variance_formula"])
    assert "df=29" in cast(str, reward["lower_bound_formula"])


def test_gate_failure_builds_valid_rejection_without_retyping_thresholds(
    valid_bundle: tuple[dict[str, object], list[dict[str, object]], dict[str, object]],
) -> None:
    plan, original_records, _ = valid_bundle
    records = copy.deepcopy(original_records)
    for record in records:
        runs = cast(list[dict[str, object]], record["runs"])
        control = runs[FROZEN_CONDITION_ORDER.index(CONSTANT_CHANNEL)]
        for segment in cast(list[dict[str, object]], control["segments"]):
            segment["reward_count"] = cast(int, segment["steps"])
            segment["early_reward_count"] = cast(int, segment["early_steps"])
        _rehash_run(control)
        _rehash_seed(record)

    artifact = build_evidence_artifact_from_records(
        plan,
        records,
        operational_metadata=_operational_metadata(),
    )
    validation = validate_evidence_artifact(artifact, plan)

    assert validation.valid, validation.errors
    assert not validation.accepted
    result = cast(dict[str, object], artifact["scientific_payload"])
    assert result["frozen_gates_passed"] is False
    assert result["outcome"] == "manual_nonpromoting"


@pytest.mark.parametrize(
    "mutation",
    (
        "primitive_without_redigest",
        "resource_with_redigest",
        "d_checkpoint_with_redigest",
        "seed_with_redigest",
        "aggregate_with_redigest",
        "acceptance_with_redigest",
        "source_with_redigest",
    ),
)
def test_artifact_rejects_hostile_mutations(
    valid_bundle: tuple[dict[str, object], list[dict[str, object]], dict[str, object]],
    mutation: str,
) -> None:
    plan, _, original_artifact = valid_bundle
    artifact = copy.deepcopy(original_artifact)
    payload = cast(dict[str, object], artifact["scientific_payload"])
    records = cast(list[dict[str, object]], payload["seed_records"])
    first_seed = records[0]
    first_run = cast(list[dict[str, object]], first_seed["runs"])[0]
    if mutation == "primitive_without_redigest":
        cast(list[dict[str, object]], first_run["segments"])[0]["reward_count"] = 0
    elif mutation == "resource_with_redigest":
        cast(dict[str, object], first_run["resource"])["final_state_bytes"] = 551
        _rehash_run(first_run)
        _rehash_seed(first_seed)
        _rehash_artifact(artifact)
    elif mutation == "d_checkpoint_with_redigest":
        checkpoint = cast(list[dict[str, object]], first_run["d_checkpoints"])[0]
        cast(list[int], checkpoint["helper_generation_post"])[1] += 1
        _rehash_run(first_run)
        _rehash_seed(first_seed)
        _rehash_artifact(artifact)
    elif mutation == "seed_with_redigest":
        cast(dict[str, object], first_seed["seed_pair"])["world_seed"] = 0
        _rehash_seed(first_seed)
        _rehash_artifact(artifact)
    elif mutation == "aggregate_with_redigest":
        aggregate = cast(dict[str, object], payload["aggregate_metrics"])
        inference = cast(dict[str, object], aggregate["paired_t_inference"])
        cast(dict[str, object], inference["reward_delta"])["degrees_freedom"] = 28
        _rehash_artifact(artifact)
    elif mutation == "acceptance_with_redigest":
        payload["accepted"] = False
        payload["outcome"] = "valid_rejection"
        _rehash_artifact(artifact)
    else:
        source = cast(dict[str, object], payload["source_sha256"])
        source[next(iter(source))] = "0" * 64
        _rehash_artifact(artifact)

    validation = validate_evidence_artifact(artifact, plan)

    assert not validation.valid
    assert validation.errors


def test_execution_refuses_invalid_plan_before_any_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(seed_count=1)
    payload = cast(dict[str, object], plan["scientific_payload"])
    payload["executed"] = True
    _rehash_plan(plan)
    called = False

    def forbidden_run(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("runner must not be reached")

    monkeypatch.setattr(evidence_module, "run_hidden_regime_condition", forbidden_run)

    with pytest.raises(ValueError, match="execution refused"):
        execute_validated_plan(plan)
    assert not called


def test_strict_json_round_trip_preserves_plan(tmp_path: Path) -> None:
    plan = _plan(seed_count=2)
    path = tmp_path / "manual-plan.json"
    path.write_text(strict_json_text(plan), encoding="utf-8")

    assert strict_json_load(path) == plan
