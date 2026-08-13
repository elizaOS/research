# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Replay, checkpoint, tamper, and real-clock integration contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation.kondo_sparse_actor_development import (
    ARM_ORDER,
    KondoSparseActorDevelopmentConfig,
    KondoSparseActorDevelopmentEvaluator,
    build_kondo_sparse_actor_development_report,
    validate_kondo_sparse_actor_development_report,
)

pytestmark = pytest.mark.integration


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000_000
        self.start = True
        self.index = 0

    def __call__(self) -> int:
        if self.start:
            self.start = False
            return self.value
        self.value += 50 + self.index
        self.index += 1
        self.start = True
        return self.value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, Mapping)
    return dict(value)


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        lhs_array = (
            np.asarray(jr.key_data(lhs))
            if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key)
            else np.asarray(lhs)
        )
        rhs_array = (
            np.asarray(jr.key_data(rhs))
            if jax.dtypes.issubdtype(rhs.dtype, jax.dtypes.prng_key)
            else np.asarray(rhs)
        )
        np.testing.assert_array_equal(lhs_array, rhs_array)


def test_prefix_checkpoint_resume_is_exact() -> None:
    config = KondoSparseActorDevelopmentConfig(num_batches=3, timing_trials=1)
    evaluator = KondoSparseActorDevelopmentEvaluator(config)
    prefix = evaluator.advance(evaluator.init())
    checkpoint = evaluator.checkpoint_payload(prefix)
    assert checkpoint["schema"] == (
        "alberta.kondo-sparse-actor-development.checkpoint.v2"
    )

    restored = evaluator.restore_checkpoint(checkpoint)
    uninterrupted = evaluator.run_to_end(prefix)
    resumed = evaluator.run_to_end(restored)

    assert uninterrupted.event_index == config.num_batches
    assert resumed.records_json == uninterrupted.records_json
    _assert_tree_equal(resumed.ordinary_parameters, uninterrupted.ordinary_parameters)
    _assert_tree_equal(resumed.uniform_parameters, uninterrupted.uniform_parameters)
    _assert_tree_equal(resumed.kondo_state, uninterrupted.kondo_state)
    _assert_tree_equal(resumed.overflow_state, uninterrupted.overflow_state)
    assert evaluator.checkpoint_payload(resumed) == evaluator.checkpoint_payload(
        uninterrupted
    )


def test_checkpoint_tamper_is_rejected_before_resume() -> None:
    config = KondoSparseActorDevelopmentConfig(num_batches=2, timing_trials=1)
    evaluator = KondoSparseActorDevelopmentEvaluator(config)
    prefix = evaluator.advance(evaluator.init())
    checkpoint = evaluator.checkpoint_payload(prefix)

    tampered = copy.deepcopy(checkpoint)
    tampered["schema"] = "alberta.kondo-sparse-actor-development.checkpoint.v1"
    body = {key: value for key, value in tampered.items() if key != "checkpoint_sha256"}
    tampered["checkpoint_sha256"] = _canonical_sha256(body)
    with pytest.raises(ValueError, match="schema/type"):
        evaluator.restore_checkpoint(tampered)

    tampered = copy.deepcopy(checkpoint)
    tampered["event_index"] = 0
    with pytest.raises(ValueError, match="digest integrity"):
        evaluator.restore_checkpoint(tampered)

    tampered = copy.deepcopy(checkpoint)
    tampered["event_index"] = 0
    body = {key: value for key, value in tampered.items() if key != "checkpoint_sha256"}
    tampered["checkpoint_sha256"] = _canonical_sha256(body)
    with pytest.raises(ValueError, match="exact prefix replay"):
        evaluator.restore_checkpoint(tampered)


def test_deterministic_tamper_fails_even_after_all_outer_digests_are_recomputed() -> None:
    config = KondoSparseActorDevelopmentConfig(num_batches=2, timing_trials=2)
    report = build_kondo_sparse_actor_development_report(
        config,
        clock_ns=_Clock(),
        clock_name="integration-test-clock",
    )
    tampered = copy.deepcopy(report)
    deterministic = _mapping(tampered["deterministic"])
    records = cast(list[dict[str, Any]], deterministic["arm_records"])
    records[0]["actor_loss"] = cast(float, records[0]["actor_loss"]) + 0.125
    deterministic["arm_records_sha256"] = _canonical_sha256(records)
    deterministic_body = {
        key: value for key, value in deterministic.items() if key != "deterministic_sha256"
    }
    deterministic["deterministic_sha256"] = _canonical_sha256(deterministic_body)
    tampered["deterministic"] = deterministic
    timing = _mapping(tampered["timing"])
    timing["deterministic_sha256"] = deterministic["deterministic_sha256"]
    timing_body = {key: value for key, value in timing.items() if key != "timing_sha256"}
    timing["timing_sha256"] = _canonical_sha256(timing_body)
    tampered["timing"] = timing
    report_body = {key: value for key, value in tampered.items() if key != "report_sha256"}
    tampered["report_sha256"] = _canonical_sha256(report_body)

    with pytest.raises(ValueError, match="deterministic replay differs"):
        validate_kondo_sparse_actor_development_report(tampered)


def test_timing_tamper_fails_structural_reconstruction_not_speed_assessment() -> None:
    config = KondoSparseActorDevelopmentConfig(num_batches=1, timing_trials=2)
    report = build_kondo_sparse_actor_development_report(
        config,
        clock_ns=_Clock(),
        clock_name="integration-test-clock",
    )
    tampered = copy.deepcopy(report)
    timing = _mapping(tampered["timing"])
    events = cast(list[dict[str, Any]], timing["events"])
    events[-1]["end_ns"] = cast(int, events[-1]["end_ns"]) + 1
    events[-1]["duration_ns"] = cast(int, events[-1]["duration_ns"]) + 1
    timing["events"] = events
    timing_body = {key: value for key, value in timing.items() if key != "timing_sha256"}
    timing["timing_sha256"] = _canonical_sha256(timing_body)
    tampered["timing"] = timing
    report_body = {key: value for key, value in tampered.items() if key != "report_sha256"}
    tampered["report_sha256"] = _canonical_sha256(report_body)

    with pytest.raises(ValueError, match="timing summary"):
        validate_kondo_sparse_actor_development_report(tampered)


def test_timing_tamper_cannot_move_an_event_before_its_predecessor() -> None:
    config = KondoSparseActorDevelopmentConfig(num_batches=1, timing_trials=1)
    report = build_kondo_sparse_actor_development_report(
        config,
        clock_ns=_Clock(),
        clock_name="integration-test-clock",
    )
    tampered = copy.deepcopy(report)
    timing = _mapping(tampered["timing"])
    events = cast(list[dict[str, Any]], timing["events"])
    prior_end = cast(int, events[0]["end_ns"])
    duration = cast(int, events[1]["duration_ns"])
    events[1]["start_ns"] = prior_end - 1
    events[1]["end_ns"] = prior_end - 1 + duration
    timing["events"] = events
    timing_body = {key: value for key, value in timing.items() if key != "timing_sha256"}
    timing["timing_sha256"] = _canonical_sha256(timing_body)
    tampered["timing"] = timing
    report_body = {key: value for key, value in tampered.items() if key != "report_sha256"}
    tampered["report_sha256"] = _canonical_sha256(report_body)

    with pytest.raises(ValueError, match="globally monotonic"):
        validate_kondo_sparse_actor_development_report(tampered)


def test_real_perf_counter_path_records_raw_nonnegative_samples_without_verdict() -> None:
    config = KondoSparseActorDevelopmentConfig(num_batches=1, timing_trials=1)
    report = build_kondo_sparse_actor_development_report(config)
    receipt = validate_kondo_sparse_actor_development_report(report)
    timing = _mapping(report["timing"])
    events = cast(list[dict[str, Any]], timing["events"])
    summaries = _mapping(timing["summaries"])

    assert receipt.valid
    assert timing["real_perf_counter_ns"] is True
    assert timing["clock_name"] == "time.perf_counter_ns"
    assert timing["clock_resolution_ns"] >= 1
    assert len(events) == len(ARM_ORDER)
    assert all(cast(int, event["duration_ns"]) >= 0 for event in events)
    for arm in ARM_ORDER:
        summary = _mapping(summaries[arm])
        assert len(cast(list[int], summary["raw_duration_ns"])) == 1
        assert summary["assessment_status"] == "not_assessed"
    assert timing["thresholds"] == []
    assert timing["verdict"] == "not_assessed"
