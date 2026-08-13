# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Checkpoint, replay, tamper, and numerical integration contracts."""

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

from alberta_framework.evaluation.kondo_actor_critic_replay_development import (
    KondoActorCriticReplayConfig,
    KondoActorCriticReplayEvaluator,
    build_kondo_actor_critic_replay_report,
    validate_kondo_actor_critic_replay_report,
)

pytestmark = pytest.mark.integration


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


def test_checkpoint_resume_is_exact_across_phase_boundary() -> None:
    config = KondoActorCriticReplayConfig(batches_per_phase=2)
    evaluator = KondoActorCriticReplayEvaluator(config)
    prefix = evaluator.init()
    for _ in range(3):
        prefix = evaluator.advance(prefix)
    assert prefix.event_index == 3
    checkpoint = evaluator.checkpoint_payload(prefix)
    assert checkpoint["schema"] == "alberta.kondo-actor-critic-replay.checkpoint.v2"
    assert len(cast(str, checkpoint["initial_snapshot_sha256"])) == 64
    assert len(cast(str, checkpoint["source_trace_sha256"])) == 64
    assert len(cast(str, checkpoint["source_prefix_sha256"])) == 64

    restored = evaluator.restore_checkpoint(checkpoint)
    uninterrupted = evaluator.run_to_end(prefix)
    resumed = evaluator.run_to_end(restored)

    assert resumed.event_index == config.total_batches
    assert resumed.records_json == uninterrupted.records_json
    _assert_tree_equal(resumed.ordinary_parameters, uninterrupted.ordinary_parameters)
    _assert_tree_equal(resumed.uniform_parameters, uninterrupted.uniform_parameters)
    _assert_tree_equal(resumed.kondo_state, uninterrupted.kondo_state)
    _assert_tree_equal(resumed.reserve_state, uninterrupted.reserve_state)
    _assert_tree_equal(resumed.ordinary_protected, uninterrupted.ordinary_protected)
    _assert_tree_equal(resumed.uniform_protected, uninterrupted.uniform_protected)
    _assert_tree_equal(resumed.kondo_protected, uninterrupted.kondo_protected)
    _assert_tree_equal(resumed.reserve_protected, uninterrupted.reserve_protected)
    assert evaluator.checkpoint_payload(resumed) == evaluator.checkpoint_payload(uninterrupted)


def test_checkpoint_digest_and_recomputed_causal_tamper_both_fail() -> None:
    config = KondoActorCriticReplayConfig(batches_per_phase=1)
    evaluator = KondoActorCriticReplayEvaluator(config)
    prefix = evaluator.advance(evaluator.init())
    checkpoint = evaluator.checkpoint_payload(prefix)

    tampered = copy.deepcopy(checkpoint)
    tampered["schema"] = "alberta.kondo-actor-critic-replay.checkpoint.v1"
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
    tampered["source_prefix_sha256"] = _canonical_sha256([])
    body = {key: value for key, value in tampered.items() if key != "checkpoint_sha256"}
    tampered["checkpoint_sha256"] = _canonical_sha256(body)
    with pytest.raises(ValueError, match="exact causal prefix"):
        evaluator.restore_checkpoint(tampered)


def test_report_validator_reconstructs_exact_causal_trace() -> None:
    config = KondoActorCriticReplayConfig(batches_per_phase=1)
    report = build_kondo_actor_critic_replay_report(config)

    receipt = validate_kondo_actor_critic_replay_report(report)

    assert receipt.valid
    assert receipt.assessment_status == "not_assessed"
    assert receipt.source_runtime_bound
    assert receipt.causal_trace_replayed
    assert receipt.exact_replay
    assert receipt.output_written is False
    assert receipt.promotion_authority is False


def test_recomputed_outer_digests_cannot_hide_record_tamper() -> None:
    config = KondoActorCriticReplayConfig(batches_per_phase=1)
    report = build_kondo_actor_critic_replay_report(config)
    tampered = copy.deepcopy(report)
    records = cast(list[dict[str, Any]], tampered["arm_records"])
    records[0]["actor_loss"] = cast(float, records[0]["actor_loss"]) + 0.125
    tampered["arm_records"] = records
    tampered["arm_records_sha256"] = _canonical_sha256(records)
    body = {key: value for key, value in tampered.items() if key != "report_sha256"}
    tampered["report_sha256"] = _canonical_sha256(body)

    with pytest.raises(ValueError, match="exact deterministic reconstruction"):
        validate_kondo_actor_critic_replay_report(tampered)


def test_recomputed_outer_digests_cannot_claim_on_policy_data() -> None:
    config = KondoActorCriticReplayConfig(batches_per_phase=1)
    report = build_kondo_actor_critic_replay_report(config)
    tampered = copy.deepcopy(report)
    tampered["source_behavior_policy_available"] = True
    tampered["on_policy"] = True
    tampered["importance_correction_applied"] = True
    body = {key: value for key, value in tampered.items() if key != "report_sha256"}
    tampered["report_sha256"] = _canonical_sha256(body)

    with pytest.raises(ValueError, match="source_behavior_policy_available"):
        validate_kondo_actor_critic_replay_report(tampered)


def test_finite_float32_learning_overflow_fails_before_report_serialization() -> None:
    config = KondoActorCriticReplayConfig(
        batches_per_phase=1,
        actor_learning_rate=3.0e38,
    )

    with pytest.raises(ValueError, match="produced nonfinite|probe loss is nonfinite"):
        build_kondo_actor_critic_replay_report(config)
