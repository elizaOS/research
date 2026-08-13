# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Checkpoint, causal replay, tamper, and JIT integration contracts."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.evaluation.kondo_actor_critic_on_policy_development import (
    KondoActorCriticOnPolicyConfig,
    KondoActorCriticOnPolicyEvaluator,
    OnPolicyManualActorState,
    build_kondo_actor_critic_on_policy_report,
    collect_on_policy_batch_kernel,
    validate_kondo_actor_critic_on_policy_report,
)

pytestmark = pytest.mark.integration


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _assert_tree_numerically_equal(left: object, right: object) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(lhs)
        right_array = np.asarray(rhs)
        if np.issubdtype(left_array.dtype, np.floating):
            np.testing.assert_allclose(
                left_array,
                right_array,
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        else:
            np.testing.assert_array_equal(left_array, right_array)


def test_collection_kernel_has_eager_jit_parity() -> None:
    config = KondoActorCriticOnPolicyConfig()
    evaluator = KondoActorCriticOnPolicyEvaluator(config)
    state = evaluator.init()
    schedule = evaluator.common_schedule(0)
    arguments = (
        state.ordinary_actor.parameters,
        state.ordinary_environment,
        evaluator.environment_parameters,
        schedule,
        jnp.asarray(1.0, dtype=jnp.float32),
        state.ordinary_actor.policy_revision,
    )

    with jax.disable_jit():
        eager = collect_on_policy_batch_kernel(*arguments)
    compiled = jax.jit(collect_on_policy_batch_kernel)(*arguments)

    _assert_tree_numerically_equal(eager, compiled)
    np.testing.assert_array_equal(eager.batch.actions, compiled.batch.actions)
    np.testing.assert_array_equal(
        eager.batch.policy_revision,
        compiled.batch.policy_revision,
    )
    assert int(np.asarray(compiled.environment.step_count)) == config.batch_size


def test_checkpoint_resume_matches_uninterrupted_exact_state() -> None:
    config = KondoActorCriticOnPolicyConfig()
    evaluator = KondoActorCriticOnPolicyEvaluator(config)
    prefix = evaluator.advance(evaluator.init())
    payload = evaluator.checkpoint_payload(prefix)
    assert payload["schema"] == "alberta.kondo-actor-critic-on-policy.checkpoint.v2"

    restored = evaluator.restore_checkpoint(payload)
    resumed = evaluator.run_to_end(restored)
    uninterrupted = evaluator.run_to_end()

    assert evaluator._state_body(restored) == evaluator._state_body(prefix)
    assert evaluator._state_body(resumed) == evaluator._state_body(uninterrupted)
    assert resumed.integrity_sha256 == uninterrupted.integrity_sha256
    assert len(resumed.records_json) == config.total_batches * 4


def test_checkpoint_integrity_and_recomputed_prefix_tamper_are_rejected() -> None:
    evaluator = KondoActorCriticOnPolicyEvaluator(KondoActorCriticOnPolicyConfig())
    prefix = evaluator.advance(evaluator.init())
    payload = evaluator.checkpoint_payload(prefix)

    tampered = copy.deepcopy(payload)
    tampered["schema"] = "alberta.kondo-actor-critic-on-policy.checkpoint.v1"
    body = {name: tampered[name] for name in tampered if name != "checkpoint_sha256"}
    tampered["checkpoint_sha256"] = _canonical_sha256(body)
    with pytest.raises(ValueError, match="schema/type"):
        evaluator.restore_checkpoint(tampered)

    tampered = copy.deepcopy(payload)
    tampered["event_index"] = 0
    with pytest.raises(ValueError, match="digest integrity"):
        evaluator.restore_checkpoint(tampered)

    tampered = copy.deepcopy(payload)
    tampered["event_index"] = 0
    tampered["common_schedule_prefix_sha256"] = _canonical_sha256([])
    body = {name: tampered[name] for name in tampered if name != "checkpoint_sha256"}
    tampered["checkpoint_sha256"] = _canonical_sha256(body)
    with pytest.raises(ValueError, match="exact causal prefix"):
        evaluator.restore_checkpoint(tampered)


def test_resealed_actor_tamper_cannot_pass_causal_advance() -> None:
    evaluator = KondoActorCriticOnPolicyEvaluator(KondoActorCriticOnPolicyConfig())
    state = evaluator.advance(evaluator.init())
    parameters = state.ordinary_actor.parameters.replace(
        output_bias=state.ordinary_actor.parameters.output_bias.at[0].add(
            jnp.float32(0.125)
        )
    )
    actor = OnPolicyManualActorState(
        parameters=parameters,
        policy_revision=state.ordinary_actor.policy_revision,
        actor_backward_count=state.ordinary_actor.actor_backward_count,
    )
    resealed = evaluator._seal_state(
        dataclasses.replace(state, ordinary_actor=actor, integrity_sha256="")
    )

    assert evaluator.validate_state(resealed, causal=False)
    assert not evaluator.validate_state(resealed)
    with pytest.raises(ValueError, match="exact causal prefix"):
        evaluator.advance(resealed)


def test_report_validator_reconstructs_every_closed_loop_arm() -> None:
    report = build_kondo_actor_critic_on_policy_report(
        KondoActorCriticOnPolicyConfig()
    )

    receipt = validate_kondo_actor_critic_on_policy_report(report)

    assert receipt.valid
    assert receipt.assessment_status == "not_assessed"
    assert receipt.source_runtime_bound
    assert receipt.causal_trace_replayed
    assert receipt.exact_replay
    assert receipt.output_written is False
    assert receipt.promotion_authority is False


def test_recomputed_report_digests_cannot_hide_causal_record_tamper() -> None:
    report = build_kondo_actor_critic_on_policy_report(
        KondoActorCriticOnPolicyConfig()
    )
    tampered = copy.deepcopy(report)
    records = cast(list[dict[str, Any]], tampered["arm_records"])
    first = records[0]
    first["actor_loss"] = cast(float, first["actor_loss"]) + 0.125
    record_body = {
        name: first[name] for name in first if name != "record_sha256"
    }
    first["record_sha256"] = _canonical_sha256(record_body)
    records[0] = first
    tampered["arm_records"] = records
    tampered["arm_records_sha256"] = _canonical_sha256(records)
    report_body = {
        name: tampered[name] for name in tampered if name != "report_sha256"
    }
    tampered["report_sha256"] = _canonical_sha256(report_body)

    with pytest.raises(ValueError, match="exact causal reconstruction"):
        validate_kondo_actor_critic_on_policy_report(tampered)


@pytest.mark.parametrize("binding", ["source_manifest", "runtime_identity"])
def test_recomputed_outer_digest_cannot_hide_binding_tamper(binding: str) -> None:
    report = build_kondo_actor_critic_on_policy_report(
        KondoActorCriticOnPolicyConfig()
    )
    tampered = copy.deepcopy(report)
    target = cast(dict[str, object], tampered[binding])
    first_key = next(iter(target))
    target[first_key] = "tampered"
    tampered[binding] = target
    body = {name: tampered[name] for name in tampered if name != "report_sha256"}
    tampered["report_sha256"] = _canonical_sha256(body)

    expected = "source manifest" if binding == "source_manifest" else "runtime identity"
    with pytest.raises(ValueError, match=expected):
        validate_kondo_actor_critic_on_policy_report(tampered)


def test_report_has_no_output_or_promotion_path() -> None:
    report = build_kondo_actor_critic_on_policy_report(
        KondoActorCriticOnPolicyConfig()
    )
    protocol = cast(Mapping[str, object], report["protocol"])

    assert report["output_written"] is False
    assert report["output_path"] is None
    assert report["evidence_claimed"] is False
    assert report["promotion_authority"] is False
    assert report["scientific_promotion_allowed"] is False
    assert protocol["output_writes"] is False
