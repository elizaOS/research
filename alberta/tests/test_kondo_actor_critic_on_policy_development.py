# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Unit contracts for the closed-loop on-policy Kondo evaluator."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation import kondo_actor_critic_on_policy_development as module
from alberta_framework.evaluation.kondo_actor_critic_on_policy_development import (
    ARM_ORDER,
    ASSESSMENT_STATUS,
    CHECKPOINT_HOST_ONLY,
    DEVELOPMENT_STATUS,
    OUTPUT_WRITES,
    PROMOTION_AUTHORITY,
    SCIENTIFIC_PROMOTION_ALLOWED,
    SPARSE_ARM_ORDER,
    KondoActorCriticOnPolicyConfig,
    KondoActorCriticOnPolicyEvaluator,
    build_kondo_actor_critic_on_policy_report,
    kondo_on_policy_protocol,
    kondo_on_policy_runtime_identity,
    kondo_on_policy_source_manifest,
    on_policy_selected_log_probability,
    sample_on_policy_actions_from_uniforms,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def config() -> KondoActorCriticOnPolicyConfig:
    return KondoActorCriticOnPolicyConfig()


@pytest.fixture(scope="module")
def report(config: KondoActorCriticOnPolicyConfig) -> dict[str, object]:
    return build_kondo_actor_critic_on_policy_report(config)


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return cast(Mapping[str, object], value)


def _list(value: object) -> list[object]:
    assert type(value) is list
    return cast(list[object], value)


def _decode_array(payload: object) -> np.ndarray[Any, Any]:
    raw = _mapping(payload)
    dtype = np.dtype(cast(str, raw["dtype"]))
    shape = tuple(cast(list[int], raw["shape"]))
    result = np.frombuffer(
        bytes.fromhex(cast(str, raw["data_hex"])),
        dtype=dtype,
    ).reshape(shape)
    assert raw["sha256"] == hashlib.sha256(result.tobytes()).hexdigest()
    return result


def test_config_protocol_and_authority_are_strict_and_nonpromoting(
    config: KondoActorCriticOnPolicyConfig,
) -> None:
    protocol = kondo_on_policy_protocol(config)

    assert KondoActorCriticOnPolicyConfig.from_config(config.to_config()) == config
    legacy_payload = config.to_config()
    legacy_payload["schema"] = "alberta.kondo-actor-critic-on-policy.config.v1"
    with pytest.raises(ValueError, match="schema"):
        KondoActorCriticOnPolicyConfig.from_config(legacy_payload)
    assert config.to_config()["schema"] == (
        "alberta.kondo-actor-critic-on-policy.config.v2"
    )
    assert protocol["schema"] == "alberta.kondo-actor-critic-on-policy.protocol.v2"
    assert DEVELOPMENT_STATUS == ASSESSMENT_STATUS == "not_assessed"
    assert CHECKPOINT_HOST_ONLY
    assert OUTPUT_WRITES is False
    assert PROMOTION_AUTHORITY is False
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert protocol["arms"] == list(ARM_ORDER)
    assert protocol["sparse_arms"] == list(SPARSE_ARM_ORDER)
    assert protocol["closed_loop_control"] is True
    assert protocol["on_policy"] is True
    assert protocol["actions_sampled_from_each_arms_own_policy"] is True
    assert protocol["actor_revision_immutable_within_batch"] is True
    assert protocol["actor_updates_only_at_batch_boundaries"] is True
    assert protocol["common_schedule_pairs_exogenous_randomness_only"] is True
    assert protocol["uniform_control_allocation_randomness_paired_across_arms"] is False
    assert protocol["trajectory_equality_assumed"] is False
    assert protocol["protected_learning_values_equal_across_arms_required"] is False
    assert protocol["output_writes"] is False
    assert protocol["assessment_status"] == "not_assessed"
    assert protocol["thresholds"] == []
    assert protocol["executed_actor_backward_mask_semantics"] == (
        "gradient-contribution-entered-executed-actor-backward"
    )
    assert protocol["sparks_joy_scope"] == "KondoSparseActorResult-only"
    assert protocol["manual_kernel_arms_are_kondo_sparse_actor_transactions"] is False
    assert protocol["ordinary_full_delight_selection_claimed"] is False
    assert "sparks_joy" not in protocol
    for name in (
        "performance_claimed",
        "compute_benefit_claimed",
        "efficacy_claimed",
        "safety_claimed",
        "policy_authority",
        "guardrail_authority",
        "promotion_authority",
        "scientific_promotion_allowed",
    ):
        assert protocol[name] is False


@pytest.mark.parametrize(
    "values",
    [
        {"seed": -1},
        {"batch_size": 7},
        {"batches_per_phase": 0},
        {"actor_feature_dim": 1},
        {"context_dim": 1},
        {"critic_dim": 129},
        {"representation_dim": 65},
        {"action_count": 3},
        {"target_rate": 0.0},
        {"target_rate": 1.0},
        {"target_rate": 0.125},
        {"reserve_count": 0},
        {"actor_learning_rate": 1.0e-45},
        {"protected_learning_rate": float("nan")},
    ],
)
def test_config_caps_and_domains_fail_closed(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        KondoActorCriticOnPolicyConfig(**values)


def test_source_and_runtime_fingerprints_cover_the_lane() -> None:
    source = kondo_on_policy_source_manifest()
    runtime = kondo_on_policy_runtime_identity()

    assert set(source) == {
        "alberta_framework/core/kondo_gate.py",
        "alberta_framework/core/kondo_sparse_actor.py",
        "alberta_framework/evaluation/kondo_actor_critic_replay_development.py",
        "alberta_framework/evaluation/kondo_actor_critic_on_policy_development.py",
    }
    assert all(len(value) == 64 for value in source.values())
    for name in (
        "python_version",
        "jax_version",
        "jaxlib_version",
        "backend",
        "backend_platform_version",
        "jax_default_prng_impl",
    ):
        assert runtime[name]
    assert cast(int, runtime["device_count"]) >= 1


def test_common_schedule_is_exact_typed_threefry_exogenous_crn(
    config: KondoActorCriticOnPolicyConfig,
) -> None:
    evaluator = KondoActorCriticOnPolicyEvaluator(config)
    schedule = evaluator.common_schedule(0)
    root = jr.key(config.seed, impl="threefry2x32")
    keys = jr.split(jr.fold_in(root, np.uint32(10_000)), 3)

    np.testing.assert_array_equal(
        schedule.key_words,
        jnp.stack(tuple(jr.key_data(key) for key in keys)),
    )
    np.testing.assert_array_equal(
        schedule.action_uniforms,
        jr.uniform(keys[0], (config.batch_size,), dtype=jnp.float32),
    )
    assert schedule.action_uniforms.dtype == jnp.float32
    assert schedule.transition_uniforms.shape == (
        config.batch_size,
        config.context_dim,
    )
    assert int(np.sum(np.asarray(schedule.failure_mask))) == 1


def test_each_arm_samples_own_policy_under_one_immutable_revision(
    config: KondoActorCriticOnPolicyConfig,
) -> None:
    evaluator = KondoActorCriticOnPolicyEvaluator(config)
    state = evaluator.advance(evaluator.init())
    before = evaluator._state_body(state)
    batches = evaluator.collect_current_batches(state)
    schedule = evaluator.common_schedule(state.event_index)
    actor_states = (
        state.ordinary_actor,
        state.uniform_actor,
        state.kondo_state,
        state.reserve_state,
    )

    for actor_state, batch in zip(actor_states, batches, strict=True):
        expected_actions = sample_on_policy_actions_from_uniforms(
            actor_state.parameters,
            batch.actor_features,
            schedule.action_uniforms,
        )
        expected_log_probability = on_policy_selected_log_probability(
            actor_state.parameters,
            batch.actor_features,
            batch.actions,
        )
        np.testing.assert_array_equal(batch.actions, expected_actions)
        np.testing.assert_array_equal(batch.action_identity, batch.actions)
        np.testing.assert_array_equal(
            batch.policy_revision,
            np.repeat(state.event_index, config.batch_size),
        )
        np.testing.assert_array_equal(
            batch.behavior_log_probability,
            expected_log_probability,
        )
        np.testing.assert_array_equal(batch.action_uniforms, schedule.action_uniforms)
    assert evaluator._state_body(state) == before


def test_updates_happen_once_only_after_batch_collection(
    config: KondoActorCriticOnPolicyConfig,
) -> None:
    evaluator = KondoActorCriticOnPolicyEvaluator(config)
    initial = evaluator.init()
    collected = evaluator.collect_current_batches(initial)
    assert all(np.all(np.asarray(item.policy_revision) == 0) for item in collected)
    assert int(np.asarray(initial.ordinary_actor.actor_backward_count)) == 0

    advanced = evaluator.advance(initial)

    assert advanced.event_index == 1
    assert int(np.asarray(advanced.ordinary_actor.policy_revision)) == 1
    assert int(np.asarray(advanced.uniform_actor.policy_revision)) == 1
    assert int(np.asarray(advanced.kondo_state.policy_revision)) == 1
    assert int(np.asarray(advanced.reserve_state.policy_revision)) == 1
    assert int(np.asarray(advanced.ordinary_actor.actor_backward_count)) == 1
    assert int(np.asarray(advanced.uniform_actor.actor_backward_count)) == 1
    assert int(np.asarray(advanced.kondo_state.actor_backward_count)) == 1
    assert int(np.asarray(advanced.reserve_state.actor_backward_count)) == 1
    for protected in (
        advanced.ordinary_protected,
        advanced.uniform_protected,
        advanced.kondo_protected,
        advanced.reserve_protected,
    ):
        assert int(np.asarray(protected.update_count)) == 1
    for environment in (
        advanced.ordinary_environment,
        advanced.uniform_environment,
        advanced.kondo_environment,
        advanced.reserve_environment,
    ):
        assert int(np.asarray(environment.step_count)) == config.batch_size
    for raw in advanced.records_json:
        record = _mapping(json.loads(raw))
        assert record["actor_revision_immutable_during_collection"] is True
        assert record["actor_updated_only_after_batch_collection"] is True
        assert record["actor_update_opportunities"] == 1
        assert record["protected_update_opportunities"] == 1


def test_forced_failures_and_guardrail_receive_full_learning(
    config: KondoActorCriticOnPolicyConfig,
    report: dict[str, object],
) -> None:
    records = [_mapping(item) for item in _list(report["arm_records"])]

    for record in records:
        failure = _decode_array(record["force_keep_mask"])
        selected = _decode_array(record["executed_actor_backward_mask"])
        assert int(np.sum(failure)) == 1
        assert np.all(~failure | selected)
        assert record["executed_actor_backward_mask_semantics"] == (
            "gradient-contribution-entered-executed-actor-backward"
        )
        assert "sparks_joy" not in record
        assert "sparks_joy_semantics" not in record
        assert record["rare_failure_rows_in_actor_backward"] == 1
        assert record["rare_failure_rows_in_protected_backward"] == 1
        assert record["rare_failure_guardrail_full_learning"] is True
        assert record["protected_rows_in_backward"] == config.batch_size
        assert record["protected_backward_leading_shape"] == config.batch_size
        assert record["protected_updates_applied"] == 1
    reserve = [
        item for item in records if item["arm"] == "kondo_top_k_reserve"
    ]
    assert all(
        int(np.sum(_decode_array(item["minimum_random_reserve"])))
        == config.reserve_count
        for item in reserve
    )


def test_closed_loop_divergence_does_not_reduce_protected_learning(
    report: dict[str, object],
) -> None:
    diagnostics = _mapping(report["diagnostics"])
    audit = [_mapping(item) for item in _list(diagnostics["trajectory_divergence_audit"])]
    per_arm = _mapping(diagnostics["per_arm"])

    assert diagnostics["trajectory_equality_assumed"] is False
    assert any(cast(int, item["unique_action_trace_count"]) > 1 for item in audit)
    assert any(cast(int, item["unique_environment_after_count"]) > 1 for item in audit)
    protected_digests = {
        _mapping(per_arm[arm])["final_protected_state_sha256"] for arm in ARM_ORDER
    }
    assert len(protected_digests) > 1
    for arm in ARM_ORDER:
        arm_values = _mapping(per_arm[arm])
        assert arm_values["protected_update_count"] == 3
        assert arm_values["rare_failures_in_protected_backward"] == 3
        assert arm_values["assessment_status"] == "not_assessed"


def test_report_resource_accounting_and_claims_remain_descriptive(
    config: KondoActorCriticOnPolicyConfig,
    report: dict[str, object],
) -> None:
    accounting = _mapping(report["logical_resource_accounting"])
    per_arm = _mapping(accounting["per_arm"])

    assert report["schema"] == "alberta.kondo-actor-critic-on-policy.report.v2"
    assert report["verdict"] == "not_assessed"
    assert report["thresholds"] == []
    assert report["output_written"] is False
    assert report["output_path"] is None
    assert report["trajectory_equality_assumed"] is False
    assert accounting["wall_clock_measured"] is False
    assert accounting["measured_flops"] is False
    for arm in ARM_ORDER:
        values = _mapping(per_arm[arm])
        assert values["closed_loop_batches_collected"] == config.total_batches
        assert values["actor_update_opportunities"] == config.total_batches
        assert values["actor_updates_applied"] == config.total_batches
        assert values["protected_update_opportunities"] == config.total_batches
        assert values["protected_updates_applied"] == config.total_batches
        assert values["protected_backward_row_slots"] == (
            config.total_batches * config.batch_size
        )
    ordinary = _mapping(per_arm["ordinary_full"])
    for arm in SPARSE_ARM_ORDER:
        sparse = _mapping(per_arm[arm])
        assert sparse["actor_backward_row_slots"] == (
            config.total_batches * config.sparse_capacity
        )
        assert sparse["actor_backward_row_slots"] < cast(
            int, ordinary["actor_backward_row_slots"]
        )
    for name in (
        "performance_claimed",
        "compute_benefit_claimed",
        "efficacy_claimed",
        "safety_claimed",
        "policy_authority",
        "guardrail_authority",
        "evidence_claimed",
        "promotion_authority",
        "scientific_promotion_allowed",
    ):
        assert report[name] is False


def test_corrupted_run_state_fails_closed(
    config: KondoActorCriticOnPolicyConfig,
) -> None:
    evaluator = KondoActorCriticOnPolicyEvaluator(config)
    state = evaluator.advance(evaluator.init())
    corrupted = dataclasses.replace(state, integrity_sha256="0" * 64)

    assert not evaluator.validate_state(corrupted)
    with pytest.raises(ValueError, match="state is invalid"):
        evaluator.advance(corrupted)


def test_module_public_surface_is_complete() -> None:
    for name in module.__all__:
        assert getattr(module, name) is not None
