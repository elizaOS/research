# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Unit contracts for the nonpromoting Kondo actor/critic replay lane."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.evaluation.kondo_actor_critic_replay_development import (
    ARM_ORDER,
    ASSESSMENT_STATUS,
    DEVELOPMENT_STATUS,
    PROMOTION_AUTHORITY,
    SCIENTIFIC_PROMOTION_ALLOWED,
    SPARSE_ARM_ORDER,
    KondoActorCriticReplayConfig,
    KondoActorCriticReplayEvaluator,
    build_kondo_actor_critic_replay_report,
    build_kondo_replay_source_batch,
    kondo_replay_protocol,
    kondo_replay_runtime_identity,
    kondo_replay_source_manifest,
    replay_protected_backward_kernel,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def config() -> KondoActorCriticReplayConfig:
    return KondoActorCriticReplayConfig(batches_per_phase=1)


@pytest.fixture(scope="module")
def report(config: KondoActorCriticReplayConfig) -> dict[str, object]:
    return build_kondo_actor_critic_replay_report(config)


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


def _assert_tree_numerically_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(lhs)
        right_array = np.asarray(rhs)
        if np.issubdtype(left_array.dtype, np.floating):
            np.testing.assert_allclose(left_array, right_array, rtol=1.0e-6, atol=1.0e-6)
        else:
            np.testing.assert_array_equal(left_array, right_array)


def test_config_protocol_and_authority_are_fail_closed(
    config: KondoActorCriticReplayConfig,
) -> None:
    payload = config.to_config()
    protocol = kondo_replay_protocol(config)

    assert DEVELOPMENT_STATUS == "not_assessed"
    assert ASSESSMENT_STATUS == "not_assessed"
    assert PROMOTION_AUTHORITY is False
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert KondoActorCriticReplayConfig.from_config(payload) == config
    legacy_payload = dict(payload)
    legacy_payload["schema"] = "alberta.kondo-actor-critic-replay.config.v1"
    with pytest.raises(ValueError, match="schema"):
        KondoActorCriticReplayConfig.from_config(legacy_payload)
    assert payload["schema"] == "alberta.kondo-actor-critic-replay.config.v2"
    assert protocol["schema"] == "alberta.kondo-actor-critic-replay.protocol.v2"
    assert protocol["arms"] == list(ARM_ORDER)
    assert protocol["sparse_arms"] == list(SPARSE_ARM_ORDER)
    assert protocol["phase_order"] == ["A1", "B", "A2"]
    assert protocol["closed_loop_control"] is False
    assert protocol["output_writes"] is False
    assert protocol["evidence_seed"] is None
    assert protocol["thresholds"] == []
    assert protocol["assessment_status"] == "not_assessed"
    assert protocol["executed_actor_backward_mask_semantics"] == (
        "gradient-contribution-entered-executed-actor-backward"
    )
    assert protocol["sparks_joy_scope"] == "KondoSparseActorResult-only"
    assert protocol["manual_kernel_arms_are_kondo_sparse_actor_transactions"] is False
    assert protocol["ordinary_full_delight_selection_claimed"] is False
    assert "sparks_joy" not in protocol
    for name in (
        "performance_claimed",
        "speedup_claimed",
        "efficacy_claimed",
        "safety_claimed",
        "policy_authority",
        "guardrail_authority",
        "promotion_authority",
        "scientific_promotion_allowed",
    ):
        assert protocol[name] is False


def test_off_policy_source_action_semantics_are_machine_readable(
    config: KondoActorCriticReplayConfig,
    report: dict[str, object],
) -> None:
    protocol = _mapping(report["protocol"])
    records = [_mapping(item) for item in _list(report["arm_records"])]

    assert report["schema"] == "alberta.kondo-actor-critic-replay.report.v2"
    assert protocol["source_action_generation"] == "evaluator-fixed-alternating-actions"
    assert protocol["source_behavior_policy_available"] is False
    assert protocol["on_policy"] is False
    assert protocol["importance_correction_applied"] is False
    assert protocol["valid_policy_gradient_efficacy_claim"] is False
    assert report["source_behavior_policy_available"] is False
    assert report["on_policy"] is False
    assert report["importance_correction_applied"] is False
    assert report["valid_policy_gradient_efficacy_claim"] is False
    for record in records:
        assert record["source_behavior_policy_available"] is False
        assert record["on_policy"] is False
        assert record["importance_correction_applied"] is False
        assert record["valid_policy_gradient_efficacy_claim"] is False
        assert "behavior_log_probability" not in record
        assert record["current_policy_log_probability_revision_binding_exact"] is True
        assert record["source_action_identity_exact"] is True
        assert record["selected_action_surprisal_semantics"] == (
            "current-policy-surprisal-of-evaluator-fixed-recorded-action"
        )
        assert record["actor_loss_semantics"] == (
            "uncorrected-off-policy-surrogate-not-policy-gradient-efficacy"
        )
    assert config.action_count == 2


@pytest.mark.parametrize(
    "values",
    [
        {"seed": -1},
        {"batch_size": 1},
        {"batch_size": 7},
        {"batches_per_phase": 0},
        {"actor_feature_dim": 1},
        {"context_dim": 1},
        {"critic_dim": 129},
        {"representation_dim": 65},
        {"action_count": 3},
        {"target_rate": 0.0},
        {"target_rate": 1.0},
        {"reserve_count": 0},
        {"reserve_count": 3},
        {"actor_learning_rate": 1.0e-45},
        {"protected_learning_rate": float("nan")},
        {"rare_failure_period": 1},
        {"rare_failure_period": 7},
    ],
)
def test_config_caps_and_domains_fail_closed(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        KondoActorCriticReplayConfig(**values)


def test_source_and_runtime_manifests_bind_complete_lane() -> None:
    source = kondo_replay_source_manifest()
    runtime = kondo_replay_runtime_identity()

    assert set(source) == {
        "alberta_framework/core/kondo_gate.py",
        "alberta_framework/core/kondo_sparse_actor.py",
        "alberta_framework/evaluation/kondo_actor_critic_replay_development.py",
    }
    assert all(len(value) == 64 for value in source.values())
    for name in (
        "python_version",
        "chex_version",
        "jax_version",
        "jaxlib_version",
        "numpy_version",
        "backend",
        "backend_platform_version",
    ):
        assert runtime[name]
    assert cast(int, runtime["device_count"]) >= 1
    assert (
        1
        <= cast(int, runtime["local_device_count"])
        <= cast(
            int,
            runtime["device_count"],
        )
    )


def test_source_trace_is_one_uninterrupted_a_b_a_sequence_with_rare_stratum(
    config: KondoActorCriticReplayConfig,
    report: dict[str, object],
) -> None:
    trace = [_mapping(item) for item in _list(report["source_trace"])]

    assert [item["event_index"] for item in trace] == list(range(config.total_batches))
    assert [item["phase"] for item in trace] == ["A1", "B", "A2"]
    assert [item["regime"] for item in trace] == ["A", "B", "A"]
    failure_count = 0
    phase_failure_counts = {"A1": 0, "B": 0, "A2": 0}
    for index, item in enumerate(trace):
        source = build_kondo_replay_source_batch(config, index)
        assert source.payload() == item
        actions = _decode_array(item["actions"])
        np.testing.assert_array_equal(
            actions,
            (np.arange(config.batch_size, dtype=np.int32) + index) % 2,
        )
        uniform = _decode_array(item["uniform_indices"]).tolist()
        assert len(uniform) == config.sparse_capacity
        assert len(set(uniform)) == config.sparse_capacity
        failures = _decode_array(item["failure_mask"])
        failure_count += int(np.sum(failures))
        phase_failure_counts[cast(str, item["phase"])] += int(np.sum(failures))
    assert 0 < failure_count < config.total_batches * config.batch_size
    assert all(count >= 1 for count in phase_failure_counts.values())


def test_one_source_and_initial_snapshot_bind_every_arm(
    config: KondoActorCriticReplayConfig,
    report: dict[str, object],
) -> None:
    records = [_mapping(item) for item in _list(report["arm_records"])]
    trace = [_mapping(item) for item in _list(report["source_trace"])]
    initial_sha = cast(str, report["initial_snapshot_sha256"])

    assert len(records) == config.total_batches * len(ARM_ORDER)
    assert len(initial_sha) == 64
    for event_index, source in enumerate(trace):
        event_records = [item for item in records if item["event_index"] == event_index]
        assert [item["arm"] for item in event_records] == list(ARM_ORDER)
        assert {item["source_batch_sha256"] for item in event_records} == {
            source["source_batch_sha256"]
        }
        assert len({item["source_actions_sha256"] for item in event_records}) == 1
        assert len({item["action_identity_sha256"] for item in event_records}) == 1
        assert {item["policy_revision_before"] for item in event_records} == {event_index}
        assert {item["policy_revision_after"] for item in event_records} == {event_index + 1}
        assert all(item["source_experience_replays_in_arm"] == 1 for item in event_records)


def test_paper_delight_and_executed_backward_mask_reconstruct_from_current_policy(
    config: KondoActorCriticReplayConfig,
    report: dict[str, object],
) -> None:
    records = [_mapping(item) for item in _list(report["arm_records"])]

    for record in records:
        current_log_probability = _decode_array(
            record["current_policy_selected_action_log_probability"]
        )
        advantage = _decode_array(record["advantage"])
        surprisal = _decode_array(record["selected_action_surprisal"])
        delight = _decode_array(record["paper_delight"])
        executed_mask = _decode_array(record["executed_actor_backward_mask"])
        np.testing.assert_array_equal(surprisal, -current_log_probability)
        np.testing.assert_array_equal(delight, advantage * surprisal)
        assert executed_mask.dtype == np.bool_
        selected = np.flatnonzero(executed_mask).tolist()
        assert record["selected_source_indices"] == selected
        assert len(selected) == record["selected_count"]
        assert record["executed_actor_backward_mask_semantics"] == (
            "gradient-contribution-entered-executed-actor-backward"
        )
        assert "sparks_joy" not in record
        assert "sparks_joy_semantics" not in record
        assert np.all(np.isfinite(delight))
        expected_count = (
            config.batch_size if record["arm"] == "ordinary_full" else config.sparse_capacity
        )
        assert len(selected) == expected_count


def test_actual_actor_microbatch_shapes_invocations_and_reserve_are_exact(
    config: KondoActorCriticReplayConfig,
    report: dict[str, object],
) -> None:
    records = [_mapping(item) for item in _list(report["arm_records"])]

    for record in records:
        arm = cast(str, record["arm"])
        expected_shape = config.batch_size if arm == "ordinary_full" else config.sparse_capacity
        assert record["actor_backward_leading_shape"] == expected_shape
        assert record["actor_compiled_backward_invocations"] == 1
        assert record["actor_update_opportunities"] == 1
        assert record["actor_updates_applied"] == 1
        assert len(cast(list[int], record["actor_backward_gather_order"])) == expected_shape
        if arm in SPARSE_ARM_ORDER:
            assert record["sparse_actor_backward"] is True
        else:
            assert record["sparse_actor_backward"] is False
    reserve_records = [item for item in records if item["arm"] == "kondo_top_k_reserve"]
    for record in reserve_records:
        reserve = _decode_array(record["minimum_random_reserve"])
        executed_mask = _decode_array(record["executed_actor_backward_mask"])
        assert int(np.sum(reserve)) == config.reserve_count
        assert np.all(~reserve | executed_mask)
        assert record["random_draw_count"] == config.batch_size


def test_all_protected_learning_is_full_batch_and_bit_identical(
    config: KondoActorCriticReplayConfig,
    report: dict[str, object],
) -> None:
    records = [_mapping(item) for item in _list(report["arm_records"])]
    loss_names = (
        "protected_total_loss",
        "baseline_loss",
        "critic_loss",
        "representation_loss",
        "world_model_loss",
        "safety_guardrail_loss",
        "protected_gradient_l2",
    )

    for event_index in range(config.total_batches):
        event_records = [item for item in records if item["event_index"] == event_index]
        for digest_name in (
            "actor_protected_inputs_sha256",
            "protected_learning_batch_sha256",
            "protected_state_before_sha256",
            "protected_state_after_sha256",
            "protected_predictions_sha256",
        ):
            assert len({item[digest_name] for item in event_records}) == 1
        internal_kondo_digests = {
            item["internal_kondo_protected_digest"]
            for item in event_records
            if item["internal_kondo_protected_digest"] is not None
        }
        assert len(internal_kondo_digests) == 1
        assert len(cast(str, next(iter(internal_kondo_digests)))) == 64
        for loss_name in loss_names:
            values = [item[loss_name] for item in event_records]
            assert all(type(value) is float and math.isfinite(value) for value in values)
            assert len(set(values)) == 1
        for item in event_records:
            assert item["protected_channels_full_batch"] is True
            assert item["protected_rows_in_backward"] == config.batch_size
            assert item["protected_backward_leading_shape"] == config.batch_size
            assert item["protected_compiled_backward_invocations"] == 1
            assert item["protected_update_opportunities"] == 1
            assert item["protected_updates_applied"] == 1
            assert (
                item["rare_failure_rows_in_protected_backward"]
                == item["rare_failure_rows_in_source"]
            )


def test_resource_accounting_is_matched_and_never_doubles_budget(
    config: KondoActorCriticReplayConfig,
    report: dict[str, object],
) -> None:
    accounting = _mapping(report["logical_resource_accounting"])
    per_arm = _mapping(accounting["per_arm"])

    assert accounting["unique_environment_batches"] == config.total_batches
    assert accounting["deterministic_training_trace_executions"] == 1
    assert accounting["training_trace_replays_per_arm"] == 1
    assert accounting["experience_double_counted_within_arm"] is False
    assert accounting["measured_flops"] is False
    assert accounting["wall_clock_measured"] is False
    for arm in ARM_ORDER:
        values = _mapping(per_arm[arm])
        assert values["source_batches_consumed"] == config.total_batches
        assert values["source_trace_replays"] == 1
        assert values["actor_update_opportunities"] == config.total_batches
        assert values["actor_updates_applied"] == config.total_batches
        assert values["actor_compiled_backward_invocations"] == config.total_batches
        assert values["protected_update_opportunities"] == config.total_batches
        assert values["protected_updates_applied"] == config.total_batches
        assert values["protected_compiled_backward_invocations"] == config.total_batches
        assert values["protected_backward_row_slots"] == (config.total_batches * config.batch_size)
    ordinary = _mapping(per_arm["ordinary_full"])
    for arm in SPARSE_ARM_ORDER:
        sparse = _mapping(per_arm[arm])
        assert sparse["actor_backward_row_slots"] == (config.total_batches * config.sparse_capacity)
        assert sparse["actor_backward_row_slots"] < ordinary["actor_backward_row_slots"]  # type: ignore[operator]


def test_rare_safety_and_recurrence_readouts_are_descriptive_only(
    report: dict[str, object],
) -> None:
    diagnostics = _mapping(report["diagnostics"])
    per_arm = _mapping(diagnostics["per_arm"])

    assert diagnostics["protected_final_state_bit_identical_across_arms"] is True
    assert diagnostics["rare_failure_coverage_is_descriptive"] is True
    assert diagnostics["recurrence_recovery_retention_thresholds"] == []
    assert diagnostics["assessment_status"] == "not_assessed"
    final_protected = set()
    for arm in ARM_ORDER:
        values = _mapping(per_arm[arm])
        recurrence = _mapping(values["recurrence_recovery_retention"])
        assert set(recurrence) == {
            "initial_a_probe_loss",
            "initial_b_probe_loss",
            "post_a1_a_probe_loss",
            "post_a1_b_probe_loss",
            "post_b_a_probe_loss",
            "post_b_b_probe_loss",
            "final_a_probe_loss",
            "final_b_probe_loss",
            "first_a_learning_delta",
            "b_adaptation_delta",
            "a_recovery_delta",
            "a_cycle_retention_delta",
            "b_retention_delta",
        }
        assert all(type(value) is float and math.isfinite(value) for value in recurrence.values())
        assert values["rare_failure_protected_learning_rows"] == values["rare_failure_source_rows"]
        assert values["assessment_status"] == "not_assessed"
        final_protected.add(values["final_protected_state_sha256"])
    assert len(final_protected) == 1
    assert report["thresholds"] == []
    assert report["verdict"] == "not_assessed"


def test_protected_backward_kernel_supports_eager_jit_and_scan(
    config: KondoActorCriticReplayConfig,
) -> None:
    evaluator = KondoActorCriticReplayEvaluator(config)
    parameters = evaluator.initial_protected_parameters
    batch = evaluator.source_batch(0).protected_batch()

    with jax.disable_jit():
        eager = replay_protected_backward_kernel(parameters, batch)
    compiled = jax.jit(replay_protected_backward_kernel)(parameters, batch)
    _assert_tree_numerically_equal(eager, compiled)

    def body(carry: jax.Array, _: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = replay_protected_backward_kernel(parameters, batch)
        return carry + jnp.asarray(1, dtype=jnp.int32), result.total_loss

    count, losses = jax.lax.scan(
        body,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.arange(2, dtype=jnp.int32),
    )
    assert int(np.asarray(count)) == 2
    loss_array = np.asarray(losses)
    np.testing.assert_array_equal(loss_array[0], loss_array[1])
    np.testing.assert_allclose(
        loss_array,
        np.repeat(np.asarray(compiled.total_loss)[None], 2, axis=0),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_every_status_and_claim_remains_fail_closed(report: dict[str, object]) -> None:
    false_fields = {
        "performance_claimed",
        "speedup_claimed",
        "efficacy_claimed",
        "safety_claimed",
        "policy_authority",
        "guardrail_authority",
        "promotion_authority",
        "scientific_promotion_allowed",
        "output_writes",
        "source_behavior_policy_available",
        "on_policy",
        "importance_correction_applied",
        "valid_policy_gradient_efficacy_claim",
    }

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for name, child in value.items():
                if name.endswith("status"):
                    assert child == "not_assessed"
                if name in false_fields:
                    assert child is False
                visit(child)
        elif type(value) is list:
            for child in cast(list[object], value):
                visit(child)

    visit(report)
