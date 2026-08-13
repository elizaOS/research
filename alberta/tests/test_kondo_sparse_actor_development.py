# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Unit contracts for the nonpromoting Kondo sparse-backward evaluator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.kondo_sparse_actor_development as development_module
from alberta_framework.evaluation.kondo_sparse_actor_development import (
    ARM_ORDER,
    ASSESSMENT_STATUS,
    DEVELOPMENT_STATUS,
    DIAGNOSTIC_ARMS,
    MATCHED_DEVELOPMENT_ARMS,
    PROMOTION_AUTHORITY,
    SCIENTIFIC_PROMOTION_ALLOWED,
    KondoSparseActorDevelopmentConfig,
    build_kondo_sparse_actor_deterministic_payload,
    build_kondo_sparse_actor_development_report,
    kondo_sparse_actor_development_protocol,
    kondo_sparse_actor_development_runtime_identity,
    kondo_sparse_actor_development_source_manifest,
    measure_kondo_sparse_actor_backward_timing,
    validate_kondo_sparse_actor_development_report,
)

pytestmark = pytest.mark.unit


class _DeterministicClock:
    def __init__(self) -> None:
        self._now = 10_000
        self._start = True
        self._sample = 0

    def __call__(self) -> int:
        if self._start:
            self._start = False
            return self._now
        duration = 11 + (self._sample % 9) * 3
        self._now += duration
        self._sample += 1
        self._start = True
        return self._now


class _BackwardClock:
    def __init__(self) -> None:
        self._values = iter((1_000, 1_010, 1_009))

    def __call__(self) -> int:
        return next(self._values)


@pytest.fixture(scope="module")
def config() -> KondoSparseActorDevelopmentConfig:
    return KondoSparseActorDevelopmentConfig(num_batches=2, timing_trials=3)


@pytest.fixture(scope="module")
def deterministic(config: KondoSparseActorDevelopmentConfig) -> dict[str, object]:
    return build_kondo_sparse_actor_deterministic_payload(config)


@pytest.fixture(scope="module")
def report(config: KondoSparseActorDevelopmentConfig) -> dict[str, object]:
    return build_kondo_sparse_actor_development_report(
        config,
        clock_ns=_DeterministicClock(),
        clock_name="deterministic-test-clock",
    )


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
    return np.frombuffer(bytes.fromhex(cast(str, raw["data_hex"])), dtype=dtype).reshape(
        shape
    )


def test_config_and_protocol_are_explicitly_nonpromoting(
    config: KondoSparseActorDevelopmentConfig,
) -> None:
    payload = config.to_config()
    protocol = kondo_sparse_actor_development_protocol(config)

    assert DEVELOPMENT_STATUS == "not_assessed"
    assert ASSESSMENT_STATUS == "not_assessed"
    assert PROMOTION_AUTHORITY is False
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert payload["assessment_status"] == "not_assessed"
    assert payload["schema"] == (
        "alberta.kondo-sparse-actor-development.config.v2"
    )
    assert payload["seed_role"] == "development-trace-and-uniform-control-only"
    assert payload["evidence_seed"] is None
    assert payload["thresholds"] == []
    assert KondoSparseActorDevelopmentConfig.from_config(payload) == config
    legacy_payload = dict(payload)
    legacy_payload["schema"] = "alberta.kondo-sparse-actor-development.config.v1"
    with pytest.raises(ValueError, match="schema"):
        KondoSparseActorDevelopmentConfig.from_config(legacy_payload)
    assert protocol["matched_development_arms"] == list(MATCHED_DEVELOPMENT_ARMS)
    assert protocol["diagnostic_only_arms"] == list(DIAGNOSTIC_ARMS)
    assert protocol["external_source_experience_equal"] is True
    assert protocol["source_actions_equal"] is True
    assert protocol["source_protected_inputs_equal"] is True
    assert protocol["closed_loop_environment_experience_measured"] is False
    assert protocol["selected_samples_equal"] is False
    assert protocol["schema"] == (
        "alberta.kondo-sparse-actor-development.protocol.v2"
    )
    assert protocol["executed_actor_backward_inclusion_semantics"] == (
        "gradient-contribution-entered-executed-actor-backward"
    )
    assert protocol["sparks_joy_scope"] == "KondoSparseActorResult-only"
    assert protocol["manual_kernel_arms_are_kondo_sparse_actor_transactions"] is False
    assert protocol["ordinary_full_delight_selection_claimed"] is False
    assert "sparks_joy_semantics" not in protocol
    assert protocol["host_screen_gather_timed"] is False
    assert protocol["accelerator_memory_measured"] is False
    assert protocol["energy_measured"] is False
    assert protocol["evidence_seed"] is None
    assert protocol["promotion_seed_eligible"] is False
    assert protocol["performance_claimed"] is False
    assert protocol["compute_saving_claimed"] is False
    assert protocol["efficacy_claimed"] is False
    assert protocol["safety_claimed"] is False
    assert protocol["policy_authority"] is False
    assert protocol["output_writes"] is False
    assert protocol["thresholds"] == []


@pytest.mark.parametrize(
    "values",
    [
        {"seed": -1},
        {"timing_order_seed": 2**32},
        {"batch_size": 1},
        {"num_batches": 0},
        {"feature_dim": 0},
        {"hidden_dim": 513},
        {"action_count": True},
        {"target_rate": 0.0},
        {"target_rate": 1.0},
        {"learning_rate": float("nan")},
        {"learning_rate": 1.0e-45},
        {"timing_trials": 129},
        {
            "batch_size": 512,
            "num_batches": 64,
            "feature_dim": 512,
            "critic_dim": 512,
            "safety_dim": 512,
        },
    ],
)
def test_config_finite_caps_fail_closed(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        KondoSparseActorDevelopmentConfig(**values)


def test_source_and_runtime_manifests_are_complete_and_stable() -> None:
    source = kondo_sparse_actor_development_source_manifest()
    runtime = kondo_sparse_actor_development_runtime_identity()

    assert set(source) == {
        "alberta_framework/core/kondo_gate.py",
        "alberta_framework/core/kondo_sparse_actor.py",
        "alberta_framework/evaluation/kondo_sparse_actor_development.py",
    }
    assert all(len(digest) == 64 for digest in source.values())
    assert runtime["python"]
    assert runtime["python_implementation"]
    assert runtime["chex"]
    assert runtime["jax"]
    assert runtime["jaxlib"]
    assert runtime["numpy"]
    assert runtime["backend"]
    assert runtime["backend_platform_version"]
    assert cast(int, runtime["device_count"]) >= 1
    assert 1 <= cast(int, runtime["local_device_count"]) <= cast(
        int,
        runtime["device_count"],
    )
    assert len(cast(list[str], runtime["device_platforms"])) == runtime["device_count"]
    assert "exact-deterministic-replay-remains-authoritative" in cast(
        str,
        runtime["identity_scope"],
    )


def test_finite_float32_update_overflow_fails_closed_before_serialization() -> None:
    config = KondoSparseActorDevelopmentConfig(
        num_batches=1,
        timing_trials=1,
        learning_rate=3.0e38,
    )

    with pytest.raises(ValueError, match="produced nonfinite"):
        build_kondo_sparse_actor_deterministic_payload(config)


def test_one_immutable_trace_and_snapshot_bind_every_arm(
    config: KondoSparseActorDevelopmentConfig,
    deterministic: dict[str, object],
) -> None:
    trace = _list(deterministic["source_trace"])
    records = _list(deterministic["arm_records"])
    initial = _mapping(deterministic["initial_parameter_snapshot"])

    assert len(trace) == config.num_batches
    assert len(records) == config.num_batches * len(ARM_ORDER)
    assert set(initial) == {
        "hidden_weight",
        "hidden_bias",
        "output_weight",
        "output_bias",
    }
    for event_index, event_raw in enumerate(trace):
        event = _mapping(event_raw)
        matching = [
            _mapping(record)
            for record in records
            if _mapping(record)["event_index"] == event_index
        ]
        assert len(matching) == len(ARM_ORDER)
        assert {record["source_batch_sha256"] for record in matching} == {
            event["source_batch_sha256"]
        }
        assert {record["policy_revision_before"] for record in matching} == {
            event_index
        }
        assert {record["policy_revision_after"] for record in matching} == {
            event_index + 1
        }
        assert len({record["source_actions_sha256"] for record in matching}) == 1
        assert len({record["source_actor_features_sha256"] for record in matching}) == 1
        assert len({record["source_protected_inputs_sha256"] for record in matching}) == 1
        assert all(record["source_experience_replays_in_arm"] == 1 for record in matching)
        assert all(
            _decode_array(record["behavior_log_probability"]).shape
            == (config.batch_size,)
            for record in matching
        )


def test_capacity_shapes_selected_counts_and_budget_are_disclosed(
    config: KondoSparseActorDevelopmentConfig,
    deterministic: dict[str, object],
) -> None:
    accounting = _mapping(deterministic["logical_resource_accounting"])
    per_arm = _mapping(accounting["per_arm"])
    ordinary = _mapping(per_arm["ordinary_full"])
    uniform = _mapping(per_arm["uniform_sparse"])
    kondo = _mapping(per_arm["kondo_top_k"])
    overflow = _mapping(per_arm["kondo_overflow_diagnostic"])

    assert deterministic["schema"] == (
        "alberta.kondo-sparse-actor-development.deterministic.v2"
    )
    assert accounting["unique_source_batches"] == config.num_batches
    assert accounting["deterministic_training_trace_executions"] == 1
    assert accounting["source_trace_replays_per_arm"] == 1
    assert accounting["experience_double_counted_within_arm"] is False
    for arm in ARM_ORDER:
        arm_accounting = _mapping(per_arm[arm])
        assert arm_accounting["source_batches_consumed"] == config.num_batches
        assert arm_accounting["training_updates"] == config.num_batches
        assert arm_accounting["compiled_training_backward_invocations"] == config.num_batches
    assert ordinary["selected_samples"] == config.num_batches * config.batch_size
    assert ordinary["backward_row_slots"] == config.num_batches * config.batch_size
    assert uniform["selected_samples"] == config.num_batches * config.sparse_capacity
    assert uniform["backward_row_slots"] == config.num_batches * config.sparse_capacity
    assert kondo["selected_samples"] == config.num_batches * config.sparse_capacity
    assert kondo["backward_row_slots"] == config.num_batches * config.sparse_capacity
    overflow_selected = cast(int, overflow["selected_samples"])
    assert overflow_selected >= config.num_batches * (
        config.sparse_capacity + 1
    )
    assert overflow_selected <= config.num_batches * (
        2 * config.sparse_capacity + 1
    )
    assert overflow["backward_row_slots"] == config.num_batches * config.batch_size
    assert accounting["measured_flops"] is False
    records = [_mapping(item) for item in _list(deterministic["arm_records"])]
    assert all("sparks_joy" not in record for record in records)


def test_paper_delight_and_screen_gather_order_reconstruct_exactly(
    deterministic: dict[str, object],
) -> None:
    records = [_mapping(item) for item in _list(deterministic["arm_records"])]
    trace = {
        cast(int, _mapping(item)["event_index"]): _mapping(item)
        for item in _list(deterministic["source_trace"])
    }
    kondo_records = [item for item in records if item["arm"] == "kondo_top_k"]

    for record in kondo_records:
        behavior = _decode_array(record["behavior_log_probability"])
        delight = _decode_array(record["delight"])
        surprisal = _decode_array(record["selected_action_surprisal"])
        event = trace[cast(int, record["event_index"])]
        advantage = _decode_array(event["return_targets"]) - _decode_array(
            event["baseline_predictions"]
        )
        np.testing.assert_array_equal(surprisal, -behavior)
        np.testing.assert_allclose(delight, advantage * surprisal, rtol=1.0e-6, atol=1.0e-6)
        assert record["screen_gather_backward_order"] == [
            "full-forward",
            "detached-screen",
            "audited-gather",
            "compiled-backward",
        ]
        assert record["backward_leading_shape"] < behavior.shape[0]
        selected = cast(list[int], record["selected_indices"])
        assert len(selected) == record["selected_count"]
        assert np.all(np.isfinite(delight))


def test_uniform_control_is_without_replacement_and_capacity_matched(
    config: KondoSparseActorDevelopmentConfig,
    deterministic: dict[str, object],
) -> None:
    trace = [_mapping(item) for item in _list(deterministic["source_trace"])]
    records = [_mapping(item) for item in _list(deterministic["arm_records"])]
    uniform = [item for item in records if item["arm"] == "uniform_sparse"]
    kondo = [item for item in records if item["arm"] == "kondo_top_k"]

    for event, uniform_record, kondo_record in zip(trace, uniform, kondo, strict=True):
        indices = _decode_array(event["uniform_indices"]).tolist()
        assert len(indices) == config.sparse_capacity
        assert len(set(indices)) == config.sparse_capacity
        assert uniform_record["selected_indices"] == indices
        assert uniform_record["backward_leading_shape"] == config.sparse_capacity
        assert kondo_record["backward_leading_shape"] == config.sparse_capacity


def test_overflow_is_explicitly_diagnostic_and_preserves_forced_rows(
    config: KondoSparseActorDevelopmentConfig,
    deterministic: dict[str, object],
) -> None:
    records = [_mapping(item) for item in _list(deterministic["arm_records"])]
    overflow = [item for item in records if item["arm"] == "kondo_overflow_diagnostic"]

    for record in overflow:
        forced = list(range(config.sparse_capacity + 1))
        assert record["diagnostic_only"] is True
        assert record["forced_indices"] == forced
        assert set(forced).issubset(set(cast(list[int], record["selected_indices"])))
        assert record["full_shape_masked_fallback"] is True
        assert record["sparse_backward"] is False
        assert record["backward_leading_shape"] == config.batch_size


def test_heldout_metrics_are_descriptive_and_do_not_update(
    deterministic: dict[str, object],
) -> None:
    diagnostics = _mapping(deterministic["heldout_diagnostics"])

    assert set(diagnostics) == set(ARM_ORDER)
    for arm in ARM_ORDER:
        values = _mapping(diagnostics[arm])
        assert values["assessment_status"] == "not_assessed"
        assert values["heldout_backward_invocations"] == 1
        assert values["heldout_parameter_updates"] == 0
        assert np.isfinite(cast(float, values["heldout_actor_loss"]))
        assert np.isfinite(cast(float, values["heldout_gradient_l2"]))
        assert np.isfinite(cast(float, values["parameter_change_l2"]))
    assert deterministic["thresholds"] == []
    assert deterministic["verdict"] == "not_assessed"


def test_timing_is_interleaved_descriptive_and_never_asserts_speedup(
    config: KondoSparseActorDevelopmentConfig,
    report: dict[str, object],
) -> None:
    timing = _mapping(report["timing"])
    events = [_mapping(item) for item in _list(timing["events"])]
    summaries = _mapping(timing["summaries"])

    assert timing["schema"] == "alberta.kondo-sparse-actor-development.timing.v2"
    assert timing["real_perf_counter_ns"] is False
    assert timing["clock_name"] == "deterministic-test-clock"
    assert timing["timing_is_descriptive_and_noisy"] is True
    assert timing["thresholds"] == []
    assert timing["verdict"] == "not_assessed"
    assert timing["host_screen_gather_timed"] is False
    assert timing["accelerator_memory_measured"] is False
    assert timing["energy_measured"] is False
    assert len(events) == config.timing_trials * len(ARM_ORDER)
    for trial in range(config.timing_trials):
        trial_events = events[trial * len(ARM_ORDER) : (trial + 1) * len(ARM_ORDER)]
        assert {event["arm"] for event in trial_events} == set(ARM_ORDER)
    for arm in ARM_ORDER:
        summary = _mapping(summaries[arm])
        raw = cast(list[int], summary["raw_duration_ns"])
        assert len(raw) == config.timing_trials
        assert summary["compiled_warmup_invocations"] == 1
        assert summary["compiled_timed_invocations"] == config.timing_trials
        assert summary["compiled_backward_invocations_total"] == config.timing_trials + 1
        ordered = sorted(raw)
        assert summary["minimum_ns"] == ordered[0]
        assert summary["maximum_ns"] == ordered[-1]
        assert summary["p50_ns"] == ordered[math.ceil(0.50 * len(ordered)) - 1]
        assert summary["p95_ns"] == ordered[math.ceil(0.95 * len(ordered)) - 1]
    assert "speedup" not in str(timing).lower()


def test_timing_measurement_rejects_a_clock_that_moves_backwards(
    config: KondoSparseActorDevelopmentConfig,
) -> None:
    with pytest.raises(ValueError, match="globally monotonic"):
        measure_kondo_sparse_actor_backward_timing(
            config,
            clock_ns=_BackwardClock(),
            clock_name="backward-test-clock",
        )


def test_timing_shape_order_and_eager_compiled_parity_are_audited(
    config: KondoSparseActorDevelopmentConfig,
    report: dict[str, object],
) -> None:
    timing = _mapping(report["timing"])
    contracts = _mapping(timing["shape_and_order_contracts"])
    parity = _mapping(timing["eager_compiled_parity"])

    assert contracts["ordinary_full_leading_shape"] == config.batch_size
    assert contracts["uniform_sparse_leading_shape"] == config.sparse_capacity
    assert contracts["kondo_top_k_leading_shape"] == config.sparse_capacity
    assert contracts["kondo_overflow_diagnostic_leading_shape"] == config.batch_size
    assert contracts["uniform_and_kondo_capacity_equal"] is True
    assert contracts["kondo_screen_detached_before_gather"] is True
    assert contracts["kondo_gather_before_compiled_backward"] is True
    assert contracts["overflow_preserves_every_forced_sample"] is True
    for arm in ARM_ORDER:
        values = _mapping(parity[arm])
        assert values["eager_compiled_numerical_parity"] is True
        assert values["eager_compiled_max_abs_delta"] <= 1.0e-6  # type: ignore[operator]


def test_report_validator_replays_only_deterministic_bytes(
    report: dict[str, object],
) -> None:
    receipt = validate_kondo_sparse_actor_development_report(report)

    assert report["schema"] == "alberta.kondo-sparse-actor-development.report.v2"
    assert receipt.valid
    assert receipt.assessment_status == "not_assessed"
    assert receipt.deterministic_replay_checked
    assert receipt.deterministic_replay_exact
    assert receipt.timing_structure_checked
    assert receipt.timing_provenance_bound
    assert not receipt.wall_clock_replayed
    assert report["deterministic_replay_includes_wall_clock"] is False
    assert report["timing_separately_provenance_bound"] is True


def test_report_builder_executes_deterministic_training_trace_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = KondoSparseActorDevelopmentConfig(num_batches=1, timing_trials=1)
    calls = 0
    original = development_module.build_kondo_sparse_actor_deterministic_payload

    def counted(
        selected_config: KondoSparseActorDevelopmentConfig | None = None,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(selected_config)

    monkeypatch.setattr(
        development_module,
        "build_kondo_sparse_actor_deterministic_payload",
        counted,
    )

    built = development_module.build_kondo_sparse_actor_development_report(
        config,
        clock_ns=_DeterministicClock(),
        clock_name="single-run-test-clock",
    )

    assert calls == 1
    assert _mapping(built["deterministic"])["assessment_status"] == "not_assessed"


def test_every_status_and_claim_field_is_fail_closed(report: dict[str, object]) -> None:
    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for name, child in value.items():
                if name.endswith("status"):
                    assert child == "not_assessed"
                if name in {
                    "performance_claimed",
                    "compute_saving_claimed",
                    "efficacy_claimed",
                    "safety_claimed",
                    "policy_authority",
                    "promotion_authority",
                    "scientific_promotion_allowed",
                    "output_writes",
                }:
                    assert child is False
                visit(child)
        elif type(value) is list:
            for child in cast(list[object], value):
                visit(child)

    visit(report)
