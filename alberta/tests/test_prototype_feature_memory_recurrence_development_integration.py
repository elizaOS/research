# mypy: disable-error-code="arg-type,attr-defined,index,no-any-return"
"""Short strict report smoke for the Prototype feature-memory recurrence lane."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.experiential_memory_policy import (
    ExperientialMemoryAdvantageGateConfig,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.prototype_agent import PrototypeAgent, PrototypeAgentState
from alberta_framework.evaluation.prototype_feature_memory_recurrence_development import (
    _CROSS_ENGINE_HORDE_FLOAT_MAX_ABS_TOLERANCE,
    _LIFECYCLE_TAG,
    ACCEPTANCE_STATUS,
    INTERPRETATION,
    LIMITATIONS,
    PROTOTYPE_FEATURE_MEMORY_RECURRENCE_REPORT_SCHEMA,
    PrototypeFeatureMemoryRecurrenceProtocol,
    _agent_config,
    _compiled_event,
    _compiled_initialize_arm,
    _horde_spec,
    _tree_bit_exact,
    prototype_feature_memory_recurrence_report_json,
    run_compiled_prototype_feature_memory_recurrence_development,
    run_prototype_feature_memory_recurrence_development,
    validate_prototype_feature_memory_recurrence_report,
)
from alberta_framework.streams.recurring_multiagent import RecurringTwoAgentWorld

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _short_protocol(
    *,
    arm_names: tuple[str, ...] = (
        "full",
        "memory_readout_blocked",
        "cue_masked_counterexample",
    ),
    memory_capacity: int = 2,
    segment_length: int = 1,
) -> PrototypeFeatureMemoryRecurrenceProtocol:
    return PrototypeFeatureMemoryRecurrenceProtocol(
        segment_length=segment_length,
        active_pair_slots=2,
        memory_capacity=memory_capacity,
        replacement_interval=1,
        metric_window=1,
        arm_names=arm_names,
    )


@pytest.fixture(scope="module")
def short_report() -> dict[str, object]:
    """Run three one-step phases through one shared static agent configuration."""

    return run_prototype_feature_memory_recurrence_development(
        _short_protocol(),
        seed=7,
    )


@pytest.fixture(scope="module")
def compiled_short_report() -> dict[str, object]:
    """Run the identical dynamic readout/cue arms through the scan path."""

    return run_compiled_prototype_feature_memory_recurrence_development(
        _short_protocol(),
        seed=7,
    )


@pytest.fixture(scope="module")
def gate_protocol() -> PrototypeFeatureMemoryRecurrenceProtocol:
    return _short_protocol(
        arm_names=(
            "conservative_outcome_gate",
            "conservative_outcome_gate_cue_masked",
        ),
        memory_capacity=4,
        segment_length=2,
    )


@pytest.fixture(scope="module")
def gate_eager_report(
    gate_protocol: PrototypeFeatureMemoryRecurrenceProtocol,
) -> dict[str, object]:
    return run_prototype_feature_memory_recurrence_development(
        gate_protocol,
        seed=13,
    )


@pytest.fixture(scope="module")
def gate_compiled_report(
    gate_protocol: PrototypeFeatureMemoryRecurrenceProtocol,
) -> dict[str, object]:
    return run_compiled_prototype_feature_memory_recurrence_development(
        gate_protocol,
        seed=13,
    )


@pytest.fixture(scope="module")
def blocked_feature_protocol() -> PrototypeFeatureMemoryRecurrenceProtocol:
    return _short_protocol(
        arm_names=("feature_promotion_blocked",),
        memory_capacity=1,
    )


@pytest.fixture(scope="module")
def blocked_feature_eager_report(
    blocked_feature_protocol: PrototypeFeatureMemoryRecurrenceProtocol,
) -> dict[str, object]:
    """Run the structurally disabled feature configuration eagerly."""

    return run_prototype_feature_memory_recurrence_development(
        blocked_feature_protocol,
        seed=11,
    )


@pytest.fixture(scope="module")
def blocked_feature_compiled_report(
    blocked_feature_protocol: PrototypeFeatureMemoryRecurrenceProtocol,
) -> dict[str, object]:
    """Exercise the second static feature configuration beyond memory capacity."""

    return run_compiled_prototype_feature_memory_recurrence_development(
        blocked_feature_protocol,
        seed=11,
    )


def _runs(report: dict[str, object]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], report["runs"])


def _numeric_leaf_differences(left: object, right: object) -> list[float]:
    """Collect exact absolute differences from two same-shaped numeric trees."""

    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right)
        return [
            difference
            for name in left
            for difference in _numeric_leaf_differences(left[name], right[name])
        ]
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        return [
            difference
            for left_item, right_item in zip(left, right, strict=True)
            for difference in _numeric_leaf_differences(left_item, right_item)
        ]
    assert type(left) is float and type(right) is float
    return [abs(left - right)]


def test_compiled_semantics_are_bit_exact_to_eager_for_dynamic_readout_and_cue_arms(
    short_report: dict[str, object],
    compiled_short_report: dict[str, object],
) -> None:
    eager_execution = cast(dict[str, object], short_report["execution"])
    compiled_execution = cast(dict[str, object], compiled_short_report["execution"])
    assert eager_execution["engine"] == "python-eager-reference"
    assert compiled_execution["engine"] == "jax-jit-scan"
    assert eager_execution["module_source_sha256"] == compiled_execution[
        "module_source_sha256"
    ]
    assert eager_execution["engine_source_sha256"] != compiled_execution[
        "engine_source_sha256"
    ]
    assert compiled_short_report["runs"] == short_report["runs"]
    eager_semantics = {
        key: value
        for key, value in short_report.items()
        if key not in {"execution", "report_sha256"}
    }
    compiled_semantics = {
        key: value
        for key, value in compiled_short_report.items()
        if key not in {"execution", "report_sha256"}
    }
    assert compiled_semantics == eager_semantics


def test_exact_default_advantage_gate_pair_has_eager_compiled_authority_parity(
    gate_eager_report: dict[str, object],
    gate_compiled_report: dict[str, object],
) -> None:
    eager_execution = cast(dict[str, object], gate_eager_report["execution"])
    compiled_execution = cast(dict[str, object], gate_compiled_report["execution"])
    assert eager_execution["engine"] == "python-eager-reference"
    assert compiled_execution["engine"] == "jax-jit-scan"
    assert validate_prototype_feature_memory_recurrence_report(
        gate_eager_report
    ).valid
    assert validate_prototype_feature_memory_recurrence_report(
        gate_compiled_report
    ).valid

    comparison = cast(dict[str, object], gate_eager_report["comparison_contract"])
    assert comparison["arm_order"] == [
        "conservative_outcome_gate",
        "conservative_outcome_gate_cue_masked",
    ]
    assert comparison["advantage_gate_additional_work_declared"] is True
    assert comparison["conservative_outcome_gate_pair_work_matched"] is True
    assert comparison[
        "conservative_outcome_gate_pair_persistent_state_shape_matched"
    ] is True

    expected_gate_config = ExperientialMemoryAdvantageGateConfig().to_config()
    runs = _runs(gate_eager_report)
    compiled_runs = _runs(gate_compiled_report)
    trace_horde_differences: list[float] = []
    metric_horde_differences: list[float] = []
    semantic_digest_differences: set[tuple[int, str]] = set()
    allowed_semantic_digest_differences = {
        "semantic_state.final_joint_state_sha256",
        "semantic_state.phase_boundaries[3].agent_state_sha256",
        "semantic_state.phase_boundaries[3].environment_state_sha256",
        "semantic_state.phase_boundaries[3].joint_state_sha256",
    }
    for run_index, (eager_run, compiled_run) in enumerate(
        zip(runs, compiled_runs, strict=True)
    ):
        assert compiled_run["agent_config"] == eager_run["agent_config"]
        assert compiled_run["resources"] == eager_run["resources"]
        assert compiled_run["work"] == eager_run["work"]
        assert compiled_run["identity_audit"] == eager_run["identity_audit"]
        assert compiled_run["metrics"]["memory"] == eager_run["metrics"]["memory"]
        eager_projection = copy.deepcopy(eager_run)
        compiled_projection = copy.deepcopy(compiled_run)
        for eager_event, compiled_event in zip(
            eager_run["trace"], compiled_run["trace"], strict=True
        ):
            assert compiled_event["action"] == eager_event["action"]
            assert compiled_event["counterfactual_base_action"] == eager_event[
                "counterfactual_base_action"
            ]
            assert compiled_event["reward"] == eager_event["reward"]
            assert compiled_event["counterfactual_reward"] == eager_event[
                "counterfactual_reward"
            ]
            assert compiled_event["counterfactual_reward_delta"] == eager_event[
                "counterfactual_reward_delta"
            ]
            assert compiled_event["memory_action_changed"] == eager_event[
                "memory_action_changed"
            ]
            gate_fields = [
                name
                for name in eager_event
                if name.startswith("memory_advantage_gate_")
            ]
            assert {
                name: compiled_event[name] for name in gate_fields
            } == {name: eager_event[name] for name in gate_fields}
            for field in (
                "horde_prediction",
                "horde_cumulant",
                "horde_squared_error",
            ):
                trace_horde_differences.extend(
                    _numeric_leaf_differences(
                        eager_event[field], compiled_event[field]
                    )
                )

        eager_metrics = eager_run["metrics"]
        compiled_metrics = compiled_run["metrics"]
        metric_horde_differences.extend(
            _numeric_leaf_differences(
                eager_metrics["phase_horde_mse"],
                compiled_metrics["phase_horde_mse"],
            )
        )
        for name in (
            "a2_entry_minus_a1_tail_horde_mse",
            "a2_horde_reacquisition_gain",
        ):
            metric_horde_differences.extend(
                _numeric_leaf_differences(
                    eager_metrics["recurrence"][name],
                    compiled_metrics["recurrence"][name],
                )
            )

        eager_semantic = eager_run["semantic_state"]
        compiled_semantic = compiled_run["semantic_state"]
        if eager_semantic["final_joint_state_sha256"] != compiled_semantic[
            "final_joint_state_sha256"
        ]:
            semantic_digest_differences.add(
                (run_index, "semantic_state.final_joint_state_sha256")
            )
        for boundary_index, (eager_boundary, compiled_boundary) in enumerate(
            zip(
                eager_semantic["phase_boundaries"],
                compiled_semantic["phase_boundaries"],
                strict=True,
            )
        ):
            for name in (
                "agent_state_sha256",
                "environment_state_sha256",
                "joint_state_sha256",
            ):
                if eager_boundary[name] != compiled_boundary[name]:
                    semantic_digest_differences.add(
                        (
                            run_index,
                            f"semantic_state.phase_boundaries[{boundary_index}].{name}",
                        )
                    )

        for projection in (eager_projection, compiled_projection):
            projection["trace_sha256"] = "cross-engine Horde floats checked separately"
            for event in projection["trace"]:
                for name in (
                    "horde_prediction",
                    "horde_cumulant",
                    "horde_squared_error",
                ):
                    event[name] = "cross-engine max-abs tolerance checked"
            projection["metrics"]["phase_horde_mse"] = (
                "cross-engine max-abs tolerance checked"
            )
            for name in (
                "a2_entry_minus_a1_tail_horde_mse",
                "a2_horde_reacquisition_gain",
            ):
                projection["metrics"]["recurrence"][name] = (
                    "cross-engine max-abs tolerance checked"
                )
            projection["semantic_state"]["final_joint_state_sha256"] = (
                "cross-engine digest equality not claimed"
            )
            final_boundary = projection["semantic_state"]["phase_boundaries"][3]
            for name in (
                "agent_state_sha256",
                "environment_state_sha256",
                "joint_state_sha256",
            ):
                final_boundary[name] = "cross-engine digest equality not claimed"
        assert compiled_projection == eager_projection

    observed_trace_horde_max_abs = max(trace_horde_differences)
    observed_metric_horde_max_abs = max(metric_horde_differences)
    assert observed_trace_horde_max_abs == pytest.approx(
        1.4901161193847656e-08,
        rel=0.0,
        abs=1.0e-15,
    )
    assert observed_metric_horde_max_abs == pytest.approx(
        5.587935447692871e-09,
        rel=0.0,
        abs=1.0e-15,
    )
    assert (
        observed_trace_horde_max_abs
        <= _CROSS_ENGINE_HORDE_FLOAT_MAX_ABS_TOLERANCE
    )
    assert (
        observed_metric_horde_max_abs
        <= _CROSS_ENGINE_HORDE_FLOAT_MAX_ABS_TOLERANCE
    )
    assert semantic_digest_differences == {
        (run_index, name)
        for run_index in range(2)
        for name in allowed_semantic_digest_differences
    }
    for execution in (eager_execution, compiled_execution):
        assert execution["focused_cross_engine_horde_float_max_abs_tolerance"] == (
            _CROSS_ENGINE_HORDE_FLOAT_MAX_ABS_TOLERANCE
        )
        assert execution["cross_engine_full_state_float_tolerance_claimed"] is False
        assert execution[
            "cross_engine_semantic_state_digest_equality_claimed"
        ] is False
    assert runs[0]["work"] == runs[1]["work"]
    assert runs[0]["resources"]["initial_state"]["total_nbytes"] == runs[1][
        "resources"
    ]["initial_state"]["total_nbytes"]
    for run in runs:
        assert run["agent_config"]["experiential_memory_advantage_gate"] == (
            expected_gate_config
        )
        gate_resource = run["resources"]["experiential_memory_advantage_gate"]
        assert gate_resource == {
            "configured": True,
            "config": expected_gate_config,
            "resources": {
                "n_actions": 2,
                "top_k": 4,
                "neighbor_action_values_interpreted": 8,
                "neighbor_reward_values_interpreted": 4,
                "neighbor_weight_values_interpreted": 4,
                "owned_persistent_state_bytes": 0,
                "random_draws_per_assessment": 0,
            },
        }
        work = run["work"]
        assert work["memory_advantage_gate_preview_assessments"] == 6
        assert work["memory_advantage_gate_committed_assessments"] == 6
        assert work["memory_advantage_gate_stale_replay_assessments"] == 1
        assert work["memory_advantage_gate_total_assessments"] == 13
        assert work["memory_advantage_gate_reported_event_assessments"] == 6
        assert work[
            "memory_advantage_gate_neighbor_action_values_interpreted"
        ] == 104
        assert work[
            "memory_advantage_gate_neighbor_reward_values_interpreted"
        ] == 52
        assert work[
            "memory_advantage_gate_neighbor_weight_values_interpreted"
        ] == 52
        assert work["memory_advantage_gate_random_draws"] == 0

        summary = run["metrics"]["memory"]["advantage_gate"]
        assert summary["configured"] is True
        assert summary["assessments_reported"] == 6
        assert (
            summary["replacement_allowed_events"] + summary["abstained_events"]
            == 6
        )
        assert sum(summary["abstention_reasons"].values()) == summary[
            "abstained_events"
        ]
        assert summary["abstention_reasons"]["unclassified"] == 0
        assert all(
            phase_summary["assessments_reported"] == 2
            for phase_summary in summary["phase"].values()
        )
        assert sum(
            event["memory_advantage_gate_replacement_allowed"]
            for event in run["trace"]
        ) == summary["replacement_allowed_events"]
        for event in run["trace"]:
            assert event["memory_advantage_gate_configured"] is True
            assert event["memory_advantage_gate_dispatch_consistent"] is True
            assert event["memory_action_changed"] is event[
                "memory_advantage_gate_replacement_allowed"
            ]


def test_compiled_blocked_feature_config_scans_past_memory_capacity(
    blocked_feature_eager_report: dict[str, object],
    blocked_feature_compiled_report: dict[str, object],
) -> None:
    report = blocked_feature_compiled_report
    validation = validate_prototype_feature_memory_recurrence_report(report)

    assert validation.valid, validation.errors
    run = _runs(report)[0]
    assert run == _runs(blocked_feature_eager_report)[0]
    assert len(run["trace"]) == 3
    assert run["agent_config"]["prototype_feature_lifecycle"][
        "replacement_interval"
    ] == 0
    assert run["resources"]["feature_memory"]["capacity_entries"] == 1
    assert run["work"]["memory_writes"] == 3
    assert run["work"]["memory_deterministic_prestate_queries"] == 6
    assert run["metrics"]["features"]["curation_commits"] == 0
    assert run["metrics"]["features"]["memory_rebinds"] == 0
    assert run["metrics"]["features"]["rows_reencoded"] == 0
    assert run["identity_audit"] == {
        "aba_replay_attempted": True,
        "agent_clock_unchanged": True,
        "environment_unchanged": True,
        "replay_update_calls": 1,
        "stale_decision_rejected": True,
        "state_bit_exact": True,
    }
    for index, event in enumerate(run["trace"]):
        assert event["environment_pre_words"] == [0, index]
        assert event["environment_post_words"] == [0, index + 1]
        assert event["prototype_pre_step_words"] == [0, index]
        assert event["prototype_post_step_words"] == [0, index + 1]
        assert event["feature_generation_pre_words"] == [0, 0]
        assert event["feature_generation_post_words"] == [0, 0]
        assert event["curation_committed"] is False
        assert event["feature_memory_rebind_applied"] is False
        assert event["transition_valid"] is True


def test_short_report_is_strict_nonpromoting_and_canonical(
    short_report: dict[str, object],
) -> None:
    validation = validate_prototype_feature_memory_recurrence_report(short_report)

    assert validation.valid, validation.errors
    assert short_report["schema_version"] == PROTOTYPE_FEATURE_MEMORY_RECURRENCE_REPORT_SCHEMA
    assert short_report["development_only"] is True
    assert short_report["scientific_promotion_allowed"] is False
    assert short_report["accepted_scientific_evidence"] is False
    assert short_report["acceptance_status"] == ACCEPTANCE_STATUS
    assert short_report["interpretation"] == INTERPRETATION
    assert short_report["limitations"] == list(LIMITATIONS)
    encoded = prototype_feature_memory_recurrence_report_json(short_report)
    assert json.loads(encoded) == short_report
    assert encoded == prototype_feature_memory_recurrence_report_json(short_report)


def test_every_arm_discards_preview_and_has_exact_matched_work_and_resources(
    short_report: dict[str, object],
) -> None:
    runs = _runs(short_report)
    comparison = cast(dict[str, object], short_report["comparison_contract"])

    assert comparison == {
        "arm_order": [
            "full",
            "memory_readout_blocked",
            "cue_masked_counterexample",
        ],
        "paired_seed": True,
        "persistent_state_shape_matched": True,
        "preview_and_transaction_work_matched": True,
        "advantage_gate_additional_work_declared": True,
        "conservative_outcome_gate_pair_work_matched": True,
        "conservative_outcome_gate_pair_persistent_state_shape_matched": True,
        "preview_state_carried": False,
        "realized_compute_or_allocator_parity_claimed": False,
        "rejected_event_short_circuit_work_parity_claimed": False,
    }
    initial_sizes: list[int] = []
    for run in runs:
        work = run["work"]
        assert work["requested_transitions"] == 3
        assert work["committed_environment_transitions"] == 3
        assert work["counterfactual_environment_calls"] == 3
        assert work["prototype_update_calls"] == 7
        assert work["discarded_preview_update_calls"] == 3
        assert work["committed_prototype_update_calls"] == 3
        assert work["identity_probe_update_calls"] == 1
        assert work["oak_update_calls"] == 6
        assert work["oak_discarded_preview_updates"] == 3
        assert work["oak_committed_updates"] == 3
        assert work["world_model_update_calls"] == 6
        assert work["world_model_discarded_preview_updates"] == 3
        assert work["world_model_committed_updates"] == 3
        assert work["horde_update_calls"] == 6
        assert work["horde_discarded_preview_updates"] == 3
        assert work["horde_committed_updates"] == 3
        assert work["memory_sidecars_supplied"] == 4
        assert work["memory_real_transition_sidecars_supplied"] == 3
        assert work["memory_stale_replay_sidecars_supplied"] == 1
        assert work["memory_deterministic_prestate_queries"] == 6
        assert work["memory_preview_prestate_query_calls"] == 3
        assert work["memory_stale_replay_prestate_query_calls"] == 1
        assert work["memory_total_prestate_query_calls"] == 10
        assert work["memory_writes"] == 3
        assert work["memory_advantage_gate_preview_assessments"] == 0
        assert work["memory_advantage_gate_committed_assessments"] == 0
        assert work["memory_advantage_gate_stale_replay_assessments"] == 0
        assert work["memory_advantage_gate_total_assessments"] == 0
        assert work["memory_advantage_gate_reported_event_assessments"] == 0
        assert work[
            "memory_advantage_gate_neighbor_action_values_interpreted"
        ] == 0
        assert work[
            "memory_advantage_gate_neighbor_reward_values_interpreted"
        ] == 0
        assert work[
            "memory_advantage_gate_neighbor_weight_values_interpreted"
        ] == 0
        assert work["memory_advantage_gate_random_draws"] == 0
        assert all(event["preview_state_discarded"] for event in run["trace"])
        assert "experiential_memory_advantage_gate" not in run["agent_config"]
        assert run["metrics"]["memory"]["advantage_gate"]["configured"] is False
        assert run["metrics"]["memory"]["advantage_gate"][
            "assessments_reported"
        ] == 0
        assert all(
            event["memory_advantage_gate_configured"] is False
            and event["memory_advantage_gate_dispatch_consistent"] is True
            for event in run["trace"]
        )
        assert all(
            event["world_model_prediction_error"] >= 0.0
            for event in run["trace"]
        )
        assert set(run["metrics"]["phase_world_model_prediction_error"]) == {
            "A1",
            "B",
            "A2",
        }
        resources = run["resources"]
        assert resources["experiential_memory_advantage_gate"] == {
            "configured": False,
            "config": None,
            "resources": None,
        }
        assert resources["initial_state"] == resources["final_state"]
        total = resources["initial_state"]["total_nbytes"]
        assert resources["phase_boundary_total_nbytes"] == [total] * 4
        assert resources["peak_total_nbytes"] == total
        world_model = resources["stable_base_world_model"]
        assert world_model["coordinates"] == "stable_base_only"
        assert world_model["generated_pair_tail_modeled"] is False
        assert world_model["observation_dim"] == 8
        assert world_model["buffer_capacity"] == 1
        assert world_model["world_model_bundle_nbytes"] == resources[
            "initial_state"
        ]["world_model_bundle_nbytes"]
        assert world_model["buffer_nbytes"] == 40
        initial_sizes.append(total)
    assert len(set(initial_sizes)) == 1


def test_exact_event_clocks_decisions_and_aba_replay_fail_closed(
    short_report: dict[str, object],
) -> None:
    for run in _runs(short_report):
        lifecycle = run["lifecycle_id"]
        for index, event in enumerate(run["trace"]):
            assert event["environment_pre_words"] == [0, index]
            assert event["environment_post_words"] == [0, index + 1]
            assert event["prototype_pre_step_words"] == [0, index]
            assert event["prototype_post_step_words"] == [0, index + 1]
            assert event["prototype_decision_id"] == [*lifecycle, 0, index]
            assert event["transition_valid"] is True
        assert run["identity_audit"] == {
            "aba_replay_attempted": True,
            "agent_clock_unchanged": True,
            "environment_unchanged": True,
            "replay_update_calls": 1,
            "stale_decision_rejected": True,
            "state_bit_exact": True,
        }


def test_full_semantic_state_digests_cover_every_phase_boundary(
    short_report: dict[str, object],
) -> None:
    for run in _runs(short_report):
        audit = cast(dict[str, Any], run["semantic_state"])
        assert audit["normalization"] == (
            "documented non-learning birth_timestamp and uptime_s leaves are "
            "zero-normalized"
        )
        boundaries = cast(list[dict[str, Any]], audit["phase_boundaries"])
        assert [boundary["label"] for boundary in boundaries] == [
            "initial",
            "after_A1",
            "after_B",
            "after_A2",
        ]
        assert [boundary["event_count"] for boundary in boundaries] == [0, 1, 2, 3]
        assert audit["final_joint_state_sha256"] == boundaries[-1][
            "joint_state_sha256"
        ]
        for boundary in boundaries:
            assert len(boundary["agent_state_sha256"]) == 64
            assert len(boundary["environment_state_sha256"]) == 64
            assert len(boundary["joint_state_sha256"]) == 64
            assert boundary["counterfactual_base_action"] in (0, 1)


def test_compiled_invalid_event_rolls_back_carry_and_stops_later_events() -> None:
    protocol = _short_protocol(arm_names=("full",), memory_capacity=1)
    world = RecurringTwoAgentWorld(
        context_length=protocol.segment_length,
        nuisance_dim=protocol.nuisance_dim,
        nuisance_scale=protocol.nuisance_scale,
    )
    horde = HordeLearner(_horde_spec(), hidden_sizes=(), step_size=0.05)
    agent = PrototypeAgent(_agent_config(protocol, feature_promotion_enabled=True))
    carry = _compiled_initialize_arm(
        agent,
        world,
        jr.key(19),
        jr.key(19 ^ 0x13579BDF),
        jnp.asarray((_LIFECYCLE_TAG, 1), dtype=jnp.uint32),
        jnp.asarray(True, dtype=jnp.bool_),
    )
    invalid_agent_state = cast(
        PrototypeAgentState,
        carry.agent_state.replace(current_action=jnp.asarray(2, dtype=jnp.int32)),
    )
    invalid_carry = carry._replace(agent_state=invalid_agent_state)

    rejected_carry, failed_trace = _compiled_event(
        agent,
        world,
        horde,
        invalid_carry,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True, dtype=jnp.bool_),
        jnp.asarray(True, dtype=jnp.bool_),
    )
    expected_rejected = invalid_carry._replace(
        life_valid=jnp.asarray(False, dtype=jnp.bool_)
    )
    assert _tree_bit_exact(rejected_carry, expected_rejected)
    assert not bool(failed_trace.action_valid)
    assert not bool(failed_trace.event_committed)

    stopped_carry, stopped_trace = _compiled_event(
        agent,
        world,
        horde,
        rejected_carry,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(True, dtype=jnp.bool_),
        jnp.asarray(True, dtype=jnp.bool_),
    )
    assert _tree_bit_exact(stopped_carry, rejected_carry)
    assert not bool(stopped_trace.actual_environment_applied)
    assert not bool(stopped_trace.preview_valid)
    assert not bool(stopped_trace.committed_update_valid)
    assert not bool(stopped_trace.event_committed)


def test_readout_blocked_arm_keeps_preview_action_while_memory_still_writes(
    short_report: dict[str, object],
) -> None:
    blocked = next(
        run for run in _runs(short_report) if run["arm"] == "memory_readout_blocked"
    )

    assert blocked["metrics"]["memory"]["query_before_write_events"] == 3
    assert blocked["metrics"]["memory"]["writes"] == 3
    assert blocked["metrics"]["memory"]["action_changes"] == 0
    assert all(not event["memory_action_changed"] for event in blocked["trace"])


def test_full_arm_supplies_analytic_uncertainty_safety_and_similarity_gates(
    short_report: dict[str, object],
) -> None:
    full = next(run for run in _runs(short_report) if run["arm"] == "full")

    assert full["metrics"]["memory"]["query_before_write_events"] == 3
    assert full["metrics"]["memory"]["writes"] == 3
    assert full["metrics"]["memory"]["retrievals_available"] == 0
    assert all(not event["memory_retrieval_available"] for event in full["trace"])
    assert any(
        "zero uncertainty and safety cost" in limitation
        for limitation in cast(list[str], short_report["limitations"])
    )
    assert any(
        "similarity gate is fixed analytically" in limitation
        for limitation in cast(list[str], short_report["limitations"])
    )


def test_cue_masked_arm_is_explicitly_a_counterexample_not_a_hidden_task_claim(
    short_report: dict[str, object],
) -> None:
    definitions = cast(list[dict[str, object]], short_report["arm_definitions"])
    counterexample = next(
        definition
        for definition in definitions
        if definition["name"] == "cue_masked_counterexample"
    )

    assert counterexample["cue_visible"] is False
    assert "counterexample" in cast(str, counterexample["role"])
    limitations = cast(list[str], short_report["limitations"])
    assert any("task cue" in limitation for limitation in limitations)


def _tamper_trace_reward(report: dict[str, object]) -> None:
    _runs(report)[0]["trace"][0]["reward"] += 0.125


def _tamper_decision_identity(report: dict[str, object]) -> None:
    _runs(report)[0]["trace"][2]["prototype_decision_id"][3] = 0


def _tamper_work(report: dict[str, object]) -> None:
    _runs(report)[0]["work"]["discarded_preview_update_calls"] = 0


def _tamper_resources(report: dict[str, object]) -> None:
    _runs(report)[0]["resources"]["final_state"]["total_nbytes"] += 4


def _tamper_identity_audit(report: dict[str, object]) -> None:
    _runs(report)[0]["identity_audit"]["state_bit_exact"] = False


def _tamper_semantic_state(report: dict[str, object]) -> None:
    _runs(report)[0]["semantic_state"]["phase_boundaries"][0][
        "agent_state_sha256"
    ] = "00" * 32


def _tamper_execution_source(report: dict[str, object]) -> None:
    cast(dict[str, object], report["execution"])["module_source_sha256"] = "00" * 32


@pytest.mark.parametrize(
    "tamper",
    [
        _tamper_trace_reward,
        _tamper_decision_identity,
        _tamper_work,
        _tamper_resources,
        _tamper_identity_audit,
        _tamper_semantic_state,
        _tamper_execution_source,
    ],
)
def test_validator_rejects_trace_identity_work_resource_and_replay_tampering(
    short_report: dict[str, object],
    tamper: Callable[[dict[str, object]], None],
) -> None:
    corrupted = copy.deepcopy(short_report)
    tamper(corrupted)

    validation = validate_prototype_feature_memory_recurrence_report(corrupted)
    assert not validation.valid
    with pytest.raises(ValueError, match="invalid recurrence report"):
        prototype_feature_memory_recurrence_report_json(corrupted)


@pytest.mark.parametrize(
    "field, value",
    [
        ("development_only", False),
        ("scientific_promotion_allowed", True),
        ("accepted_scientific_evidence", True),
        ("acceptance_status", "accepted"),
        ("report_sha256", "00" * 32),
    ],
)
def test_validator_rejects_relabeling_and_digest_tampering(
    short_report: dict[str, object],
    field: str,
    value: object,
) -> None:
    corrupted = copy.deepcopy(short_report)
    corrupted[field] = value

    assert not validate_prototype_feature_memory_recurrence_report(corrupted).valid
