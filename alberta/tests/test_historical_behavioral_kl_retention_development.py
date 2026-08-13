"""Contracts for the pure-stdlib historical behavioral-KL L0 probe."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
import struct
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.evaluation import (
    historical_behavioral_kl_retention_development as historical_module,
)
from alberta_framework.evaluation.historical_behavioral_kl_retention_development import (
    ARM_NAMES,
    ARM_ROUTINGS,
    ARTIFACT_AUTHORITY,
    ASSESSMENT_STATUS,
    BENCHMARK_EXECUTION_AUTHORITY,
    CANDIDATE_COMPONENT_NAMES,
    CONFIG_SCHEMA,
    CPO_REPRODUCTION,
    DEVELOPMENT_ONLY,
    EVIDENCE_CLAIMED,
    GLOBAL_SHRINK_APPLIED,
    MRCL_REPRODUCTION,
    OUTPUT_WRITES_ALLOWED,
    REPORT_SCHEMA,
    RNG_USED,
    SCIENTIFIC_PROMOTION_ALLOWED,
    THRESHOLDS_FROZEN,
    VLM_GENERALIZATION_CLAIMED,
    ActorState,
    ArmRun,
    ContextualBanditEvent,
    HistoricalBehavioralKLRetentionConfig,
    HistoricalBehavioralKLRetentionReport,
    HistoricalBehavioralKLSource,
    MatchedArmAudit,
    ParetoCoordinate,
    PrefixStepTrace,
    ResourceSummary,
    RetainedRealStateAnchor,
    ScalingSummary,
    WorkSummary,
    build_historical_behavioral_kl_source,
    run_historical_behavioral_kl_retention_development,
    validate_historical_behavioral_kl_retention_report,
    validate_historical_behavioral_kl_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module")
def small_config() -> HistoricalBehavioralKLRetentionConfig:
    return HistoricalBehavioralKLRetentionConfig(
        a_prefix_steps=8,
        b_interference_steps=6,
        step_size=0.4,
        historical_kl_weight=2.0,
        current_kl_weight=2.0,
        movement_l2_weight=0.25,
        max_abs_parameter=20.0,
        max_report_bytes=1_000_000,
    )


@pytest.fixture(scope="module")
def small_report(
    small_config: HistoricalBehavioralKLRetentionConfig,
) -> HistoricalBehavioralKLRetentionReport:
    return run_historical_behavioral_kl_retention_development(small_config)


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _state_bits(state: ActorState) -> tuple[bytes, bytes, int]:
    return (
        _float_bits(state.parameters[0]),
        _float_bits(state.parameters[1]),
        state.accepted_updates,
    )


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("schema", "wrong", ValueError),
        ("schema", cast(Any, type("Text", (str,), {})(CONFIG_SCHEMA)), ValueError),
        ("a_prefix_steps", True, TypeError),
        ("a_prefix_steps", 0, ValueError),
        ("a_prefix_steps", 4_097, ValueError),
        ("b_interference_steps", False, TypeError),
        ("b_interference_steps", 0, ValueError),
        ("step_size", 1, TypeError),
        ("step_size", float("nan"), ValueError),
        ("step_size", float("inf"), ValueError),
        ("step_size", -0.0, ValueError),
        ("step_size", 1.1, ValueError),
        ("historical_kl_weight", 0.0, ValueError),
        ("current_kl_weight", True, TypeError),
        ("movement_l2_weight", 101.0, ValueError),
        ("max_abs_parameter", 0.5, ValueError),
        ("max_report_bytes", True, TypeError),
        ("max_report_bytes", 1_023, ValueError),
        ("max_report_bytes", 16_000_001, ValueError),
    ],
)
def test_config_rejects_aliases_nonfinite_values_and_cap_violations(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        HistoricalBehavioralKLRetentionConfig(**{field: value})  # type: ignore[arg-type]


def test_module_source_imports_only_python_standard_library() -> None:
    source_path = Path(inspect.getsourcefile(historical_module) or "")
    parsed = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "statistics",
        "struct",
        "pathlib",
        "typing",
    }


def test_source_is_exact_real_a_prefix_and_one_matched_b_stream(
    small_config: HistoricalBehavioralKLRetentionConfig,
) -> None:
    source = build_historical_behavioral_kl_source(small_config)
    replay = build_historical_behavioral_kl_source(small_config)
    assert validate_historical_behavioral_kl_source(source) == ()
    assert source.input_sha256 == replay.input_sha256
    assert source.generator_contract_sha256 == replay.generator_contract_sha256
    assert len(source.a_events) == small_config.a_prefix_steps
    assert len(source.b_events) == small_config.b_interference_steps
    for step, event in enumerate(source.a_events):
        assert type(event) is ContextualBanditEvent
        assert event.ordinal == step
        assert event.state == (1.0, 0.0)
        assert event.rewards_by_action == (0.0, 1.0)
    for step, event in enumerate(source.b_events):
        assert event.ordinal == step
        assert event.state == (1.0, 1.0)
        assert event.rewards_by_action == (1.0, 0.0)
    assert tuple(inspect.signature(historical_module._task_objective_gradient).parameters) == (
        "parameters",
        "state",
        "rewards",
    )


def test_source_validator_rejects_type_config_signed_zero_nonfinite_and_order_tamper(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    source = small_report.source
    assert validate_historical_behavioral_kl_source(
        cast(HistoricalBehavioralKLSource, object())
    ) == ("source type differs",)
    wrong_config = dataclasses.replace(source, config=cast(Any, object()))
    assert validate_historical_behavioral_kl_source(wrong_config) == (
        "source config type differs",
    )

    event = source.a_events[0]
    signed_event = dataclasses.replace(event, rewards_by_action=(-0.0, 1.0))
    assert signed_event.rewards_by_action == event.rewards_by_action
    assert _float_bits(signed_event.rewards_by_action[0]) != _float_bits(
        event.rewards_by_action[0]
    )
    signed_source = dataclasses.replace(
        source,
        a_events=(signed_event, *source.a_events[1:]),
    )
    signed_errors = validate_historical_behavioral_kl_source(signed_source)
    assert "source does not reconstruct bit-exactly" in signed_errors
    assert "source input digest does not bind source events" in signed_errors

    nonfinite_event = dataclasses.replace(event, state=(float("nan"), 0.0))
    nonfinite_source = dataclasses.replace(
        source,
        a_events=(nonfinite_event, *source.a_events[1:]),
    )
    assert "source contains non-finite values" in validate_historical_behavioral_kl_source(
        nonfinite_source
    )

    reversed_source = dataclasses.replace(source, b_events=tuple(reversed(source.b_events)))
    assert "source does not reconstruct bit-exactly" in (
        validate_historical_behavioral_kl_source(reversed_source)
    )


def test_report_is_explicitly_nonpromoting_nonreproducing_and_nonwriting(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    report = small_report
    assert validate_historical_behavioral_kl_retention_report(report) == ()
    assert REPORT_SCHEMA == "alberta.historical-behavioral-kl-retention.development.v1"
    assert DEVELOPMENT_ONLY is True
    assert ASSESSMENT_STATUS == "not_assessed"
    assert SCIENTIFIC_PROMOTION_ALLOWED is False
    assert BENCHMARK_EXECUTION_AUTHORITY is False
    assert ARTIFACT_AUTHORITY is False
    assert OUTPUT_WRITES_ALLOWED is False
    assert EVIDENCE_CLAIMED is False
    assert THRESHOLDS_FROZEN is False
    assert CPO_REPRODUCTION is False
    assert MRCL_REPRODUCTION is False
    assert VLM_GENERALIZATION_CLAIMED is False
    assert RNG_USED is False
    assert GLOBAL_SHRINK_APPLIED is False
    assert report.posthoc_recovery_updates == 0
    assert report.arm_names == ARM_NAMES
    assert report.candidate_component_names == CANDIDATE_COMPONENT_NAMES
    forbidden_summary_fields = {
        "winner",
        "default",
        "verdict",
        "threshold",
        "accepted",
        "frontier",
        "dominated",
    }
    for value_type in (
        HistoricalBehavioralKLRetentionReport,
        ParetoCoordinate,
    ):
        assert not forbidden_summary_fields & {
            field.name for field in dataclasses.fields(value_type)
        }


def test_a_anchor_is_bound_to_consumed_real_event_and_is_not_a_dream(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    anchor = small_report.retained_a_anchor
    event = small_report.source.a_events[-1]
    assert type(anchor) is RetainedRealStateAnchor
    assert anchor.kind == "retained_real_A_state_anchor"
    assert anchor.state == event.state
    assert anchor.source_event_ordinal == event.ordinal
    assert anchor.source_event_sha256 == historical_module._sha256(event)
    assert anchor.world_value_snapshot_used is False
    assert anchor.synthetic_world_grounding_used is False
    assert anchor.is_dream is False


def test_common_prefix_is_single_pass_and_every_arm_starts_bit_identically(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    report = small_report
    assert len(report.prefix_trace) == report.config.a_prefix_steps
    assert report.initial_state.accepted_updates == 0
    assert report.frozen_a_state.accepted_updates == report.config.a_prefix_steps
    for step, record in enumerate(report.prefix_trace):
        assert type(record) is PrefixStepTrace
        assert record.step == step
        assert record.update_applied is True
        assert record.parameter_address_mask == (True, True)
        if step == 0:
            assert record.parameters_pre == report.initial_state.parameters
        else:
            assert record.parameters_pre == report.prefix_trace[step - 1].parameters_post
        assert record.parameters_post == tuple(
            record.parameters_pre[index] + record.parameter_delta[index]
            for index in range(2)
        )
    assert report.prefix_trace[-1].parameters_post == report.frozen_a_state.parameters
    frozen_bits = _state_bits(report.frozen_a_state)
    for arm in report.arms:
        assert _state_bits(arm.initial_state) == frozen_bits
        assert arm.initial_state.accepted_updates == report.config.a_prefix_steps
        assert arm.final_state.accepted_updates == (
            report.config.a_prefix_steps + report.config.b_interference_steps
        )


def test_candidate_compute_and_b_event_stream_are_exactly_matched_across_arms(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    report = small_report
    assert tuple(arm.name for arm in report.arms) == ARM_NAMES
    assert tuple(arm.routing for arm in report.arms) == ARM_ROUTINGS
    event_streams = tuple(
        tuple(record.source_event_sha256 for record in arm.trace) for arm in report.arms
    )
    assert all(stream == event_streams[0] for stream in event_streams)
    first_candidates = tuple(
        (arm.trace[0].candidate_objectives, arm.trace[0].candidate_gradients)
        for arm in report.arms
    )
    assert all(candidate == first_candidates[0] for candidate in first_candidates)

    for arm, audit in zip(report.arms, report.matched_arm_audits, strict=True):
        assert type(audit) is MatchedArmAudit
        assert len(arm.trace) == report.config.b_interference_steps
        assert audit.arm_name == arm.name
        assert audit.routing == arm.routing
        assert audit.events_consumed == report.config.b_interference_steps
        assert audit.updates_attempted == report.config.b_interference_steps
        assert audit.updates_applied == report.config.b_interference_steps
        assert audit.candidate_objectives_per_event == len(CANDIDATE_COMPONENT_NAMES)
        assert audit.candidate_gradient_float64_scalars_per_event == 8
        assert audit.parameters_addressed_per_event == 2
        assert audit.posthoc_recovery_updates == 0
        assert audit.rng_draws == 0
        for step, record in enumerate(arm.trace):
            assert record.step == step
            assert len(record.candidate_objectives) == 4
            assert len(record.candidate_gradients) == 4
            assert all(len(gradient) == 2 for gradient in record.candidate_gradients)
            assert record.routing == arm.routing
            assert record.parameter_address_mask == (True, True)
            assert record.update_applied is True


def test_historical_and_current_state_kl_candidate_gradients_are_not_aliases(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    differences = []
    for arm in small_report.arms:
        for record in arm.trace:
            historical_gradient = record.candidate_gradients[1]
            current_gradient = record.candidate_gradients[2]
            differences.append(
                tuple(_float_bits(value) for value in historical_gradient)
                != tuple(_float_bits(value) for value in current_gradient)
            )
            assert historical_gradient[1] == 0.0
    assert any(differences)


def test_every_update_is_exactly_the_fixed_route_over_all_candidates(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    for arm in small_report.arms:
        for record in arm.trace:
            expected_objective, expected_gradient = historical_module._route_candidates(
                record.candidate_objectives,
                record.candidate_gradients,
                arm.routing,
            )
            assert _float_bits(record.routed_objective) == _float_bits(expected_objective)
            assert tuple(_float_bits(value) for value in record.routed_gradient) == tuple(
                _float_bits(value) for value in expected_gradient
            )
            expected_delta = tuple(
                -small_report.config.step_size * value
                for value in record.routed_gradient
            )
            assert tuple(_float_bits(value) for value in record.parameter_delta) == tuple(
                _float_bits(value) for value in expected_delta
            )


def test_raw_metrics_and_unclassified_pareto_coordinates_bind_to_trace(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    for arm, coordinate in zip(
        small_report.arms,
        small_report.pareto_coordinates,
        strict=True,
    ):
        first = arm.trace[0]
        last = arm.trace[-1]
        metrics = arm.metrics
        assert metrics.a_return_before_b == first.a_return_pre
        assert metrics.a_return_after_b == last.a_return_post
        assert metrics.a_return_delta == last.a_return_post - first.a_return_pre
        assert metrics.a_forgetting == first.a_return_pre - last.a_return_post
        assert metrics.a_margin_before_b == first.a_margin_pre
        assert metrics.a_margin_after_b == last.a_margin_post
        assert metrics.b_return_before_b == first.b_return_pre
        assert metrics.b_return_after_b == last.b_return_post
        assert metrics.b_plasticity_gain == last.b_return_post - first.b_return_pre
        assert metrics.b_margin_before_b == first.b_margin_pre
        assert metrics.b_margin_after_b == last.b_margin_post
        assert metrics.historical_a_kl_final == last.historical_a_kl_post
        assert metrics.current_b_kl_final == last.current_b_kl_post
        assert metrics.movement_l2_final == last.movement_l2_post
        assert coordinate.arm_name == arm.name
        assert coordinate.retained_a_return == metrics.a_return_after_b
        assert coordinate.a_forgetting == metrics.a_forgetting
        assert coordinate.b_return == metrics.b_return_after_b
        assert coordinate.b_plasticity_gain == metrics.b_plasticity_gain


def test_work_resource_scaling_and_hashes_are_exact_and_bounded(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    report = small_report
    b_arm_steps = len(ARM_NAMES) * report.config.b_interference_steps
    assert type(report.work) is WorkSummary
    assert report.work.prefix_task_objective_evaluations == report.config.a_prefix_steps
    assert report.work.b_task_objective_evaluations == b_arm_steps
    assert report.work.historical_kl_objective_evaluations == b_arm_steps
    assert report.work.current_kl_objective_evaluations == b_arm_steps
    assert report.work.movement_l2_objective_evaluations == b_arm_steps
    assert report.work.total_candidate_objective_evaluations == 4 * b_arm_steps
    assert report.work.total_candidate_gradient_float64_scalars == 8 * b_arm_steps
    assert report.work.prefix_parameter_updates == report.config.a_prefix_steps
    assert report.work.routed_parameter_updates == b_arm_steps
    assert report.work.total_parameter_updates == (
        report.config.a_prefix_steps + b_arm_steps
    )
    assert report.work.addressed_parameter_float64_scalars == 2 * (
        report.config.a_prefix_steps + b_arm_steps
    )
    assert report.work.frozen_policy_probability_evaluations == 2 * len(ARM_NAMES)
    assert report.work.rng_draws == 0
    assert report.work.global_shrink_evaluations == 0

    assert type(report.resource) is ResourceSummary
    assert report.resource.parameter_count == 2
    assert report.resource.actor_state_logical_nbytes == 24
    assert report.resource.retained_anchor_logical_nbytes == 16
    assert report.resource.total_trace_records == (
        report.config.a_prefix_steps + b_arm_steps
    )
    assert report.resource.canonical_source_nbytes == len(
        historical_module._canonical_bytes(report.source)
    )
    trace_payload = (report.prefix_trace, tuple(arm.trace for arm in report.arms))
    assert report.resource.canonical_trace_nbytes == len(
        historical_module._canonical_bytes(trace_payload)
    )
    assert len(historical_module._canonical_bytes(report)) <= report.config.max_report_bytes
    assert report.resource.report_cap_enforced is True

    assert type(report.scaling) is ScalingSummary
    assert report.scaling.actor_parameter_count == 2
    assert report.scaling.historical_anchor_count == 1
    assert report.scaling.historical_anchor_float64_scalars == 2
    assert report.scaling.candidate_gradient_scalars_per_b_event == 8
    assert report.scaling.empirical_scale_claimed is False
    hashes = {
        "implementation_source_sha256": report.implementation_source_sha256,
        "config_sha256": report.config_sha256,
        "source_sha256": report.source_sha256,
        "initial_state_sha256": report.initial_state_sha256,
        "frozen_a_state_sha256": report.frozen_a_state_sha256,
        "retained_anchor_sha256": report.retained_anchor_sha256,
        "arm_states_sha256": report.arm_states_sha256,
        "trace_sha256": report.trace_sha256,
        "work_sha256": report.work_sha256,
        "resource_sha256": report.resource_sha256,
        "scaling_sha256": report.scaling_sha256,
    }
    for digest in hashes.values():
        assert type(digest) is str
        assert len(digest) == 64
        int(digest, 16)


def test_validators_reconstruct_config_after_post_init_mutation(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    alias = HistoricalBehavioralKLRetentionConfig(a_prefix_steps=8, b_interference_steps=6)
    object.__setattr__(alias, "a_prefix_steps", True)
    alias_source = dataclasses.replace(small_report.source, config=alias)
    assert validate_historical_behavioral_kl_source(alias_source) == (
        "source config fields are not canonical",
    )
    alias_report = dataclasses.replace(small_report, config=alias)
    assert validate_historical_behavioral_kl_retention_report(alias_report) == (
        "report config fields are not canonical",
    )

    nonfinite = HistoricalBehavioralKLRetentionConfig(a_prefix_steps=8, b_interference_steps=6)
    object.__setattr__(nonfinite, "step_size", float("nan"))
    with pytest.raises(ValueError):
        run_historical_behavioral_kl_retention_development(nonfinite)


def test_report_validator_guards_nested_types_before_dereference(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    assert validate_historical_behavioral_kl_retention_report(
        cast(HistoricalBehavioralKLRetentionReport, object())
    ) == ("report type differs",)
    cases = (
        (
            dataclasses.replace(small_report, config=cast(Any, object())),
            "report config type differs",
        ),
        (
            dataclasses.replace(small_report, source=cast(Any, object())),
            "report source type differs",
        ),
        (
            dataclasses.replace(small_report, initial_state=cast(Any, object())),
            "initial state type differs",
        ),
        (
            dataclasses.replace(small_report, retained_a_anchor=cast(Any, object())),
            "retained A anchor type differs",
        ),
        (
            dataclasses.replace(small_report, prefix_trace=cast(Any, object())),
            "prefix trace types differ",
        ),
        (
            dataclasses.replace(small_report, arms=cast(Any, object())),
            "arm types or cardinality differ",
        ),
        (
            dataclasses.replace(small_report, work=cast(Any, object())),
            "work summary type differs",
        ),
    )
    for malformed, expected_error in cases:
        assert expected_error in validate_historical_behavioral_kl_retention_report(
            malformed
        )


def test_report_validator_rejects_signed_zero_nonfinite_route_hash_and_resource_tamper(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    arm = small_report.arms[0]
    record = arm.trace[0]
    historical_gradient = record.candidate_gradients[1]
    assert historical_gradient[0] == 0.0
    changed_gradients = (
        record.candidate_gradients[0],
        (-0.0, historical_gradient[1]),
        record.candidate_gradients[2],
        record.candidate_gradients[3],
    )
    signed_record = dataclasses.replace(record, candidate_gradients=changed_gradients)
    signed_arm = dataclasses.replace(arm, trace=(signed_record, *arm.trace[1:]))
    signed_report = dataclasses.replace(
        small_report,
        arms=(signed_arm, *small_report.arms[1:]),
    )
    signed_errors = validate_historical_behavioral_kl_retention_report(signed_report)
    assert "report does not reconstruct bit-exactly" in signed_errors
    assert "trace digest differs" in signed_errors

    nonfinite_record = dataclasses.replace(record, routed_objective=float("nan"))
    nonfinite_arm = dataclasses.replace(arm, trace=(nonfinite_record, *arm.trace[1:]))
    nonfinite_report = dataclasses.replace(
        small_report,
        arms=(nonfinite_arm, *small_report.arms[1:]),
    )
    assert "report contains non-finite values" in (
        validate_historical_behavioral_kl_retention_report(nonfinite_report)
    )

    changed_route_arm = dataclasses.replace(arm, routing=ARM_ROUTINGS[1])
    route_report = dataclasses.replace(
        small_report,
        arms=(changed_route_arm, *small_report.arms[1:]),
    )
    assert "report does not reconstruct bit-exactly" in (
        validate_historical_behavioral_kl_retention_report(route_report)
    )

    hash_errors = validate_historical_behavioral_kl_retention_report(
        dataclasses.replace(small_report, trace_sha256="0" * 64)
    )
    assert "trace digest differs" in hash_errors

    resource_errors = validate_historical_behavioral_kl_retention_report(
        dataclasses.replace(
            small_report,
            resource=dataclasses.replace(
                small_report.resource,
                parameter_count=True,
            ),
        )
    )
    assert "report does not reconstruct bit-exactly" in resource_errors
    assert "resource digest differs" in resource_errors


def test_limitations_require_exact_string_types(
    small_report: HistoricalBehavioralKLRetentionReport,
) -> None:
    class TextAlias(str):
        pass

    limitations = (TextAlias(small_report.limitations[0]), *small_report.limitations[1:])
    assert limitations == small_report.limitations
    errors = validate_historical_behavioral_kl_retention_report(
        dataclasses.replace(small_report, limitations=limitations)
    )
    assert "report limitations differ" in errors


def test_caps_fail_closed_without_writes() -> None:
    tiny_report_cap = HistoricalBehavioralKLRetentionConfig(
        a_prefix_steps=1,
        b_interference_steps=1,
        max_report_bytes=1_024,
    )
    with pytest.raises(RuntimeError, match="max_report_bytes"):
        run_historical_behavioral_kl_retention_development(tiny_report_cap)

    tight_parameter_cap = HistoricalBehavioralKLRetentionConfig(
        a_prefix_steps=32,
        b_interference_steps=1,
        max_abs_parameter=1.0,
    )
    with pytest.raises(RuntimeError, match="parameter"):
        run_historical_behavioral_kl_retention_development(tight_parameter_cap)


def test_behavioral_kl_is_stable_when_sigmoid_rounds_to_probability_boundary() -> None:
    probabilities = []
    for parameters in ((100.0, 100.0), (-100.0, -100.0)):
        kl_value, gradient, probability = historical_module._behavioral_kl_component(
            parameters,
            (1.0, 1.0),
            0.5,
        )
        assert math.isfinite(kl_value)
        assert kl_value >= 0.0
        assert all(math.isfinite(value) for value in gradient)
        assert 0.0 <= probability <= 1.0
        probabilities.append(probability)
    assert probabilities[0] == 1.0

    high_weight = HistoricalBehavioralKLRetentionConfig(
        historical_kl_weight=8.0,
        current_kl_weight=8.0,
        max_abs_parameter=100.0,
    )
    report = run_historical_behavioral_kl_retention_development(high_weight)
    assert validate_historical_behavioral_kl_retention_report(report) == ()


def test_public_builders_reject_config_substitutes() -> None:
    with pytest.raises(TypeError, match="HistoricalBehavioralKLRetentionConfig"):
        build_historical_behavioral_kl_source(cast(Any, object()))
    with pytest.raises(TypeError, match="HistoricalBehavioralKLRetentionConfig"):
        run_historical_behavioral_kl_retention_development(cast(Any, object()))


def test_report_schema_has_raw_coordinates_without_classification_fields() -> None:
    assert {field.name for field in dataclasses.fields(ParetoCoordinate)} == {
        "arm_name",
        "retained_a_return",
        "a_forgetting",
        "b_return",
        "b_plasticity_gain",
    }
    assert {field.name for field in dataclasses.fields(MatchedArmAudit)} == {
        "arm_name",
        "routing",
        "b_event_stream_sha256",
        "events_consumed",
        "updates_attempted",
        "updates_applied",
        "candidate_objectives_per_event",
        "candidate_gradient_float64_scalars_per_event",
        "parameters_addressed_per_event",
        "posthoc_recovery_updates",
        "rng_draws",
    }
    assert all(type(arm) is ArmRun for arm in run_historical_behavioral_kl_retention_development(
        HistoricalBehavioralKLRetentionConfig(a_prefix_steps=2, b_interference_steps=2)
    ).arms)
