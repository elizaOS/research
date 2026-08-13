"""Contracts for the causal pre-action versus retrospective inverse L0 lane."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import math
from collections.abc import Callable
from types import MappingProxyType
from typing import cast

import pytest

from alberta_framework.evaluation import inverse_vs_preaction_behavior_development as lane
from alberta_framework.evaluation.inverse_vs_preaction_behavior_development import (
    ARTIFACT_AUTHORITY,
    ASSESSMENT_STATUS,
    BENCHMARK_EXECUTION_AUTHORITY,
    BRANCH_NAMES,
    DEVELOPMENT_ONLY,
    DEVELOPMENT_SCHEMA,
    EVIDENCE_CLAIMED,
    EVIDENCE_LEVEL,
    OUTPUT_WRITES_ALLOWED,
    SCIENTIFIC_PROMOTION_ALLOWED,
    THRESHOLDS_DEFINED,
    InverseVsPreactionBehaviorConfig,
    run_inverse_vs_preaction_behavior_development,
    validate_inverse_vs_preaction_behavior_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return run_inverse_vs_preaction_behavior_development(
        InverseVsPreactionBehaviorConfig(
            prefix_steps=32,
            branch_steps=16,
            entry_window=8,
        )
    )


def _branches(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        cast(str, branch["branch"]): branch
        for branch in cast(list[dict[str, object]], report["branch_results"])
    }


def _summary(
    report: dict[str, object],
    branch: str,
    window: str,
) -> dict[str, object]:
    return cast(dict[str, object], _branches(report)[branch][window])


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("run_id", "unversioned", ValueError),
        ("prefix_steps", True, TypeError),
        ("prefix_steps", 15, ValueError),
        ("prefix_steps", 17, ValueError),
        ("branch_steps", 8, ValueError),
        ("branch_steps", 17, ValueError),
        ("entry_window", 0, ValueError),
        ("entry_window", 33, ValueError),
        ("pseudocount", 1, TypeError),
        ("pseudocount", float("nan"), ValueError),
        ("pseudocount", 1_000_001.0, ValueError),
        ("max_total_source_events", 2, ValueError),
        ("max_logical_state_bytes", 2, ValueError),
        ("max_report_bytes", False, TypeError),
    ],
)
def test_config_is_exact_bounded_and_roundtrips(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        InverseVsPreactionBehaviorConfig(**{field: value})  # type: ignore[arg-type]

    config = InverseVsPreactionBehaviorConfig()
    assert InverseVsPreactionBehaviorConfig.from_config(config.to_config()) == config
    with pytest.raises(TypeError, match="exact JSON object"):
        InverseVsPreactionBehaviorConfig.from_config(
            MappingProxyType(config.to_config())
        )
    malformed = config.to_config()
    malformed["extra"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        InverseVsPreactionBehaviorConfig.from_config(malformed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("prefix_steps", 64.0),
        ("branch_steps", 32.0),
        ("entry_window", True),
        ("pseudocount", 1),
        ("max_report_bytes", True),
    ],
)
def test_config_rejects_canonical_numeric_and_boolean_aliases(
    field: str,
    replacement: object,
) -> None:
    payload = InverseVsPreactionBehaviorConfig().to_config()
    payload[field] = replacement
    with pytest.raises((TypeError, ValueError), match="invalid config payload"):
        InverseVsPreactionBehaviorConfig.from_config(payload)


def test_matched_source_changes_exactly_the_evaluator_owned_factor() -> None:
    physical_bits = {"control": 0, "policy": 0, "physics": 0}
    previous_posts: dict[str, lane._Observation | None] = {
        "control": None,
        "policy": None,
        "physics": None,
    }
    action_rows: dict[str, list[tuple[int, int]]] = {
        "control": [],
        "policy": [],
        "physics": [],
    }
    for step in range(16):
        control = lane._source_event(
            step,
            physical_bit=physical_bits["control"],
            partner_policy_drift=False,
            physical_law_drift=False,
        )
        policy = lane._source_event(
            step,
            physical_bit=physical_bits["policy"],
            partner_policy_drift=True,
            physical_law_drift=False,
        )
        physics = lane._source_event(
            step,
            physical_bit=physical_bits["physics"],
            partner_policy_drift=False,
            physical_law_drift=True,
        )
        assert control.pre_observation.cue == policy.pre_observation.cue
        assert control.pre_observation.cue == physics.pre_observation.cue
        assert policy.partner_action == 1 - control.partner_action
        assert physics.partner_action == control.partner_action
        assert policy.post_observation.physical_bit == (
            policy.pre_observation.physical_bit ^ policy.partner_action
        )
        assert physics.post_observation.cue == control.post_observation.cue
        for name, event in (
            ("control", control),
            ("policy", policy),
            ("physics", physics),
        ):
            if previous_posts[name] is not None:
                assert event.pre_observation == previous_posts[name]
            previous_posts[name] = event.post_observation
            physical_bits[name] = event.post_observation.physical_bit
            action_rows[name].append((event.pre_observation.cue, event.partner_action))

    assert physical_bits == {"control": 0, "policy": 0, "physics": 0}
    assert action_rows["control"] == action_rows["physics"]
    assert all(
        policy_action == 1 - control_action
        for (_, control_action), (_, policy_action) in zip(
            action_rows["control"],
            action_rows["policy"],
            strict=True,
        )
    )

    for cue in range(2):
        actions = [
            action for row_cue, action in action_rows["control"] if row_cue == cue
        ]
        assert actions.count(cue) == 6
        assert actions.count(1 - cue) == 2


def test_pre_action_api_cannot_receive_action_or_post_observation() -> None:
    freeze_signature = inspect.signature(lane._freeze_pre_action)
    assert tuple(freeze_signature.parameters) == (
        "state",
        "pre_observation",
        "pseudocount",
    )
    frozen_fields = {field.name for field in dataclasses.fields(lane._FrozenPreActionPrediction)}
    assert "partner_action" not in frozen_fields
    assert "post_observation" not in frozen_fields
    assert "inverse_probabilities" not in frozen_fields

    state = lane._initial_state()
    pre = lane._Observation(cue=0, physical_bit=0)
    frozen = lane._freeze_pre_action(state, pre, 1.0)
    assert frozen.partner_action_revealed is False
    assert frozen.post_observation_revealed is False
    assert frozen.inverse_distribution_available is False

    inverse_input_fields = {
        field.name for field in dataclasses.fields(lane._RevealedObservationPair)
    }
    assert "partner_action" not in inverse_input_fields
    assert tuple(inspect.signature(lane._retrospective_inverse_distribution).parameters) == (
        "state",
        "revealed_pair",
        "pseudocount",
    )

    unrevealed = lane._RevealedObservationPair(
        pre_observation=pre,
        post_observation=lane._Observation(cue=1, physical_bit=0),
        frozen_pre_action_sha256=lane._sha256(frozen.to_data()),
        outcome_revealed=False,
    )
    with pytest.raises(ValueError, match="outcome-revealed"):
        lane._retrospective_inverse_distribution(state, unrevealed, 1.0)


def test_report_binds_the_reveal_order_and_retrospective_semantics(
    report: dict[str, object],
) -> None:
    expected_order = [
        "observe_pre_observation",
        "freeze_pre_action_behavior_distribution",
        "freeze_action_conditional_world_predictions",
        "freeze_causal_action_marginal_world_prediction",
        "reveal_partner_action",
        "reveal_post_observation",
        "form_retrospective_inverse_distribution",
        "score_frozen_predictions",
        "commit_one_update_per_model",
    ]
    for branch in _branches(report).values():
        witness = cast(dict[str, object], branch["first_event_timing_witness"])
        assert witness["operation_order"] == expected_order
        assert expected_order.index("reveal_post_observation") < expected_order.index(
            "form_retrospective_inverse_distribution"
        )
        pre = cast(dict[str, object], witness["pre_action_payload"])
        assert pre["partner_action_revealed"] is False
        assert pre["post_observation_revealed"] is False
        assert pre["inverse_distribution_available"] is False
        assert witness["retrospective_inverse_requires_post_observation"] is True
        assert witness["inverse_decision_time_available"] is False
        assert witness["inverse_input_contains_partner_action_label"] is False
        inverse_input = cast(
            dict[str, object],
            witness["inverse_input_payload_without_action_label"],
        )
        assert "partner_action" not in inverse_input

    visible = cast(dict[str, object], report["learner_visible_input_contract"])
    assert visible["branch_or_task_identifier"] is False
    assert visible["post_observation_available_to_pre_action_behavior_predictor"] is False
    assert visible["post_observation_required_by_retrospective_inverse_head"] is True


def test_every_branch_starts_from_one_bit_exact_common_prefix_without_reset(
    report: dict[str, object],
) -> None:
    states = cast(dict[str, object], report["states"])
    prefix = cast(dict[str, object], states["after_common_prefix"])
    prefix_sha = prefix["content_sha256"]
    branches = _branches(report)
    assert tuple(branches) == BRANCH_NAMES
    assert {
        branch["starts_from_common_prefix_state_sha256"] for branch in branches.values()
    } == {prefix_sha}

    prefix_result = cast(dict[str, object], report["common_prefix"])
    prefix_source = cast(dict[str, object], prefix_result["source_trace"])
    assert prefix_source["event_count"] == 32
    assert prefix_source["stored_event_count"] == 0
    assert report["task_identifiers_exposed"] is False
    assert report["resets_exposed"] is False
    resource = cast(dict[str, object], report["resource"])
    assert resource["learner_resets"] == 0
    assert resource["learner_task_or_branch_identifiers"] == 0
    assert resource["passes_over_each_source_event"] == 1
    assert resource["replay_capacity"] == 0


def test_interventions_are_external_and_do_not_conflate_policy_with_physics(
    report: dict[str, object],
) -> None:
    interventions = {
        name: cast(dict[str, object], branch["evaluator_only_intervention"])
        for name, branch in _branches(report).items()
    }
    assert interventions == {
        "control": {
            "partner_policy_mapping_changed": False,
            "physical_transition_law_changed": False,
        },
        "partner_policy_drift": {
            "partner_policy_mapping_changed": True,
            "physical_transition_law_changed": False,
        },
        "physical_law_drift": {
            "partner_policy_mapping_changed": False,
            "physical_transition_law_changed": True,
        },
    }


def test_raw_entry_metrics_localize_the_two_kinds_of_change(
    report: dict[str, object],
) -> None:
    control = _summary(report, "control", "entry")
    policy = _summary(report, "partner_policy_drift", "entry")
    physics = _summary(report, "physical_law_drift", "entry")
    control_behavior = cast(dict[str, object], control["behavior"])
    policy_behavior = cast(dict[str, object], policy["behavior"])
    physics_behavior = cast(dict[str, object], physics["behavior"])
    control_inverse = cast(dict[str, object], control["retrospective_inverse"])
    physics_inverse = cast(dict[str, object], physics["retrospective_inverse"])
    control_world = cast(dict[str, object], control["world"])
    policy_world = cast(dict[str, object], policy["world"])
    physics_world = cast(dict[str, object], physics["world"])

    assert policy_behavior["nll"] > control_behavior["nll"]  # type: ignore[operator]
    assert policy_behavior["brier"] > control_behavior["brier"]  # type: ignore[operator]
    assert physics_behavior == control_behavior
    assert physics_inverse["nll"] > control_inverse["nll"]  # type: ignore[operator]
    assert physics_inverse["brier"] > control_inverse["brier"]  # type: ignore[operator]
    assert physics_world["conditional_post_bit_squared_error"] > control_world[
        "conditional_post_bit_squared_error"
    ]  # type: ignore[operator]
    assert policy_world["causal_action_marginal_post_bit_squared_error"] > control_world[
        "causal_action_marginal_post_bit_squared_error"
    ]  # type: ignore[operator]
    assert physics_world["causal_action_marginal_post_bit_squared_error"] > control_world[
        "causal_action_marginal_post_bit_squared_error"
    ]  # type: ignore[operator]


def test_confusions_are_raw_complete_counts_and_deltas_recompute(
    report: dict[str, object],
) -> None:
    branches = _branches(report)
    for branch in branches.values():
        for window, expected_steps in (("entry", 8), ("full", 16)):
            summary = cast(dict[str, object], branch[window])
            assert summary["steps"] == expected_steps
            for model in ("behavior", "retrospective_inverse"):
                channel = cast(dict[str, object], summary[model])
                confusion = cast(
                    list[list[int]],
                    channel["confusion_rows_true_columns_predicted"],
                )
                assert sum(sum(row) for row in confusion) == expected_steps
                assert 0.0 <= cast(float, channel["argmax_accuracy"]) <= 1.0
                assert math.isfinite(cast(float, channel["nll"]))
                assert math.isfinite(cast(float, channel["brier"]))

    deltas = cast(dict[str, dict[str, dict[str, object]]], report["branch_minus_control_deltas"])
    for branch_name in ("partner_policy_drift", "physical_law_drift"):
        for window in ("entry", "full"):
            actual = deltas[branch_name][window]
            expected = lane._metric_delta(
                cast(dict[str, object], branches[branch_name][window]),
                cast(dict[str, object], branches["control"][window]),
            )
            assert actual == expected


def test_state_work_trajectory_and_scaling_receipts_are_exact(
    report: dict[str, object],
) -> None:
    resource = cast(dict[str, object], report["resource"])
    work = cast(dict[str, object], report["work"])
    assert resource["behavior_table_cells"] == 4
    assert resource["inverse_table_cells"] == 32
    assert resource["world_count_table_cells"] == 4
    assert resource["world_one_count_table_cells"] == 4
    assert resource["persistent_integer_scalars"] == 45
    assert resource["logical_preallocated_state_nbytes"] == 360
    assert resource["state_size_fixed_across_steps"] is True
    assert resource["total_source_events"] == 80
    assert resource["stored_trace_events"] == 0
    assert resource["randomness_calls"] == 0
    assert resource["per_event_prediction_scaling"] == "O(A)"
    assert resource["persistent_state_scaling"] == "O(C*A + O^2*A + P*A)"
    assert work["common_prefix_events_consumed"] == 32
    assert work["counterfactual_continuation_events_consumed"] == 48
    assert work["total_evaluator_source_events_consumed"] == 80
    assert work["behavior_distributions_frozen"] == 80
    assert work["action_conditional_world_cells_predicted"] == 160
    assert work["causal_action_marginals_frozen"] == 80
    assert work["retrospective_inverse_distributions_formed_after_reveal"] == 80
    assert work["behavior_updates_committed"] == 80
    assert work["inverse_updates_committed"] == 80
    assert work["world_updates_committed"] == 80
    assert work["source_event_replays"] == 0
    unhashed_work = dict(work)
    work_sha = unhashed_work.pop("work_contract_sha256")
    assert work_sha == lane._sha256(unhashed_work)

    for segment in [
        cast(dict[str, object], report["common_prefix"]),
        *_branches(report).values(),
    ]:
        source_trace = cast(dict[str, object], segment["source_trace"])
        trajectory = cast(dict[str, object], segment["trajectory_trace"])
        for descriptor in (source_trace, trajectory):
            assert descriptor["stored_event_count"] == 0
            assert cast(int, descriptor["event_count"]) > 0
            assert cast(int, descriptor["canonical_nbytes"]) > 2
            assert cast(int, descriptor["maximum_transient_event_canonical_nbytes"]) > 0
            digest = cast(str, descriptor["sha256"])
            assert len(digest) == 64

    assert len(cast(str, report["source_manifest_sha256"])) == 64
    assert len(cast(str, report["trajectory_manifest_sha256"])) == 64
    assert resource["final_report_canonical_nbytes"] == lane._canonical_nbytes(report)


def test_report_is_strict_deterministic_reconstructable_l0(
    report: dict[str, object],
) -> None:
    rerun = run_inverse_vs_preaction_behavior_development(
        InverseVsPreactionBehaviorConfig(
            prefix_steps=32,
            branch_steps=16,
            entry_window=8,
        )
    )
    assert rerun == report
    assert validate_inverse_vs_preaction_behavior_report(report) == ()
    assert report["schema"] == DEVELOPMENT_SCHEMA
    assert report["development_only"] is DEVELOPMENT_ONLY is True
    assert report["assessment_status"] == ASSESSMENT_STATUS == "not_assessed"
    assert report["evidence_level"] == EVIDENCE_LEVEL == "L0"
    assert report["scientific_promotion_allowed"] is SCIENTIFIC_PROMOTION_ALLOWED is False
    assert report["benchmark_execution_authority"] is BENCHMARK_EXECUTION_AUTHORITY is False
    assert report["artifact_authority"] is ARTIFACT_AUTHORITY is False
    assert report["output_writes_allowed"] is OUTPUT_WRITES_ALLOWED is False
    assert report["evidence_claimed"] is EVIDENCE_CLAIMED is False
    assert report["thresholds_defined"] is THRESHOLDS_DEFINED is False
    assert report["descriptive_claims_only"] is True
    assert tuple(inspect.signature(run_inverse_vs_preaction_behavior_development).parameters) == (
        "config",
    )

    implementation = (
        inspect.getsource(run_inverse_vs_preaction_behavior_development)
        + inspect.getsource(validate_inverse_vs_preaction_behavior_report)
    ).lower()
    assert "winner" not in implementation
    assert "verdict" not in implementation


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("task_identifiers_exposed",), 0),
        (("resets_exposed",), 0),
        (("resource", "randomness_calls"), False),
        (("resource", "logical_preallocated_state_nbytes"), 360.0),
        (("branch_results", 0, "entry", "behavior", "nll"), -0.0),
    ],
)
def test_resealed_canonical_type_and_signed_zero_tampering_is_rejected(
    report: dict[str, object],
    path: tuple[object, ...],
    replacement: object,
) -> None:
    tampered = copy.deepcopy(report)
    tampered.pop("integrity")
    cursor: object = tampered
    for key in path[:-1]:
        if type(key) is int:
            cursor = cast(list[object], cursor)[key]
        else:
            cursor = cast(dict[str, object], cursor)[cast(str, key)]
    final_key = path[-1]
    if type(final_key) is int:
        cast(list[object], cursor)[final_key] = replacement
    else:
        cast(dict[str, object], cursor)[cast(str, final_key)] = replacement
    resealed = lane._seal_report(tampered)
    errors = validate_inverse_vs_preaction_behavior_report(resealed)
    assert "report does not reconstruct with exact canonical types and bytes" in errors


def test_resealed_source_state_and_trajectory_hash_tampering_is_rejected(
    report: dict[str, object],
) -> None:
    def mutate_source(value: dict[str, object]) -> None:
        cast(dict[str, object], value["source_contract"])["post_cue"] = "tampered"

    def mutate_state(value: dict[str, object]) -> None:
        cast(
            dict[str, object],
            cast(dict[str, object], value["states"])["after_common_prefix"],
        )["content_sha256"] = "0" * 64

    def mutate_trajectory(value: dict[str, object]) -> None:
        cast(
            dict[str, object],
            cast(list[dict[str, object]], value["branch_results"])[0][
                "trajectory_trace"
            ],
        )["sha256"] = "f" * 64

    mutators: tuple[Callable[[dict[str, object]], None], ...] = (
        mutate_source,
        mutate_state,
        mutate_trajectory,
    )
    for mutator in mutators:
        tampered = copy.deepcopy(report)
        tampered.pop("integrity")
        mutator(tampered)
        resealed = lane._seal_report(tampered)
        assert validate_inverse_vs_preaction_behavior_report(resealed)


def test_report_size_limit_is_enforced_after_exact_serialization() -> None:
    with pytest.raises(ValueError, match="report exceeds max_report_bytes"):
        run_inverse_vs_preaction_behavior_development(
            InverseVsPreactionBehaviorConfig(max_report_bytes=1)
        )


def test_runner_rejects_falsey_or_nonexact_config_substitutes() -> None:
    class FalseySubstitute:
        def __bool__(self) -> bool:
            return False

    malformed_values: tuple[object, ...] = (False, 0, {}, FalseySubstitute())
    for malformed in malformed_values:
        with pytest.raises(
            TypeError,
            match="exact InverseVsPreactionBehaviorConfig",
        ):
            run_inverse_vs_preaction_behavior_development(malformed)  # type: ignore[arg-type]
