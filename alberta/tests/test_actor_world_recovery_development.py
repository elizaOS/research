"""Contracts for the deterministic L0 actor/world recovery scaffold."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import math
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from alberta_framework.evaluation import actor_world_recovery_development as recovery_module
from alberta_framework.evaluation.actor_world_recovery_development import (
    ARTIFACT_AUTHORITY,
    ASSESSMENT_STATUS,
    BENCHMARK_EXECUTION_AUTHORITY,
    DEVELOPMENT_ONLY,
    DEVELOPMENT_SCHEMA,
    EVIDENCE_CLAIMED,
    EVIDENCE_LEVEL,
    OUTPUT_WRITES_ALLOWED,
    RECOVERY_ARMS,
    SCIENTIFIC_PROMOTION_ALLOWED,
    THRESHOLDS_DEFINED,
    ActorWorldRecoveryConfig,
    run_actor_world_recovery_development,
    validate_actor_world_recovery_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return run_actor_world_recovery_development(
        ActorWorldRecoveryConfig(
            task_a_online_steps=4,
            task_b_online_steps=8,
            recovery_updates=3,
            actor_step_size=0.2,
        )
    )


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("run_id", "unversioned", ValueError),
        ("task_a_online_steps", True, TypeError),
        ("task_a_online_steps", 1, ValueError),
        ("task_a_online_steps", 3, ValueError),
        ("task_b_online_steps", 3, ValueError),
        ("recovery_updates", 0, ValueError),
        ("actor_step_size", 1, TypeError),
        ("actor_step_size", float("nan"), ValueError),
        ("actor_step_size", 1.1, ValueError),
        ("max_online_steps", 2, ValueError),
        ("max_recovery_gradient_evaluations", 2, ValueError),
        ("max_persistent_state_bytes", False, TypeError),
    ],
)
def test_config_is_exact_bounded_and_fail_closed(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ActorWorldRecoveryConfig(**{field: value})  # type: ignore[arg-type]

    config = ActorWorldRecoveryConfig()
    assert ActorWorldRecoveryConfig.from_config(config.to_config()) == config
    with pytest.raises(TypeError, match="exact JSON object"):
        ActorWorldRecoveryConfig.from_config(MappingProxyType(config.to_config()))
    malformed = config.to_config()
    malformed["extra"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        ActorWorldRecoveryConfig.from_config(malformed)


def test_actor_class_has_an_explicit_joint_solution_witness(
    report: dict[str, object],
) -> None:
    witness = cast(dict[str, object], report["actor_capacity_witness"])
    assert witness["weights"] == [[1.0, -2.0], [-1.0, 2.0]]
    assert witness["task_greedy_actions"] == [0, 1]
    assert witness["evaluator_preferred_actions"] == [0, 1]
    assert witness["task_observation_dot_product"] == 1.0
    assert witness["task_observations_are_nonorthogonal"] is True
    assert witness["joint_solution_constructed"] is True
    assert witness["capacity_failure_excluded_for_these_two_contexts"] is True

    witness_state = dataclasses.replace(
        recovery_module._initial_state(),
        actor_weights=((1.0, -2.0), (-1.0, 2.0)),
    )
    assert recovery_module._probe(witness_state, 0)["actor_greedy_action"] == 0
    assert recovery_module._probe(witness_state, 1)["actor_greedy_action"] == 1


def test_online_component_probes_are_preupdate_unseeded_and_separate(
    report: dict[str, object],
) -> None:
    snapshots = cast(dict[str, dict[str, object]], report["snapshots"])
    initial_a = cast(dict[str, object], snapshots["initial"]["task_a_probe"])
    learned_a = cast(
        dict[str, object],
        snapshots["after_task_a_before_interference"]["task_a_probe"],
    )
    post_b_a = cast(
        dict[str, object],
        snapshots["after_task_b_interference"]["task_a_probe"],
    )

    assert initial_a["world_reward_predictions"] == [0.0, 0.0]
    assert initial_a["value_return_predictions"] == [0.0, 0.0]
    assert cast(float, initial_a["world_reward_mse"]) > 0.0
    assert cast(float, initial_a["value_return_mse"]) > 0.0
    assert learned_a["world_reward_mse"] == 0.0
    assert learned_a["world_next_observation_mse"] == 0.0
    assert learned_a["value_return_mse"] == 0.0
    assert post_b_a["world_reward_mse"] == 0.0
    assert post_b_a["world_next_observation_mse"] == 0.0
    assert post_b_a["value_return_mse"] == 0.0
    for probe in (initial_a, learned_a, post_b_a):
        for field in (
            "world_reward_mse",
            "world_next_observation_mse",
            "value_return_mse",
            "actor_margin",
            "actor_greedy_return",
            "actor_expected_return",
        ):
            assert math.isfinite(cast(float, probe[field]))

    runner_source = inspect.getsource(recovery_module._run_online_phase)
    prediction_position = runner_source.index("pre_reward, pre_next = _world_prediction")
    update_position = runner_source.index("updated = _online_update")
    assert prediction_position < update_position
    limitations = " ".join(cast(list[str], report["limitations"]))
    assert "oracle-like context routing" in limitations


def test_online_trace_digest_is_streamed_without_retaining_events() -> None:
    events = [
        {"step": 0, "post_state_canonical_nbytes": 11, "prediction": 0.0},
        {"step": 1, "post_state_canonical_nbytes": 13, "prediction": -0.0},
    ]
    trace = recovery_module._OnlineTraceAccumulator()
    for event in events:
        trace.append(event)
    descriptor = trace.descriptor()
    assert descriptor["online_event_count"] == 2
    assert descriptor["stored_online_event_count"] == 0
    assert descriptor["online_trace_sha256"] == recovery_module._sha256(events)
    assert descriptor["online_trace_canonical_nbytes"] == recovery_module._serialized_nbytes(
        events
    )
    assert trace.maximum_state_canonical_nbytes_observed == 13


def test_imagined_arms_share_bytes_and_real_control_has_matched_separate_labels(
    report: dict[str, object],
) -> None:
    dataset = cast(dict[str, object], report["grounded_recovery_dataset"])
    arms = {
        cast(str, arm["arm"]): arm
        for arm in cast(list[dict[str, object]], report["recovery_arms"])
    }
    assert tuple(arms) == RECOVERY_ARMS
    assert dataset["policy_gradient_and_dream_imitation_imagined_bytes_identical"] is True
    assert dataset["competent_real_labels_are_separate_from_imagined_action_enumeration"] is True
    assert dataset["imagined_transition_content_matches_anchor"] is True
    assert dataset["grounding_exact"] is True
    assert dataset["external_environment_authenticated"] is False

    imagined_sha = dataset["imagined_content_sha256"]
    assert arms["policy_gradient"]["dataset_content_sha256"] == imagined_sha
    assert arms["graded_dream_imitation"]["dataset_content_sha256"] == imagined_sha
    assert (
        arms["competent_real_cloning"]["dataset_content_sha256"]
        == dataset["competent_real_content_sha256"]
    )
    assert (
        arms["competent_real_cloning"]["dataset_content_sha256"]
        != arms["policy_gradient"]["dataset_content_sha256"]
    )

    start_hashes = {
        cast(str, arm["starts_from_post_interference_state_sha256"])
        for arm in arms.values()
    }
    assert len(start_hashes) == 1
    assert {arm["dataset_entry_count"] for arm in arms.values()} == {2}
    assert {arm["objective_gradient_evaluations"] for arm in arms.values()} == {3}
    assert {arm["dataset_entries_presented"] for arm in arms.values()} == {6}
    assert all(arm["world_value_state_unchanged"] is True for arm in arms.values())

    grading = cast(dict[str, object], dataset["dream_grading_receipt"])
    assert grading["terminal_outcomes_revealed_before_grading"] is True
    assert grading["realized_first"] is True
    assert grading["realized_terminal_returns"] == [1.0, -1.0]
    assert grading["grades"] == [2.0, 0.0]
    grading_unhashed = dict(grading)
    grading_sha = grading_unhashed.pop("receipt_sha256")
    assert grading_sha == recovery_module._sha256(grading_unhashed)
    real_labels = cast(dict[str, object], dataset["competent_real_label_receipt"])
    assert real_labels["evaluator_label_authenticated"] is True
    assert real_labels["external_environment_authenticated"] is False
    assert real_labels["labels"] == [0, 0]
    real_unhashed = dict(real_labels)
    real_sha = real_unhashed.pop("receipt_sha256")
    assert real_sha == recovery_module._sha256(real_unhashed)


def test_objective_equations_work_and_raw_outcomes_are_bound_without_a_winner(
    report: dict[str, object],
) -> None:
    contracts = cast(dict[str, dict[str, object]], report["objective_contracts"])
    assert contracts["policy_gradient"]["equation"] == (
        "J_pg = sum_a pi(a|x) * G_imagined(a)"
    )
    assert contracts["graded_dream_imitation"]["equation"] == (
        "J_dream = (1/N) * sum_i grade_i * log pi(a_i|x)"
    )
    assert contracts["competent_real_cloning"]["equation"] == (
        "J_real = (1/N) * sum_i log pi(a_star_i|x)"
    )
    work = cast(dict[str, object], report["work"])
    assert work["recovery_objective_gradient_evaluations"] == 9
    assert work["recovery_analytic_backward_evaluations"] == 9
    assert work["recovery_dataset_entries_presented"] == 18
    assert work["recovery_actor_parameter_scalars_addressed"] == 36
    assert work["recovery_world_updates"] == 0
    assert work["recovery_value_updates"] == 0
    assert work["equal_gradient_and_dataset_budget_across_arms"] is True

    for arm in cast(list[dict[str, object]], report["recovery_arms"]):
        final_probe = cast(dict[str, object], arm["final_task_a_probe"])
        for field in ("actor_margin", "actor_greedy_return", "actor_expected_return"):
            assert math.isfinite(cast(float, final_probe[field]))
        first = cast(dict[str, object], arm["first_update_receipt"])
        last = cast(dict[str, object], arm["last_update_receipt"])
        assert first["dataset_entries_read"] == last["dataset_entries_read"] == 2
        assert first["actor_parameter_scalars_addressed"] == 4
        assert last["actor_parameter_scalars_addressed"] == 4

    runner_and_validator = (
        inspect.getsource(run_actor_world_recovery_development)
        + inspect.getsource(validate_actor_world_recovery_report)
    ).lower()
    assert "winner" not in runner_and_validator
    assert report["thresholds_defined"] is False
    assert "winner" not in report


def test_report_is_strict_l0_deterministic_and_reconstructable(
    report: dict[str, object],
) -> None:
    assert report["schema_version"] == DEVELOPMENT_SCHEMA
    assert report["development_only"] is DEVELOPMENT_ONLY is True
    assert report["assessment_status"] == ASSESSMENT_STATUS == "not_assessed"
    assert report["evidence_level"] == EVIDENCE_LEVEL == "L0"
    assert report["scientific_promotion_allowed"] is SCIENTIFIC_PROMOTION_ALLOWED is False
    assert report["benchmark_execution_authority"] is BENCHMARK_EXECUTION_AUTHORITY is False
    assert report["artifact_authority"] is ARTIFACT_AUTHORITY is False
    assert report["output_writes_allowed"] is OUTPUT_WRITES_ALLOWED is False
    assert report["evidence_claimed"] is EVIDENCE_CLAIMED is False
    assert report["thresholds_defined"] is THRESHOLDS_DEFINED is False
    assert validate_actor_world_recovery_report(report) == ()

    replay = run_actor_world_recovery_development(
        ActorWorldRecoveryConfig(
            task_a_online_steps=4,
            task_b_online_steps=8,
            recovery_updates=3,
            actor_step_size=0.2,
        )
    )
    assert replay == report


def test_hashes_resources_scaling_and_no_output_surface_are_exact(
    report: dict[str, object],
) -> None:
    source = cast(dict[str, object], report["source"])
    source_path = Path(recovery_module.__file__)
    assert source["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert report["report_sha256"] == recovery_module._report_sha256(report)

    resources = cast(dict[str, object], report["resources"])
    assert resources["persistent_state_logical_scalars"] == 30
    assert resources["canonical_report_bytes"] == recovery_module._serialized_nbytes(report)
    assert resources["canonical_report_bytes"] <= cast(
        int, resources["canonical_report_byte_limit"]
    )
    assert cast(int, resources["maximum_observed_state_canonical_bytes"]) <= cast(
        int, resources["persistent_state_byte_limit"]
    )
    dataset = cast(dict[str, object], report["grounded_recovery_dataset"])
    assert resources["grounded_dataset_canonical_bytes"] == dataset["canonical_nbytes"]
    assert dataset["canonical_nbytes"] == recovery_module._serialized_nbytes(dataset)
    unhashed_dataset = dict(dataset)
    supplied_dataset_sha = unhashed_dataset.pop("dataset_envelope_sha256")
    assert supplied_dataset_sha == recovery_module._sha256(unhashed_dataset)
    assert dataset["canonical_nbytes"] <= cast(
        int, resources["grounded_dataset_byte_limit"]
    )
    scaling = cast(dict[str, object], report["scaling"])
    assert scaling["world_logical_scalars_at_configured_size"] == 16
    assert scaling["value_logical_scalars_at_configured_size"] == 8
    assert scaling["actor_logical_scalars_at_configured_size"] == 4
    assert scaling["clock_logical_scalars"] == 2
    assert 16 + 8 + 4 + 2 == 30
    assert scaling["learner_persistent_state_fixed_with_respect_to_lifetime"] is True
    assert scaling["evaluator_trace_accumulator_fixed_with_respect_to_lifetime"] is True
    assert scaling["general_feature_scaling_assessed"] is False

    trajectory = cast(dict[str, object], report["trajectory"])
    assert trajectory["stored_online_event_count"] == 0
    assert trajectory["logical_digest_state_bytes"] == 32
    assert resources["online_trace_logical_digest_state_bytes"] == 32
    assert resources["maximum_transient_online_event_canonical_bytes"] == trajectory[
        "maximum_transient_event_canonical_nbytes"
    ]

    public_source = inspect.getsource(run_actor_world_recovery_development)
    assert "write_text" not in public_source
    assert "open(" not in public_source


def test_rehashed_finite_forgery_and_malformed_types_fail_reconstruction(
    report: dict[str, object],
) -> None:
    forged = copy.deepcopy(report)
    snapshots = cast(dict[str, dict[str, object]], forged["snapshots"])
    probe = cast(dict[str, object], snapshots["after_task_b_interference"]["task_a_probe"])
    probe["actor_margin"] = cast(float, probe["actor_margin"]) + 0.125
    forged["report_sha256"] = recovery_module._report_sha256(forged)
    assert validate_actor_world_recovery_report(forged) == (
        "report differs from deterministic reconstruction",
    )

    forged_source = copy.deepcopy(report)
    cast(dict[str, object], forged_source["source"])["source_sha256"] = "0" * 64
    forged_source["report_sha256"] = recovery_module._report_sha256(forged_source)
    assert validate_actor_world_recovery_report(forged_source) == (
        "report differs from deterministic reconstruction",
    )

    signed_zero = copy.deepcopy(report)
    initial = cast(
        dict[str, object],
        cast(dict[str, dict[str, object]], signed_zero["snapshots"])["initial"][
            "task_a_probe"
        ],
    )
    reward_predictions = cast(list[float], initial["world_reward_predictions"])
    assert reward_predictions[0] == 0.0
    reward_predictions[0] = -0.0
    signed_zero["report_sha256"] = recovery_module._report_sha256(signed_zero)
    assert validate_actor_world_recovery_report(signed_zero) == (
        "report differs from deterministic reconstruction",
    )

    assert validate_actor_world_recovery_report(MappingProxyType(report)) == (
        "report must be an exact JSON object",
    )


def test_runtime_resource_caps_reject_without_writing() -> None:
    with pytest.raises(ValueError, match="dataset exceeds"):
        run_actor_world_recovery_development(
            ActorWorldRecoveryConfig(max_dataset_bytes=1)
        )
    with pytest.raises(ValueError, match="state exceeds"):
        run_actor_world_recovery_development(
            ActorWorldRecoveryConfig(max_persistent_state_bytes=1)
        )
    with pytest.raises(ValueError, match="report exceeds"):
        run_actor_world_recovery_development(
            ActorWorldRecoveryConfig(max_report_bytes=1)
        )
