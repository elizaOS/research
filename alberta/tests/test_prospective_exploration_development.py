# mypy: disable-error-code="arg-type,attr-defined,no-any-return,no-untyped-call"
"""Closed-loop, replay, checkpoint, and report contracts for WP5 exploration."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

import jax
import numpy as np
import pytest

from alberta_framework.evaluation import prospective_exploration_development as dev

pytestmark = [pytest.mark.integration, pytest.mark.development]


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_module() -> Any:
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def evaluator() -> dev.ProspectiveExplorationDevelopmentEvaluator:
    return dev.ProspectiveExplorationDevelopmentEvaluator(
        dev.ProspectiveExplorationDevelopmentConfig()
    )


@pytest.fixture(scope="module")
def final_state(
    evaluator: dev.ProspectiveExplorationDevelopmentEvaluator,
) -> dev.ProspectiveExplorationRunState:
    return evaluator._reconstruct(dev.FIXED_HORIZON)


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return dev.build_prospective_exploration_development_report()


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return cast(Mapping[str, object], value)


def test_frozen_protocol_compares_all_modes_without_threshold_or_authority() -> None:
    config = dev.ProspectiveExplorationDevelopmentConfig()
    protocol = dev.prospective_exploration_protocol(config)
    assert dev.ProspectiveExplorationDevelopmentConfig.from_config(config.to_config()) == config
    assert config.horizon == dev.FIXED_HORIZON == 8
    assert config.checkpoint_split == dev.FIXED_CHECKPOINT_SPLIT == 3
    assert tuple(protocol["comparators"]) == dev.MODE_ORDER
    assert protocol["thresholds"] == []
    assert protocol["winner_selection"] is False
    assert protocol["efficacy_claimed"] is False
    assert protocol["output_path"] is None
    assert protocol["artifact_writer_available"] is False
    assert protocol["promotion_authority"] is False
    assert protocol["scientific_promotion_allowed"] is False
    boundary = _mapping(protocol["ranking_and_safety_boundary"])
    assert boundary["selector_internal_mask"] == "fixed all true for ranking only"
    assert boundary["selector_owns_actual_action_admissibility"] is False
    assert boundary["actual_admissibility_owner"] == "caller-owned hard shield"
    score = _mapping(protocol["score_estimation"])
    assert score["counterfactual_reward_input"] is False
    assert score["latent_progress_target_input"] is False
    assert score["oracle_score_input"] is False

    for change in (
        {"seed": 1},
        {"selector_seed": 1},
        {"horizon": 9},
        {"checkpoint_split": 4},
        {"delayed_investments_required": 4},
        {"ensemble_size": 4},
        {"epsilon": 0.2},
        {"metric_cap": 9.0},
    ):
        with pytest.raises(ValueError):
            dev.ProspectiveExplorationDevelopmentConfig(**change)


def test_initial_arms_have_independent_owners_but_matched_rng_and_parameters(
    evaluator: dev.ProspectiveExplorationDevelopmentEvaluator,
) -> None:
    state = evaluator.initial_state()
    assert evaluator.validate_state(state)
    assert tuple(arm.mode for arm in state.arms) == dev.MODE_ORDER
    assert len({id(arm.environment) for arm in state.arms}) == len(dev.MODE_ORDER)
    owners = {
        tuple(np.asarray(arm.environment.environment_owner_digest).tolist())
        for arm in state.arms
    }
    assert len(owners) == len(dev.MODE_ORDER)
    reference = state.arms[0]
    for arm in state.arms[1:]:
        np.testing.assert_array_equal(arm.estimator.weights, reference.estimator.weights)
        np.testing.assert_array_equal(
            jax.random.key_data(arm.estimator.rng_key),
            jax.random.key_data(reference.estimator.rng_key),
        )
        np.testing.assert_array_equal(
            jax.random.key_data(arm.selector.rng_key),
            jax.random.key_data(reference.selector.rng_key),
        )
        assert not np.array_equal(
            arm.estimator.estimator_owner_digest,
            reference.estimator.estimator_owner_digest,
        )


def test_raw_trace_closes_score_rank_shield_execute_update_chain(
    evaluator: dev.ProspectiveExplorationDevelopmentEvaluator,
    final_state: dev.ProspectiveExplorationRunState,
) -> None:
    records = evaluator.records(final_state)
    assert len(records) == dev.FIXED_HORIZON * len(dev.MODE_ORDER) == 48
    assert Counter(record["mode"] for record in records) == {
        mode: dev.FIXED_HORIZON for mode in dev.MODE_ORDER
    }
    fallbacks = 0
    owners_by_mode: dict[str, tuple[object, ...]] = {}
    for record in records:
        event = cast(int, record["event_index"])
        mode = cast(str, record["mode"])
        estimate = _mapping(record["causal_estimates"])
        ranking = _mapping(record["ranking"])
        shield = _mapping(record["caller_owned_hard_shield"])
        transition = _mapping(record["observed_transition"])
        update = _mapping(record["estimator_update"])
        assert record["schema"] == (
            "alberta.prospective-exploration-development.trace.v2"
        )
        assert record["source_event_words"] == [0, event + 1]
        assert record["candidate_actions"] == [0, 1, 2, 3]
        assert record["candidate_budget"] == 4
        assert estimate["estimator_revision_words"] == [0, event]
        assert estimate["causal_online_estimate"] is True
        assert estimate["oracle_input_used"] is False
        assert estimate["derived_from_this_arm_history_only"] is True
        assert ranking["permissive_internal_mask"] == [True, True, True, True]
        assert ranking["selector"] == "ProspectiveExploration"
        assert "selected_expected_improvement_surprisal_score" in ranking
        assert "selected_prospective_delight" not in ranking
        assert ranking["owns_actual_admissibility"] is False
        assert ranking["pre_decision_words"] == [0, event]
        assert ranking["post_decision_words"] == [0, event + 1]
        assert ranking["logical_uniform_draws"] == 5
        assert shield["applied_after_ranking"] is True
        assert shield["applied_before_environment"] is True
        assert shield["action_available"] is True
        assert transition["action"] == shield["executable_action"]
        assert transition["decision_words"] == ranking["post_decision_words"]
        assert transition["estimator_pre_revision_words"] == [0, event]
        assert update["called"] is True
        assert update["applied"] is True
        assert update["pre_revision_words"] == [0, event]
        assert update["post_revision_words"] == [0, event + 1]
        assert record["physical_action_dispatched"] is False
        fallbacks += int(shield["fallback_used"] is True)
        owners_by_mode.setdefault(
            mode,
            tuple(cast(list[object], estimate["estimator_owner_digest"])),
        )
        assert owners_by_mode[mode] == tuple(
            cast(list[object], estimate["estimator_owner_digest"])
        )
    assert fallbacks > 0
    assert len(set(owners_by_mode.values())) == len(dev.MODE_ORDER)


def test_matched_candidate_rng_update_resources_are_actual(
    evaluator: dev.ProspectiveExplorationDevelopmentEvaluator,
    final_state: dev.ProspectiveExplorationRunState,
) -> None:
    resources = evaluator.resource_report(final_state)
    assert resources["logical_opportunities_matched"] is True
    assert resources["common_paired_exogenous_normal_draws"] == 24
    per_arm = _mapping(resources["per_arm"])
    signatures: list[object] = []
    for mode in dev.MODE_ORDER:
        arm = _mapping(per_arm[mode])
        logical = _mapping(arm["logical"])
        assert logical == {
            "candidate_slots": 32,
            "candidate_generation_random_draws": 0,
            "selector_uniform_draws": 40,
            "estimator_initialization_normal_draws": 72,
            "estimator_score_scalars": 128,
            "estimator_update_opportunities": 8,
            "estimator_updates_applied": 8,
            "estimator_update_random_draws": 0,
            "estimator_update_parameter_opportunities": 144,
            "hard_shield_calls": 8,
            "environment_step_opportunities": 8,
            "environment_steps_applied": 8,
        }
        signatures.append(logical)
        persistent = _mapping(arm["persistent_array_bytes"])
        assert cast(int, persistent["total"]) > 0
        assert cast(int, arm["record_json_logical_bytes"]) > 0
    assert all(signature == signatures[0] for signature in signatures)


def test_resealed_composite_state_tamper_fails_component_and_exact_prefix_validation(
    evaluator: dev.ProspectiveExplorationDevelopmentEvaluator,
    final_state: dev.ProspectiveExplorationRunState,
) -> None:
    first = final_state.arms[0]
    changed_environment = first.environment.replace(
        stable_signal=first.environment.stable_signal + np.float32(0.01)
    )
    changed_arm = dataclasses.replace(first, environment=changed_environment)
    changed_state = dataclasses.replace(
        final_state,
        arms=(changed_arm, *final_state.arms[1:]),
        integrity_sha256="",
    )
    resealed = evaluator._seal_state(changed_state)
    assert evaluator._validate_state_structure(resealed)
    assert not evaluator.validate_state(resealed)

    invalid_environment = first.environment.replace(
        delayed_progress=np.asarray(99, dtype=np.int32)
    )
    invalid_arm = dataclasses.replace(first, environment=invalid_environment)
    invalid_state = dataclasses.replace(
        final_state,
        arms=(invalid_arm, *final_state.arms[1:]),
        integrity_sha256="",
    )
    assert not evaluator._validate_state_structure(evaluator._seal_state(invalid_state))


def test_full_in_memory_checkpoint_restore_resume_and_tamper_fail_closed(
    evaluator: dev.ProspectiveExplorationDevelopmentEvaluator,
    final_state: dev.ProspectiveExplorationRunState,
) -> None:
    prefix = evaluator._reconstruct(dev.FIXED_CHECKPOINT_SPLIT)
    checkpoint = evaluator.checkpoint_payload(prefix)
    transported = json.loads(json.dumps(checkpoint, allow_nan=False))
    restored = evaluator.restore_checkpoint(transported)
    resumed = evaluator.run_to_end(restored)
    assert evaluator._state_body(resumed) == evaluator._state_body(final_state)
    assert resumed.integrity_sha256 == final_state.integrity_sha256
    assert checkpoint["in_memory_only"] is True
    assert checkpoint["output_path"] is None
    assert checkpoint["artifact_writer_available"] is False
    assert checkpoint["promotion_authority"] is False

    forged = copy.deepcopy(transported)
    composite = cast(dict[str, Any], forged["full_composite_state"])
    arms = cast(list[dict[str, Any]], composite["arms"])
    environment = cast(dict[str, Any], arms[0]["environment"])
    environment["delayed_progress"] = 99
    body = {name: forged[name] for name in forged if name != "checkpoint_sha256"}
    forged["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="exact causal prefix"):
        evaluator.restore_checkpoint(forged)


def test_report_is_raw_reconstructable_nonassessing_and_exercises_both_stresses(
    report: dict[str, object],
) -> None:
    receipt = dev.validate_prospective_exploration_development_report(report)
    assert receipt.valid
    assert receipt.assessment_status == "not_assessed"
    assert receipt.exact_causal_replay
    assert receipt.checkpoint_resume_exact
    assert receipt.raw_trace_reconstructable
    assert receipt.matched_budgets
    assert receipt.independent_environment_owners
    assert receipt.output_written is False
    assert receipt.promotion_authority is False
    assert report["thresholds"] == []
    assert report["winner_selected"] is False
    assert report["efficacy_claimed"] is False
    assert report["safety_claimed"] is False
    assert report["output_path"] is None
    assert report["artifact_writer_available"] is False
    assert report["scientific_promotion_allowed"] is False
    diagnostics = _mapping(report["diagnostics"])
    assert diagnostics["each_score_derived_from_own_executed_history"] is True
    assert diagnostics["oracle_score_input_used"] is False
    assert diagnostics["caller_hard_shield_owns_actual_admissibility"] is True
    assert diagnostics["logical_candidate_rng_update_budgets_matched"] is True
    summaries = _mapping(report["summaries"])
    random_summary = _mapping(summaries["random"])
    assert cast(int, random_summary["delayed_investment_executions"]) >= 3
    assert cast(int, random_summary["delayed_collection_executions"]) >= 1
    assert cast(int, random_summary["noisy_tv_executions"]) >= 1
    assert all(
        _mapping(summaries[mode])["assessment_status"] == "not_assessed"
        for mode in dev.MODE_ORDER
    )


def test_resealed_reward_tamper_cannot_reconstruct_report_summary(
    report: dict[str, object],
) -> None:
    forged = copy.deepcopy(report)
    records = cast(list[dict[str, object]], forged["records"])
    transition = cast(dict[str, object], records[0]["observed_transition"])
    transition["reward"] = 9.0
    record_body = {
        name: records[0][name] for name in records[0] if name != "record_sha256"
    }
    records[0]["record_sha256"] = dev._canonical_sha256(record_body)
    forged["records_sha256"] = dev._canonical_sha256(records)
    report_body = {
        name: forged[name] for name in forged if name != "report_sha256"
    }
    forged["report_sha256"] = dev._canonical_sha256(report_body)
    with pytest.raises(ValueError, match="summaries do not reconstruct"):
        dev.validate_prospective_exploration_development_report(forged)
