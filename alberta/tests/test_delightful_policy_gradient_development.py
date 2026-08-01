# mypy: disable-error-code="call-arg"
"""Development contracts for matched Delightful Policy Gradient benchmarks."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

import alberta_framework.benchmarks.delightful_policy_gradient_development as module
from alberta_framework.benchmarks.delightful_policy_gradient_development import (
    BENCHMARK_ORDER,
    DELIGHTFUL_POLICY_GRADIENT_DEVELOPMENT_SCHEMA,
    POLICY_GRADIENT_MODES,
    DelightfulPolicyGradientDevelopmentConfig,
    delightful_policy_gradient_development_report_json,
    run_delightful_policy_gradient_development,
    validate_delightful_policy_gradient_development_report,
)

pytestmark = pytest.mark.development


def _small_config() -> DelightfulPolicyGradientDevelopmentConfig:
    return DelightfulPolicyGradientDevelopmentConfig(
        seeds=(0,),
        phase_length=2,
        adaptation_window=1,
        recovery_window=1,
        final_window=1,
        worst_window=1,
    )


@pytest.fixture(scope="module")
def development_report() -> dict[str, object]:
    return run_delightful_policy_gradient_development(_small_config())


def _runs(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    return cast(list[Mapping[str, object]], report["runs"])


def _run(
    report: Mapping[str, object],
    benchmark: str,
    mode: str,
) -> Mapping[str, object]:
    return next(
        run for run in _runs(report) if run["benchmark"] == benchmark and run["mode"] == mode
    )


def _redigest(report: dict[str, object]) -> None:
    without_digest = dict(report)
    without_digest.pop("report_digest", None)
    digest = cast(dict[str, object], report["report_digest"])
    digest["sha256"] = module._canonical_sha256(without_digest)


def test_protocol_roundtrip_is_strict_nonpromoting_and_plan_ordered() -> None:
    config = _small_config()
    payload = config.to_config()
    assert DelightfulPolicyGradientDevelopmentConfig.from_config(payload) == config
    assert payload["schema_version"] == DELIGHTFUL_POLICY_GRADIENT_DEVELOPMENT_SCHEMA
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["kondo_implemented"] is False
    assert payload["benchmark_order"] == list(BENCHMARK_ORDER)
    assert payload["policy_gradient_modes"] == list(POLICY_GRADIENT_MODES)
    assert payload["regime_schedule"] == [0, 1, 0]
    assert payload["actor_trace_lambda"] == 0.0
    assert "sparks_joy" not in cast(str, payload["claim_scope"])
    assert any("sparks_joy" in claim for claim in cast(list[str], payload["excluded_claims"]))

    promoted = copy.deepcopy(payload)
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="changed claims"):
        DelightfulPolicyGradientDevelopmentConfig.from_config(promoted)
    kondo = copy.deepcopy(payload)
    kondo["kondo_implemented"] = True
    with pytest.raises(ValueError, match="changed claims"):
        DelightfulPolicyGradientDevelopmentConfig.from_config(kondo)
    reordered = copy.deepcopy(payload)
    reordered["benchmark_order"] = list(reversed(BENCHMARK_ORDER))
    with pytest.raises(ValueError, match="changed claims"):
        DelightfulPolicyGradientDevelopmentConfig.from_config(reordered)
    numeric_type_changed = copy.deepcopy(payload)
    numeric_type_changed["max_input_magnitude"] = 1_000_000
    with pytest.raises(ValueError, match="changed claims"):
        DelightfulPolicyGradientDevelopmentConfig.from_config(numeric_type_changed)


@pytest.mark.parametrize(
    "overrides",
    (
        {"seeds": ()},
        {"seeds": (0, 0)},
        {"seeds": (True,)},
        {"phase_length": True},
        {"phase_length": 0},
        {"phase_length": 2**31 // 3 + 1},
        {"adaptation_window": 3, "phase_length": 2},
        {"diagnostic_recovery_fraction": 1.1},
        {"rare_probability_cutoff": 0.0},
        {"actor_step_size": float("nan")},
        {"policy_temperature": 0.0},
    ),
)
def test_protocol_rejects_ambiguous_or_nonfinite_controls(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        DelightfulPolicyGradientDevelopmentConfig(**overrides)  # type: ignore[arg-type]


def test_report_is_deterministic_versioned_and_strictly_valid(
    development_report: dict[str, object],
) -> None:
    validation = validate_delightful_policy_gradient_development_report(development_report)
    assert validation.valid, validation.errors
    assert development_report["development_only"] is True
    assert development_report["scientific_promotion_allowed"] is False
    assert development_report["kondo_implemented"] is False
    assert development_report["benchmark_order"] == list(BENCHMARK_ORDER)
    assert delightful_policy_gradient_development_report_json(
        development_report
    ) == delightful_policy_gradient_development_report_json(development_report)
    assert len(cast(Mapping[str, str], development_report["report_digest"])["sha256"]) == 64


def test_modes_are_matched_by_config_initialization_and_common_random_numbers(
    development_report: dict[str, object],
) -> None:
    expected_order = [
        (benchmark, mode) for benchmark in BENCHMARK_ORDER for mode in POLICY_GRADIENT_MODES
    ]
    assert [(run["benchmark"], run["mode"]) for run in _runs(development_report)] == expected_order
    for benchmark in BENCHMARK_ORDER:
        ordinary = _run(development_report, benchmark, "ordinary_pg")
        delightful = _run(development_report, benchmark, "delightful_pg")
        ordinary_config = dict(cast(Mapping[str, object], ordinary["agent_config"]))
        delightful_config = dict(cast(Mapping[str, object], delightful["agent_config"]))
        assert ordinary_config.pop("mode") == "ordinary_pg"
        assert delightful_config.pop("mode") == "delightful_pg"
        assert ordinary_config == delightful_config
        assert ordinary["initial_state_sha256"] == delightful["initial_state_sha256"]

        ordinary_trace = cast(Mapping[str, object], ordinary["trace"])
        delightful_trace = cast(Mapping[str, object], delightful["trace"])
        assert ordinary_trace["action_rng_words"] == delightful_trace["action_rng_words"]
        assert ordinary_trace["environment_rng_words"] == delightful_trace["environment_rng_words"]
        assert (
            cast(list[int], ordinary_trace["action"])[0]
            == cast(list[int], delightful_trace["action"])[0]
        )
        assert (
            cast(list[list[float]], ordinary_trace["behavior_policy"])[0]
            == cast(list[list[float]], delightful_trace["behavior_policy"])[0]
        )


def test_traces_are_prequential_uninterrupted_and_environment_reconstructible(
    development_report: dict[str, object],
) -> None:
    steps = _small_config().total_steps
    for run in _runs(development_report):
        trace = cast(Mapping[str, object], run["trace"])
        assert trace["step"] == list(range(steps))
        assert trace["phase_index"] == [0, 0, 1, 1, 2, 2]
        assert trace["regime_id"] == [0, 0, 1, 1, 0, 0]
        assert trace["transition_count_after"] == list(range(1, steps + 1))
        assert trace["applied"] == [True] * steps
        np.testing.assert_array_equal(trace["behavior_policy"], trace["target_policy"])
        policies = np.asarray(trace["behavior_policy"], dtype=np.float32)
        actions = np.asarray(trace["action"], dtype=np.int32)
        selected = policies[np.arange(steps), actions]
        np.testing.assert_allclose(
            np.asarray(trace["action_probability"], dtype=np.float32),
            selected,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            np.asarray(trace["behavior_log_probability"], dtype=np.float32),
            np.log(selected),
            atol=1.0e-6,
        )
        if run["benchmark"] == "six_state_riverswim":
            states = cast(list[int], trace["state_index"])
            next_states = cast(list[int], trace["next_state_index"])
            assert states[0] == 0
            assert states[1:] == next_states[:-1]


def test_metrics_cover_lifetime_adaptation_recovery_safety_retention_and_dg(
    development_report: dict[str, object],
) -> None:
    steps = _small_config().total_steps
    for run in _runs(development_report):
        performance = cast(Mapping[str, object], run["performance"])
        assert len(cast(list[float], performance["phase_mean_rewards"])) == 3
        assert len(cast(list[float], performance["adaptation_auc"])) == 2
        assert len(cast(list[int | None], performance["recovery_steps"])) == 2
        assert isinstance(performance["recurrence_retention_delta"], float)
        assert float(cast(float, performance["safety_cost_rate"])) >= 0.0
        assert 0.0 <= float(cast(float, performance["safety_violation_rate"])) <= 1.0

        dg = cast(Mapping[str, object], run["dg_diagnostics"])
        assert 0.0 <= float(cast(float, dg["effective_sample_size"])) <= steps
        assert 0.0 <= float(cast(float, dg["effective_sample_fraction"])) <= 1.0
        assert 0.0 <= float(cast(float, dg["positive_delight_rate"])) <= 1.0
        assert 0.0 <= float(cast(float, dg["negative_delight_rate"])) <= 1.0
        strata = cast(Mapping[str, Mapping[str, object]], dg["strata"])
        stratum_count = sum(cast(int, strata[name]["count"]) for name in strata)
        assert stratum_count <= steps
        if run["mode"] == "ordinary_pg":
            assert float(cast(float, dg["mean_gate_weight"])) == 1.0

        gambling = cast(Mapping[str, object], run["heteroskedastic_gambling"])
        assert gambling["applicable"] is (run["benchmark"] == "contextual_heteroskedastic_gambling")


def test_operation_and_resource_accounting_are_exact_and_non_kondo(
    development_report: dict[str, object],
) -> None:
    steps = _small_config().total_steps
    for run in _runs(development_report):
        operations = cast(Mapping[str, object], run["operations"])
        assert operations["accounting_scope"] == (
            "logical algorithm events; excludes JAX validation/compiler operations, "
            "FLOPs, energy, and wall clock"
        )
        assert operations["wall_clock_measured"] is False
        assert cast(int, operations["environment_transitions"]) == steps
        assert cast(int, operations["executed_action_samples"]) == steps
        assert cast(int, operations["pending_action_samples"]) == 1
        assert cast(int, operations["total_action_samples"]) == steps + 1
        assert cast(int, operations["actor_gate_evaluations"]) == steps
        assert cast(int, operations["actor_update_attempts"]) == steps
        assert cast(int, operations["actor_update_commits"]) == steps
        assert cast(int, operations["critic_update_commits"]) == steps
        assert cast(int, operations["rejected_transactions"]) == 0
        assert "backward_skips" not in operations

        resources = cast(Mapping[str, object], run["resources"])
        assert int(cast(int, resources["persistent_state_nbytes"])) > 0
        assert int(cast(int, resources["logical_trace_nbytes"])) > 0
        assert int(cast(int, resources["peak_owned_logical_nbytes"])) == (
            int(cast(int, resources["persistent_state_nbytes"]))
            + int(cast(int, resources["environment_state_nbytes"]))
            + int(cast(int, resources["logical_trace_nbytes"]))
        )
        assert resources["replay_capacity"] == 0

    comparisons = cast(list[Mapping[str, object]], development_report["paired_comparisons"])
    assert [comparison["benchmark"] for comparison in comparisons] == list(BENCHMARK_ORDER)
    assert all(comparison["verdict"] is None for comparison in comparisons)


def test_validator_rejects_digest_semantics_crn_and_numeric_type_tampering(
    development_report: dict[str, object],
) -> None:
    digest_tampered = copy.deepcopy(development_report)
    digest_tampered["interpretation"] = "promoted"
    validation = validate_delightful_policy_gradient_development_report(digest_tampered)
    assert not validation.valid
    assert any("interpretation" in error for error in validation.errors)
    assert any("digest" in error for error in validation.errors)

    reward_tampered = copy.deepcopy(development_report)
    first_run = cast(dict[str, object], cast(list[object], reward_tampered["runs"])[0])
    trace = cast(dict[str, object], first_run["trace"])
    rewards = cast(list[float], trace["reward"])
    rewards[0] += 0.5
    _redigest(reward_tampered)
    validation = validate_delightful_policy_gradient_development_report(reward_tampered)
    assert not validation.valid
    assert any("reward" in error or "performance" in error for error in validation.errors)

    crn_tampered = copy.deepcopy(development_report)
    runs = cast(list[dict[str, object]], crn_tampered["runs"])
    delightful = runs[1]
    delightful_trace = cast(dict[str, object], delightful["trace"])
    action_keys = cast(list[list[int]], delightful_trace["action_rng_words"])
    action_keys[0][0] ^= 1
    _redigest(crn_tampered)
    validation = validate_delightful_policy_gradient_development_report(crn_tampered)
    assert not validation.valid
    assert any("action" in error or "CRN" in error for error in validation.errors)

    type_tampered = copy.deepcopy(development_report)
    first_run = cast(dict[str, object], cast(list[object], type_tampered["runs"])[0])
    trace = cast(dict[str, object], first_run["trace"])
    cast(list[object], trace["reward"])[0] = "1.0"
    _redigest(type_tampered)
    validation = validate_delightful_policy_gradient_development_report(type_tampered)
    assert not validation.valid
    assert any("scalar types" in error for error in validation.errors)

    state_tampered = copy.deepcopy(development_report)
    first_run = cast(dict[str, object], cast(list[object], state_tampered["runs"])[0])
    trace = cast(dict[str, object], first_run["trace"])
    cast(list[int], trace["action"])[0] = _small_config().total_steps
    _redigest(state_tampered)
    validation = validate_delightful_policy_gradient_development_report(state_tampered)
    assert not validation.valid
    assert any("out of range" in error for error in validation.errors)

    fingerprint_tampered = copy.deepcopy(development_report)
    runs = cast(list[dict[str, object]], fingerprint_tampered["runs"])
    for run in runs[:2]:
        run["initial_state_sha256"] = "0" * 64
    _redigest(fingerprint_tampered)
    validation = validate_delightful_policy_gradient_development_report(fingerprint_tampered)
    assert not validation.valid
    assert any("initial state does not reconstruct" in error for error in validation.errors)

    accounting_type_tampered = copy.deepcopy(development_report)
    first_run = cast(
        dict[str, object],
        cast(list[object], accounting_type_tampered["runs"])[0],
    )
    operations = cast(dict[str, object], first_run["operations"])
    operations["environment_transitions"] = float(_small_config().total_steps)
    _redigest(accounting_type_tampered)
    validation = validate_delightful_policy_gradient_development_report(accounting_type_tampered)
    assert not validation.valid
    assert any("operation accounting" in error for error in validation.errors)


def test_validator_structurally_forbids_promotion_and_kondo(
    development_report: dict[str, object],
) -> None:
    promoted = copy.deepcopy(development_report)
    promoted["scientific_promotion_allowed"] = True
    _redigest(promoted)
    validation = validate_delightful_policy_gradient_development_report(promoted)
    assert not validation.valid
    assert any("promotion" in error for error in validation.errors)

    kondo = copy.deepcopy(development_report)
    kondo["kondo_implemented"] = True
    _redigest(kondo)
    validation = validate_delightful_policy_gradient_development_report(kondo)
    assert not validation.valid
    assert any("Kondo" in error for error in validation.errors)

    extra = copy.deepcopy(development_report)
    extra["accepted"] = True
    _redigest(extra)
    validation = validate_delightful_policy_gradient_development_report(extra)
    assert not validation.valid
    assert any("top-level" in error for error in validation.errors)
