# mypy: disable-error-code="attr-defined,no-untyped-call,redundant-cast"
"""Unit contracts for the strict WP9 dynamics-adaptation diagnostic."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation import embodied_dynamics_adaptation_development as dev

pytestmark = [pytest.mark.unit, pytest.mark.development]


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_module() -> Any:
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def evaluator() -> dev.EmbodiedDynamicsAdaptationEvaluator:
    return dev.EmbodiedDynamicsAdaptationEvaluator(
        dev.EmbodiedDynamicsAdaptationConfig()
    )


@pytest.fixture(scope="module")
def final_state(
    evaluator: dev.EmbodiedDynamicsAdaptationEvaluator,
) -> dev.EmbodiedDynamicsAdaptationRunState:
    return evaluator.run_to_end()


def _records(
    state: dev.EmbodiedDynamicsAdaptationRunState,
) -> list[dict[str, object]]:
    return [cast(dict[str, object], json.loads(raw)) for raw in state.records_json]


def test_frozen_config_protocol_and_authority_are_strict() -> None:
    config = dev.EmbodiedDynamicsAdaptationConfig()
    protocol = dev.embodied_dynamics_protocol(config)

    assert dev.EmbodiedDynamicsAdaptationConfig.from_config(config.to_config()) == config
    assert config.total_events == 12
    assert dev.DEVELOPMENT_STATUS == dev.ASSESSMENT_STATUS == "not_assessed"
    assert dev.OUTPUT_WRITES is False
    assert dev.PHYSICAL_DISPATCH_AUTHORITY is False
    assert dev.DEPLOYMENT_AUTHORITY is False
    assert dev.PROMOTION_AUTHORITY is False
    assert dev.SCIENTIFIC_PROMOTION_ALLOWED is False
    assert protocol["a_b_a_phases"] == ["A_initial", "B", "A_return"]
    change = cast(dict[str, object], protocol["change_family_diagnostic"])
    assert change == {
        "name": "asymmetric_coupled_family_C",
        "declared_separately": True,
        "executed": True,
        "development_data_consumed": True,
        "untouched_held_out": False,
        "ever_promotable": False,
    }
    assert protocol["thresholds"] == []
    assert protocol["output_path"] is None
    for name in (
        "performance_claimed",
        "adaptation_efficacy_claimed",
        "safety_claimed",
        "deployment_authority",
        "evidence_claimed",
        "promotion_authority",
        "scientific_promotion_allowed",
    ):
        assert protocol[name] is False


@pytest.mark.parametrize(
    "change",
    [
        {"seed": -1},
        {"phase_steps": 4},
        {"change_family_steps": 4},
        {"adaptive_base_step_size": 0.2},
        {"adaptive_average_reward_step_size": 0.0},
        {"adaptive_option_step_size": float("nan")},
        {"adaptive_option_model_step_size": 0.0},
        {"discount": 0.99},
    ],
)
def test_config_retuning_and_nonfinite_values_fail_closed(
    change: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        dev.EmbodiedDynamicsAdaptationConfig(**change)  # type: ignore[arg-type]


def test_source_runtime_and_typed_threefry_exogenous_schedule(
    evaluator: dev.EmbodiedDynamicsAdaptationEvaluator,
) -> None:
    source = dev.embodied_dynamics_source_manifest()
    runtime = dev.embodied_dynamics_runtime_identity()
    assert set(source) == {
        "alberta_framework/core/embodied_safety_envelope.py",
        "alberta_framework/core/multi_head_learner.py",
        "alberta_framework/core/oak.py",
        "alberta_framework/core/options.py",
        "alberta_framework/core/optimizers.py",
        "alberta_framework/core/prototype_agent.py",
        "alberta_framework/evaluation/embodied_dynamics_adaptation_development.py",
    }
    assert all(len(value) == 64 for value in source.values())
    assert runtime["jax_default_prng_impl"]

    event = evaluator.common_event(4)
    root = jr.key(evaluator.config.seed, impl="threefry2x32")
    keys = jr.split(jr.fold_in(root, np.uint32(10_004)), 4)
    np.testing.assert_array_equal(
        event.key_words,
        jnp.stack(tuple(jr.key_data(key) for key in keys)),
    )
    assert event.key_words.dtype == jnp.uint32
    assert event.key_words.shape == (4, 2)
    assert int(event.regime_code) == 1
    assert int(event.fault_code) == dev.FAULT_STALE_TELEMETRY
    payload = evaluator.common_schedule_payload()
    assert len(payload) == evaluator.config.total_events
    assert all(
        item["paired_scope"] == "exogenous_dynamics_sensor_fault_latency_only"
        for item in payload
    )


def test_dynamics_invalid_input_is_noop_and_eager_jit_scan_are_exact(
    evaluator: dev.EmbodiedDynamicsAdaptationEvaluator,
) -> None:
    initial = dev.initial_embodied_dynamics_state()
    event = evaluator.common_event(0)
    command = dev._primitive_command(1)
    bad = cast(
        dev.EmbodiedDynamicsState,
        initial.replace(
            joint_position=jnp.asarray((jnp.nan, 0.0), dtype=jnp.float32)
        ),
    )
    rejected = dev.embodied_dynamics_step_kernel(bad, command, event)
    assert not bool(rejected.applied)
    assert dev._tree_bits_equal(rejected.state, bad)

    with jax.disable_jit():
        eager = dev.embodied_dynamics_step_kernel(initial, command, event)
    compiled = jax.jit(dev.embodied_dynamics_step_kernel)(initial, command, event)
    assert dev._tree_bits_equal(eager, compiled)
    parity = dev._dynamics_kernel_parity(evaluator)
    assert parity["single_step_eager_jit_exact"] is True
    assert parity["iterative_scan_float32_parity"] is True
    assert parity["iterative_scan_bit_exact_claimed"] is False
    assert parity["scan_jit_exact"] is True
    assert parity["full_orchestration_host_only"] is True


def test_initial_policy_parameters_match_but_endogenous_rng_is_independent(
    evaluator: dev.EmbodiedDynamicsAdaptationEvaluator,
) -> None:
    snapshot = evaluator.initial_snapshot_payload()
    assert snapshot["initial_learned_parameters_equal"] is True
    assert snapshot["policy_rng_states_independent"] is True


def test_trace_enforces_envelope_command_transition_boundary(
    final_state: dev.EmbodiedDynamicsAdaptationRunState,
) -> None:
    records = _records(final_state)
    assert len(records) == 24
    blocked = [record for record in records if record["action_available"] is False]
    executed = [record for record in records if record["action_available"] is True]
    fallbacks = [record for record in records if record["fallback_used"] is True]
    assert blocked and executed and fallbacks
    for record in blocked:
        assert record["available_command"] is None
        assert record["simulated_command_executed"] is False
        assert record["dynamics_advanced"] is False
        assert record["learner_transition"] is None
        assert record["prototype_update_called"] is False
        assert record["executed_primitive_action"] is None
    for record in executed:
        assert record["available_command"] is not None
        assert record["dynamics_advanced"] is True
        assert record["learner_transition"] is not None
        assert record["prototype_update_called"] is True
        assert record["prototype_update_applied"] is True
        transition = cast(dict[str, object], record["learner_transition"])
        assert transition["action"] == record["executed_primitive_action"]
    changed = [
        record
        for record in fallbacks
        if record["cached_action_replacement_changed_action"] is True
    ]
    assert changed
    for record in changed:
        assert record["cached_action_replacement_attempted"] is True
        assert record["cached_action_replacement_committed"] is True
        assert record["executed_primitive_action"] == 0
    assert all(record["physical_command_dispatched"] is False for record in records)


def test_adaptive_frozen_witness_and_matched_opportunities_are_actual(
    evaluator: dev.EmbodiedDynamicsAdaptationEvaluator,
    final_state: dev.EmbodiedDynamicsAdaptationRunState,
) -> None:
    diagnostics = dev._diagnostics(evaluator, final_state)
    assert diagnostics["common_schedule_paired_exogenous_only"] is True
    assert diagnostics["policy_action_randomness_paired"] is False
    assert diagnostics["adaptive_and_frozen_trajectory_diverged"] is True
    assert diagnostics["initial_learned_parameters_equal"] is True
    assert diagnostics["adaptive_parameters_changed"] is True
    assert diagnostics["frozen_parameters_unchanged"] is True
    assert diagnostics["update_calls_matched"] is True
    assert diagnostics["update_call_opportunities_matched"] is True
    assert diagnostics["update_call_opportunities_per_arm"] == 12
    assert diagnostics["available_update_calls_matched"] is True
    assert diagnostics["safety_availability_diverged_in_this_trace"] is False
    assert diagnostics["adaptive_update_calls"] == diagnostics["frozen_update_calls"]
    assert diagnostics["adaptive_skips"] == diagnostics["frozen_skips"]
    assert diagnostics["no_action_no_command_or_transition_contract"] is True
    assert diagnostics["every_changed_fallback_rebound_public_credit_owner"] is True
    assert diagnostics["a_b_a_observed"] is True
    assert diagnostics["change_family_observed"] is True
    assert diagnostics["untouched_held_out_data"] is False


def test_resealed_composite_state_tamper_fails_exact_causal_validation(
    evaluator: dev.EmbodiedDynamicsAdaptationEvaluator,
    final_state: dev.EmbodiedDynamicsAdaptationRunState,
) -> None:
    changed_dynamics = cast(
        dev.EmbodiedDynamicsState,
        final_state.adaptive.dynamics_state.replace(
            joint_position=final_state.adaptive.dynamics_state.joint_position.at[0].add(
                jnp.asarray(0.01, dtype=jnp.float32)
            )
        ),
    )
    changed_arm = dataclasses.replace(
        final_state.adaptive,
        dynamics_state=changed_dynamics,
    )
    changed = dataclasses.replace(final_state, adaptive=changed_arm, integrity_sha256="")
    resealed = evaluator._seal_state(changed)
    assert evaluator.validate_state(resealed, causal=False)
    assert not evaluator.validate_state(resealed)


def test_records_are_finite_canonical_json_and_hash_chained(
    final_state: dev.EmbodiedDynamicsAdaptationRunState,
) -> None:
    records = _records(final_state)
    encoded = json.dumps(records, allow_nan=False, separators=(",", ":"), sort_keys=True)
    assert encoded
    heads: dict[str, str] = {}
    for record in records:
        body = {name: record[name] for name in record if name != "record_sha256"}
        assert record["record_sha256"] == hashlib.sha256(
            json.dumps(
                body,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        arm = cast(str, record["arm"])
        if arm in heads:
            assert record["causal_parent_sha256"] == heads[arm]
        heads[arm] = record["record_sha256"]
