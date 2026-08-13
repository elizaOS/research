"""Focused production-state checks for the future-utility state gate."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import CompositionalFeatureLearner
from alberta_framework.evaluation import (
    _compositional_future_utility_calibration_engine as engine,
)
from alberta_framework.evaluation import (
    _compositional_future_utility_state_gate as state_gate,
)
from alberta_framework.evaluation import compositional_control_life_development as control

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def production_executions() -> Iterator[
    dict[str, tuple[engine.FutureUtilityArmSpec, control.CompositionalControlLifeArmExecution]]
]:
    protocol = control.build_short_test_protocol()
    source = control.build_bound_compositional_control_life_source(
        protocol,
        observation_key=jr.key(901),
        exploration_key=jr.key(902),
        random_action_key=jr.key(903),
        learner_key=jr.key(904),
    )
    historical = control.learner_config_for_arm(
        "dovetail_coverage_ancestor_headroom_leftpack"
    )
    arms = (
        engine.FutureUtilityArmSpec(
            name="current_mix0_decay095_none",
            role="short unnormalized production integration",
            mix=0.0,
            trace_decay=0.95,
            normalization="none",
        ),
        engine.FutureUtilityArmSpec(
            name="horizon_mix1_decay883_uncertainty_age",
            role="short normalized production integration",
            mix=1.0,
            trace_decay=0.999215304851532,
            normalization="uncertainty_age",
        ),
    )
    executions: dict[
        str,
        tuple[engine.FutureUtilityArmSpec, control.CompositionalControlLifeArmExecution],
    ] = {}
    for arm in arms:
        learner = CompositionalFeatureLearner.from_config(
            engine.build_future_utility_learner_config(historical, arm)
        )
        execution = control.execute_compositional_control_life_arm(
            protocol,
            learner,
            source.learner_key,
            source.observations,
            source.phase_indices,
            source.exploration_mask,
            source.random_actions,
            composed_readout_enabled=True,
        )
        executions[arm.name] = (arm, execution)
    yield executions


def _replace_state(
    execution: control.CompositionalControlLifeArmExecution,
    *,
    initial: bool,
    field: str,
    value: object,
) -> control.CompositionalControlLifeArmExecution:
    state = execution.initial_state if initial else execution.final_state
    changed = state.replace(**{field: value})  # type: ignore[attr-defined]
    return dataclasses.replace(
        execution,
        **{"initial_state" if initial else "final_state": changed},
    )


def _gate(
    arm: engine.FutureUtilityArmSpec,
    execution: control.CompositionalControlLifeArmExecution,
) -> state_gate.FutureUtilityStateGateReceipt:
    return state_gate.validate_future_utility_state_gate(
        execution,
        future_utility_mix=arm.mix,
        future_utility_trace_decay=arm.trace_decay,
        future_utility_normalization=arm.normalization,
    )


def test_short_real_production_arms_pass_with_exact_bounded_receipts(
    production_executions: dict[
        str,
        tuple[engine.FutureUtilityArmSpec, control.CompositionalControlLifeArmExecution],
    ],
) -> None:
    receipts = {
        name: _gate(arm, execution)
        for name, (arm, execution) in production_executions.items()
    }
    current = receipts["current_mix0_decay095_none"]
    horizon = receipts["horizon_mix1_decay883_uncertainty_age"]

    assert current.normalization_moment_policy == "disabled-exact-zero"
    assert horizon.normalization_moment_policy == "enabled-bounded-endpoint"
    assert current.expected_raw_energy_f32_bits != horizon.expected_raw_energy_f32_bits
    for receipt in receipts.values():
        assert receipt.initial_fields_all_zero
        assert receipt.all_fields_finite
        assert receipt.contribution_mode_zero_marginal_traces
        assert receipt.raw_slots_untouched_by_curation
        assert receipt.raw_energy_bits_exact
        assert receipt.normalization_moment_policy_exact
        assert receipt.utility_event_final_rows_exact
        assert len(receipt.initial_subset_sha256) == 64
        assert len(receipt.final_subset_sha256) == 64
        assert receipt.initial_subset_sha256 != receipt.final_subset_sha256


def test_manifest_hash_and_nonclaims_are_exact_and_authority_free() -> None:
    assert tuple(spec.name for spec in state_gate.STATE_FIELD_MANIFEST) == (
        "utilities",
        "utility_contribution_trace",
        "utility_error_trace",
        "utility_feature_trace",
        "utility_feature_energy_trace",
        "utility_signal_second_moment",
        "task_activity_ema",
        "candidate_utilities",
        "candidate_utility_contribution_trace",
        "candidate_utility_feature_trace",
        "candidate_utility_feature_energy_trace",
        "candidate_utility_signal_second_moment",
    )
    assert len(state_gate.STATE_FIELD_MANIFEST) == 12
    assert state_gate.state_field_manifest_sha256() == (
        state_gate.STATE_FIELD_MANIFEST_SHA256
    )
    assert state_gate.NONCLAIMS == (
        "per-step-contribution-transition-not-proven",
        "candidate-trace-transition-not-proven",
        "mixed-utility-equation-not-proven",
        "normalization-use-in-ranking-not-proven",
        "trace-reset-and-promotion-transfer-not-proven",
    )
    assert state_gate.DEVELOPMENT_ONLY
    assert not state_gate.PANEL_EXECUTION_AUTHORIZED
    assert not state_gate.ROOT_ISSUANCE_AUTHORIZED
    assert not state_gate.RESULT_AUTHORIZED
    assert not state_gate.OUTPUT_WRITES_ALLOWED
    assert not state_gate.EVIDENCE_AUTHORIZED
    assert not state_gate.SCIENTIFIC_PROMOTION_ALLOWED


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("utility_contribution_trace", jnp.zeros((2, 10), dtype=jnp.float32)),
        ("utility_signal_second_moment", jnp.zeros((11,), dtype=jnp.int32)),
        (
            "candidate_utility_feature_energy_trace",
            jnp.full((8,), jnp.nan, dtype=jnp.float32),
        ),
    ),
)
def test_gate_rejects_wrong_shape_dtype_or_finiteness(
    production_executions: dict[
        str,
        tuple[engine.FutureUtilityArmSpec, control.CompositionalControlLifeArmExecution],
    ],
    field: str,
    value: object,
) -> None:
    arm, execution = production_executions["current_mix0_decay095_none"]
    broken = _replace_state(execution, initial=False, field=field, value=value)
    with pytest.raises((TypeError, ValueError), match="shape|dtype|finite"):
        _gate(arm, broken)


def test_gate_rejects_nonzero_genesis_and_contribution_mode_marginal_state(
    production_executions: dict[
        str,
        tuple[engine.FutureUtilityArmSpec, control.CompositionalControlLifeArmExecution],
    ],
) -> None:
    arm, execution = production_executions["current_mix0_decay095_none"]
    genesis = _replace_state(
        execution,
        initial=True,
        field="utilities",
        value=execution.initial_state.utilities.at[0].set(jnp.float32(1.0)),
    )
    with pytest.raises(ValueError, match="genesis"):
        _gate(arm, genesis)

    marginal = _replace_state(
        execution,
        initial=False,
        field="utility_feature_trace",
        value=execution.final_state.utility_feature_trace.at[0].set(jnp.float32(1.0)),
    )
    with pytest.raises(ValueError, match="marginal"):
        _gate(arm, marginal)


def test_gate_rejects_raw_slot_curation_and_raw_energy_bit_drift(
    production_executions: dict[
        str,
        tuple[engine.FutureUtilityArmSpec, control.CompositionalControlLifeArmExecution],
    ],
) -> None:
    arm, execution = production_executions["current_mix0_decay095_none"]
    trace = execution.events.curation_trace
    active_change = np.array(trace.active_change_mask, copy=True)
    root_change = np.array(trace.root_change_mask, copy=True)
    active_change[0, 0] = True
    root_change[0, 0] = True
    broken_trace = trace.replace(  # type: ignore[attr-defined]
        active_change_mask=active_change,
        root_change_mask=root_change,
    )
    broken_events = execution.events._replace(curation_trace=broken_trace)
    with pytest.raises(ValueError, match="raw active"):
        _gate(arm, dataclasses.replace(execution, events=broken_events))

    energy = execution.final_state.utility_feature_energy_trace.at[0].set(
        jnp.nextafter(
            execution.final_state.utility_feature_energy_trace[0],
            jnp.float32(jnp.inf),
        )
    )
    broken_energy = _replace_state(
        execution,
        initial=False,
        field="utility_feature_energy_trace",
        value=energy,
    )
    with pytest.raises(ValueError, match="energy"):
        _gate(arm, broken_energy)


def test_gate_rejects_normalization_moment_policy_drift(
    production_executions: dict[
        str,
        tuple[engine.FutureUtilityArmSpec, control.CompositionalControlLifeArmExecution],
    ],
) -> None:
    current_arm, current_execution = production_executions[
        "current_mix0_decay095_none"
    ]
    enabled_when_disabled = _replace_state(
        current_execution,
        initial=False,
        field="utility_signal_second_moment",
        value=current_execution.final_state.utility_signal_second_moment.at[0].set(
            jnp.float32(1.0)
        ),
    )
    with pytest.raises(ValueError, match="disabled normalization"):
        _gate(current_arm, enabled_when_disabled)

    normalized_arm, normalized_execution = production_executions[
        "horizon_mix1_decay883_uncertainty_age"
    ]
    missing_raw_moment = _replace_state(
        normalized_execution,
        initial=False,
        field="utility_signal_second_moment",
        value=normalized_execution.final_state.utility_signal_second_moment.at[0].set(
            jnp.float32(0.0)
        ),
    )
    with pytest.raises(ValueError, match="raw active.*moment"):
        _gate(normalized_arm, missing_raw_moment)


def test_gate_rejects_utility_event_final_row_mismatch(
    production_executions: dict[
        str,
        tuple[engine.FutureUtilityArmSpec, control.CompositionalControlLifeArmExecution],
    ],
) -> None:
    arm, execution = production_executions["current_mix0_decay095_none"]
    utilities = np.array(execution.events.raw_active_utilities, copy=True)
    utilities[-1, 0] = np.nextafter(utilities[-1, 0], np.float32(np.inf))
    broken_events = execution.events._replace(
        raw_active_utilities=utilities  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="final row"):
        _gate(arm, dataclasses.replace(execution, events=broken_events))


def test_subset_hash_covers_every_declared_field(
    production_executions: dict[
        str,
        tuple[engine.FutureUtilityArmSpec, control.CompositionalControlLifeArmExecution],
    ],
) -> None:
    _arm, execution = production_executions["current_mix0_decay095_none"]
    baseline = state_gate.future_utility_state_subset_sha256(execution.final_state)
    for spec in state_gate.STATE_FIELD_MANIFEST:
        value = getattr(execution.final_state, spec.name)
        changed = value.at[(0,) * value.ndim].set(
            jnp.nextafter(value[(0,) * value.ndim], jnp.float32(jnp.inf))
        )
        state = execution.final_state.replace(  # type: ignore[attr-defined]
            **{spec.name: changed}
        )
        assert state_gate.future_utility_state_subset_sha256(state) != baseline


def test_full_v3_raw_energy_pins_are_frozen() -> None:
    assert state_gate.expected_raw_energy_f32_bits(0.95, 8_998) == 0x419FFFF4
    assert (
        state_gate.expected_raw_energy_f32_bits(0.999215304851532, 8_998)
        == 0x449F2936
    )
