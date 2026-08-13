"""Public execution-analysis boundary for compositional control lives."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Mapping
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import CompositionalFeatureLearner
from alberta_framework.evaluation import (
    _compositional_future_utility_calibration_engine as engine,
)
from alberta_framework.evaluation import compositional_control_life_development as control
from alberta_framework.evaluation.generated_birth_identity_scrub_epoch import (
    generated_birth_identity_scrub_epoch_core_state_sha256,
)

pytestmark = pytest.mark.integration

_SOURCE_ARM = "dovetail_coverage_ancestor_headroom_leftpack"


@pytest.fixture(scope="module")
def future_utility_execution() -> Iterator[
    tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ]
]:
    protocol = control.build_short_test_protocol()
    source = control.build_bound_compositional_control_life_source(
        protocol,
        observation_key=jr.key(1_901),
        exploration_key=jr.key(1_902),
        random_action_key=jr.key(1_903),
        learner_key=jr.key(1_904),
    )
    arm = engine.FutureUtilityArmSpec(
        name="future_mix1_decay095_none",
        role="public arm-analysis integration fixture",
        mix=1.0,
        trace_decay=0.95,
        normalization="none",
    )
    learner = CompositionalFeatureLearner.from_config(
        engine.build_future_utility_learner_config(
            control.learner_config_for_arm(_SOURCE_ARM),
            arm,
        )
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
    yield protocol, source, execution


def test_real_future_utility_execution_has_exact_public_receipts(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution
    receipt = control.validate_compositional_control_life_arm_execution(
        protocol,
        execution,
        pinned_curation_due_mask=source.curation_due_mask,
    )
    analysis = control.analyze_compositional_control_life_arm_execution(
        protocol,
        execution,
        curation_geometry_arm_name=_SOURCE_ARM,
        pinned_curation_due_mask=source.curation_due_mask,
    )

    assert receipt.to_config() == {
        "schema": control.ARM_EXECUTION_RECEIPT_SCHEMA,
        "total_steps": protocol.total_steps,
        "initial_state_sha256": execution.initial_state_sha256,
        "final_state_sha256": execution.final_state_sha256,
        "trace_sha256": execution.trace_sha256,
        "expected_persistent_state_nbytes": 2_072,
        "initial_persistent_state_nbytes": 2_072,
        "final_persistent_state_nbytes": 2_072,
        "final_step_count": protocol.total_steps,
        "final_step_words_uint32": [0, protocol.total_steps],
        "final_replacement_phase": protocol.total_steps % control.CURATION_INTERVAL,
        "initial_state_finite": True,
        "final_state_finite": True,
        "all_lifetime_counters_valid": True,
        "all_lifetime_capacity_available": True,
        "all_ranking_contracts_valid": True,
        "all_core_predictions_match_full_q": True,
        "initial_target_signature_counts_zero": True,
        "scientific_promotion_allowed": False,
        "evidence_authorized": False,
        "output_writes_allowed": False,
    }
    payload = cast(dict[str, Any], analysis.to_config())
    assert payload["schema"] == control.ARM_ANALYSIS_RECEIPT_SCHEMA
    assert payload["curation_geometry_arm_name"] == _SOURCE_ARM
    assert payload["execution_receipt"] == receipt.to_config()
    assert list(payload["curation_totals"]) == list(control.CURATION_COUNT_NAMES)
    assert payload["curation_totals"]["curation_due"] == protocol.total_steps // 32
    assert set(payload["active_structural_trajectories"]) == set(
        control.SIGNATURE_NAMES
    )
    assert set(payload["candidate_structural_trajectories"]) == set(
        control.SIGNATURE_NAMES
    )
    assert payload["curation_decision_audit"]["due_curation_event_count"] == (
        protocol.total_steps // 32
    )
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload

    geometry = engine.FutureUtilityEndpointGeometry(
        phase_order=control.PHASE_ORDER,
        phase_lengths=protocol.phase_lengths,
        target_names=("A", "B", "C"),
        curation_interval=control.CURATION_INTERVAL,
    )
    active = cast(
        Mapping[str, Mapping[str, object]],
        analysis.active_structural_trajectories,
    )
    endpoints = engine.build_future_utility_primary_endpoints(
        geometry,
        execution.events,
        active_trajectories={name: active[name] for name in ("A", "B", "C")},
        curation_totals=dict(
            zip(control.CURATION_COUNT_NAMES, analysis.curation_totals, strict=True)
        ),
        curation_audit=analysis.curation_decision_audit,
        pinned_due_mask=source.curation_due_mask,
    )
    margin_passes = cast(Mapping[str, object], endpoints["margin_passes"])
    assert margin_passes["due_curation_event_count"] == (
        protocol.total_steps // 32
    )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        ("initial_state_sha256", "0" * 64, "initial state SHA-256"),
        ("final_state_sha256", "0" * 64, "final state SHA-256"),
        ("trace_sha256", "0" * 64, "event trace SHA-256"),
        ("expected_persistent_state_nbytes", 2_076, "byte accounting"),
    ),
)
def test_execution_validator_rejects_digest_and_byte_tampering(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
    field: str,
    replacement: object,
    match: str,
) -> None:
    protocol, source, execution = future_utility_execution
    broken = dataclasses.replace(cast(Any, execution), **{field: replacement})
    with pytest.raises(ValueError, match=match):
        control.validate_compositional_control_life_arm_execution(
            protocol,
            broken,
            pinned_curation_due_mask=source.curation_due_mask,
        )


def test_execution_validator_rejects_clock_count_and_event_tampering(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution

    final_state = execution.final_state.replace(  # type: ignore[attr-defined]
        step_words=jnp.asarray([0, protocol.total_steps - 1], dtype=jnp.uint32)
    )
    broken_clock = dataclasses.replace(
        execution,
        final_state=final_state,
        final_state_sha256=(
            generated_birth_identity_scrub_epoch_core_state_sha256(final_state)
        ),
    )
    with pytest.raises(ValueError, match="lifetime clocks"):
        control.validate_compositional_control_life_arm_execution(
            protocol,
            broken_clock,
            pinned_curation_due_mask=source.curation_due_mask,
        )

    initial_counts = execution.initial_active_signature_counts.at[0].set(1)
    broken_counts = dataclasses.replace(
        execution,
        initial_active_signature_counts=initial_counts,
    )
    with pytest.raises(ValueError, match="genesis structure"):
        control.validate_compositional_control_life_arm_execution(
            protocol,
            broken_counts,
            pinned_curation_due_mask=source.curation_due_mask,
        )

    lifetime_valid = np.array(execution.events.lifetime_counter_valid, copy=True)
    lifetime_valid[0] = False
    events = execution.events._replace(
        lifetime_counter_valid=cast(Any, lifetime_valid)
    )
    broken_events = dataclasses.replace(
        execution,
        events=events,
        trace_sha256=control._array_tree_sha256(events),
    )
    with pytest.raises(ValueError, match="all_lifetime_counters_valid"):
        control.validate_compositional_control_life_arm_execution(
            protocol,
            broken_events,
            pinned_curation_due_mask=source.curation_due_mask,
        )


def test_analysis_binds_curation_geometry_and_returns_fresh_payloads(
    future_utility_execution: tuple[
        control.CompositionalControlLifeProtocol,
        control.BoundCompositionalControlLifeSource,
        control.CompositionalControlLifeArmExecution,
    ],
) -> None:
    protocol, source, execution = future_utility_execution
    with pytest.raises(ValueError, match="curation_geometry_arm_name"):
        control.analyze_compositional_control_life_arm_execution(
            protocol,
            execution,
            curation_geometry_arm_name="not-declared",
            pinned_curation_due_mask=source.curation_due_mask,
        )
    with pytest.raises(ValueError, match="headroom|left-pack"):
        control.analyze_compositional_control_life_arm_execution(
            protocol,
            execution,
            curation_geometry_arm_name="myopic_full",
            pinned_curation_due_mask=source.curation_due_mask,
        )

    analysis = control.analyze_compositional_control_life_arm_execution(
        protocol,
        execution,
        curation_geometry_arm_name=_SOURCE_ARM,
        pinned_curation_due_mask=source.curation_due_mask,
    )
    first = cast(dict[str, Any], analysis.to_config())
    first["active_structural_trajectories"]["A"]["ever_present"] = "tampered"
    first["curation_decision_audit"]["due_curation_records"].clear()
    second = cast(dict[str, Any], analysis.to_config())
    assert second["active_structural_trajectories"]["A"]["ever_present"] != "tampered"
    assert len(second["curation_decision_audit"]["due_curation_records"]) == (
        protocol.total_steps // 32
    )
    with pytest.raises(ValueError, match="authority"):
        dataclasses.replace(analysis, evidence_authorized=True)
