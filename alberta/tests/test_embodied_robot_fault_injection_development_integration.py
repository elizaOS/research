# mypy: disable-error-code="no-untyped-call"
"""Slow exact-replay and checkpoint tests for the WP9 robot fault audit."""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import jax
import pytest

from alberta_framework.evaluation import embodied_robot_fault_injection_development as dev

pytestmark = [pytest.mark.integration, pytest.mark.development, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_module() -> Any:
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def report() -> dev.RobotFaultInjectionReport:
    return dev.run_embodied_robot_fault_injection_development()


@pytest.fixture(scope="module")
def checkpoint_bundle() -> tuple[dict[str, object], dict[str, object]]:
    return dev.make_embodied_robot_fault_injection_checkpoint(dev.FIXED_CHECKPOINT_SPLIT)


def test_report_is_canonical_exact_replay_with_eager_jit_scan_parity(
    report: dev.RobotFaultInjectionReport,
) -> None:
    payload = report.payload()
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    assert report.deterministic_payload_digest in encoded
    assert report.assessment == "not_assessed"
    assert report.summary.event_count == 30
    assert report.summary.physical_dispatch_count == 0
    assert report.summary.learner_reset_count == 0
    assert report.summary.learner_state_mutations == 0
    assert not report.caller_authentication_performed
    assert report.external_caller_authentication_required
    assert report.synthetic_telemetry_audit_schedule
    assert not report.dynamics_simulation_performed
    assert not report.geometry_proof
    assert not report.learner_adaptation_latency_available
    assert report.recovery_delays_are_envelope_action_availability_only
    assert report.simulated_command_execution_is_accounting_only
    assert report.shadow_success_input_is_action_availability_proxy
    assert report.summary.simulated_command_execution_is_accounting_only
    assert report.summary.shadow_success_input_is_action_availability_proxy
    assert report.kernel_parity["single_event_eager_jit_equal"] is True
    assert report.kernel_parity["scan_final_state_equal"] is True
    assert report.kernel_parity["scan_outputs_equal"] is True
    assert report.held_out_change_family.executed is False


def test_external_anchor_json_roundtrip_resumes_exact_report(
    report: dev.RobotFaultInjectionReport,
    checkpoint_bundle: tuple[dict[str, object], dict[str, object]],
) -> None:
    checkpoint, anchor = checkpoint_bundle
    transported_checkpoint = json.loads(json.dumps(checkpoint, allow_nan=False))
    transported_anchor = json.loads(json.dumps(anchor, allow_nan=False))
    resumed = dev.resume_embodied_robot_fault_injection_checkpoint(
        transported_checkpoint,
        transported_anchor,
    )
    assert resumed.payload() == report.payload()


def test_checkpoint_tamper_fails_even_with_outer_reseal(
    checkpoint_bundle: tuple[dict[str, object], dict[str, object]],
) -> None:
    checkpoint, anchor = checkpoint_bundle
    forged = copy.deepcopy(checkpoint)
    trace = forged["trace_prefix"]
    assert isinstance(trace, list)
    first = trace[0]
    assert isinstance(first, dict)
    first["action_available"] = not first["action_available"]
    unsigned = dict(forged)
    unsigned.pop("checkpoint_digest")
    forged["checkpoint_digest"] = dev._digest(unsigned)
    with pytest.raises(ValueError, match="external trust anchor"):
        dev.resume_embodied_robot_fault_injection_checkpoint(forged, anchor)

    forged_anchor = copy.deepcopy(anchor)
    forged_anchor["checkpoint_digest"] = forged["checkpoint_digest"]
    with pytest.raises(ValueError, match="causal prefix replay"):
        dev.resume_embodied_robot_fault_injection_checkpoint(forged, forged_anchor)


def test_anchor_revision_tamper_and_report_reseal_are_rejected(
    report: dev.RobotFaultInjectionReport,
    checkpoint_bundle: tuple[dict[str, object], dict[str, object]],
) -> None:
    checkpoint, anchor = checkpoint_bundle
    stale_anchor = copy.deepcopy(anchor)
    revision = stale_anchor["envelope_revision"]
    assert type(revision) is int
    stale_anchor["envelope_revision"] = revision - 1
    with pytest.raises(ValueError, match="revision differs"):
        dev.resume_embodied_robot_fault_injection_checkpoint(checkpoint, stale_anchor)

    changed = dataclasses.replace(report, assessment="accepted")
    resealed = dataclasses.replace(
        changed,
        deterministic_payload_digest=dev._digest(changed.payload(include_digest=False)),
    )
    errors = dev.validate_embodied_robot_fault_injection_report(resealed)
    assert "assessment must remain not_assessed" in errors


@pytest.mark.parametrize("value", [-1, 31, True, 1.0])
def test_checkpoint_split_requires_strict_in_range_integer(value: object) -> None:
    with pytest.raises(ValueError, match="strict integer"):
        dev.make_embodied_robot_fault_injection_checkpoint(value)  # type: ignore[arg-type]
