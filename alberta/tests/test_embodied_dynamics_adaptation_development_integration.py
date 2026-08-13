# mypy: disable-error-code="no-untyped-call"
"""Checkpoint, replay, tamper, and report contracts for WP9 dynamics adaptation."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import jax
import pytest

from alberta_framework.evaluation import embodied_dynamics_adaptation_development as dev

pytestmark = [pytest.mark.integration, pytest.mark.development, pytest.mark.slow]


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
def checkpoint(
    evaluator: dev.EmbodiedDynamicsAdaptationEvaluator,
) -> dict[str, object]:
    prefix = evaluator._reconstruct(dev.FIXED_CHECKPOINT_SPLIT)
    return evaluator.checkpoint_payload(prefix)


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return dev.build_embodied_dynamics_adaptation_report()


def test_full_composite_checkpoint_restore_and_resume_are_exact(
    evaluator: dev.EmbodiedDynamicsAdaptationEvaluator,
    checkpoint: dict[str, object],
) -> None:
    transported = json.loads(json.dumps(checkpoint, allow_nan=False))
    restored = evaluator.restore_checkpoint(transported)
    resumed = evaluator.run_to_end(restored)
    uninterrupted = evaluator.run_to_end()

    assert checkpoint["host_only"] is True
    assert checkpoint["full_composite_state"] is True
    assert checkpoint["physical_dispatch_count"] == 0
    assert checkpoint["deployment_authority"] is False
    assert checkpoint["promotion_authority"] is False
    assert evaluator._state_body(resumed) == evaluator._state_body(uninterrupted)
    assert resumed.integrity_sha256 == uninterrupted.integrity_sha256


def test_checkpoint_tamper_fails_even_after_outer_digest_reseal(
    evaluator: dev.EmbodiedDynamicsAdaptationEvaluator,
    checkpoint: dict[str, object],
) -> None:
    forged = copy.deepcopy(checkpoint)
    composite = cast(dict[str, object], forged["composite_state"])
    composite["event_index"] = 0
    body = {name: forged[name] for name in forged if name != "checkpoint_sha256"}
    forged["checkpoint_sha256"] = dev._canonical_sha256(body)
    with pytest.raises(ValueError, match="exact causal prefix"):
        evaluator.restore_checkpoint(forged)

    stale = copy.deepcopy(checkpoint)
    stale["source_manifest_sha256"] = "0" * 64
    body = {name: stale[name] for name in stale if name != "checkpoint_sha256"}
    stale["checkpoint_sha256"] = dev._canonical_sha256(body)
    with pytest.raises(ValueError, match="source/runtime binding"):
        evaluator.restore_checkpoint(stale)


def test_report_reconstructs_every_transition_and_remains_nonassessing(
    report: dict[str, object],
) -> None:
    receipt = dev.validate_embodied_dynamics_adaptation_report(report)

    assert receipt.valid
    assert receipt.assessment_status == "not_assessed"
    assert receipt.source_runtime_bound
    assert receipt.exact_causal_replay
    assert receipt.checkpoint_resume_exact
    assert receipt.output_written is False
    assert receipt.physical_dispatch_count == 0
    assert receipt.deployment_authority is False
    assert receipt.promotion_authority is False
    assert report["output_path"] is None
    assert report["artifact_writer_available"] is False
    assert report["thresholds"] == []
    assert report["adaptation_efficacy_claimed"] is False
    assert report["safety_claimed"] is False
    assert report["physical_safety_certificate"] is False
    assert report["evidence_claimed"] is False


def test_resealed_record_or_binding_tamper_cannot_pass_exact_report_replay(
    report: dict[str, object],
) -> None:
    forged = copy.deepcopy(report)
    records = cast(list[dict[str, object]], forged["records"])
    records[0]["reward"] = 99.0
    body = {
        name: records[0][name]
        for name in records[0]
        if name != "record_sha256"
    }
    records[0]["record_sha256"] = dev._canonical_sha256(body)
    forged["records_sha256"] = dev._canonical_sha256(records)
    report_body = {
        name: forged[name] for name in forged if name != "report_sha256"
    }
    forged["report_sha256"] = dev._canonical_sha256(report_body)
    with pytest.raises(ValueError, match="exact causal replay"):
        dev.validate_embodied_dynamics_adaptation_report(forged)

    forged = copy.deepcopy(report)
    source = cast(dict[str, object], forged["source_manifest"])
    source[next(iter(source))] = "0" * 64
    report_body = {
        name: forged[name] for name in forged if name != "report_sha256"
    }
    forged["report_sha256"] = dev._canonical_sha256(report_body)
    with pytest.raises(ValueError, match="source manifest"):
        dev.validate_embodied_dynamics_adaptation_report(forged)
