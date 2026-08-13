# mypy: disable-error-code="arg-type,call-arg,type-var"
"""Unit contracts for consolidated-memory transfer stress evaluation."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.evaluation.consolidated_memory_transfer import (
    CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS,
    CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS,
    CONSOLIDATED_MEMORY_TRANSFER_PROMOTION_AUTHORITY,
    CONSOLIDATED_MEMORY_TRANSFER_SCIENTIFIC_PROMOTION_ALLOWED,
    ConsolidatedMemoryTransferConfig,
    ConsolidatedMemoryTransferEvaluator,
    ConsolidatedMemoryTransferProtocol,
    default_consolidated_memory_transfer_config,
    default_consolidated_memory_transfer_protocol,
    frozen_consolidated_memory_state_sha256,
    frozen_consolidated_memory_transfer_protocol_sha256,
    frozen_consolidated_memory_transfer_run_state_sha256,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def evaluator() -> ConsolidatedMemoryTransferEvaluator:
    return ConsolidatedMemoryTransferEvaluator()


def _tree_equal(left: object, right: object) -> bool:
    return bool(jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, left, right)))


def test_default_config_and_protocol_are_strict_frozen_development_contracts() -> None:
    config = default_consolidated_memory_transfer_config()
    protocol = default_consolidated_memory_transfer_protocol()
    assert config.to_config()["assessment_status"] == "not-assessed"
    assert config.to_config()["performance_thresholds_applied"] is False
    assert config.to_config()["promotion_authority"] is False
    assert len(protocol.events) == 17
    assert tuple(dict.fromkeys(event.phase_id for event in protocol.events)) == (
        "initial",
        "interference",
        "return",
    )
    roles = {event.role for event in protocol.events}
    assert {
        "compatible-recurrence",
        "misleading-probe",
        "semantic-generation-shift",
        "provenance-mismatch",
        "procedural-outcome-shift",
        "eviction-pressure",
        "retained-semantic-return",
        "stale-skill-probe",
        "procedural-recovery",
    } <= roles
    assert protocol.to_config()["query_precedes_write"] is True
    assert protocol.to_config()["regime_labels_visible_to_memory"] is False
    assert len(frozen_consolidated_memory_transfer_protocol_sha256()) == 64
    assert not CONSOLIDATED_MEMORY_TRANSFER_PROMOTION_AUTHORITY
    assert not CONSOLIDATED_MEMORY_TRANSFER_SCIENTIFIC_PROMOTION_ALLOWED
    assert CONSOLIDATED_MEMORY_TRANSFER_DEVELOPMENT_STATUS.endswith("not-assessed")
    assert CONSOLIDATED_MEMORY_TRANSFER_ASSESSMENT_STATUS == "not-assessed"

    reconstructed_config = ConsolidatedMemoryTransferConfig.from_config(config.to_config())
    reconstructed_protocol = ConsolidatedMemoryTransferProtocol.from_config(protocol.to_config())
    assert reconstructed_config == config
    assert reconstructed_protocol == protocol


def test_noncanonical_or_changed_protocol_is_rejected() -> None:
    protocol = default_consolidated_memory_transfer_protocol()
    changed_event = dataclasses.replace(protocol.events[0], expected_target=(9.0, 9.0))
    changed = dataclasses.replace(protocol, events=(changed_event, *protocol.events[1:]))
    with pytest.raises(ValueError, match="frozen v1"):
        ConsolidatedMemoryTransferEvaluator(protocol=changed)

    config_payload = default_consolidated_memory_transfer_config().to_config()
    config_payload = dict(config_payload)
    config_payload["performance_thresholds_applied"] = True
    with pytest.raises(ValueError, match="fixed fields"):
        ConsolidatedMemoryTransferConfig.from_config(config_payload)


def test_empty_source_bound_snapshot_is_immutable_and_matched_arms_are_exact(
    evaluator: ConsolidatedMemoryTransferEvaluator,
) -> None:
    initial = evaluator.initial_memory_state
    initial_sha = frozen_consolidated_memory_state_sha256(initial)
    run = evaluator.advance(evaluator.init(), steps=6)
    assert int(run.event_index) == 6
    assert _tree_equal(run.full_memory_state, run.retrieval_ablation_state)
    assert frozen_consolidated_memory_state_sha256(initial) == initial_sha
    assert int(initial.operation_count) == 0
    assert int(run.full_memory_state.operation_count) == 6
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        run.event_index = jnp.asarray(0, dtype=jnp.int32)


def test_checkpoint_resume_replays_prefix_and_reaches_exact_final_state(
    evaluator: ConsolidatedMemoryTransferEvaluator,
) -> None:
    partial = evaluator.advance(evaluator.init(), steps=8)
    checkpoint = evaluator.checkpoint_payload(partial)
    restored = evaluator.restore_checkpoint(checkpoint)
    assert _tree_equal(partial, restored)
    assert checkpoint["run_state_sha256"] == (
        frozen_consolidated_memory_transfer_run_state_sha256(partial)
    )

    resumed = evaluator.advance(restored, steps=99)
    uninterrupted = evaluator.advance(evaluator.init(), steps=99)
    assert int(resumed.event_index) == 17
    assert int(resumed.full_memory_state.operation_count) == 17
    assert _tree_equal(resumed, uninterrupted)


def test_checkpoint_rejects_position_state_source_runtime_and_replay_tampering(
    evaluator: ConsolidatedMemoryTransferEvaluator,
) -> None:
    partial = evaluator.advance(evaluator.init(), steps=5)
    checkpoint = evaluator.checkpoint_payload(partial)

    position_tamper = dict(checkpoint)
    position_tamper["event_index"] = 6
    with pytest.raises(ValueError, match="run-state SHA|prefix replay"):
        evaluator.restore_checkpoint(position_tamper)

    state_tamper = dict(checkpoint)
    full_payload = dict(cast(Mapping[str, object], state_tamper["full_memory"]))
    full_state = cast(Any, full_payload["state"])
    full_payload["state"] = dataclasses.replace(
        full_state,
        operation_count=full_state.operation_count + jnp.asarray(1, dtype=jnp.int32),
    )
    state_tamper["full_memory"] = full_payload
    with pytest.raises(ValueError, match="state SHA"):
        evaluator.restore_checkpoint(state_tamper)

    source_tamper = copy.deepcopy(checkpoint)
    source_tamper["source_sha256"] = {"tampered.py": "0" * 64}
    with pytest.raises(ValueError, match="binding differs"):
        evaluator.restore_checkpoint(source_tamper)

    runtime_tamper = copy.deepcopy(checkpoint)
    runtime_tamper["runtime_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="binding differs"):
        evaluator.restore_checkpoint(runtime_tamper)


def test_checkpoint_rejects_self_consistent_but_wrong_prefix_state(
    evaluator: ConsolidatedMemoryTransferEvaluator,
) -> None:
    prefix_four = evaluator.advance(evaluator.init(), steps=4)
    prefix_five = evaluator.advance(prefix_four, steps=1)
    checkpoint = evaluator.checkpoint_payload(prefix_five)
    forged = dict(checkpoint)
    bindings = {
        "source_digest": evaluator.config.source_digest,
        "semantic_namespace_digest": evaluator.config.semantic_namespace_digest,
        "representation_revision": evaluator.config.representation_revision,
        "source_revision": evaluator.config.source_revision,
    }
    forged["full_memory"] = evaluator.memory.checkpoint_payload(
        prefix_four.full_memory_state, **bindings
    )
    forged["retrieval_ablation"] = evaluator.memory.checkpoint_payload(
        prefix_four.retrieval_ablation_state, **bindings
    )
    forged_state = dataclasses.replace(prefix_four, event_index=jnp.asarray(5, dtype=jnp.int32))
    forged["run_state_sha256"] = frozen_consolidated_memory_transfer_run_state_sha256(forged_state)
    with pytest.raises(ValueError, match="prefix replay"):
        evaluator.restore_checkpoint(forged)
