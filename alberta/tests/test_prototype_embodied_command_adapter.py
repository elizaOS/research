# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Cross-layer contracts for discrete Prototype commands and envelope settlement."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_embodied_safety_envelope import (
    LIFECYCLE,
    MODEL,
    OPTIMIZER,
    PARTNER,
    SOURCE,
    _telemetry,
    _words,
)
from test_embodied_safety_envelope import (
    _config as _envelope_config,
)
from test_embodied_safety_envelope import (
    _state as _envelope_state,
)
from test_prototype_consolidated_memory import _decision_input, _digest
from test_prototype_consolidated_semantic_memory import (
    _agent as _semantic_agent,
)
from test_prototype_consolidated_semantic_memory import (
    _initial as _semantic_initial,
)

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.prototype_consolidated_semantic_memory import (
    PrototypeConsolidatedSemanticMemoryState,
)
from alberta_framework.core.prototype_embodied_command_adapter import (
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CALLER_AUTHENTICATION,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CHECKPOINT_HOST_ONLY,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_EAGER_SUPPORTED,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_EVIDENCE_AUTHORITY,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_JIT_PREPARE_SUPPORTED,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_JIT_SETTLE_SUPPORTED,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_LEARNING_AUTHORITY,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_PHYSICAL_DISPATCH_AUTHORITY,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_PROMOTION_AUTHORITY,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_SAFETY_AUTHORITY,
    PROTOTYPE_EMBODIED_COMMAND_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED,
    DiscreteEmbodiedPrimitiveCommand,
    PrototypeEmbodiedCommandAdapter,
    PrototypeEmbodiedCommandAdapterConfig,
    PrototypeEmbodiedCommandPreparationInput,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clear_jax_caches() -> Iterator[None]:
    yield
    jax.clear_caches()


def _tree_bits_equal(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree:
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return False
        if left_array.dtype == jnp.float32:
            left_array = jax.lax.bitcast_convert_type(left_array, jnp.uint32)
            right_array = jax.lax.bitcast_convert_type(right_array, jnp.uint32)
        if not bool(jnp.array_equal(left_array, right_array)):
            return False
    return True


def _safe_a() -> DiscreteEmbodiedPrimitiveCommand:
    return DiscreteEmbodiedPrimitiveCommand(
        joint_position=(0.2, 0.3),
        joint_velocity=(0.1, 0.2),
        joint_torque=(0.3, 0.4),
        workspace_position=(0.2, 0.2, 1.0),
        collision_clearance=0.4,
    )


def _safe_b() -> DiscreteEmbodiedPrimitiveCommand:
    return DiscreteEmbodiedPrimitiveCommand(
        joint_position=(-0.2, -0.3),
        joint_velocity=(-0.1, -0.2),
        joint_torque=(-0.3, -0.4),
        workspace_position=(-0.2, -0.2, 1.0),
        collision_clearance=0.6,
    )


def _unsafe() -> DiscreteEmbodiedPrimitiveCommand:
    return DiscreteEmbodiedPrimitiveCommand(
        joint_position=(2.0, 0.3),
        joint_velocity=(0.1, 0.2),
        joint_torque=(0.3, 0.4),
        workspace_position=(0.2, 0.2, 1.0),
        collision_clearance=0.4,
    )


def _started_semantic() -> tuple[Any, PrototypeConsolidatedSemanticMemoryState]:
    semantic = _semantic_agent()
    initial = _semantic_initial(semantic)
    with jax.disable_jit():
        state = semantic.start(
            initial,
            jnp.zeros((1,), dtype=jnp.float32),
            decision_input=_decision_input(initial.composition),
        ).state
    return semantic, state


def _fallback_overrides(
    command: DiscreteEmbodiedPrimitiveCommand,
) -> dict[str, object]:
    return {
        "fallback_joint_position": command.joint_position,
        "fallback_joint_velocity": command.joint_velocity,
        "fallback_joint_torque": command.joint_torque,
        "fallback_workspace_position": command.workspace_position,
        "fallback_collision_clearance": command.collision_clearance,
    }


def _adapter_state(
    *,
    unsafe_proposal: bool = False,
    map_fallback: bool = True,
    disconnected: bool = False,
    restrict_fallback: bool = False,
    partial_semantic: bool = False,
    max_decisions: int = 32,
) -> tuple[
    PrototypeEmbodiedCommandAdapter,
    Any,
    PrototypeEmbodiedCommandPreparationInput,
    int,
    int,
]:
    semantic, semantic_state = _started_semantic()
    selected = int(semantic_state.composition.prototype.current_action)
    fallback = 1 - selected

    if restrict_fallback:
        mask = tuple(index == selected for index in range(2))
        with jax.disable_jit():
            restricted = semantic.composition.decide(
                semantic_state.composition,
                decision_input=_decision_input(semantic_state.composition, mask=mask),
            ).state
        semantic_state = PrototypeConsolidatedSemanticMemoryState(
            composition=restricted
        )
        selected = int(restricted.prototype.current_action)
        fallback = 1 - selected

    if partial_semantic:
        composition = semantic_state.composition
        forced = semantic.composition.prototype.replace_cached_primitive_action(
            composition.prototype,
            decision_id=composition.prototype.current_decision_id,
            decision_observation=composition.prototype.current_representation,
            proposed_action=jnp.asarray(fallback, dtype=jnp.int32),
            safety_action_mask=composition.dispatch_owner.hard_safety_action_mask,
        )
        assert bool(forced.committed)
        partial = composition.replace(
            prototype=forced.state,
            dispatch_owner=semantic.composition._dispatch_owner_record(
                available=jnp.asarray(True, dtype=jnp.bool_),
                prototype_decision_id=composition.prototype.current_decision_id,
                selected_action=jnp.asarray(fallback, dtype=jnp.int32),
                hard_safety_action_mask=(
                    composition.dispatch_owner.hard_safety_action_mask
                ),
            ),
        )
        assert bool(semantic.composition.validate_state(partial))
        semantic_state = PrototypeConsolidatedSemanticMemoryState(composition=partial)
        selected, fallback = fallback, selected

    bank = [_safe_a(), _safe_b()]
    if unsafe_proposal:
        bank[selected] = _unsafe()
    fallback_command = bank[fallback]
    envelope_overrides: dict[str, object] = {
        "max_decisions": max_decisions,
        "max_committed_actions": max_decisions,
    }
    if map_fallback:
        envelope_overrides.update(_fallback_overrides(fallback_command))
    envelope_config = _envelope_config(**envelope_overrides)
    config = PrototypeEmbodiedCommandAdapterConfig(
        semantic=semantic.config,
        envelope=envelope_config,
        command_bank=tuple(bank),
    )
    adapter = PrototypeEmbodiedCommandAdapter(config)
    state = adapter.init(semantic_state, _envelope_state(adapter.envelope))
    preparation = PrototypeEmbodiedCommandPreparationInput(
        telemetry=_telemetry(connected=not disconnected),
        envelope_decision_id=_words(1),
        envelope_action_id=_words(1),
        control_tick=_words(12),
        control_deadline_tick=_words(15),
        model_version=MODEL,
        optimizer_version=OPTIMIZER,
        lifecycle_version=LIFECYCLE,
        untrusted_reward=jnp.asarray(7.0, dtype=jnp.float32),
        partner_metadata_digest=PARTNER,
        learned_cost_estimate=jnp.asarray(-1_000.0, dtype=jnp.float32),
    )
    return adapter, state, preparation, selected, fallback


def test_fixed_bank_has_unique_bit_exact_payload_identity_without_geometry_claim() -> None:
    semantic, _ = _started_semantic()
    duplicate = _safe_a()
    with pytest.raises(ValueError, match="unique bit-exact"):
        PrototypeEmbodiedCommandAdapterConfig(
            semantic=semantic.config,
            envelope=_envelope_config(),
            command_bank=(duplicate, duplicate),
        )

    positive_zero = dataclasses.replace(_safe_a(), collision_clearance=0.0)
    negative_zero = dataclasses.replace(_safe_a(), collision_clearance=-0.0)
    config = PrototypeEmbodiedCommandAdapterConfig(
        semantic=semantic.config,
        envelope=_envelope_config(),
        command_bank=(positive_zero, negative_zero),
    )
    assert config.command_bank[0].float32_identity() != config.command_bank[1].float32_identity()
    assert config.to_config()["command_geometry_certificate"] is False
    assert PrototypeEmbodiedCommandAdapterConfig.from_config(config.to_config()) == config


def test_prepare_binds_current_owner_mask_envelope_request_digest_and_clock() -> None:
    adapter, state, preparation, selected, _ = _adapter_state()
    eager = adapter.prepare(state, preparation)
    compiled = jax.jit(adapter.prepare)(state, preparation)
    assert _tree_bits_equal(eager, compiled)
    assert bool(eager.diagnostics.prepared)
    assert bool(adapter.state_valid(eager.state))
    assert tuple(np.asarray(eager.receipt_words)) == (0, 1)
    pending = eager.state.pending
    owner = state.semantic.composition.dispatch_owner
    assert int(pending.selected_action) == selected
    assert _tree_bits_equal(pending.prototype_decision_id, owner.prototype_decision_id)
    assert _tree_bits_equal(
        pending.hard_safety_action_mask,
        owner.hard_safety_action_mask,
    )
    assert _tree_bits_equal(pending.envelope_source_checksum, state.envelope.state_checksum)
    assert _tree_bits_equal(pending.envelope_decision_id, preparation.envelope_decision_id)
    assert _tree_bits_equal(pending.telemetry.telemetry_id, preparation.telemetry.telemetry_id)
    assert _tree_bits_equal(pending.model_version, MODEL)
    assert _tree_bits_equal(pending.adapter_config_digest, adapter.config_digest)

    duplicate = adapter.prepare(eager.state, preparation)
    assert not bool(duplicate.diagnostics.prepared)
    assert _tree_bits_equal(duplicate.state, eager.state)

    exhausted_clock = adapter._with_binding_checksum(
        state.replace(
            receipt_clock_words=jnp.full((2,), 0xFFFFFFFF, dtype=jnp.uint32),
            binding_checksum=jnp.zeros((2,), dtype=jnp.uint32),
        )
    )
    assert bool(adapter.state_valid(exhausted_clock))
    exhausted = adapter.prepare(exhausted_clock, preparation)
    assert not bool(exhausted.diagnostics.receipt_clock_available)
    assert not bool(exhausted.diagnostics.prepared)
    assert _tree_bits_equal(exhausted.state, exhausted_clock)


def test_real_proposed_accepted_result_consumes_exact_receipt_and_replay_is_noop() -> None:
    adapter, state, preparation, selected, _ = _adapter_state()
    prepared = adapter.prepare(state, preparation).state
    envelope_result = adapter.evaluate_pending(prepared)
    assert bool(envelope_result.transaction_applied)
    assert bool(envelope_result.proposed_accepted)
    assert not bool(envelope_result.fallback_used)

    eager = adapter.settle(prepared, envelope_result)
    compiled = jax.jit(adapter.settle)(prepared, envelope_result)
    assert _tree_bits_equal(eager, compiled)
    assert bool(eager.diagnostics.transaction_committed)
    assert bool(eager.diagnostics.receipt_consumed)
    assert int(eager.diagnostics.mapped_action) == selected
    assert bool(eager.diagnostics.mapped_action_matches_selected_proposal)
    assert int(eager.action) == selected
    assert not bool(eager.state.pending.available)
    assert _tree_bits_equal(eager.state.envelope, envelope_result.state)
    assert _tree_bits_equal(eager.state.semantic, prepared.semantic)

    replay = adapter.settle(eager.state, envelope_result)
    assert not bool(replay.diagnostics.transaction_committed)
    assert _tree_bits_equal(replay.state, eager.state)
    duplicate_prepare = adapter.prepare(eager.state, preparation)
    assert not bool(duplicate_prepare.diagnostics.prepared)
    assert _tree_bits_equal(duplicate_prepare.state, eager.state)


def test_real_certified_fallback_maps_bit_exactly_and_settles_semantic_owner() -> None:
    adapter, state, preparation, selected, fallback = _adapter_state(
        unsafe_proposal=True,
        map_fallback=True,
    )
    prepared = adapter.prepare(state, preparation).state
    envelope_result = adapter.evaluate_pending(prepared)
    assert bool(envelope_result.transaction_applied)
    assert bool(envelope_result.action_available)
    assert not bool(envelope_result.proposed_accepted)
    assert bool(envelope_result.fallback_used)
    assert _tree_bits_equal(envelope_result.command, adapter.envelope.fallback_command)

    settled = adapter.settle(prepared, envelope_result)
    diagnostics = settled.diagnostics
    assert bool(diagnostics.envelope_result_exact)
    assert int(diagnostics.command_match_count) == 1
    assert int(diagnostics.mapped_action) == fallback
    assert fallback != selected
    assert bool(diagnostics.mapped_action_admitted_by_bound_mask)
    assert bool(diagnostics.semantic_settlement_committed)
    assert bool(diagnostics.receipt_consumed)
    assert int(settled.action) == fallback
    assert int(settled.state.semantic.composition.prototype.current_action) == fallback
    assert int(settled.state.semantic.composition.dispatch_owner.selected_action) == fallback
    assert _tree_bits_equal(settled.state.envelope, envelope_result.state)
    assert not bool(diagnostics.learning_applied)
    assert not bool(diagnostics.evidence_written)
    assert not bool(diagnostics.random_generator_consumed)


def test_unmapped_or_mask_disallowed_real_fallback_is_whole_state_noop() -> None:
    adapter, state, preparation, _, _ = _adapter_state(
        unsafe_proposal=True,
        map_fallback=False,
    )
    prepared = adapter.prepare(state, preparation).state
    envelope_result = adapter.evaluate_pending(prepared)
    assert bool(envelope_result.fallback_used)
    unmapped = adapter.settle(prepared, envelope_result)
    assert bool(unmapped.diagnostics.envelope_result_exact)
    assert int(unmapped.diagnostics.command_match_count) == 0
    assert not bool(unmapped.diagnostics.transaction_committed)
    assert not bool(unmapped.diagnostics.receipt_consumed)
    assert not bool(unmapped.semantic.composition.diagnostics.transaction_committed)
    assert _tree_bits_equal(unmapped.semantic.state, prepared.semantic)
    assert _tree_bits_equal(unmapped.state, prepared)

    restricted, restricted_state, restricted_input, _, fallback = _adapter_state(
        unsafe_proposal=True,
        map_fallback=True,
        restrict_fallback=True,
    )
    restricted_prepared = restricted.prepare(restricted_state, restricted_input).state
    restricted_result = restricted.evaluate_pending(restricted_prepared)
    assert bool(restricted_result.fallback_used)
    disallowed = restricted.settle(restricted_prepared, restricted_result)
    assert int(disallowed.diagnostics.mapped_action) == fallback
    assert not bool(disallowed.diagnostics.mapped_action_admitted_by_bound_mask)
    assert not bool(disallowed.diagnostics.semantic_settlement_committed)
    assert not bool(disallowed.diagnostics.transaction_committed)
    assert not bool(disallowed.semantic.composition.diagnostics.transaction_committed)
    assert _tree_bits_equal(
        disallowed.semantic.state,
        restricted_prepared.semantic,
    )
    assert _tree_bits_equal(disallowed.state, restricted_prepared)


def test_no_action_adopts_envelope_log_and_preserves_semantic_owner_for_fresh_attempt() -> None:
    adapter, state, preparation, _, _ = _adapter_state(disconnected=True)
    prepared = adapter.prepare(state, preparation).state
    envelope_result = adapter.evaluate_pending(prepared)
    assert bool(envelope_result.transaction_applied)
    assert not bool(envelope_result.action_available)
    settled = adapter.settle(prepared, envelope_result)
    assert bool(settled.diagnostics.transaction_committed)
    assert bool(settled.diagnostics.semantic_owner_retry_preserved)
    assert bool(settled.diagnostics.envelope_only_state_committed)
    assert bool(settled.diagnostics.attempt_receipt_closed)
    assert not bool(settled.diagnostics.receipt_consumed)
    assert int(settled.action) == -1
    assert _tree_bits_equal(settled.state.semantic, prepared.semantic)
    assert _tree_bits_equal(settled.state.envelope, envelope_result.state)
    assert not _tree_bits_equal(settled.state.envelope, prepared.envelope)
    assert not bool(settled.state.pending.available)
    assert not bool(settled.state.has_settled_prototype_decision)

    stale_retry = adapter.prepare(settled.state, preparation)
    assert not bool(stale_retry.diagnostics.prepared)
    fresh_input = preparation.replace(
        telemetry=_telemetry(identity=2, sample_tick=20),
        envelope_decision_id=_words(2),
        envelope_action_id=_words(2),
        control_tick=_words(22),
        control_deadline_tick=_words(25),
    )
    fresh_retry = adapter.prepare(settled.state, fresh_input)
    assert bool(fresh_retry.diagnostics.prepared)
    assert tuple(np.asarray(fresh_retry.receipt_words)) == (0, 2)
    assert _tree_bits_equal(
        fresh_retry.state.pending.prototype_decision_id,
        prepared.pending.prototype_decision_id,
    )


def test_exact_stop_only_result_persists_latch_after_ordinary_capacity_exhaustion() -> None:
    adapter, state, preparation, _, _ = _adapter_state(
        disconnected=True,
        max_decisions=1,
    )
    first_prepared = adapter.prepare(state, preparation).state
    first_result = adapter.evaluate_pending(first_prepared)
    first = adapter.settle(first_prepared, first_result)
    assert bool(first.diagnostics.envelope_only_state_committed)
    assert int(first.state.envelope.decision_count) == 1
    semantic_before_stop = first.state.semantic

    stop_input = preparation.replace(
        telemetry=_telemetry(
            identity=2,
            sample_tick=20,
            emergency_stop=True,
        ),
        envelope_decision_id=_words(2),
        envelope_action_id=_words(2),
        control_tick=_words(22),
        control_deadline_tick=_words(25),
    )
    stop_prepared = adapter.prepare(first.state, stop_input).state
    stop_result = adapter.evaluate_pending(stop_prepared)
    assert not bool(stop_result.decision_capacity_available)
    assert not bool(stop_result.transaction_applied)
    assert bool(stop_result.emergency_stop_latch_applied)
    assert bool(stop_result.state.emergency_stop_latched)

    stopped = adapter.settle(stop_prepared, stop_result)
    assert bool(stopped.diagnostics.transaction_committed)
    assert bool(stopped.diagnostics.envelope_only_state_committed)
    assert bool(stopped.diagnostics.stop_only_latch_committed)
    assert bool(stopped.diagnostics.semantic_owner_retry_preserved)
    assert bool(stopped.diagnostics.attempt_receipt_closed)
    assert not bool(stopped.diagnostics.receipt_consumed)
    assert int(stopped.action) == -1
    assert _tree_bits_equal(stopped.state.semantic, semantic_before_stop)
    assert _tree_bits_equal(stopped.state.envelope, stop_result.state)
    assert bool(stopped.state.envelope.emergency_stop_latched)
    assert int(stopped.state.envelope.emergency_stop_latch_count) == 1
    assert int(stopped.state.envelope.revision) == int(first.state.envelope.revision) + 1
    assert int(stopped.state.envelope.decision_count) == 1
    assert not bool(stopped.state.pending.available)
    assert not bool(stopped.state.has_settled_prototype_decision)
    assert not bool(stopped.diagnostics.learning_applied)
    assert not bool(stopped.diagnostics.physical_dispatch_authority)


def test_corrupt_stale_partial_and_tampered_results_are_whole_state_noops() -> None:
    adapter, state, preparation, _, _ = _adapter_state()
    prepared = adapter.prepare(state, preparation).state
    envelope_result = adapter.evaluate_pending(prepared)
    tampered_command = envelope_result.command.replace(
        joint_position=envelope_result.command.joint_position.at[0].add(
            jnp.asarray(0.25, dtype=jnp.float32)
        )
    )
    tampered = envelope_result.replace(command=tampered_command)
    rejected = adapter.settle(prepared, tampered)
    assert not bool(rejected.diagnostics.envelope_result_exact)
    assert not bool(rejected.diagnostics.transaction_committed)
    assert _tree_bits_equal(rejected.state, prepared)

    corrupt_pending = prepared.replace(
        pending=prepared.pending.replace(
            checksum=prepared.pending.checksum.at[0].add(jnp.uint32(1))
        )
    )
    corrupt = adapter.settle(corrupt_pending, envelope_result)
    assert not bool(corrupt.diagnostics.source_state_valid)
    assert not bool(corrupt.diagnostics.transaction_committed)
    assert _tree_bits_equal(corrupt.state, corrupt_pending)

    stale_input = preparation.replace(envelope_decision_id=jnp.zeros((2,), dtype=jnp.uint32))
    stale_prepare = adapter.prepare(state, stale_input)
    assert not bool(stale_prepare.diagnostics.prepared)
    assert _tree_bits_equal(stale_prepare.state, state)

    partial_adapter, partial_state, partial_input, _, _ = _adapter_state(
        unsafe_proposal=True,
        map_fallback=True,
        partial_semantic=True,
    )
    partial_prepared = partial_adapter.prepare(partial_state, partial_input).state
    partial_result = partial_adapter.evaluate_pending(partial_prepared)
    assert bool(partial_result.fallback_used)
    partial = partial_adapter.settle(partial_prepared, partial_result)
    assert bool(
        partial.semantic.composition.diagnostics.procedural_cancellation_required
    )
    assert not bool(
        partial.semantic.composition.diagnostics.procedural_cancellation_applied
    )
    assert not bool(partial.diagnostics.semantic_settlement_committed)
    assert not bool(partial.diagnostics.transaction_committed)
    assert _tree_bits_equal(partial.state, partial_prepared)


def test_checkpoint_resource_execution_mode_authority_and_public_exports() -> None:
    adapter, state, preparation, _, _ = _adapter_state()
    prepared = adapter.prepare(state, preparation).state
    payload = adapter.checkpoint_payload(prepared)
    restored = adapter.restore_checkpoint(
        payload,
        semantic_source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        semantic_representation_revision=0,
        semantic_source_revision=0,
        envelope_source_digest=SOURCE,
        trusted_envelope_state_revision=prepared.envelope.revision,
        trusted_envelope_state_digest=payload["envelope"]["state_digest"],
        trusted_adapter_state_digest=payload["state_sha256"],
    )
    assert _tree_bits_equal(restored, prepared)
    assert bool(adapter.state_valid(restored))

    tampered_payload = dict(payload)
    tampered_payload["receipt_clock_words"] = jnp.asarray((0, 2), dtype=jnp.uint32)
    with pytest.raises(ValueError, match="invalid, stale, or tampered"):
        adapter.restore_checkpoint(
            tampered_payload,
            semantic_source_digest=_digest("source"),
            semantic_namespace_digest=_digest("namespace"),
            semantic_representation_revision=0,
            semantic_source_revision=0,
            envelope_source_digest=SOURCE,
            trusted_envelope_state_revision=prepared.envelope.revision,
            trusted_envelope_state_digest=payload["envelope"]["state_digest"],
            trusted_adapter_state_digest=payload["state_sha256"],
        )

    budget = adapter.resource_budget(prepared)
    assert budget.persistent_state_nbytes > budget.pending_receipt_nbytes > 0
    assert budget.static_command_bank_nbytes == 2 * (3 * 2 + 3 + 1) * 4
    assert budget.maximum_pending_receipts == 1
    assert budget.envelope_recomputations_per_settlement == 1
    assert budget.envelope_state_commits_per_exact_no_action == 1
    assert budget.stop_latch_preservations_per_exact_stop_only_result == 1
    assert budget.semantic_settlement_delegations_per_settlement == 1
    assert budget.physical_dispatches_per_operation == 0
    assert budget.learning_state_mutations_per_operation == 0
    assert budget.evidence_writes_per_operation == 0
    assert budget.random_generator_calls_per_operation == 0
    assert budget.checkpoint_host_only is True
    assert budget.eager_prepare_and_settle is True
    assert budget.jit_prepare_and_settle is True
    assert budget.command_geometry_certificate is False

    constants = (
        PROTOTYPE_EMBODIED_COMMAND_ADAPTER_PHYSICAL_DISPATCH_AUTHORITY,
        PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CALLER_AUTHENTICATION,
        PROTOTYPE_EMBODIED_COMMAND_ADAPTER_LEARNING_AUTHORITY,
        PROTOTYPE_EMBODIED_COMMAND_ADAPTER_EVIDENCE_AUTHORITY,
        PROTOTYPE_EMBODIED_COMMAND_ADAPTER_SAFETY_AUTHORITY,
        PROTOTYPE_EMBODIED_COMMAND_ADAPTER_PROMOTION_AUTHORITY,
        PROTOTYPE_EMBODIED_COMMAND_ADAPTER_SCIENTIFIC_PROMOTION_ALLOWED,
    )
    assert constants == (False,) * len(constants)
    assert PROTOTYPE_EMBODIED_COMMAND_ADAPTER_CHECKPOINT_HOST_ONLY is True
    assert PROTOTYPE_EMBODIED_COMMAND_ADAPTER_EAGER_SUPPORTED is True
    assert PROTOTYPE_EMBODIED_COMMAND_ADAPTER_JIT_PREPARE_SUPPORTED is True
    assert PROTOTYPE_EMBODIED_COMMAND_ADAPTER_JIT_SETTLE_SUPPORTED is True
    for namespace in (core, alberta):
        assert namespace.PrototypeEmbodiedCommandAdapter is PrototypeEmbodiedCommandAdapter
        assert namespace.DiscreteEmbodiedPrimitiveCommand is DiscreteEmbodiedPrimitiveCommand
