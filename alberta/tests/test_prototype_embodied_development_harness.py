# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Whole-agent contracts for the bounded Prototype embodiment harness."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
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
from test_embodied_safety_envelope import _config as _envelope_config
from test_embodied_safety_envelope import _state as _envelope_state
from test_grounded_imagination_composition import REVISION_ONE, SUPPORT, _system
from test_prototype_consolidated_memory import (
    _agent as _procedural_agent,
)
from test_prototype_consolidated_memory import _decision_input, _digest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.prototype_consolidated_semantic_memory import (
    PrototypeConsolidatedSemanticMemoryAgent,
    PrototypeConsolidatedSemanticMemoryConfig,
)
from alberta_framework.core.prototype_embodied_command_adapter import (
    DiscreteEmbodiedPrimitiveCommand,
    PrototypeEmbodiedCommandAdapter,
    PrototypeEmbodiedCommandAdapterConfig,
    PrototypeEmbodiedCommandPreparationInput,
)
from alberta_framework.core.prototype_embodied_development_harness import (
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CHECKPOINT_HOST_ONLY,
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_EVIDENCE_AUTHORITY,
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_GROUNDED_INTERNAL_JIT,
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_JIT_SETTLE_SUPPORTED,
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PHYSICAL_DISPATCH_AUTHORITY,
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PREPARE_HOST_ORCHESTRATED,
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PROMOTION_AUTHORITY,
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SAFETY_AUTHORITY,
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SCIENTIFIC_PROMOTION_ALLOWED,
    PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SHADOW_RESULT_AUTHENTICATED,
    DeterministicPrimitivePlant,
    DeterministicPrimitivePlantConfig,
    PrototypeEmbodiedDevelopmentHarness,
    PrototypeEmbodiedDevelopmentHarnessPreparationInput,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def grounded_system() -> Iterator[tuple[Any, Any, Any]]:
    value = _system()
    yield value
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
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        elif left_array.dtype == jnp.float32:
            left_array = jax.lax.bitcast_convert_type(left_array, jnp.uint32)
            right_array = jax.lax.bitcast_convert_type(right_array, jnp.uint32)
        if not bool(jnp.array_equal(left_array, right_array)):
            return False
    return True


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(array.nbytes)
    return total


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
    return dataclasses.replace(_safe_a(), joint_position=(2.0, 0.3))


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


def _started_semantic() -> tuple[Any, Any]:
    base = _procedural_agent().config
    stomp = dataclasses.replace(base.prototype.oak.stomp, observation_dim=3)
    oak = dataclasses.replace(base.prototype.oak, stomp=stomp)
    composition = dataclasses.replace(
        base,
        prototype=dataclasses.replace(base.prototype, oak=oak),
    )
    semantic = PrototypeConsolidatedSemanticMemoryAgent(
        PrototypeConsolidatedSemanticMemoryConfig(
            composition=composition,
            raw_observation_dim=2,
        )
    )
    initial = semantic.init(
        jr.key(7),
        source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        representation_revision=0,
        source_revision=0,
        lifecycle_id=jnp.asarray((17, 19), dtype=jnp.uint32),
    )
    with jax.disable_jit():
        started = semantic.start(
            initial,
            jnp.zeros((2,), dtype=jnp.float32),
            decision_input=_decision_input(initial.composition),
        ).state
    return semantic, started


@dataclasses.dataclass(frozen=True)
class _Rig:
    harness: PrototypeEmbodiedDevelopmentHarness
    state: Any
    model_state: Any
    selected: int
    fallback: int


def _rig(
    grounded_system: tuple[Any, Any, Any],
    *,
    unsafe_proposal: bool = False,
    max_decisions: int = 8,
    max_plant_transitions: int = 4,
) -> _Rig:
    grounded, grounded_state, model_state = grounded_system
    semantic, semantic_state = _started_semantic()
    selected = int(semantic_state.composition.prototype.current_action)
    fallback = 1 - selected
    bank = [_safe_a(), _safe_b()]
    rewards = [0.25, 0.5]
    if unsafe_proposal:
        bank[selected] = _unsafe()
    envelope_config = _envelope_config(
        max_decisions=max_decisions,
        max_committed_actions=max_decisions,
        **_fallback_overrides(bank[fallback]),
    )
    adapter = PrototypeEmbodiedCommandAdapter(
        PrototypeEmbodiedCommandAdapterConfig(
            semantic=semantic.config,
            envelope=envelope_config,
            command_bank=tuple(bank),
        )
    )
    adapter_state = adapter.init(
        semantic_state,
        _envelope_state(adapter.envelope),
    )
    plant = DeterministicPrimitivePlant(
        DeterministicPrimitivePlantConfig(
            observation_lower=(-10.0, -10.0),
            observation_upper=(10.0, 10.0),
            primitive_deltas=((1.0, 0.0), (0.0, 1.0)),
            primitive_rewards=tuple(rewards),
            max_transitions=max_plant_transitions,
        )
    )
    harness = PrototypeEmbodiedDevelopmentHarness(adapter, plant, grounded)
    state = harness.init(
        adapter_state,
        plant.init(jnp.zeros((2,), dtype=jnp.float32)),
        grounded_state,
    )
    return _Rig(harness, state, model_state, selected, fallback)


def _preparation(
    rig: _Rig,
    *,
    identity: int = 1,
    connected: bool = True,
    emergency_stop: bool = False,
) -> PrototypeEmbodiedDevelopmentHarnessPreparationInput:
    tick_base = 10 * identity
    envelope = PrototypeEmbodiedCommandPreparationInput(
        telemetry=_telemetry(
            identity=identity,
            sample_tick=tick_base,
            connected=connected,
            emergency_stop=emergency_stop,
        ),
        envelope_decision_id=_words(identity),
        envelope_action_id=_words(identity),
        control_tick=_words(tick_base + 2),
        control_deadline_tick=_words(tick_base + 5),
        model_version=MODEL,
        optimizer_version=OPTIMIZER,
        lifecycle_version=LIFECYCLE,
        untrusted_reward=jnp.asarray(7.0, dtype=jnp.float32),
        partner_metadata_digest=PARTNER,
        learned_cost_estimate=jnp.asarray(-1_000.0, dtype=jnp.float32),
    )
    return PrototypeEmbodiedDevelopmentHarnessPreparationInput(
        envelope=envelope,
        model_state=rig.model_state,
        action_support_counts=SUPPORT,
        source_revision_words=REVISION_ONE,
        region_ids=jnp.zeros((1, 2), dtype=jnp.int32),
        safety_admitted=jnp.ones((1, 2), dtype=jnp.bool_),
        protected=jnp.zeros((1, 2), dtype=jnp.bool_),
    )


def _prepared(
    rig: _Rig,
    *,
    state: Any | None = None,
    identity: int = 1,
    connected: bool = True,
    emergency_stop: bool = False,
) -> Any:
    result = rig.harness.prepare(
        rig.state if state is None else state,
        _preparation(
            rig,
            identity=identity,
            connected=connected,
            emergency_stop=emergency_stop,
        ),
    )
    assert bool(result.diagnostics.prepared)
    assert bool(rig.harness.state_valid(result.state))
    return result


def test_accepted_action_has_exact_causal_commit_and_jit_parity(
    grounded_system: tuple[Any, Any, Any],
) -> None:
    rig = _rig(grounded_system)
    prepared = _prepared(rig)
    assert _tree_bits_equal(prepared.state.plant, rig.state.plant)
    assert _tree_bits_equal(prepared.state.shadow, rig.state.shadow)

    envelope = rig.harness.evaluate_pending_envelope(prepared.state)
    assert bool(envelope.transaction_applied)
    assert bool(envelope.action_available)
    assert bool(envelope.proposed_accepted)
    eager = rig.harness.settle(prepared.state, envelope, prepared.shadow)
    compiled = jax.jit(rig.harness.settle)(
        prepared.state,
        envelope,
        prepared.shadow,
    )
    assert _tree_bits_equal(eager, compiled)
    assert bool(eager.diagnostics.transaction_committed)
    assert bool(eager.diagnostics.action_transaction_committed)
    assert bool(eager.diagnostics.plant_proposal_requested)
    assert bool(eager.transition.proposal_applied)
    assert bool(eager.transition.committed)
    assert bool(eager.diagnostics.semantic_transition_requested)
    assert bool(eager.diagnostics.semantic_transition_committed)
    assert bool(eager.diagnostics.semantic_prototype_learning_retained)
    assert bool(eager.diagnostics.semantic_successor_plant_bound)
    assert bool(eager.diagnostics.semantic_successor_rearmed)
    assert not bool(eager.diagnostics.plant_capacity_exhausted_after_commit)
    assert int(eager.action) == rig.selected
    expected = np.asarray(
        rig.harness.plant.config.primitive_deltas[rig.selected],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(eager.state.plant.observation), expected)
    assert int(eager.state.plant.transition_count) == 1
    prototype = eager.state.adapter.semantic.composition.prototype
    owner = eager.state.adapter.semantic.composition.dispatch_owner
    np.testing.assert_array_equal(
        np.asarray(prototype.current_raw_observation[:2]),
        np.asarray(eager.state.plant.observation),
    )
    np.testing.assert_array_equal(np.asarray(prototype.step_words), (0, 1))
    np.testing.assert_array_equal(np.asarray(prototype.oak_state.step_words), (0, 1))
    assert bool(owner.available)
    assert int(owner.selected_action) == int(prototype.current_action)
    assert _tree_bits_equal(owner.prototype_decision_id, prototype.current_decision_id)
    assert _tree_bits_equal(eager.state.shadow, prepared.shadow.state)
    assert not bool(eager.state.pending.available)
    assert bool(eager.state.last_commit.available)
    assert int(eager.state.last_commit.selected_action) == rig.selected
    assert int(eager.state.last_commit.executed_action) == rig.selected
    assert _tree_bits_equal(
        eager.state.last_commit.shadow_result_content_tag,
        prepared.state.pending.shadow_result_content_tag,
    )
    assert bool(eager.state.last_commit.plant_transition.committed)
    assert int(eager.diagnostics.learning_updates_applied_by_real_adapter) == 0
    assert int(eager.diagnostics.prototype_learning_updates_adopted) == 1
    assert not bool(eager.diagnostics.physical_dispatch_authority)
    assert bool(rig.harness.state_valid(eager.state))

    second_prepared = _prepared(rig, state=eager.state, identity=2)
    second_envelope = rig.harness.evaluate_pending_envelope(second_prepared.state)
    assert bool(second_envelope.action_available)
    second = rig.harness.settle(
        second_prepared.state,
        second_envelope,
        second_prepared.shadow,
    )
    assert bool(second.diagnostics.action_transaction_committed)
    assert bool(second.diagnostics.semantic_successor_rearmed)
    assert int(second.state.plant.transition_count) == 2
    second_prototype = second.state.adapter.semantic.composition.prototype
    np.testing.assert_array_equal(
        np.asarray(second_prototype.current_raw_observation[:2]),
        np.asarray(second.state.plant.observation),
    )
    np.testing.assert_array_equal(np.asarray(second_prototype.step_words), (0, 2))
    np.testing.assert_array_equal(
        np.asarray(second_prototype.oak_state.step_words),
        (0, 2),
    )
    assert bool(rig.harness.state_valid(second.state))


def test_certified_fallback_maps_actual_primitive_and_shadow_pairing(
    grounded_system: tuple[Any, Any, Any],
) -> None:
    rig = _rig(grounded_system, unsafe_proposal=True)
    prepared = _prepared(rig)
    envelope = rig.harness.evaluate_pending_envelope(prepared.state)
    assert bool(envelope.transaction_applied)
    assert bool(envelope.action_available)
    assert not bool(envelope.proposed_accepted)
    assert bool(envelope.fallback_used)
    assert bool(envelope.fallback_certified)

    result = rig.harness.settle(prepared.state, envelope, prepared.shadow)
    assert bool(result.diagnostics.action_transaction_committed)
    assert bool(result.diagnostics.semantic_successor_rearmed)
    assert int(result.action) == rig.fallback
    assert int(result.state.last_commit.selected_action) == rig.selected
    assert int(result.state.last_commit.executed_action) == rig.fallback
    assert int(result.state.last_commit.plant_transition.primitive_action) == rig.fallback
    assert _tree_bits_equal(result.state.last_commit.executed_command, envelope.command)
    assert _tree_bits_equal(
        result.state.last_commit.shadow_result_content_tag,
        prepared.state.pending.shadow_result_content_tag,
    )
    expected = np.asarray(
        rig.harness.plant.config.primitive_deltas[rig.fallback],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(result.state.plant.observation), expected)
    np.testing.assert_array_equal(
        np.asarray(
            result.state.adapter.semantic.composition.prototype.current_raw_observation[:2]
        ),
        expected,
    )
    assert bool(rig.harness.state_valid(result.state))


def test_exact_no_action_adopts_only_envelope_even_if_shadow_payload_differs(
    grounded_system: tuple[Any, Any, Any],
) -> None:
    rig = _rig(grounded_system)
    prepared = _prepared(rig, connected=False)
    envelope = rig.harness.evaluate_pending_envelope(prepared.state)
    assert bool(envelope.transaction_applied)
    assert not bool(envelope.action_available)
    tampered_shadow = prepared.shadow.replace(
        diagnostics=prepared.shadow.diagnostics.replace(
            real_environment_authenticated=jnp.asarray(True, dtype=jnp.bool_)
        )
    )
    result = rig.harness.settle(prepared.state, envelope, tampered_shadow)

    assert bool(result.diagnostics.envelope_only_transaction_committed)
    assert bool(result.diagnostics.transaction_committed)
    assert not bool(result.diagnostics.action_transaction_committed)
    assert not bool(result.transition.requested)
    assert not bool(result.transition.proposal_applied)
    assert not bool(result.transition.committed)
    assert not bool(result.diagnostics.semantic_transition_requested)
    assert not bool(result.diagnostics.semantic_transition_committed)
    assert int(result.diagnostics.prototype_learning_updates_adopted) == 0
    assert int(result.action) == -1
    assert _tree_bits_equal(result.state.plant, prepared.state.plant)
    assert _tree_bits_equal(result.state.shadow, prepared.state.shadow)
    assert _tree_bits_equal(
        result.state.adapter.semantic,
        prepared.state.adapter.semantic,
    )
    assert not bool(result.state.pending.available)
    assert bool(result.diagnostics.no_action_plant_unchanged)
    assert bool(result.diagnostics.no_action_shadow_unchanged)
    assert not bool(result.state.last_commit.available)

    retry = _prepared(rig, state=result.state, identity=2)
    retry_envelope = rig.harness.evaluate_pending_envelope(retry.state)
    retried = rig.harness.settle(retry.state, retry_envelope, retry.shadow)
    assert bool(retried.diagnostics.action_transaction_committed)
    assert bool(retried.diagnostics.semantic_successor_rearmed)
    assert int(retried.state.plant.transition_count) == 1


def test_stop_only_after_capacity_exhaustion_latches_without_plant_or_shadow(
    grounded_system: tuple[Any, Any, Any],
) -> None:
    rig = _rig(grounded_system, max_decisions=1)
    first = _prepared(rig, connected=False)
    first_envelope = rig.harness.evaluate_pending_envelope(first.state)
    first_settlement = rig.harness.settle(first.state, first_envelope, first.shadow)
    assert bool(first_settlement.diagnostics.envelope_only_transaction_committed)
    assert int(first_settlement.state.adapter.envelope.decision_count) == 1

    stopped = _prepared(
        rig,
        state=first_settlement.state,
        identity=2,
        emergency_stop=True,
    )
    stopped_envelope = rig.harness.evaluate_pending_envelope(stopped.state)
    assert not bool(stopped_envelope.decision_capacity_available)
    assert not bool(stopped_envelope.transaction_applied)
    assert bool(stopped_envelope.emergency_stop_latch_applied)
    result = rig.harness.settle(stopped.state, stopped_envelope, stopped.shadow)

    assert bool(result.diagnostics.envelope_only_transaction_committed)
    assert bool(result.diagnostics.emergency_stop_latch_preserved)
    assert bool(result.state.adapter.envelope.emergency_stop_latched)
    assert not bool(result.transition.requested)
    assert not bool(result.transition.proposal_applied)
    assert not bool(result.transition.committed)
    assert not bool(result.diagnostics.semantic_transition_requested)
    assert int(result.diagnostics.prototype_learning_updates_adopted) == 0
    assert _tree_bits_equal(result.state.plant, stopped.state.plant)
    assert _tree_bits_equal(result.state.shadow, stopped.state.shadow)
    assert _tree_bits_equal(
        result.state.adapter.semantic,
        stopped.state.adapter.semantic,
    )


def test_plant_capacity_halts_scheduling_without_inventing_environment_boundary(
    grounded_system: tuple[Any, Any, Any],
) -> None:
    rig = _rig(grounded_system, max_plant_transitions=1)
    prepared = _prepared(rig)
    envelope = rig.harness.evaluate_pending_envelope(prepared.state)
    result = rig.harness.settle(prepared.state, envelope, prepared.shadow)

    assert bool(result.diagnostics.action_transaction_committed)
    assert not bool(result.transition.truncated)
    assert not bool(result.transition.terminated)
    assert float(result.transition.discount) == 1.0
    assert bool(result.diagnostics.semantic_successor_rearmed)
    assert bool(result.diagnostics.plant_capacity_exhausted_after_commit)
    assert int(result.diagnostics.prototype_learning_updates_adopted) == 1
    prototype = result.state.adapter.semantic.composition.prototype
    owner = result.state.adapter.semantic.composition.dispatch_owner
    assert bool(prototype.started)
    assert bool(owner.available)
    np.testing.assert_array_equal(
        np.asarray(prototype.current_raw_observation[:2]),
        np.asarray(result.state.plant.observation),
    )
    np.testing.assert_array_equal(np.asarray(prototype.step_words), (0, 1))
    np.testing.assert_array_equal(np.asarray(prototype.oak_state.step_words), (0, 1))
    assert int(result.state.plant.transition_count) == 1
    assert bool(rig.harness.state_valid(result.state))

    refused = rig.harness.prepare(result.state, _preparation(rig, identity=2))
    assert not bool(refused.diagnostics.prepared)
    assert not bool(refused.diagnostics.plant_capacity_available)
    assert _tree_bits_equal(refused.state, result.state)


def test_shadow_mismatch_outer_rollback_distinguishes_proposal_from_commit_and_replay(
    grounded_system: tuple[Any, Any, Any],
) -> None:
    rig = _rig(grounded_system)
    prepared = _prepared(rig)
    envelope = rig.harness.evaluate_pending_envelope(prepared.state)
    mismatched_shadow = prepared.shadow.replace(
        diagnostics=prepared.shadow.diagnostics.replace(
            scientific_promotion_allowed=jnp.asarray(True, dtype=jnp.bool_)
        )
    )
    rejected_shadow = rig.harness.settle(
        prepared.state,
        envelope,
        mismatched_shadow,
    )
    assert bool(rejected_shadow.transition.requested)
    assert bool(rejected_shadow.transition.proposal_applied)
    assert not bool(rejected_shadow.transition.committed)
    assert bool(rejected_shadow.diagnostics.semantic_transition_requested)
    assert bool(rejected_shadow.diagnostics.semantic_transition_committed)
    assert int(rejected_shadow.diagnostics.prototype_learning_updates_adopted) == 0
    assert not bool(rejected_shadow.diagnostics.transaction_committed)
    assert not bool(rejected_shadow.diagnostics.shadow_result_content_matches_receipt)
    assert _tree_bits_equal(rejected_shadow.state, prepared.state)
    assert bool(rejected_shadow.state.pending.available)

    tampered_envelope = envelope.replace(
        unavailable_reason=envelope.unavailable_reason + jnp.asarray(1, dtype=jnp.int32)
    )
    rejected_envelope = rig.harness.settle(
        prepared.state,
        tampered_envelope,
        prepared.shadow,
    )
    assert not bool(rejected_envelope.diagnostics.envelope_result_exact)
    assert not bool(rejected_envelope.transition.requested)
    assert not bool(rejected_envelope.transition.proposal_applied)
    assert not bool(rejected_envelope.transition.committed)
    assert _tree_bits_equal(rejected_envelope.state, prepared.state)

    committed = rig.harness.settle(prepared.state, envelope, prepared.shadow)
    assert bool(committed.diagnostics.action_transaction_committed)
    replay = rig.harness.settle(committed.state, envelope, prepared.shadow)
    assert not bool(replay.diagnostics.transaction_committed)
    assert not bool(replay.transition.requested)
    assert not bool(replay.transition.proposal_applied)
    assert not bool(replay.transition.committed)
    assert _tree_bits_equal(replay.state, committed.state)


def test_pending_checkpoint_resume_is_exact_and_tamper_fails_closed(
    grounded_system: tuple[Any, Any, Any],
) -> None:
    rig = _rig(grounded_system)
    prepared = _prepared(rig)
    envelope = rig.harness.evaluate_pending_envelope(prepared.state)
    payload = rig.harness.checkpoint_payload(prepared.state)
    adapter_payload = cast(dict[str, Any], payload["adapter"])
    envelope_payload = cast(dict[str, Any], adapter_payload["envelope"])

    restored = rig.harness.restore_checkpoint(
        payload,
        semantic_source_digest=_digest("source"),
        semantic_namespace_digest=_digest("namespace"),
        semantic_representation_revision=0,
        semantic_source_revision=0,
        envelope_source_digest=SOURCE,
        trusted_envelope_state_revision=prepared.state.adapter.envelope.revision,
        trusted_envelope_state_digest=envelope_payload["state_digest"],
        trusted_adapter_state_digest=adapter_payload["state_sha256"],
        trusted_harness_state_digest=payload["state_sha256"],
    )
    assert _tree_bits_equal(restored, prepared.state)
    original = rig.harness.settle(prepared.state, envelope, prepared.shadow)
    resumed = rig.harness.settle(restored, envelope, prepared.shadow)
    assert _tree_bits_equal(original, resumed)

    plant = payload["plant"]
    tampered = {
        **payload,
        "plant": plant.replace(
            observation=plant.observation.at[0].set(
                plant.observation[0] + jnp.asarray(0.5, dtype=jnp.float32)
            )
        ),
    }
    with pytest.raises(ValueError, match="invalid, stale, or tampered"):
        rig.harness.restore_checkpoint(
            tampered,
            semantic_source_digest=_digest("source"),
            semantic_namespace_digest=_digest("namespace"),
            semantic_representation_revision=0,
            semantic_source_revision=0,
            envelope_source_digest=SOURCE,
            trusted_envelope_state_revision=prepared.state.adapter.envelope.revision,
            trusted_envelope_state_digest=envelope_payload["state_digest"],
            trusted_adapter_state_digest=adapter_payload["state_sha256"],
            trusted_harness_state_digest=payload["state_sha256"],
        )


def test_config_resources_authority_and_public_exports_are_exact(
    grounded_system: tuple[Any, Any, Any],
) -> None:
    rig = _rig(grounded_system)
    config = rig.harness.to_config()
    assert config["owned_adapter_states"] == 1
    assert config["owned_plant_states"] == 1
    assert config["owned_grounded_composition_states"] == 1
    assert config["shadow_candidate_state_persisted_while_pending"] is False
    assert config["shadow_result_authenticated"] is False
    assert config["real_successor_transition"] == "one_exact_plant_transition"
    assert config["synthetic_reward_source"] == "plant_primitive_reward"
    assert config["plant_capacity_exhaustion"] == "halt_prepare_with_live_successor"
    assert config["budget_exhaustion_implies_environment_boundary"] is False
    assert config["scientific_promotion_allowed"] is False
    restored = PrototypeEmbodiedDevelopmentHarness.from_config(config)
    assert restored.to_config() == config
    with pytest.raises(ValueError, match="fields"):
        PrototypeEmbodiedDevelopmentHarness.from_config({**config, "extra": True})

    budget = rig.harness.resource_budget(rig.state)
    assert budget.persistent_state_nbytes == _tree_nbytes(rig.state)
    assert budget.pending_receipts == 1
    assert budget.plant_capacity_halts_prepare
    assert budget.budget_exhaustion_environment_boundaries_inferred == 0
    assert budget.logical_shadow_steps_per_prepare == 1
    assert budget.logical_plant_proposals_per_settle == 1
    assert budget.maximum_semantic_transition_calls_per_settle == 1
    assert budget.maximum_prototype_learning_updates_per_settle == 1
    assert budget.no_action_semantic_transition_calls_per_settle == 0
    assert budget.maximum_plant_transitions_per_settle == 1
    assert budget.shadow_recomputations_per_settle == 0
    assert budget.physical_dispatches_per_operation == 0
    assert budget.real_adapter_learning_updates_per_settle == 0
    assert budget.evidence_writes_per_operation == 0
    assert budget.persistent_growth_per_operation_bytes == 0
    assert budget.checkpoint_host_only
    assert budget.prepare_host_orchestrated
    assert budget.grounded_internal_jit
    assert budget.jit_settle_supported
    assert not budget.shadow_result_authenticated
    assert budget.shadow_content_integrity_only
    assert not budget.safety_authority
    assert not budget.evidence_authority
    assert not budget.promotion_authority
    assert not budget.scientific_promotion_allowed

    for module in (alberta, core):
        assert module.PrototypeEmbodiedDevelopmentHarness is (
            PrototypeEmbodiedDevelopmentHarness
        )
        assert module.DeterministicPrimitivePlant is DeterministicPrimitivePlant
    assert PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_CHECKPOINT_HOST_ONLY
    assert PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PREPARE_HOST_ORCHESTRATED
    assert PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_GROUNDED_INTERNAL_JIT
    assert PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_JIT_SETTLE_SUPPORTED
    assert not PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SHADOW_RESULT_AUTHENTICATED
    assert not PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PHYSICAL_DISPATCH_AUTHORITY
    assert not PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SAFETY_AUTHORITY
    assert not PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_EVIDENCE_AUTHORITY
    assert not PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_PROMOTION_AUTHORITY
    assert not PROTOTYPE_EMBODIED_DEVELOPMENT_HARNESS_SCIENTIFIC_PROMOTION_ALLOWED
