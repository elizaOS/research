# mypy: disable-error-code="attr-defined,call-arg,arg-type,no-any-return,operator,type-var"
"""Red-first executed-P lineage contract for the Kondo sparse actor."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_external_learned_state_live_memory_action_stack_adapter import (
    MASK,
)
from test_external_learned_state_live_memory_action_stack_adapter import (
    _adapter as _action_stack_adapter,
)
from test_external_learned_state_live_memory_action_stack_adapter import (
    _feedback as _action_stack_feedback,
)
from test_external_learned_state_live_memory_action_stack_adapter import (
    _start as _action_stack_start,
)
from test_external_learned_state_live_memory_adapter import _event_input, _transition

from alberta_framework.core.kondo_executed_action_lineage_bridge import (
    KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA,
    KONDO_EXECUTED_ACTION_LINEAGE_BRIDGE_SCHEMA,
    KondoExecutedActionCompactAdoptionBatch,
    KondoExecutedActionLineageBridge,
    KondoExecutedActionLineageBridgeConfig,
    KondoExecutedActionLineageResult,
    KondoExecutedActionProposalBatch,
)
from alberta_framework.core.kondo_gate import KondoGateConfig
from alberta_framework.core.kondo_sparse_actor import (
    KondoActorParameters,
    KondoActorProtectedInputs,
    KondoSparseActorConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BATCH_SIZE = 2
FEATURE_DIM = 3


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> object:
    with jax.disable_jit():
        yield


def _actor_config() -> KondoSparseActorConfig:
    return KondoSparseActorConfig(
        feature_dim=FEATURE_DIM,
        hidden_dim=4,
        action_count=2,
        critic_dim=2,
        safety_dim=2,
        learning_rate=0.025,
        gate=KondoGateConfig(
            batch_size=BATCH_SIZE,
            mode="top_k_rate",
            target_rate=1.0,
            price=0.0,
            temperature=0.1,
            max_screenings=100,
        ),
    )


def _parameters(*, output_shift: float = 0.0) -> KondoActorParameters:
    return KondoActorParameters(
        hidden_weight=jnp.asarray(
            (
                (0.20, -0.10, 0.05, 0.30),
                (-0.15, 0.25, 0.10, -0.20),
                (0.05, 0.15, -0.30, 0.10),
            ),
            dtype=jnp.float32,
        ),
        hidden_bias=jnp.asarray((0.01, -0.02, 0.03, 0.04), dtype=jnp.float32),
        output_weight=jnp.asarray(
            (
                (0.30, -0.20),
                (-0.10, 0.25),
                (0.20, 0.05),
                (-0.15, 0.10),
            ),
            dtype=jnp.float32,
        ),
        output_bias=jnp.asarray((0.02 + output_shift, -0.03), dtype=jnp.float32),
    )


def _bridge() -> tuple[KondoExecutedActionLineageBridge, object]:
    action_stack = _action_stack_adapter()
    bridge = KondoExecutedActionLineageBridge(
        KondoExecutedActionLineageBridgeConfig(
            actor=_actor_config(),
            action_stack=action_stack.config,
        )
    )
    state = bridge.actor.init(
        _parameters(),
        jr.key(701, impl="threefry2x32"),
    )
    return bridge, state


def _features() -> jax.Array:
    return jnp.asarray(
        ((1.0, 0.0, 0.5), (0.0, 1.0, -0.5)),
        dtype=jnp.float32,
    )


def _protected() -> KondoActorProtectedInputs:
    baseline = jnp.asarray((0.2, -0.1), dtype=jnp.float32)
    return KondoActorProtectedInputs(
        critic_features=jnp.asarray(((0.1, 0.2), (0.3, 0.4)), dtype=jnp.float32),
        baseline_predictions=baseline,
        return_targets=baseline + jnp.asarray((1.0, -0.5), dtype=jnp.float32),
        safety_features=jnp.asarray(((0.5, 0.6), (0.7, 0.8)), dtype=jnp.float32),
    )


def _tree_exact(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        lhs_array = jnp.asarray(lhs)
        rhs_array = jnp.asarray(rhs)
        if jax.dtypes.issubdtype(lhs_array.dtype, jax.dtypes.prng_key):
            lhs_array = jr.key_data(lhs_array)
            rhs_array = jr.key_data(rhs_array)
        np.testing.assert_array_equal(np.asarray(lhs_array), np.asarray(rhs_array))


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree.leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            array = jr.key_data(array)
        total += int(np.asarray(array).nbytes)
    return total


def _prepared_decisions(
    bridge: KondoExecutedActionLineageBridge,
    *,
    seed_offset: int = 0,
) -> tuple[object, ...]:
    prepared = []
    for row in range(BATCH_SIZE):
        action_stack = bridge.action_stacks[row]
        state = _action_stack_start(action_stack, 710 + seed_offset + row)
        candidate = action_stack.prepare_memory_transition(
            state,
            _transition(
                state.coordinator_state,
                next_observation=(0.4 + 0.1 * row, -0.2),
            ),
            _event_input(provenance=720 + seed_offset + row),
            MASK,
        )
        assert bool(candidate.preparation_valid)
        prepared.append(candidate)
    return tuple(prepared)


def _sample(
    bridge: KondoExecutedActionLineageBridge,
    actor_state: object,
    prepared: tuple[object, ...],
) -> KondoExecutedActionProposalBatch:
    return bridge.sample_proposals(
        actor_state,
        _features(),
        jr.split(jr.key(730, impl="threefry2x32"), BATCH_SIZE),
        action_stack_memory_preparations=prepared,
    )


def _adopt_proposals(
    bridge: KondoExecutedActionLineageBridge,
    prepared: tuple[object, ...],
    proposal: KondoExecutedActionProposalBatch,
    *,
    memory_path: bool = False,
) -> tuple[object, ...]:
    adopted = []
    for row, candidate in enumerate(prepared):
        action_stack = bridge.action_stacks[row]
        memory_prototype = (
            candidate.memory_candidate_state.coordinator_state.inner_state.prototype_state
        )
        if memory_path:
            selected = memory_prototype
            planner_before = candidate.memory_candidate_state.action_binding.memory_action
            consumed = jnp.asarray(False, dtype=jnp.bool_)
        else:
            prototype = action_stack.coordinator.inner.prototype
            replacement = prototype.replace_cached_primitive_action(
                memory_prototype,
                decision_id=memory_prototype.current_decision_id,
                decision_observation=memory_prototype.current_representation,
                proposed_action=proposal.selected_actions[row],
                safety_action_mask=candidate.hard_action_mask,
            )
            assert bool(replacement.committed)
            selected = replacement.state
            planner_before = proposal.selected_actions[row]
            consumed = jnp.asarray(True, dtype=jnp.bool_)
        finalized = action_stack.bind_final_action(
            candidate,
            selected,
            planner_action_before_mask=planner_before,
            planner_candidate_words=proposal.proposal_digest_words[row],
            planner_consumed=consumed,
        )
        assert bool(finalized.finalization_valid)
        receipt = action_stack.integrity_receipt(finalized)
        result = action_stack.adopt_finalized_transition(
            candidate.source_state,
            finalized,
            receipt,
        )
        assert bool(result.diagnostics.transaction_applied)
        adopted.append(result)
    return tuple(adopted)


def _next_preparations(
    bridge: KondoExecutedActionLineageBridge,
    adopted: tuple[object, ...],
) -> tuple[object, ...]:
    prepared = []
    for row, result in enumerate(adopted):
        action_stack = bridge.action_stacks[row]
        state = result.state
        feedback = (
            _action_stack_feedback(state, learn=False)
            if bool(state.action_binding.memory_feedback_required)
            else None
        )
        candidate = action_stack.prepare_memory_transition(
            state,
            _transition(
                state.coordinator_state,
                next_observation=(0.8, -0.4 - 0.1 * row),
            ),
            _event_input(provenance=740 + row),
            MASK,
            feedback,
        )
        assert bool(candidate.preparation_valid)
        prepared.append(candidate)
    return tuple(prepared)


def _valid_fixture() -> tuple[
    KondoExecutedActionLineageBridge,
    object,
    KondoExecutedActionProposalBatch,
    tuple[object, ...],
    tuple[object, ...],
]:
    bridge, state = _bridge()
    prepared = _prepared_decisions(bridge)
    proposal = _sample(bridge, state, prepared)
    adopted = _adopt_proposals(bridge, prepared, proposal)
    next_prepared = _next_preparations(bridge, adopted)
    return bridge, state, proposal, adopted, next_prepared


def test_config_and_proposal_bind_exact_unmasked_actor_snapshot() -> None:
    bridge, state = _bridge()
    prepared = _prepared_decisions(bridge)
    proposal = _sample(bridge, state, prepared)
    payload = bridge.to_config()

    assert payload["schema"] == KONDO_EXECUTED_ACTION_LINEAGE_BRIDGE_SCHEMA
    assert payload["hard_action_mask_support"] == "exact-all-true-only"
    assert payload["proposal_integrity"] == "unkeyed-rederived-source-bound"
    assert payload["compact_pending_adoption_lineage_supported"] is True
    assert payload["compact_pair_preflight_supported"] is True
    assert payload["compact_pending_mutable_owner_snapshots"] == 0
    assert (
        payload["compact_pending_integrity"]
        == "issuer-reconstructed-unkeyed-content-bound"
    )
    assert payload["compact_historic_tree_reconstruction_at_step"] is False
    assert payload["lineage_mode_codes"] == {
        "full-current-reconstruction": 0,
        "compact-issued-carry-forward": 1,
    }
    assert payload["caller_authenticated"] is False
    assert payload["physical_execution_authenticated"] is False
    assert payload["dispatch_authority"] is False
    assert payload["safety_execution_claimed"] is False
    assert payload["critic_execution_claimed"] is False
    assert payload["evidence_authority"] is False
    assert payload["promotion_authority"] is False

    assert proposal.sampling_keys.shape == (BATCH_SIZE,)
    assert jax.dtypes.issubdtype(proposal.sampling_keys.dtype, jax.dtypes.prng_key)
    assert proposal.proposal_digest_words.shape == (BATCH_SIZE, 8)
    assert bool(jnp.all(proposal.proposal_digest_words != 0))
    np.testing.assert_array_equal(
        proposal.action_stack_memory_preparation_words,
        jnp.stack([item.content_tag_words for item in prepared]),
    )
    np.testing.assert_array_equal(
        proposal.action_stack_memory_candidate_binding_words,
        jnp.stack(
            [
                item.memory_candidate_state.action_binding.content_tag_words
                for item in prepared
            ]
        ),
    )
    np.testing.assert_array_equal(
        proposal.policy_revision,
        jnp.full((BATCH_SIZE,), state.policy_revision, dtype=jnp.int32),
    )
    expected_behavior = bridge.actor.behavior_log_probability(
        state,
        proposal.actor_features,
        proposal.selected_actions,
    )
    np.testing.assert_array_equal(
        jax.lax.bitcast_convert_type(proposal.behavior_log_probability, jnp.uint32),
        jax.lax.bitcast_convert_type(expected_behavior, jnp.uint32),
    )


def test_distinct_per_row_action_owners_round_trip_and_execute_exact_lineage() -> None:
    base = _action_stack_adapter().config
    second = dataclasses.replace(
        base,
        final_action_owner_digest=tuple(
            int(word) ^ 0x01010101 for word in base.final_action_owner_digest
        ),
    )
    config = KondoExecutedActionLineageBridgeConfig(
        actor=_actor_config(),
        action_stack=base,
        action_stack_rows=(base, second),
    )
    restored = KondoExecutedActionLineageBridgeConfig.from_config(config.to_config())
    assert restored.to_config() == config.to_config()
    bridge = KondoExecutedActionLineageBridge(restored)
    assert len(bridge.action_stacks) == BATCH_SIZE
    assert (
        bridge.action_stacks[0].config.final_action_owner_digest
        != bridge.action_stacks[1].config.final_action_owner_digest
    )
    state = bridge.actor.init(
        _parameters(),
        jr.key(702, impl="threefry2x32"),
    )
    prepared = _prepared_decisions(bridge, seed_offset=20)
    proposal = _sample(bridge, state, prepared)
    adopted = _adopt_proposals(bridge, prepared, proposal)
    next_prepared = _next_preparations(bridge, adopted)
    result = bridge.step(state, proposal, adopted, next_prepared, _protected())

    assert bool(jnp.all(result.diagnostics.actor_eligible))
    assert int(result.work.actor_step_calls) == 1
    assert bool(jnp.all(result.actor_result.sparks_joy))


def test_per_row_action_owner_configuration_fails_closed() -> None:
    base = _action_stack_adapter().config
    second = dataclasses.replace(
        base,
        final_action_owner_digest=tuple(
            int(word) ^ 0x02020202 for word in base.final_action_owner_digest
        ),
    )
    with pytest.raises(ValueError, match="fixed-batch tuple"):
        KondoExecutedActionLineageBridgeConfig(
            actor=_actor_config(),
            action_stack=base,
            action_stack_rows=(base,),
        )
    with pytest.raises(ValueError, match=r"action_stack_rows\[0\]"):
        KondoExecutedActionLineageBridgeConfig(
            actor=_actor_config(),
            action_stack=base,
            action_stack_rows=(second, base),
        )

    payload = KondoExecutedActionLineageBridgeConfig(
        actor=_actor_config(),
        action_stack=base,
        action_stack_rows=(base, second),
    ).to_config()
    malformed = dict(payload)
    malformed["action_stack_rows"] = {"row": base.to_config()}
    with pytest.raises(ValueError, match="exact list"):
        KondoExecutedActionLineageBridgeConfig.from_config(malformed)
    relabeled = dict(payload)
    relabeled["action_stack_row_ownership"] = "trusted-caller"
    with pytest.raises(ValueError, match="fixed semantics differ"):
        KondoExecutedActionLineageBridgeConfig.from_config(relabeled)


def test_nontrivial_mask_and_legacy_key_fail_before_proposal_sampling() -> None:
    bridge, state = _bridge()
    prepared = _prepared_decisions(bridge)
    nontrivial = prepared[0].replace(
        hard_action_mask=prepared[0].hard_action_mask.at[1].set(False)
    )
    with pytest.raises(ValueError, match="all-true"):
        bridge.sample_proposals(
            state,
            _features(),
            jr.split(jr.key(751, impl="threefry2x32"), BATCH_SIZE),
            action_stack_memory_preparations=(nontrivial, prepared[1]),
        )
    with pytest.raises(TypeError, match="typed PRNG"):
        bridge.sample_proposals(
            state,
            _features(),
            jnp.stack((jr.PRNGKey(1), jr.PRNGKey(2))),
            action_stack_memory_preparations=prepared,
        )


def test_exact_executed_lineage_calls_actor_once_and_returns_actual_nested_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, state, proposal, adopted, next_prepared = _valid_fixture()
    protected = _protected()
    calls = 0
    original: Callable[..., Any] = bridge.actor.step

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge.actor, "step", counted)
    result = bridge.step(state, proposal, adopted, next_prepared, protected)

    assert type(result) is KondoExecutedActionLineageResult
    assert calls == 1
    assert int(result.work.actor_step_calls) == 1
    assert int(result.work.adoption_integrity_reconstructions) == BATCH_SIZE
    assert int(result.work.action_stack_learner_evaluations) == 0
    assert int(result.work.planner_model_evaluations) == 0
    assert bool(jnp.all(result.diagnostics.actor_eligible))
    assert bool(result.actor_result.transaction_applied)
    assert bool(jnp.all(result.actor_result.sparks_joy))
    assert not hasattr(result, "sparks_joy")
    assert "sparks_joy" not in {field.name for field in dataclasses.fields(type(result))}
    _tree_exact(result.actor_result.protected, protected)
    _tree_exact(result.protected, protected)
    assert int(result.actor_result.protected_slots) == BATCH_SIZE


def test_compact_adoption_matches_full_lineage_without_mutable_owner_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, state, proposal, adopted, next_prepared = _valid_fixture()
    protected = _protected()

    compact = bridge.compact_adoption_receipts(proposal, adopted)
    with monkeypatch.context() as patcher:
        patcher.setattr(
            bridge.actor,
            "step",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("compact preflight must not enter actor backward")
            ),
        )
        preflight = bridge.preflight_compact(
            state,
            proposal,
            compact,
            next_prepared,
        )
    assert bool(jnp.all(preflight.actor_eligible))
    full_result = bridge.step(state, proposal, adopted, next_prepared, protected)
    compact_result = bridge.step_compact(
        state,
        proposal,
        compact,
        next_prepared,
        protected,
    )

    assert type(compact) is KondoExecutedActionCompactAdoptionBatch
    assert bridge.to_config()["compact_pending_mutable_owner_snapshots"] == 0
    assert (
        KONDO_EXECUTED_ACTION_COMPACT_ADOPTION_SCHEMA
        == "alberta.kondo-executed-action-compact-adoption.v1"
    )
    assert all(
        hasattr(getattr(compact, field.name), "shape")
        for field in dataclasses.fields(type(compact))
    )
    assert not hasattr(compact, "state")
    assert not hasattr(compact, "finalized")
    assert bool(jnp.all(compact.integrity_receipt_words != 0))
    assert _tree_nbytes(compact) < _tree_nbytes(adopted)
    _tree_exact(compact_result.actor_result, full_result.actor_result)
    _tree_exact(compact_result.protected, protected)
    assert int(compact_result.work.actor_step_calls) == 1
    assert int(compact_result.work.adoption_integrity_reconstructions) == 0
    assert int(compact_result.work.compact_adoption_receipt_validations) == BATCH_SIZE
    assert bool(jnp.all(compact_result.diagnostics.actor_eligible))
    assert bool(jnp.all(full_result.diagnostics.lineage_mode == 0))
    assert bool(jnp.all(full_result.diagnostics.historic_adoption_reconstructed))
    assert not bool(
        jnp.any(full_result.diagnostics.compact_carry_forward_certificate_valid)
    )
    assert bool(jnp.all(compact_result.diagnostics.lineage_mode == 1))
    assert not bool(
        jnp.any(compact_result.diagnostics.historic_adoption_reconstructed)
    )
    assert not bool(jnp.any(compact_result.diagnostics.adoption_result_exact))
    assert not bool(
        jnp.any(compact_result.diagnostics.memory_preparation_integrity_rederived)
    )
    assert not bool(jnp.any(compact_result.diagnostics.memory_candidate_binding_exact))
    assert bool(
        jnp.all(compact_result.diagnostics.compact_carry_forward_certificate_valid)
    )
    assert bool(jnp.all(compact_result.actor_result.sparks_joy))
    assert not hasattr(compact_result, "sparks_joy")


def test_compact_adoption_tamper_and_foreign_destination_fail_closed() -> None:
    bridge, state, proposal, adopted, next_prepared = _valid_fixture()
    compact = bridge.compact_adoption_receipts(proposal, adopted)

    masked = proposal.replace(
        hard_action_masks=proposal.hard_action_masks.at[0, 1].set(False),
        proposal_digest_words=jnp.zeros_like(proposal.proposal_digest_words),
    )
    masked = _retag_proposal(bridge, masked)
    with pytest.raises(ValueError, match="all-true"):
        bridge.compact_adoption_receipts(masked, adopted)

    corrupted = compact.replace(
        content_tag_words=compact.content_tag_words.at[:, 0].add(
            jnp.asarray(1, dtype=jnp.uint32)
        )
    )
    corrupted_result = bridge.step_compact(
        state,
        proposal,
        corrupted,
        next_prepared,
        _protected(),
    )
    assert not bool(
        jnp.any(corrupted_result.diagnostics.compact_carry_forward_certificate_valid)
    )
    assert not bool(jnp.any(corrupted_result.diagnostics.actor_eligible))
    assert int(corrupted_result.work.actor_step_calls) == 1
    assert not bool(jnp.any(corrupted_result.actor_result.sparks_joy))
    assert not bool(corrupted_result.actor_result.sparse_backward_used)
    assert not bool(corrupted_result.actor_result.full_shape_masked_backward_used)
    assert int(corrupted_result.actor_result.backward_batch_size) == 0
    _tree_exact(corrupted_result.actor_result.state, state)

    foreign = _prepared_decisions(bridge, seed_offset=200)
    foreign_destination = bridge.action_stack_source_state_digest_words(foreign[0])
    resealed = compact.replace(
        destination_state_words=compact.destination_state_words.at[0].set(
            foreign_destination
        ),
        content_tag_words=jnp.zeros_like(compact.content_tag_words),
    )
    resealed = resealed.replace(
        content_tag_words=bridge._compact_adoption_tags(resealed)
    )
    foreign_result = bridge.step_compact(
        state,
        proposal,
        resealed,
        next_prepared,
        _protected(),
    )
    assert bool(foreign_result.diagnostics.proposal_integrity_rederived[0])
    assert not bool(
        foreign_result.diagnostics.compact_carry_forward_certificate_valid[0]
    )
    assert not bool(foreign_result.diagnostics.next_preparation_source_exact[0])
    assert not bool(foreign_result.diagnostics.actor_eligible[0])


def _retag_proposal(
    bridge: KondoExecutedActionLineageBridge,
    proposal: KondoExecutedActionProposalBatch,
) -> KondoExecutedActionProposalBatch:
    return proposal.replace(
        proposal_digest_words=bridge.rederive_proposal_digest_words(proposal)
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "digest",
        "key",
        "snapshot",
        "revision",
        "log_probability",
        "action",
        "decision_identity",
        "source_state",
        "memory_preparation",
        "candidate_binding",
        "hard_mask",
        "replay",
    ),
)
def test_proposal_tamper_replay_and_foreign_snapshot_fail_closed(mutation: str) -> None:
    bridge, state, proposal, adopted, next_prepared = _valid_fixture()
    if mutation == "digest":
        proposal = proposal.replace(
            proposal_digest_words=proposal.proposal_digest_words.at[0, 0].add(
                jnp.asarray(1, dtype=jnp.uint32)
            )
        )
    elif mutation == "key":
        proposal = _retag_proposal(
            bridge,
            proposal.replace(
                sampling_keys=proposal.sampling_keys.at[0].set(
                    jr.key(991, impl="threefry2x32")
                )
            ),
        )
    elif mutation == "snapshot":
        foreign_state = bridge.actor.init(
            _parameters(output_shift=0.5),
            jr.key(701, impl="threefry2x32"),
        )
        proposal = proposal.replace(
            actor_state_words=bridge.actor_state_digest_words(foreign_state)
        )
        proposal = _retag_proposal(bridge, proposal)
    elif mutation == "revision":
        proposal = _retag_proposal(
            bridge,
            proposal.replace(
                policy_revision=proposal.policy_revision.at[0].add(
                    jnp.asarray(1, dtype=jnp.int32)
                )
            ),
        )
    elif mutation == "log_probability":
        proposal = _retag_proposal(
            bridge,
            proposal.replace(
                behavior_log_probability=proposal.behavior_log_probability.at[0].add(
                    jnp.asarray(0.125, dtype=jnp.float32)
                )
            ),
        )
    elif mutation == "action":
        proposal = _retag_proposal(
            bridge,
            proposal.replace(
                selected_actions=proposal.selected_actions.at[0].set(
                    1 - proposal.selected_actions[0]
                )
            ),
        )
    elif mutation == "decision_identity":
        proposal = _retag_proposal(
            bridge,
            proposal.replace(
                action_stack_decision_identities=(
                    proposal.action_stack_decision_identities.at[0, 0].add(
                        jnp.asarray(1, dtype=jnp.uint32)
                    )
                )
            ),
        )
    elif mutation == "source_state":
        proposal = _retag_proposal(
            bridge,
            proposal.replace(
                action_stack_source_state_words=(
                    proposal.action_stack_source_state_words.at[0, 0].add(
                        jnp.asarray(1, dtype=jnp.uint32)
                    )
                )
            ),
        )
    elif mutation == "memory_preparation":
        proposal = _retag_proposal(
            bridge,
            proposal.replace(
                action_stack_memory_preparation_words=(
                    proposal.action_stack_memory_preparation_words.at[0, 0].add(
                        jnp.asarray(1, dtype=jnp.uint32)
                    )
                )
            ),
        )
    elif mutation == "candidate_binding":
        proposal = _retag_proposal(
            bridge,
            proposal.replace(
                action_stack_memory_candidate_binding_words=(
                    proposal.action_stack_memory_candidate_binding_words.at[0, 0].add(
                        jnp.asarray(1, dtype=jnp.uint32)
                    )
                )
            ),
        )
    elif mutation == "hard_mask":
        proposal = _retag_proposal(
            bridge,
            proposal.replace(
                hard_action_masks=proposal.hard_action_masks.at[0, 1].set(False)
            ),
        )
    elif mutation == "replay":
        proposal = proposal.replace(
            actor_features=proposal.actor_features.at[1].set(proposal.actor_features[0]),
            sampling_keys=proposal.sampling_keys.at[1].set(proposal.sampling_keys[0]),
            selected_actions=proposal.selected_actions.at[1].set(proposal.selected_actions[0]),
            behavior_log_probability=proposal.behavior_log_probability.at[1].set(
                proposal.behavior_log_probability[0]
            ),
            policy_revision=proposal.policy_revision.at[1].set(proposal.policy_revision[0]),
            action_stack_decision_identities=(
                proposal.action_stack_decision_identities.at[1].set(
                    proposal.action_stack_decision_identities[0]
                )
            ),
            action_stack_source_state_words=(
                proposal.action_stack_source_state_words.at[1].set(
                    proposal.action_stack_source_state_words[0]
                )
            ),
            action_stack_memory_preparation_words=(
                proposal.action_stack_memory_preparation_words.at[1].set(
                    proposal.action_stack_memory_preparation_words[0]
                )
            ),
            action_stack_memory_candidate_binding_words=(
                proposal.action_stack_memory_candidate_binding_words.at[1].set(
                    proposal.action_stack_memory_candidate_binding_words[0]
                )
            ),
            hard_action_masks=proposal.hard_action_masks.at[1].set(
                proposal.hard_action_masks[0]
            ),
            proposal_digest_words=proposal.proposal_digest_words.at[1].set(
                proposal.proposal_digest_words[0]
            ),
        )
    result = bridge.step(state, proposal, adopted, next_prepared, _protected())

    assert not bool(jnp.all(result.diagnostics.actor_eligible))
    assert not bool(result.diagnostics.actor_eligible[0 if mutation != "replay" else 1])


def test_checksum_valid_foreign_post_memory_source_splice_is_rejected() -> None:
    bridge, state, proposal, adopted, next_prepared = _valid_fixture()
    foreign = _prepared_decisions(bridge, seed_offset=100)
    spliced = proposal.replace(
        action_stack_source_state_words=(
            proposal.action_stack_source_state_words.at[0].set(
                bridge.action_stack_source_state_digest_words(foreign[0])
            )
        ),
        action_stack_memory_preparation_words=(
            proposal.action_stack_memory_preparation_words.at[0].set(
                foreign[0].content_tag_words
            )
        ),
        action_stack_memory_candidate_binding_words=(
            proposal.action_stack_memory_candidate_binding_words.at[0].set(
                foreign[0].memory_candidate_state.action_binding.content_tag_words
            )
        ),
        action_stack_decision_identities=(
            proposal.action_stack_decision_identities.at[0].set(
                foreign[0].memory_candidate_state.action_binding.prototype_decision_id
            )
        ),
        hard_action_masks=proposal.hard_action_masks.at[0].set(
            foreign[0].hard_action_mask
        ),
    )
    spliced = _retag_proposal(bridge, spliced)
    assert bool(
        jnp.all(
            spliced.proposal_digest_words
            == bridge.rederive_proposal_digest_words(spliced)
        )
    )

    result = bridge.step(state, spliced, adopted, next_prepared, _protected())

    assert bool(result.diagnostics.proposal_integrity_rederived[0])
    assert not bool(result.diagnostics.action_stack_source_exact[0])
    assert not bool(result.diagnostics.actor_eligible[0])


def test_all_invalid_lineage_executes_zero_backward_and_preserves_actor_exactly() -> None:
    bridge, state, proposal, adopted, next_prepared = _valid_fixture()
    bad = proposal.replace(
        behavior_log_probability=proposal.behavior_log_probability
        + jnp.asarray(0.25, dtype=jnp.float32)
    )
    bad = _retag_proposal(bridge, bad)
    protected = _protected()
    result = bridge.step(state, bad, adopted, next_prepared, protected)

    assert not bool(jnp.any(result.diagnostics.actor_eligible))
    assert int(result.work.actor_step_calls) == 1
    assert not bool(jnp.any(result.actor_result.sparks_joy))
    assert not bool(result.actor_result.sparse_backward_used)
    assert not bool(result.actor_result.full_shape_masked_backward_used)
    assert int(result.actor_result.backward_batch_size) == 0
    assert not bool(result.actor_result.transaction_applied)
    _tree_exact(result.actor_result.state, state)
    _tree_exact(result.actor_result.protected, protected)
    _tree_exact(result.protected, protected)


def test_memory_path_and_tampered_final_or_next_transition_are_actor_ineligible() -> None:
    bridge, state = _bridge()
    prepared = _prepared_decisions(bridge)
    proposal = _sample(bridge, state, prepared)
    memory_adopted = _adopt_proposals(
        bridge,
        prepared,
        proposal,
        memory_path=True,
    )
    memory_next = _next_preparations(bridge, memory_adopted)
    memory_result = bridge.step(
        state,
        proposal,
        memory_adopted,
        memory_next,
        _protected(),
    )
    assert not bool(jnp.any(memory_result.diagnostics.actor_path_selected))
    assert not bool(jnp.any(memory_result.diagnostics.actor_eligible))
    _tree_exact(memory_result.actor_result.state, state)

    bridge, state, proposal, adopted, next_prepared = _valid_fixture()
    wrong_binding = adopted[0].finalized.final_action_binding.replace(
        final_action=1 - adopted[0].finalized.final_action_binding.final_action
    )
    wrong_finalized = adopted[0].finalized.replace(final_action_binding=wrong_binding)
    wrong_adopted = (adopted[0].replace(finalized=wrong_finalized), adopted[1])
    final_result = bridge.step(
        state,
        proposal,
        wrong_adopted,
        next_prepared,
        _protected(),
    )
    assert not bool(final_result.diagnostics.adoption_result_exact[0])
    assert not bool(final_result.diagnostics.actor_eligible[0])

    wrong_transition = dataclasses.replace(
        next_prepared[0].transition,
        action=1 - next_prepared[0].transition.action,
    )
    changed = next_prepared[0].replace(
        transition=wrong_transition,
        content_tag_words=jnp.zeros((8,), dtype=jnp.uint32),
    )
    changed = changed.replace(
        content_tag_words=bridge.rederive_next_preparation_digest_words(changed)
    )
    transition_result = bridge.step(
        state,
        proposal,
        adopted,
        (changed, next_prepared[1]),
        _protected(),
    )
    assert bool(transition_result.diagnostics.next_preparation_integrity_rederived[0])
    assert not bool(transition_result.diagnostics.next_transition_action_exact[0])
    assert not bool(transition_result.diagnostics.actor_eligible[0])
