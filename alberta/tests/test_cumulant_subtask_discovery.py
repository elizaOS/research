# mypy: disable-error-code="no-untyped-def,type-var"
"""Unit contracts for bounded proposal-only cumulant discovery v1."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.cumulant_subtask_discovery import (
    CUMULANT_SOURCE_CONTROLLABLE_EVENT,
    CUMULANT_SOURCE_FEATURE_CHANGE,
    CUMULANT_SOURCE_HAND_AUTHORED,
    CUMULANT_SOURCE_PREDICTION_BOTTLENECK,
    CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
    CUMULANT_SUBTASK_DISCOVERY_AUTHORITY,
    CUMULANT_SUBTASK_DISCOVERY_CHECKPOINT_SCHEMA,
    CUMULANT_SUBTASK_DISCOVERY_GO_NO_GO_AUTHORITY,
    CUMULANT_SUBTASK_DISCOVERY_PROMOTION_AUTHORITY,
    CUMULANT_SUBTASK_DISCOVERY_SCIENTIFIC_PROMOTION_ALLOWED,
    CumulantSubtaskDiscovery,
    CumulantSubtaskDiscoveryConfig,
    CumulantSubtaskDiscoveryResult,
    CumulantSubtaskDiscoveryState,
)

pytestmark = pytest.mark.unit

RAW_DIM = 3
PROBE_DIM = 2
N_ACTIONS = 2
OPTION_BUDGET = 4
GENERATION = 3
SOURCE_DIGEST = jnp.asarray([0x1234, 0x5678], dtype=jnp.uint32)
HAND_IDENTITY = jnp.asarray([0xCAFE, 0xBEEF], dtype=jnp.uint32)


def _config(**changes: Any) -> CumulantSubtaskDiscoveryConfig:
    values: dict[str, Any] = {
        "raw_feature_dim": RAW_DIM,
        "probe_feature_dim": PROBE_DIM,
        "n_actions": N_ACTIONS,
        "controllable_event_dim": 1,
        "transition_atom_dim": 1,
        "prediction_bottleneck_dim": 1,
        "option_budget": OPTION_BUDGET,
        "family_quotas": (1, 1, 1, 1),
        "controllable_event_descriptors": (
            (CUMULANT_SOURCE_CONTROLLABLE_EVENT, 0, 1, 10),
        ),
        "feature_change_descriptors": (
            (CUMULANT_SOURCE_FEATURE_CHANGE, 0, 1, 20),
        ),
        "reward_transition_descriptors": (
            (CUMULANT_SOURCE_REWARD_TRANSITION_ATOM, 0, 1, 30),
        ),
        "prediction_bottleneck_descriptors": (
            (CUMULANT_SOURCE_PREDICTION_BOTTLENECK, 0, 1, 40),
        ),
        "incumbent_descriptors": ((90, 0, 1, 900),),
        "hand_comparator_descriptors": tuple(
            (CUMULANT_SOURCE_HAND_AUTHORED, index, 1, 100 + index)
            for index in range(OPTION_BUDGET)
        ),
        "hand_comparator_identity": tuple(int(cell) for cell in HAND_IDENTITY),
        "reward_task_weights": (0.5,),
        "model_task_weights": (0.5,),
        "probe_step_size": 0.1,
        "shadow_step_size": 1.0e-6,
        "learnability_evidence_floor": 1,
        "controllability_evidence_floor_per_action": 1,
        "novelty_evidence_floor": 1,
        "contribution_evidence_floor": 1,
        "bottleneck_evidence_floor": 1,
        "learnability_threshold": 0.0,
        "baseline_variance_floor": 1.0e-8,
        "controllability_threshold": 0.0,
        "novelty_threshold": 1.0e-8,
        "contribution_threshold": 0.0,
        "bottleneck_epistemic_floor": 0.0,
        "bottleneck_progress_floor": 0.0,
        "bottleneck_aleatoric_ceiling": 1.0,
        "max_observations": 32,
    }
    values.update(changes)
    return CumulantSubtaskDiscoveryConfig(**values)


def _snapshot(step: int) -> dict[str, jax.Array]:
    action = float(step % N_ACTIONS)
    value = float(step)
    return {
        "raw": jnp.asarray([value, value * value + 0.25, 1.0 + action], jnp.float32),
        "event": jnp.asarray([1.0 + 2.0 * action + 0.1 * value], jnp.float32),
        "atom": jnp.asarray([0.5 + action + 0.2 * value], jnp.float32),
        "bottleneck": jnp.asarray([2.0 - action + 0.15 * value], jnp.float32),
        "probe": jnp.asarray([1.0, 0.1 * value], jnp.float32),
        "incumbent": jnp.asarray([20.0 + value], jnp.float32),
        "hand": jnp.asarray([value + index for index in range(OPTION_BUDGET)], jnp.float32),
    }


def _transition_id(step: int) -> jax.Array:
    return jnp.asarray([0xD15C, step], dtype=jnp.uint32)


def _arm_inputs(step: int) -> dict[str, Any]:
    current = _snapshot(step)
    return {
        "current_raw_features": current["raw"],
        "current_raw_available": jnp.ones((RAW_DIM,), dtype=jnp.bool_),
        "current_controllable_events": current["event"],
        "current_controllable_events_available": jnp.ones((1,), dtype=jnp.bool_),
        "current_transition_atoms": current["atom"],
        "current_transition_atoms_available": jnp.full(
            (1,), step > 0, dtype=jnp.bool_
        ),
        "current_bottleneck_values": current["bottleneck"],
        "current_bottleneck_available": jnp.ones((1,), dtype=jnp.bool_),
        "probe_features": current["probe"],
        "current_incumbent_values": current["incumbent"],
        "current_incumbent_available": jnp.ones((1,), dtype=jnp.bool_),
        "current_hand_values": current["hand"],
        "current_hand_available": jnp.ones((OPTION_BUDGET,), dtype=jnp.bool_),
        "hand_comparator_identity": HAND_IDENTITY,
        "reward_base_predictions": jnp.zeros((1,), dtype=jnp.float32),
        "model_base_predictions": jnp.zeros((1,), dtype=jnp.float32),
        "action": jnp.asarray(step % N_ACTIONS, dtype=jnp.int32),
        "behavior_propensity": jnp.asarray(0.5, dtype=jnp.float32),
        "randomized": jnp.asarray(True),
        "transition_id": _transition_id(step),
        "semantic_generation": jnp.asarray(GENERATION, dtype=jnp.int32),
        "source_digest": SOURCE_DIGEST,
    }


def _observe_inputs(step: int) -> dict[str, Any]:
    successor = _snapshot(step + 1)
    intervention = jnp.zeros((N_ACTIONS,), dtype=jnp.bool_).at[step % N_ACTIONS].set(True)
    return {
        "next_raw_features": successor["raw"],
        "next_raw_available": jnp.ones((RAW_DIM,), dtype=jnp.bool_),
        "next_controllable_events": successor["event"],
        "next_controllable_events_available": jnp.ones((1,), dtype=jnp.bool_),
        "next_transition_atoms": successor["atom"],
        "next_transition_atoms_available": jnp.ones((1,), dtype=jnp.bool_),
        "next_bottleneck_values": successor["bottleneck"],
        "next_bottleneck_available": jnp.ones((1,), dtype=jnp.bool_),
        "bottleneck_epistemic": jnp.asarray([0.5], dtype=jnp.float32),
        "bottleneck_progress": jnp.asarray([0.25], dtype=jnp.float32),
        "bottleneck_aleatoric": jnp.asarray([0.1], dtype=jnp.float32),
        "bottleneck_evidence_available": jnp.ones((1,), dtype=jnp.bool_),
        "randomized_action_evidence": intervention,
        "next_incumbent_values": successor["incumbent"],
        "next_incumbent_available": jnp.ones((1,), dtype=jnp.bool_),
        "next_hand_values": successor["hand"],
        "next_hand_available": jnp.ones((OPTION_BUDGET,), dtype=jnp.bool_),
        "hand_comparator_identity": HAND_IDENTITY,
        "reward_targets": jnp.zeros((1,), dtype=jnp.float32),
        "reward_targets_available": jnp.ones((1,), dtype=jnp.bool_),
        "model_targets": jnp.zeros((1,), dtype=jnp.float32),
        "model_targets_available": jnp.ones((1,), dtype=jnp.bool_),
        "transition_id": _transition_id(step),
        "semantic_generation": jnp.asarray(GENERATION, dtype=jnp.int32),
        "source_digest": SOURCE_DIGEST,
    }


def _step(
    discovery: CumulantSubtaskDiscovery,
    state: CumulantSubtaskDiscoveryState,
    step: int,
) -> CumulantSubtaskDiscoveryResult:
    arm = discovery.arm(state, **_arm_inputs(step))
    return discovery.observe(state, arm, **_observe_inputs(step))


def _ready_result() -> tuple[CumulantSubtaskDiscovery, CumulantSubtaskDiscoveryResult]:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(7), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    result: CumulantSubtaskDiscoveryResult | None = None
    for step in range(8):
        result = _step(discovery, state, step)
        state = result.state
        if bool(result.discovered.ready):
            return discovery, result
    assert result is not None
    pytest.fail("permissive unit trajectory did not produce an exact-quota bundle")


def test_config_roundtrip_is_strict_and_rejects_quota_or_identity_collisions() -> None:
    config = _config()
    assert CumulantSubtaskDiscoveryConfig.from_config(config.to_config()) == config
    payload = config.to_config()
    payload["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema_version"):
        CumulantSubtaskDiscoveryConfig.from_config(payload)
    with pytest.raises(ValueError, match="sum exactly"):
        _config(family_quotas=(1, 1, 1, 2))
    with pytest.raises(ValueError, match="collide"):
        _config(incumbent_descriptors=((90, 0, 1, 10),))
    with pytest.raises(ValueError, match="signed-int32"):
        _config(incumbent_descriptors=((90, 0, 1, 2**40),))
    with pytest.raises(ValueError, match="duplicate canonical semantics"):
        _config(
            feature_change_descriptors=(
                (CUMULANT_SOURCE_FEATURE_CHANGE, 0, 1, 10),
            )
        )
    invalid_hand = list(_config().hand_comparator_descriptors)
    invalid_hand[0] = (CUMULANT_SOURCE_HAND_AUTHORED, 0, 0, 100)
    with pytest.raises(ValueError, match="polarity"):
        _config(hand_comparator_descriptors=tuple(invalid_hand))
    with pytest.raises(ValueError, match="float32 total mass exactly 1"):
        _config(
            reward_task_weights=(
                0.33390195325043454,
                0.3787617501559222,
                0.28733629659364307,
            ),
            model_task_weights=(),
        )


def test_global_resource_ceiling_includes_projection_and_pair_state() -> None:
    per_family = 256
    families = tuple(
        tuple((family, 0, 1, 10_000 * family + index) for index in range(per_family))
        for family in range(4)
    )
    with pytest.raises(ValueError, match="fixed cell ceiling"):
        _config(
            raw_feature_dim=4_096,
            probe_feature_dim=4_096,
            option_budget=1_024,
            family_quotas=(per_family, per_family, per_family, per_family),
            controllable_event_descriptors=families[0],
            feature_change_descriptors=families[1],
            reward_transition_descriptors=families[2],
            prediction_bottleneck_descriptors=families[3],
            hand_comparator_descriptors=tuple(
                (CUMULANT_SOURCE_HAND_AUTHORED, index, 1, 100_000 + index)
                for index in range(1_024)
            ),
        )


def test_init_is_key_deterministic_bound_and_declares_no_authority() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    first = discovery.init(
        jr.key(1), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    replay = discovery.init(
        jr.key(1), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    other = discovery.init(
        jr.key(2), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    chex.assert_trees_all_equal(first, replay)
    assert not np.array_equal(first.random_projections, other.random_projections)
    assert bool(
        discovery.validate_state(
            first, semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
        )
    )
    budget = discovery.resource_budget
    assert budget.random_draws_at_init == OPTION_BUDGET * RAW_DIM
    assert budget.random_generator_calls_at_init == 1
    assert budget.random_generator_calls_per_arm == 0
    assert budget.random_generator_calls_per_observe == 0
    assert budget.projection_checksum_cells_per_state_validation == OPTION_BUDGET * RAW_DIM
    assert budget.state_validation_calls_per_arm == 1
    assert budget.state_validation_calls_per_observe == 2
    assert budget.pair_novelty_cells == _config().candidate_count**2
    assert budget.backward_passes_per_observe == 0
    leaves = jax.tree_util.tree_leaves(first)
    assert budget.persistent_logical_scalars == sum(leaf.size for leaf in leaves)
    assert budget.persistent_state_nbytes == sum(
        leaf.size * leaf.dtype.itemsize for leaf in leaves
    )
    assert not CUMULANT_SUBTASK_DISCOVERY_AUTHORITY
    assert not CUMULANT_SUBTASK_DISCOVERY_PROMOTION_AUTHORITY
    assert not CUMULANT_SUBTASK_DISCOVERY_GO_NO_GO_AUTHORITY
    assert not CUMULANT_SUBTASK_DISCOVERY_SCIENTIFIC_PROMOTION_ALLOWED


def test_projection_and_state_tampering_fail_validation() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(3), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    tampered = dataclasses.replace(
        state,
        random_projections=state.random_projections.at[0, 0].add(1.0),
    )
    assert not bool(
        discovery.validate_state(
            tampered, semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
        )
    )


def test_scalar_identity_and_action_inputs_reject_lossy_coercions() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    with pytest.raises(TypeError, match="semantic_generation.*int32"):
        discovery.init(
            jr.key(31),
            semantic_generation=3.9,  # type: ignore[arg-type]
            source_digest=SOURCE_DIGEST,
        )
    with pytest.raises(ValueError, match="signed-int32"):
        discovery.init(
            jr.key(31),
            semantic_generation=2**40,
            source_digest=SOURCE_DIGEST,
        )
    state = discovery.init(
        jr.key(31), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    inputs = _arm_inputs(0)
    inputs["action"] = jnp.asarray(1.7, dtype=jnp.float32)
    with pytest.raises(TypeError, match="action.*int32"):
        discovery.arm(state, **inputs)
    inputs = _arm_inputs(0)
    inputs["randomized"] = jnp.asarray(1, dtype=jnp.int32)
    with pytest.raises(TypeError, match="randomized.*bool"):
        discovery.arm(state, **inputs)
    assert not bool(
        discovery.validate_state(
            state,
            semantic_generation=GENERATION + 1,
            source_digest=SOURCE_DIGEST,
        )
    )


def test_reward_transition_atom_is_born_then_scores_only_the_next_transition() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(4), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    reward_index = 2
    first = _step(discovery, state, 0)
    assert bool(first.diagnostics.reward_births_this_transition[reward_index])
    assert int(first.state.learnability_counts[reward_index]) == 0
    assert int(first.state.task_contribution_counts[reward_index, 0]) == 0
    second = _step(discovery, first.state, 1)
    assert int(second.state.learnability_counts[reward_index]) == 1
    assert int(second.state.task_contribution_counts[reward_index, 0]) == 1


def test_unavailable_cells_freeze_only_their_evidence() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(5), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    arm = discovery.arm(state, **_arm_inputs(0))
    inputs = _observe_inputs(0)
    inputs["next_bottleneck_available"] = jnp.zeros((1,), dtype=jnp.bool_)
    inputs["bottleneck_evidence_available"] = jnp.zeros((1,), dtype=jnp.bool_)
    result = discovery.observe(state, arm, **inputs)
    assert bool(result.diagnostics.transaction_applied)
    assert int(result.state.learnability_counts[0]) == 1
    assert int(result.state.bottleneck_evidence_counts[3]) == 0


def test_persistent_high_aleatoric_bottleneck_evidence_is_vetoed() -> None:
    discovery = CumulantSubtaskDiscovery(
        _config(
            bottleneck_evidence_floor=3,
            bottleneck_epistemic_floor=0.5,
            bottleneck_progress_floor=0.5,
            bottleneck_aleatoric_ceiling=0.25,
        )
    )
    state = discovery.init(
        jr.key(50), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    result: CumulantSubtaskDiscoveryResult | None = None
    for step in range(5):
        arm = discovery.arm(state, **_arm_inputs(step))
        inputs = _observe_inputs(step)
        inputs["bottleneck_epistemic"] = jnp.asarray([0.9], dtype=jnp.float32)
        inputs["bottleneck_progress"] = jnp.asarray([0.8], dtype=jnp.float32)
        inputs["bottleneck_aleatoric"] = jnp.asarray([0.75], dtype=jnp.float32)
        result = discovery.observe(state, arm, **inputs)
        assert bool(result.diagnostics.transaction_applied)
        state = result.state

    assert result is not None
    bottleneck_index = 3
    assert int(state.bottleneck_evidence_counts[bottleneck_index]) == 5
    mean_aleatoric = (
        state.bottleneck_aleatoric_sums[bottleneck_index]
        / state.bottleneck_evidence_counts[bottleneck_index]
    )
    assert float(mean_aleatoric) == pytest.approx(0.75)
    assert not bool(result.diagnostics.bottleneck_ready[bottleneck_index])
    assert not bool(result.diagnostics.all_local_gates_ready[bottleneck_index])


def test_identical_to_incumbent_candidate_fails_novelty_gate() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(501), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    arm = discovery.arm(state, **_arm_inputs(0))
    inputs = _observe_inputs(0)
    inputs["next_incumbent_values"] = inputs["next_controllable_events"].copy()
    result = discovery.observe(state, arm, **inputs)

    event_index = 0
    assert bool(result.diagnostics.transaction_applied)
    assert int(result.state.incumbent_novelty_counts[event_index, 0]) == 1
    assert float(result.state.incumbent_novelty_sums[event_index, 0]) == 0.0
    assert float(result.diagnostics.novelty_scores[event_index]) == 0.0
    assert not bool(result.diagnostics.novelty_against_incumbents_ready[event_index])


def test_pair_novelty_blocks_a_later_family_candidate_against_prior_selection() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    candidates = discovery.config.candidate_count
    pair_ready = ~jnp.eye(candidates, dtype=jnp.bool_)
    pair_ready = pair_ready.at[0, 1].set(False).at[1, 0].set(False)

    selected, selected_mask, counts, ready = discovery._select_discovered(
        jnp.ones((candidates,), dtype=jnp.bool_),
        pair_ready,
        jnp.ones((candidates,), dtype=jnp.float32),
    )

    np.testing.assert_array_equal(selected, np.asarray([0, -1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(
        selected_mask, np.asarray([True, False, True, True], dtype=np.bool_)
    )
    np.testing.assert_array_equal(counts, np.asarray([1, 0, 1, 1], dtype=np.int32))
    assert not bool(ready)


def test_equal_scores_use_descriptor_lexicographic_order_not_declaration_order() -> None:
    lexicographic = (CUMULANT_SOURCE_CONTROLLABLE_EVENT, 0, 1, 10)
    later = (CUMULANT_SOURCE_CONTROLLABLE_EVENT, 0, 1, 11)

    def selected_descriptors(
        rows: tuple[tuple[int, int, int, int], ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        discovery = CumulantSubtaskDiscovery(
            _config(controllable_event_descriptors=rows)
        )
        candidates = discovery.config.candidate_count
        selected, _, _, ready = discovery._select_discovered(
            jnp.ones((candidates,), dtype=jnp.bool_),
            ~jnp.eye(candidates, dtype=jnp.bool_),
            jnp.ones((candidates,), dtype=jnp.float32),
        )
        assert bool(ready)
        descriptors = np.asarray(discovery.config.candidate_descriptors, dtype=np.int32)
        return np.asarray(selected), descriptors[np.asarray(selected)]

    forward_indices, forward = selected_descriptors((later, lexicographic))
    reverse_indices, reverse = selected_descriptors((lexicographic, later))

    assert int(forward_indices[0]) == 1
    assert int(reverse_indices[0]) == 0
    np.testing.assert_array_equal(forward, reverse)
    np.testing.assert_array_equal(forward[0], np.asarray(lexicographic, dtype=np.int32))


def test_controllability_uses_only_declared_randomized_propensity_evidence() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(51), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    arm_inputs = _arm_inputs(0)
    arm_inputs["randomized"] = jnp.asarray(False)
    arm = discovery.arm(state, **arm_inputs)
    result = discovery.observe(state, arm, **_observe_inputs(0))
    assert bool(result.diagnostics.transaction_applied)
    np.testing.assert_array_equal(
        result.state.action_evidence_counts,
        np.zeros_like(result.state.action_evidence_counts),
    )
    assert int(result.state.learnability_counts[0]) == 1

    selected_evidence_missing = _observe_inputs(0)
    selected_evidence_missing["randomized_action_evidence"] = jnp.asarray(
        [False, True], dtype=jnp.bool_
    )
    randomized_arm = discovery.arm(state, **_arm_inputs(0))
    missing = discovery.observe(state, randomized_arm, **selected_evidence_missing)
    assert bool(missing.diagnostics.transaction_applied)
    np.testing.assert_array_equal(
        missing.state.action_evidence_counts,
        np.zeros_like(missing.state.action_evidence_counts),
    )

    invalid_inputs = _arm_inputs(0)
    invalid_inputs["behavior_propensity"] = jnp.asarray(0.0, dtype=jnp.float32)
    invalid = discovery.arm(state, **invalid_inputs)
    assert not bool(invalid.available)


def test_controllability_requires_the_per_action_evidence_floor() -> None:
    discovery = CumulantSubtaskDiscovery(
        _config(controllability_evidence_floor_per_action=2)
    )
    state = discovery.init(
        jr.key(511), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    result: CumulantSubtaskDiscoveryResult | None = None
    for step in range(3):
        result = _step(discovery, state, step)
        assert bool(result.diagnostics.transaction_applied)
        state = result.state

    assert result is not None
    np.testing.assert_array_equal(
        state.action_evidence_counts[0], np.asarray([2, 1], dtype=np.int32)
    )
    assert float(result.diagnostics.controllability_scores[0]) > 0.0
    assert not bool(result.diagnostics.controllability_ready[0])


def test_controllability_uses_varying_propensity_weighted_conditional_means() -> None:
    discovery = CumulantSubtaskDiscovery(
        _config(controllability_evidence_floor_per_action=2)
    )
    state = discovery.init(
        jr.key(512), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    actions = (0, 0, 1, 1)
    propensities = (0.25, 0.5, 0.2, 0.8)
    outcomes = (2.0, 6.0, -1.0, 3.0)
    current_event = float(_snapshot(0)["event"][0])
    result: CumulantSubtaskDiscoveryResult | None = None

    for step, (action, propensity, outcome) in enumerate(
        zip(actions, propensities, outcomes, strict=True)
    ):
        arm_inputs = _arm_inputs(step)
        arm_inputs["current_controllable_events"] = jnp.asarray(
            [current_event], dtype=jnp.float32
        )
        arm_inputs["action"] = jnp.asarray(action, dtype=jnp.int32)
        arm_inputs["behavior_propensity"] = jnp.asarray(propensity, dtype=jnp.float32)
        arm = discovery.arm(state, **arm_inputs)
        assert bool(arm.available)

        observe_inputs = _observe_inputs(step)
        observe_inputs["next_controllable_events"] = jnp.asarray(
            [outcome], dtype=jnp.float32
        )
        observe_inputs["randomized_action_evidence"] = jax.nn.one_hot(
            action, N_ACTIONS, dtype=jnp.bool_
        )
        result = discovery.observe(state, arm, **observe_inputs)
        assert bool(result.diagnostics.transaction_applied)
        state = result.state
        current_event = outcome

    assert result is not None
    expected_sums = np.asarray([20.0, -1.25], dtype=np.float32)
    expected_masses = np.asarray([6.0, 6.25], dtype=np.float32)
    expected_means = expected_sums / expected_masses
    np.testing.assert_allclose(
        state.action_outcome_weighted_sums[0], expected_sums, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        state.action_importance_masses[0], expected_masses, rtol=1.0e-6
    )
    np.testing.assert_array_equal(
        state.action_evidence_counts[0], np.asarray([2, 2], dtype=np.int32)
    )
    assert float(result.diagnostics.controllability_scores[0]) == pytest.approx(
        float(np.max(expected_means) - np.min(expected_means)), rel=1.0e-6
    )
    assert bool(result.diagnostics.controllability_ready[0])


def test_missing_task_channel_is_not_renormalized_into_contribution_readiness() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(52), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    first = _step(discovery, state, 0)
    arm = discovery.arm(first.state, **_arm_inputs(1))
    inputs = _observe_inputs(1)
    inputs["reward_targets_available"] = jnp.zeros((1,), dtype=jnp.bool_)
    result = discovery.observe(first.state, arm, **inputs)
    assert bool(result.diagnostics.transaction_applied)
    assert int(result.state.task_contribution_counts[0, 0]) == 0
    assert int(result.state.task_contribution_counts[0, 1]) == 1
    assert not bool(result.diagnostics.contribution_ready[0])


def test_fixed_mass_contribution_score_does_not_renormalize_missing_task() -> None:
    discovery = CumulantSubtaskDiscovery(
        _config(reward_task_weights=(0.25,), model_task_weights=(0.75,))
    )
    state = discovery.init(
        jr.key(521), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    first_arm = discovery.arm(state, **_arm_inputs(0))
    first_inputs = _observe_inputs(0)
    first_inputs["next_controllable_events"] = jnp.asarray([2.0], dtype=jnp.float32)
    first = discovery.observe(state, first_arm, **first_inputs)
    assert bool(first.diagnostics.transaction_applied)

    event_index = 0
    state_with_known_shadow = dataclasses.replace(
        first.state,
        reward_shadow_weights=first.state.reward_shadow_weights.at[event_index, 0].set(
            0.5
        ),
    )
    assert bool(
        discovery.validate_state(
            state_with_known_shadow,
            semantic_generation=GENERATION,
            source_digest=SOURCE_DIGEST,
        )
    )
    arm_inputs = _arm_inputs(1)
    arm_inputs["current_controllable_events"] = jnp.asarray([2.0], dtype=jnp.float32)
    arm = discovery.arm(state_with_known_shadow, **arm_inputs)
    assert bool(arm.available)
    inputs = _observe_inputs(1)
    inputs["reward_targets"] = jnp.asarray([1.0], dtype=jnp.float32)
    inputs["model_targets_available"] = jnp.zeros((1,), dtype=jnp.bool_)
    result = discovery.observe(state_with_known_shadow, arm, **inputs)

    assert bool(result.diagnostics.transaction_applied)
    np.testing.assert_array_equal(
        result.state.task_contribution_counts[event_index],
        np.asarray([1, 0], dtype=np.int32),
    )
    assert float(result.state.task_contribution_sums[event_index, 0]) == pytest.approx(1.0)
    assert float(result.diagnostics.contribution_scores[event_index]) == pytest.approx(
        0.25
    )
    assert not bool(result.diagnostics.contribution_ready[event_index])


def test_equal_probe_and_baseline_error_rejects_irreducible_noise() -> None:
    discovery = CumulantSubtaskDiscovery(_config(learnability_threshold=0.1))
    state = discovery.init(
        jr.key(53), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    first = _step(discovery, state, 0)
    noise_like = dataclasses.replace(
        first.state,
        probe_squared_error_sums=jnp.full(
            (first.state.probe_squared_error_sums.shape[0],), 1_000.0, jnp.float32
        ),
        baseline_squared_error_sums=jnp.full(
            (first.state.baseline_squared_error_sums.shape[0],), 1_000.0, jnp.float32
        ),
    )
    assert bool(
        discovery.validate_state(
            noise_like,
            semantic_generation=GENERATION,
            source_digest=SOURCE_DIGEST,
        )
    )
    result = _step(discovery, noise_like, 1)
    assert bool(result.diagnostics.transaction_applied)
    assert not bool(jnp.any(result.diagnostics.learnability_ready))


@pytest.mark.slow
def test_seeded_irreducible_noisy_tv_stream_fails_learnability_gate() -> None:
    observations = 96
    discovery = CumulantSubtaskDiscovery(
        _config(
            probe_step_size=0.02,
            learnability_evidence_floor=observations,
            learnability_threshold=0.2,
            baseline_variance_floor=0.5,
            max_observations=observations + 1,
        )
    )
    state = discovery.init(
        jr.key(531), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    noise = jnp.where(
        jr.bernoulli(jr.key(532), shape=(observations + 1,)),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(-1.0, dtype=jnp.float32),
    )

    def compiled_step(
        current_state: CumulantSubtaskDiscoveryState,
        arm_inputs: dict[str, Any],
        observe_inputs: dict[str, Any],
    ) -> CumulantSubtaskDiscoveryResult:
        arm = discovery.arm(current_state, **arm_inputs)
        return discovery.observe(current_state, arm, **observe_inputs)

    run_step = jax.jit(compiled_step)
    result: CumulantSubtaskDiscoveryResult | None = None
    for step in range(observations):
        arm_inputs = _arm_inputs(step)
        arm_inputs["current_controllable_events"] = noise[step].reshape((1,))
        arm_inputs["probe_features"] = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
        observe_inputs = _observe_inputs(step)
        observe_inputs["next_controllable_events"] = noise[step + 1].reshape((1,))
        result = run_step(state, arm_inputs, observe_inputs)
        assert bool(result.diagnostics.transaction_applied)
        state = result.state

    assert result is not None
    event_index = 0
    assert int(state.learnability_counts[event_index]) == observations
    baseline_mse = (
        state.baseline_squared_error_sums[event_index]
        / state.learnability_counts[event_index]
    )
    probe_mse = (
        state.probe_squared_error_sums[event_index]
        / state.learnability_counts[event_index]
    )
    assert float(baseline_mse) >= 0.5
    assert float(probe_mse) >= 0.8 * float(baseline_mse)
    assert float(result.diagnostics.learnability_scores[event_index]) < 0.2
    assert not bool(result.diagnostics.learnability_ready[event_index])


def test_nonfinite_or_stale_inputs_reject_atomically() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(6), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    arm = discovery.arm(state, **_arm_inputs(0))
    nonfinite = _observe_inputs(0)
    nonfinite["reward_targets"] = jnp.asarray([jnp.nan], dtype=jnp.float32)
    rejected = discovery.observe(state, arm, **nonfinite)
    assert not bool(rejected.diagnostics.transaction_valid)
    chex.assert_trees_all_equal(rejected.state, state)
    accepted = discovery.observe(state, arm, **_observe_inputs(0))
    stale = discovery.observe(accepted.state, arm, **_observe_inputs(0))
    assert not bool(stale.diagnostics.transaction_valid)
    chex.assert_trees_all_equal(stale.state, accepted.state)


def test_forged_arm_mutation_rejects_atomically() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(61), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    arm = discovery.arm(state, **_arm_inputs(0))
    forged = dataclasses.replace(
        arm,
        frozen_reward_inserted_predictions=arm.frozen_reward_inserted_predictions.at[
            0, 0
        ].add(0.5),
    )
    rejected = discovery.observe(state, forged, **_observe_inputs(0))

    assert not bool(rejected.diagnostics.arm_cache_valid)
    assert not bool(rejected.diagnostics.arm_valid)
    assert not bool(rejected.diagnostics.transaction_valid)
    assert not bool(rejected.diagnostics.transaction_applied)
    chex.assert_trees_all_equal(rejected.state, state)
    for bundle in (
        rejected.discovered,
        rejected.random_comparator,
        rejected.hand_comparator,
    ):
        assert not bool(bundle.ready)
        np.testing.assert_array_equal(
            bundle.selected_candidate_indices,
            np.full((OPTION_BUDGET,), -1, dtype=np.int32),
        )


def test_fixed_quotas_emit_only_a_complete_exact_budget_bundle() -> None:
    _, result = _ready_result()
    assert bool(result.discovered.ready)
    np.testing.assert_array_equal(
        result.diagnostics.family_selected_counts, np.ones((4,), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        result.discovered.selected_family_ids,
        np.asarray(
            [
                CUMULANT_SOURCE_CONTROLLABLE_EVENT,
                CUMULANT_SOURCE_FEATURE_CHANGE,
                CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
                CUMULANT_SOURCE_PREDICTION_BOTTLENECK,
            ],
            dtype=np.int32,
        ),
    )
    assert result.random_comparator.selected_cumulants.shape == (OPTION_BUDGET,)
    assert result.hand_comparator.selected_cumulants.shape == (OPTION_BUDGET,)


def test_materialization_uses_compact_tail_and_rejects_stale_or_tampered_bundle() -> None:
    discovery, result = _ready_result()
    raw = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    augmented = discovery.materialize(raw, result.discovered)
    np.testing.assert_array_equal(augmented[:RAW_DIM], raw)
    np.testing.assert_array_equal(
        augmented[RAW_DIM:], result.discovered.selected_cumulants
    )
    np.testing.assert_array_equal(
        result.discovered.tail_slot_indices,
        np.arange(RAW_DIM, RAW_DIM + OPTION_BUDGET, dtype=np.int32),
    )
    stale = discovery.materialize(
        raw,
        result.discovered,
        state_observation_count=result.discovered.state_observation_count + 1,
    )
    np.testing.assert_array_equal(stale[RAW_DIM:], np.zeros((OPTION_BUDGET,)))
    tampered = dataclasses.replace(
        result.discovered,
        selected_cumulants=result.discovered.selected_cumulants.at[0].add(1.0),
    )
    rejected = discovery.materialize(raw, tampered)
    np.testing.assert_array_equal(rejected[RAW_DIM:], np.zeros((OPTION_BUDGET,)))


def test_checkpoint_pytree_roundtrip_and_schema_or_state_tampering_reject() -> None:
    discovery, result = _ready_result()
    leaves, structure = jax.tree_util.tree_flatten(result.state)
    restored_tree = jax.tree_util.tree_unflatten(structure, leaves)
    chex.assert_trees_all_equal(restored_tree, result.state)
    payload = discovery.checkpoint_payload(
        result.state,
        semantic_generation=GENERATION,
        source_digest=SOURCE_DIGEST,
    )
    assert payload["schema_version"] == CUMULANT_SUBTASK_DISCOVERY_CHECKPOINT_SCHEMA
    restored = discovery.restore_checkpoint(
        payload,
        semantic_generation=GENERATION,
        source_digest=SOURCE_DIGEST,
    )
    chex.assert_trees_all_equal(restored, result.state)
    bad_schema = dict(payload)
    bad_schema["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema_version"):
        discovery.restore_checkpoint(
            bad_schema,
            semantic_generation=GENERATION,
            source_digest=SOURCE_DIGEST,
        )
    bad_state = dict(payload)
    bad_state["state"] = dataclasses.replace(
        result.state,
        random_projections=result.state.random_projections.at[0, 0].add(1.0),
    )
    with pytest.raises(ValueError, match="state digest"):
        discovery.restore_checkpoint(
            bad_state,
            semantic_generation=GENERATION,
            source_digest=SOURCE_DIGEST,
        )
    valid_value_tamper = dict(payload)
    valid_value_tamper["state"] = dataclasses.replace(
        result.state,
        probe_weights=result.state.probe_weights.at[0, 0].add(0.25),
    )
    with pytest.raises(ValueError, match="state digest"):
        discovery.restore_checkpoint(
            valid_value_tamper,
            semantic_generation=GENERATION,
            source_digest=SOURCE_DIGEST,
        )


def test_missing_comparator_cell_suppresses_all_three_matched_bundles() -> None:
    discovery, ready = _ready_result()
    state = ready.state
    step = int(state.observation_count)
    arm = discovery.arm(state, **_arm_inputs(step))
    inputs = _observe_inputs(step)
    inputs["next_hand_available"] = jnp.asarray(
        [True, True, True, False], dtype=jnp.bool_
    )
    result = discovery.observe(state, arm, **inputs)
    assert bool(result.diagnostics.transaction_applied)
    assert bool(jnp.all(result.diagnostics.family_selected_counts == 1))
    assert not bool(result.discovered.ready)
    assert not bool(result.random_comparator.ready)
    assert not bool(result.hand_comparator.ready)


def test_capacity_cap_is_valid_neutral() -> None:
    config = _config(max_observations=1)
    discovery = CumulantSubtaskDiscovery(config)
    state = discovery.init(
        jr.key(8), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    first = _step(discovery, state, 0)
    arm = discovery.arm(first.state, **_arm_inputs(1))
    capped = discovery.observe(first.state, arm, **_observe_inputs(1))
    assert bool(capped.diagnostics.transaction_valid)
    assert bool(capped.diagnostics.capacity_capped)
    assert not bool(capped.diagnostics.transaction_applied)
    assert not bool(capped.discovered.ready)
    chex.assert_trees_all_equal(capped.state, first.state)


@pytest.mark.slow
def test_arm_observe_and_materialize_are_jittable() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(9), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )
    arm_inputs = _arm_inputs(0)
    observe_inputs = _observe_inputs(0)
    arm = jax.jit(lambda value: discovery.arm(value, **arm_inputs))(state)
    result = jax.jit(
        lambda value, frozen: discovery.observe(value, frozen, **observe_inputs)
    )(state, arm)
    assert bool(result.diagnostics.transaction_applied)
    augmented = jax.jit(discovery.materialize)(
        _snapshot(1)["raw"], result.random_comparator
    )
    assert augmented.shape == (RAW_DIM + OPTION_BUDGET,)


@pytest.mark.slow
def test_two_step_scan_preserves_exact_transition_transactions() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    initial = discovery.init(
        jr.key(10), semantic_generation=GENERATION, source_digest=SOURCE_DIGEST
    )

    def body(state: CumulantSubtaskDiscoveryState, step: jax.Array):
        current = step.astype(jnp.float32)
        action = step % N_ACTIONS
        current_raw = jnp.asarray(
            [current, current * current + 0.25, 1.0 + action], dtype=jnp.float32
        )
        next_step = step + 1
        successor = next_step.astype(jnp.float32)
        next_action = next_step % N_ACTIONS
        next_raw = jnp.asarray(
            [successor, successor * successor + 0.25, 1.0 + next_action],
            dtype=jnp.float32,
        )
        arm = discovery.arm(
            state,
            current_raw_features=current_raw,
            current_raw_available=jnp.ones((RAW_DIM,), dtype=jnp.bool_),
            current_controllable_events=jnp.asarray(
                [1.0 + 2.0 * action + 0.1 * current], jnp.float32
            ),
            current_controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
            current_transition_atoms=jnp.asarray(
                [0.5 + action + 0.2 * current], jnp.float32
            ),
            current_transition_atoms_available=jnp.asarray([step > 0]),
            current_bottleneck_values=jnp.asarray(
                [2.0 - action + 0.15 * current], jnp.float32
            ),
            current_bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
            probe_features=jnp.asarray([1.0, 0.1 * current], jnp.float32),
            current_incumbent_values=jnp.asarray([20.0 + current], jnp.float32),
            current_incumbent_available=jnp.ones((1,), dtype=jnp.bool_),
            current_hand_values=current + jnp.arange(OPTION_BUDGET, dtype=jnp.float32),
            current_hand_available=jnp.ones((OPTION_BUDGET,), dtype=jnp.bool_),
            hand_comparator_identity=HAND_IDENTITY,
            reward_base_predictions=jnp.zeros((1,), jnp.float32),
            model_base_predictions=jnp.zeros((1,), jnp.float32),
            action=action,
            behavior_propensity=jnp.asarray(0.5, jnp.float32),
            randomized=jnp.asarray(True),
            transition_id=jnp.stack((jnp.uint32(0xD15C), step.astype(jnp.uint32))),
            semantic_generation=jnp.asarray(GENERATION, jnp.int32),
            source_digest=SOURCE_DIGEST,
        )
        intervention = jax.nn.one_hot(action, N_ACTIONS, dtype=jnp.bool_)
        result = discovery.observe(
            state,
            arm,
            next_raw_features=next_raw,
            next_raw_available=jnp.ones((RAW_DIM,), jnp.bool_),
            next_controllable_events=jnp.asarray(
                [1.0 + 2.0 * next_action + 0.1 * successor], jnp.float32
            ),
            next_controllable_events_available=jnp.ones((1,), jnp.bool_),
            next_transition_atoms=jnp.asarray(
                [0.5 + next_action + 0.2 * successor], jnp.float32
            ),
            next_transition_atoms_available=jnp.ones((1,), jnp.bool_),
            next_bottleneck_values=jnp.asarray(
                [2.0 - next_action + 0.15 * successor], jnp.float32
            ),
            next_bottleneck_available=jnp.ones((1,), jnp.bool_),
            bottleneck_epistemic=jnp.asarray([0.5], jnp.float32),
            bottleneck_progress=jnp.asarray([0.25], jnp.float32),
            bottleneck_aleatoric=jnp.asarray([0.1], jnp.float32),
            bottleneck_evidence_available=jnp.ones((1,), jnp.bool_),
            randomized_action_evidence=intervention,
            next_incumbent_values=jnp.asarray([20.0 + successor], jnp.float32),
            next_incumbent_available=jnp.ones((1,), jnp.bool_),
            next_hand_values=successor + jnp.arange(OPTION_BUDGET, dtype=jnp.float32),
            next_hand_available=jnp.ones((OPTION_BUDGET,), jnp.bool_),
            hand_comparator_identity=HAND_IDENTITY,
            reward_targets=jnp.zeros((1,), jnp.float32),
            reward_targets_available=jnp.ones((1,), jnp.bool_),
            model_targets=jnp.zeros((1,), jnp.float32),
            model_targets_available=jnp.ones((1,), jnp.bool_),
            transition_id=jnp.stack((jnp.uint32(0xD15C), step.astype(jnp.uint32))),
            semantic_generation=jnp.asarray(GENERATION, jnp.int32),
            source_digest=SOURCE_DIGEST,
        )
        return result.state, result.diagnostics.transaction_applied

    final, applied = jax.jit(lambda state: jax.lax.scan(body, state, jnp.arange(2)))(
        initial
    )
    np.testing.assert_array_equal(applied, np.ones((2,), dtype=np.bool_))
    assert int(final.observation_count) == 2
