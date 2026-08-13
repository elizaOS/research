# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def"
"""WP7.2 causal cumulant/subtask discovery integration contracts.

These are mechanism and isolation tests only.  They are not scientific
evidence and confer no curation, promotion, or go/no-go authority.
"""

from __future__ import annotations

from typing import Any, NamedTuple

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
    CUMULANT_SOURCE_RANDOM_PROJECTION,
    CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
    CUMULANT_SUBTASK_DISCOVERY_AUTHORITY,
    CUMULANT_SUBTASK_DISCOVERY_CHECKPOINT_SCHEMA,
    CUMULANT_SUBTASK_DISCOVERY_CONFIG_SCHEMA,
    CUMULANT_SUBTASK_DISCOVERY_GO_NO_GO_AUTHORITY,
    CUMULANT_SUBTASK_DISCOVERY_PROMOTION_AUTHORITY,
    CUMULANT_SUBTASK_DISCOVERY_RANKING_SEMANTICS,
    CUMULANT_SUBTASK_DISCOVERY_SCIENTIFIC_PROMOTION_ALLOWED,
    CumulantSubtaskDiscovery,
    CumulantSubtaskDiscoveryArm,
    CumulantSubtaskDiscoveryConfig,
    CumulantSubtaskDiscoveryDiagnostics,
    CumulantSubtaskDiscoveryResourceBudget,
    CumulantSubtaskDiscoveryResult,
    CumulantSubtaskDiscoveryState,
    CumulantSubtaskProposalBundle,
)
from alberta_framework.core.oak import OaKAgent, OaKConfig
from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import PrototypeAgent, PrototypeAgentConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]

RAW_DIM = 4
PROBE_DIM = 3
N_ACTIONS = 2
EVENT_DIM = 2
ATOM_DIM = 2
BOTTLENECK_DIM = 2
OPTION_BUDGET = 4
SEMANTIC_GENERATION = 7
SOURCE_DIGEST = jnp.asarray([0xA17E2, 0x51D3], dtype=jnp.uint32)
HAND_IDENTITY = jnp.asarray([0xCAFE, 0xF00D], dtype=jnp.uint32)


def _config() -> CumulantSubtaskDiscoveryConfig:
    """Small over-complete universe: two candidates compete for each quota."""

    return CumulantSubtaskDiscoveryConfig(
        raw_feature_dim=RAW_DIM,
        probe_feature_dim=PROBE_DIM,
        n_actions=N_ACTIONS,
        controllable_event_dim=EVENT_DIM,
        transition_atom_dim=ATOM_DIM,
        prediction_bottleneck_dim=BOTTLENECK_DIM,
        option_budget=OPTION_BUDGET,
        family_quotas=(1, 1, 1, 1),
        controllable_event_descriptors=(
            (CUMULANT_SOURCE_CONTROLLABLE_EVENT, 0, 1, 100),
            (CUMULANT_SOURCE_CONTROLLABLE_EVENT, 1, -1, 101),
        ),
        feature_change_descriptors=(
            (CUMULANT_SOURCE_FEATURE_CHANGE, 0, 1, 200),
            (CUMULANT_SOURCE_FEATURE_CHANGE, 1, -1, 201),
        ),
        reward_transition_descriptors=(
            (CUMULANT_SOURCE_REWARD_TRANSITION_ATOM, 0, 1, 300),
            (CUMULANT_SOURCE_REWARD_TRANSITION_ATOM, 1, -1, 301),
        ),
        prediction_bottleneck_descriptors=(
            (CUMULANT_SOURCE_PREDICTION_BOTTLENECK, 0, 1, 400),
            (CUMULANT_SOURCE_PREDICTION_BOTTLENECK, 1, -1, 401),
        ),
        incumbent_descriptors=((91, 0, 1, 500),),
        hand_comparator_descriptors=tuple(
            (CUMULANT_SOURCE_HAND_AUTHORED, index, 1, 600 + index)
            for index in range(OPTION_BUDGET)
        ),
        hand_comparator_identity=(
            int(HAND_IDENTITY[0]),
            int(HAND_IDENTITY[1]),
        ),
        reward_task_weights=(0.5,),
        model_task_weights=(0.5,),
        probe_step_size=0.1,
        shadow_step_size=0.1,
        learnability_evidence_floor=1,
        controllability_evidence_floor_per_action=1,
        novelty_evidence_floor=1,
        contribution_evidence_floor=1,
        bottleneck_evidence_floor=1,
        learnability_threshold=0.0,
        controllability_threshold=0.0,
        novelty_threshold=1.0e-12,
        contribution_threshold=0.0,
        bottleneck_epistemic_floor=0.0,
        bottleneck_progress_floor=0.0,
        bottleneck_aleatoric_ceiling=10.0,
        max_observations=64,
    )


def _snapshot(step: int) -> dict[str, jax.Array]:
    """Deterministic, action-correlated finite source values at state ``step``."""

    parity = -1.0 if step % 2 else 1.0
    phase = float((step % 3) - 1)
    value = float(step)
    return {
        "raw": jnp.asarray(
            [0.25 * value, 0.1 * value * value, parity, phase], dtype=jnp.float32
        ),
        "events": jnp.asarray([0.4 * value + parity, phase + 0.25], dtype=jnp.float32),
        "atoms": jnp.asarray(
            [0.3 * value + 0.5 * parity, parity * (0.5 + 0.1 * value)],
            dtype=jnp.float32,
        ),
        "bottleneck": jnp.asarray(
            [0.2 * value + parity, 0.3 * phase - 0.1 * value], dtype=jnp.float32
        ),
        "probe": jnp.asarray([1.0, parity, 0.2 * value], dtype=jnp.float32),
        "incumbent": jnp.asarray([0.17 * value + 0.03 * parity], dtype=jnp.float32),
        "hand": jnp.asarray(
            [parity, phase, 0.1 * value, parity * phase], dtype=jnp.float32
        ),
    }


def _transition_id(step: int) -> jax.Array:
    return jnp.asarray([0xD15C0, step], dtype=jnp.uint32)


def _arm_inputs(
    step: int,
    *,
    transition_atoms_available: bool | None = None,
) -> dict[str, Any]:
    current = _snapshot(step)
    atom_available = step > 0 if transition_atoms_available is None else transition_atoms_available
    return {
        "current_raw_features": current["raw"],
        "current_raw_available": jnp.ones((RAW_DIM,), dtype=jnp.bool_),
        "current_controllable_events": current["events"],
        "current_controllable_events_available": jnp.ones((EVENT_DIM,), dtype=jnp.bool_),
        "current_transition_atoms": current["atoms"],
        "current_transition_atoms_available": jnp.full(
            (ATOM_DIM,), atom_available, dtype=jnp.bool_
        ),
        "current_bottleneck_values": current["bottleneck"],
        "current_bottleneck_available": jnp.ones((BOTTLENECK_DIM,), dtype=jnp.bool_),
        "probe_features": current["probe"],
        "current_incumbent_values": current["incumbent"],
        "current_incumbent_available": jnp.ones((1,), dtype=jnp.bool_),
        "current_hand_values": current["hand"],
        "current_hand_available": jnp.ones((OPTION_BUDGET,), dtype=jnp.bool_),
        "hand_comparator_identity": HAND_IDENTITY,
        "reward_base_predictions": jnp.asarray([0.05 * step], dtype=jnp.float32),
        "model_base_predictions": jnp.asarray([-0.04 * step], dtype=jnp.float32),
        "action": jnp.asarray(step % N_ACTIONS, dtype=jnp.int32),
        "behavior_propensity": jnp.asarray(0.5, dtype=jnp.float32),
        "randomized": jnp.asarray(True),
        "transition_id": _transition_id(step),
        "semantic_generation": jnp.asarray(SEMANTIC_GENERATION, dtype=jnp.int32),
        "source_digest": SOURCE_DIGEST,
    }


def _observe_inputs(
    step: int,
    *,
    bottleneck_available: bool = True,
) -> dict[str, Any]:
    successor = _snapshot(step + 1)
    action_evidence = jnp.zeros((N_ACTIONS,), dtype=jnp.bool_).at[step % N_ACTIONS].set(True)
    return {
        "next_raw_features": successor["raw"],
        "next_raw_available": jnp.ones((RAW_DIM,), dtype=jnp.bool_),
        "next_controllable_events": successor["events"],
        "next_controllable_events_available": jnp.ones((EVENT_DIM,), dtype=jnp.bool_),
        "next_transition_atoms": successor["atoms"],
        "next_transition_atoms_available": jnp.ones((ATOM_DIM,), dtype=jnp.bool_),
        "next_bottleneck_values": successor["bottleneck"],
        "next_bottleneck_available": jnp.full(
            (BOTTLENECK_DIM,), bottleneck_available, dtype=jnp.bool_
        ),
        "bottleneck_epistemic": jnp.asarray([0.5, 0.7], dtype=jnp.float32),
        "bottleneck_progress": jnp.asarray([0.2, 0.3], dtype=jnp.float32),
        "bottleneck_aleatoric": jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        "bottleneck_evidence_available": jnp.full(
            (BOTTLENECK_DIM,), bottleneck_available, dtype=jnp.bool_
        ),
        "randomized_action_evidence": action_evidence,
        "next_incumbent_values": successor["incumbent"],
        "next_incumbent_available": jnp.ones((1,), dtype=jnp.bool_),
        "next_hand_values": successor["hand"],
        "next_hand_available": jnp.ones((OPTION_BUDGET,), dtype=jnp.bool_),
        "hand_comparator_identity": HAND_IDENTITY,
        "reward_targets": jnp.asarray([0.4 + 0.07 * step], dtype=jnp.float32),
        "reward_targets_available": jnp.ones((1,), dtype=jnp.bool_),
        "model_targets": jnp.asarray([-0.3 + 0.05 * step], dtype=jnp.float32),
        "model_targets_available": jnp.ones((1,), dtype=jnp.bool_),
        "transition_id": _transition_id(step),
        "semantic_generation": jnp.asarray(SEMANTIC_GENERATION, dtype=jnp.int32),
        "source_digest": SOURCE_DIGEST,
    }


def _observe_step(
    discovery: CumulantSubtaskDiscovery,
    state: CumulantSubtaskDiscoveryState,
    step: int,
    *,
    bottleneck_available: bool = True,
) -> tuple[CumulantSubtaskDiscoveryArm, CumulantSubtaskDiscoveryResult]:
    arm = discovery.arm(state, **_arm_inputs(step))
    result = discovery.observe(
        state,
        arm,
        **_observe_inputs(step, bottleneck_available=bottleneck_available),
    )
    return arm, result


def _materialize_typed_keys(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)
        return value

    return jax.tree.map(convert, tree)


class _ReadyCohorts(NamedTuple):
    discovery: CumulantSubtaskDiscovery
    result: CumulantSubtaskDiscoveryResult
    next_step: int


@pytest.fixture(scope="module")
def ready_cohorts() -> _ReadyCohorts:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(81),
        semantic_generation=SEMANTIC_GENERATION,
        source_digest=SOURCE_DIGEST,
    )
    last_result: CumulantSubtaskDiscoveryResult | None = None
    for step in range(48):
        _, last_result = _observe_step(discovery, state, step)
        assert bool(last_result.diagnostics.transaction_applied)
        state = last_result.state
        if bool(last_result.discovered.ready):
            return _ReadyCohorts(discovery, last_result, step + 1)
    assert last_result is not None
    pytest.fail(
        "permissive four-family trajectory never produced a complete proposal; "
        f"family counts={np.asarray(last_result.diagnostics.family_selected_counts)}, "
        f"local gates={np.asarray(last_result.diagnostics.all_local_gates_ready)}"
    )


def test_wp72_api_surface_is_versioned_and_has_no_authority() -> None:
    config = _config()
    discovery = CumulantSubtaskDiscovery(config)

    assert CUMULANT_SUBTASK_DISCOVERY_CONFIG_SCHEMA
    assert CUMULANT_SUBTASK_DISCOVERY_CHECKPOINT_SCHEMA
    assert CUMULANT_SUBTASK_DISCOVERY_RANKING_SEMANTICS
    assert CumulantSubtaskDiscoveryState is not None
    assert CumulantSubtaskDiscoveryArm is not None
    assert CumulantSubtaskDiscoveryDiagnostics is not None
    assert CumulantSubtaskProposalBundle is not None
    assert CumulantSubtaskDiscoveryResult is not None
    assert CumulantSubtaskDiscoveryResourceBudget is not None
    assert CumulantSubtaskDiscoveryConfig.from_config(discovery.to_config()) == config

    budget = discovery.resource_budget
    assert budget.option_budget == OPTION_BUDGET
    assert budget.candidate_count == config.candidate_count
    assert budget.backward_passes_per_observe == 0
    assert budget.consumer_updates_per_observe == 0
    assert budget.router_calls_per_observe == 0
    assert budget.horde_updates_per_observe == 0
    assert budget.option_updates_per_observe == 0
    assert budget.promotion_decisions_per_observe == 0
    assert not CUMULANT_SUBTASK_DISCOVERY_AUTHORITY
    assert not CUMULANT_SUBTASK_DISCOVERY_PROMOTION_AUTHORITY
    assert not CUMULANT_SUBTASK_DISCOVERY_GO_NO_GO_AUTHORITY
    assert not CUMULANT_SUBTASK_DISCOVERY_SCIENTIFIC_PROMOTION_ALLOWED
    assert not budget.curation_authority
    assert not budget.promotion_authority
    assert not budget.go_no_go_authority
    assert not budget.scientific_promotion_allowed


def test_arm_observe_is_forward_shifted_and_reward_birth_has_no_same_step_evidence() -> None:
    config = _config()
    discovery = CumulantSubtaskDiscovery(config)
    state0 = discovery.init(
        jr.key(1),
        semantic_generation=SEMANTIC_GENERATION,
        source_digest=SOURCE_DIGEST,
    )

    arm0, first = _observe_step(discovery, state0, 0)
    assert bool(arm0.available)
    assert bool(first.diagnostics.transaction_valid)
    assert bool(first.diagnostics.transaction_applied)
    assert int(first.state.observation_count) == 1
    np.testing.assert_array_equal(first.state.last_transition_id, _transition_id(0))

    current = _snapshot(0)
    successor = _snapshot(1)
    expected_successor_candidates = jnp.concatenate(
        (
            successor["events"] * jnp.asarray([1.0, -1.0], dtype=jnp.float32),
            (successor["raw"][:2] - current["raw"][:2])
            * jnp.asarray([1.0, -1.0], dtype=jnp.float32),
            successor["atoms"] * jnp.asarray([1.0, -1.0], dtype=jnp.float32),
            successor["bottleneck"] * jnp.asarray([1.0, -1.0], dtype=jnp.float32),
        )
    )
    chex.assert_trees_all_close(
        first.state.last_candidate_values,
        expected_successor_candidates,
        atol=1.0e-6,
    )

    descriptors = np.asarray(config.candidate_descriptors, dtype=np.int32)
    reward_indices = np.flatnonzero(
        descriptors[:, 0] == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM
    )
    np.testing.assert_array_equal(
        np.asarray(first.diagnostics.reward_births_this_transition)[reward_indices],
        np.ones((len(reward_indices),), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        np.asarray(first.state.task_contribution_counts)[reward_indices],
        np.zeros((len(reward_indices), config.task_count), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(first.state.reward_shadow_weights)[reward_indices],
        np.zeros((len(reward_indices), len(config.reward_task_weights)), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(first.state.model_shadow_weights)[reward_indices],
        np.zeros((len(reward_indices), len(config.model_task_weights)), dtype=np.float32),
    )

    misaligned_inputs = _arm_inputs(1)
    misaligned_inputs["current_controllable_events"] = (
        misaligned_inputs["current_controllable_events"].at[0].add(1.0)
    )
    misaligned_arm = discovery.arm(first.state, **misaligned_inputs)
    assert not bool(misaligned_arm.available)

    arm1, second = _observe_step(discovery, first.state, 1)
    assert bool(arm1.available)
    chex.assert_trees_all_close(
        arm1.current_candidate_values,
        first.state.last_candidate_values,
        atol=1.0e-6,
    )
    chex.assert_trees_all_equal(
        arm1.current_candidate_available,
        first.state.last_candidate_available,
    )
    assert bool(second.diagnostics.transaction_applied)
    np.testing.assert_array_equal(
        np.asarray(second.state.task_contribution_counts)[reward_indices],
        np.ones((len(reward_indices), config.task_count), dtype=np.int32),
    )


def test_all_four_fixed_quotas_and_matched_budgets_are_exact(
    ready_cohorts: _ReadyCohorts,
) -> None:
    config = ready_cohorts.discovery.config
    result = ready_cohorts.result

    assert bool(result.diagnostics.bundle_ready)
    np.testing.assert_array_equal(
        result.diagnostics.family_quotas,
        np.asarray(config.family_quotas, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.diagnostics.family_selected_counts,
        np.asarray(config.family_quotas, dtype=np.int32),
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

    expected_tail_slots = np.arange(RAW_DIM, RAW_DIM + OPTION_BUDGET, dtype=np.int32)
    for bundle in (result.discovered, result.random_comparator, result.hand_comparator):
        assert bool(bundle.ready)
        assert np.asarray(bundle.selected_descriptors).shape == (OPTION_BUDGET, 4)
        assert np.asarray(bundle.selected_cumulants).shape == (OPTION_BUDGET,)
        np.testing.assert_array_equal(bundle.tail_slot_indices, expected_tail_slots)

    np.testing.assert_array_equal(
        result.random_comparator.selected_family_ids,
        np.full((OPTION_BUDGET,), CUMULANT_SOURCE_RANDOM_PROJECTION, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.hand_comparator.selected_family_ids,
        np.full((OPTION_BUDGET,), CUMULANT_SOURCE_HAND_AUTHORED, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.random_comparator.selected_descriptors,
        ready_cohorts.discovery.random_comparator_descriptors,
    )
    np.testing.assert_array_equal(
        result.hand_comparator.selected_descriptors,
        ready_cohorts.discovery.hand_comparator_descriptors,
    )
    assert not np.array_equal(
        np.asarray(result.discovered.selected_candidate_indices),
        np.asarray(result.discovered.tail_slot_indices),
    )


def test_missing_family_never_emits_a_partial_discovered_proposal() -> None:
    discovery = CumulantSubtaskDiscovery(_config())
    state = discovery.init(
        jr.key(9),
        semantic_generation=SEMANTIC_GENERATION,
        source_digest=SOURCE_DIGEST,
    )
    last: CumulantSubtaskDiscoveryResult | None = None
    for step in range(40):
        arm = discovery.arm(
            state,
            **{
                **_arm_inputs(step),
                "current_bottleneck_available": jnp.zeros(
                    (BOTTLENECK_DIM,), dtype=jnp.bool_
                ),
            },
        )
        last = discovery.observe(
            state,
            arm,
            **_observe_inputs(step, bottleneck_available=False),
        )
        assert bool(last.diagnostics.transaction_applied)
        assert not bool(last.discovered.ready)
        state = last.state

    assert last is not None
    families = np.asarray(discovery.config.candidate_descriptors, dtype=np.int32)[:, 0]
    local_ready = np.asarray(last.diagnostics.all_local_gates_ready)
    for family in (
        CUMULANT_SOURCE_CONTROLLABLE_EVENT,
        CUMULANT_SOURCE_FEATURE_CHANGE,
        CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
    ):
        assert np.any(local_ready[families == family])
    assert not np.any(local_ready[families == CUMULANT_SOURCE_PREDICTION_BOTTLENECK])
    np.testing.assert_array_equal(
        last.diagnostics.family_selected_counts,
        np.asarray([1, 1, 1, 0], dtype=np.int32),
    )
    assert int(jnp.sum(last.diagnostics.selected_mask)) == 3
    assert np.all(np.asarray(last.discovered.selected_candidate_indices) == -1)
    assert np.all(np.asarray(last.discovered.selected_family_ids) == -1)
    assert np.all(np.asarray(last.discovered.selected_descriptors) == 0)
    assert np.all(np.asarray(last.discovered.selected_scores) == 0.0)
    assert np.all(np.asarray(last.discovered.selected_cumulants) == 0.0)


def _matched_stomp_config(bundle: CumulantSubtaskProposalBundle) -> STOMPConfig:
    specs = tuple(
        SubtaskSpec(
            feature_index=int(feature_index),
            threshold=0.5,
            pseudo_reward_scale=1.0,
            max_option_steps=8,
        )
        for feature_index in np.asarray(bundle.tail_slot_indices)
    )
    return STOMPConfig(
        subtask_specs=specs,
        observation_dim=RAW_DIM + OPTION_BUDGET,
        n_primitive_actions=N_ACTIONS,
        base_step_size=0.05,
        option_step_size=0.05,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )


def test_compact_tail_indices_feed_three_identically_budgeted_fresh_stomp_cohorts(
    ready_cohorts: _ReadyCohorts,
) -> None:
    discovery = ready_cohorts.discovery
    bundles = (
        ready_cohorts.result.discovered,
        ready_cohorts.result.random_comparator,
        ready_cohorts.result.hand_comparator,
    )
    expected_tail_slots = tuple(range(RAW_DIM, RAW_DIM + OPTION_BUDGET))
    configs = tuple(_matched_stomp_config(bundle) for bundle in bundles)

    assert configs[0] == configs[1] == configs[2]
    for config in configs:
        assert config.n_options == OPTION_BUDGET
        assert tuple(spec.feature_index for spec in config.subtask_specs) == expected_tail_slots
        assert all(spec.feature_index < config.observation_dim for spec in config.subtask_specs)
        assert all(spec.threshold == 0.5 for spec in config.subtask_specs)
        assert all(spec.pseudo_reward_scale == 1.0 for spec in config.subtask_specs)
        assert all(spec.max_option_steps == 8 for spec in config.subtask_specs)

    raw0 = _snapshot(ready_cohorts.next_step)["raw"]
    raw1 = _snapshot(ready_cohorts.next_step + 1)["raw"]
    updates = []
    for seed, (bundle, config) in enumerate(zip(bundles, configs, strict=True), start=31):
        observation0 = discovery.materialize(raw0, bundle)
        observation1 = discovery.materialize(raw1, bundle)
        assert observation0.shape == (RAW_DIM + OPTION_BUDGET,)
        np.testing.assert_array_equal(observation0[:RAW_DIM], raw0)
        np.testing.assert_array_equal(observation0[RAW_DIM:], bundle.selected_cumulants)

        agent = STOMPAgent(config)
        started = agent.start(agent.init(jr.key(seed)), observation0)
        update = agent.update(
            started,
            jnp.asarray(0.25, dtype=jnp.float32),
            observation1,
            jnp.asarray(0.9, dtype=jnp.float32),
        )
        updates.append(update)

    assert len(updates) == 3
    for update in updates:
        assert int(update.state.step_count) == 1
        assert bool(jnp.isfinite(update.td_error))
        assert bool(jnp.isfinite(update.average_reward))
        assert bool(jnp.isfinite(update.pseudo_reward))
        assert bool(jnp.isfinite(update.planning_td_error))


def test_stale_arm_cannot_emit_a_proposal_or_change_current_state(
    ready_cohorts: _ReadyCohorts,
) -> None:
    discovery = ready_cohorts.discovery
    current = ready_cohorts.result.state
    step = ready_cohorts.next_step
    arm = discovery.arm(current, **_arm_inputs(step))
    accepted = discovery.observe(current, arm, **_observe_inputs(step))
    assert bool(accepted.diagnostics.transaction_applied)
    assert bool(accepted.discovered.ready)

    proposal = accepted.discovered
    assert bool(
        discovery.validate_proposal_bundle(
            proposal,
            semantic_generation=proposal.semantic_generation,
            source_digest=proposal.source_digest,
            canonical_digest=proposal.canonical_digest,
            transition_id=proposal.transition_id,
            state_observation_count=proposal.state_observation_count,
        )
    )
    stale_revision = proposal.state_observation_count + jnp.asarray(1, dtype=jnp.int32)
    assert not bool(
        discovery.validate_proposal_bundle(
            proposal,
            semantic_generation=proposal.semantic_generation,
            source_digest=proposal.source_digest,
            canonical_digest=proposal.canonical_digest,
            transition_id=proposal.transition_id,
            state_observation_count=stale_revision,
        )
    )
    stale_materialization = discovery.materialize(
        _snapshot(step + 1)["raw"],
        proposal,
        semantic_generation=proposal.semantic_generation,
        source_digest=proposal.source_digest,
        canonical_digest=proposal.canonical_digest,
        transition_id=proposal.transition_id,
        state_observation_count=stale_revision,
    )
    np.testing.assert_array_equal(
        stale_materialization[RAW_DIM:],
        np.zeros((OPTION_BUDGET,), dtype=np.float32),
    )

    stale = discovery.observe(accepted.state, arm, **_observe_inputs(step))
    assert not bool(stale.diagnostics.transaction_valid)
    assert not bool(stale.diagnostics.transaction_applied)
    chex.assert_trees_all_equal(stale.state, accepted.state)
    for bundle in (stale.discovered, stale.random_comparator, stale.hand_comparator):
        assert not bool(bundle.ready)
        assert np.all(np.asarray(bundle.selected_candidate_indices) == -1)


def test_discovery_calls_cannot_mutate_oak_or_prototype_state(
    ready_cohorts: _ReadyCohorts,
) -> None:
    bundle = ready_cohorts.result.discovered
    stomp_config = _matched_stomp_config(bundle)
    zero_observation = jnp.zeros((stomp_config.observation_dim,), dtype=jnp.float32)

    oak = OaKAgent(OaKConfig(stomp=stomp_config))
    oak_state = oak.start(oak.init(jr.key(71)), zero_observation)
    prototype = PrototypeAgent(PrototypeAgentConfig(oak=OaKConfig(stomp=stomp_config)))
    prototype_state = prototype.start(prototype.init(jr.key(72)), zero_observation)
    oak_before = _materialize_typed_keys(oak_state)
    prototype_before = _materialize_typed_keys(prototype_state)

    discovery = ready_cohorts.discovery
    step = ready_cohorts.next_step
    arm = discovery.arm(ready_cohorts.result.state, **_arm_inputs(step))
    observed = discovery.observe(
        ready_cohorts.result.state,
        arm,
        **_observe_inputs(step),
    )
    assert bool(observed.diagnostics.transaction_applied)
    materialized = discovery.materialize(_snapshot(step + 1)["raw"], observed.discovered)
    assert materialized.shape == (RAW_DIM + OPTION_BUDGET,)

    chex.assert_trees_all_equal(oak_before, _materialize_typed_keys(oak_state))
    chex.assert_trees_all_equal(
        prototype_before,
        _materialize_typed_keys(prototype_state),
    )
