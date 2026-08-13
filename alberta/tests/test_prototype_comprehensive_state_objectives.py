"""Unit contracts for the Prototype/comprehensive WP3 transaction."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.comprehensive_state_objectives import (
    ComprehensiveStateObjectives,
    ComprehensiveStateObjectivesConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.prototype_comprehensive_state_objectives import (
    PROTOTYPE_COMPREHENSIVE_OBJECTIVES_EVIDENCE_LEVEL,
    PROTOTYPE_COMPREHENSIVE_OBJECTIVES_MAX_TRANSITIONS,
    PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OUTCOME_STATUS,
    PrototypeComprehensiveObjectivesState,
    PrototypeComprehensiveStateObjectives,
    PrototypeComprehensiveTargetReceipt,
    measure_prototype_comprehensive_objectives_state_nbytes,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
)
from alberta_framework.core.state_builder import OnlineGatedStateBuilderConfig

pytestmark = pytest.mark.unit

RAW_DIM = 2
HIDDEN_DIM = 1
FEATURE_DIM = RAW_DIM + HIDDEN_DIM
N_ACTIONS = 2


@pytest.fixture(autouse=True)
def _eager_jax() -> Any:
    with jax.disable_jit():
        yield


def _builder_config() -> OnlineGatedStateBuilderConfig:
    return OnlineGatedStateBuilderConfig(
        observation_dim=RAW_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        include_raw_observation=True,
        step_size=0.05,
        gradient_clip=5.0,
        initialization_scale=0.1,
    )


def _prototype(*, learn_from_model: bool = False) -> PrototypeAgent:
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(SubtaskSpec(feature_index=0),),
                    observation_dim=FEATURE_DIM,
                    n_primitive_actions=N_ACTIONS,
                    base_hidden_sizes=(),
                    base_step_size=0.02,
                    option_step_size=0.02,
                    epsilon_base=0.0,
                    epsilon_option=0.0,
                )
            ),
            state_builder=_builder_config(),
            representation_gradient_mixer=(
                RepresentationGradientMixerConfig(
                    representation_dim=FEATURE_DIM,
                    mode="behavior_only",
                )
                if learn_from_model
                else None
            ),
        )
    )


def _objectives(
    *,
    representation_dim: int = FEATURE_DIM,
    observation_target_dim: int = RAW_DIM,
) -> ComprehensiveStateObjectives:
    return ComprehensiveStateObjectives(
        ComprehensiveStateObjectivesConfig(
            representation_dim=representation_dim,
            observation_target_dim=observation_target_dim,
            n_actions=N_ACTIONS,
            gvf_discounts=(0.25, 0.75, 0.95),
            initialization_scale=0.08,
            representation_gradient_clip=10.0,
        )
    )


def _adapter() -> PrototypeComprehensiveStateObjectives:
    return PrototypeComprehensiveStateObjectives(_prototype(), _objectives())


def _transition(
    state: PrototypeComprehensiveObjectivesState,
    next_observation: jax.Array,
    *,
    reward: float = 0.3,
    discount: float = 0.9,
    decision_id: jax.Array | None = None,
) -> PrototypeTransition:
    prototype = state.prototype_state
    return PrototypeTransition(  # type: ignore[call-arg]
        observation=prototype.current_raw_observation,
        action=prototype.current_action,
        decision_id=(prototype.current_decision_id if decision_id is None else decision_id),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
    )


def _targets(
    adapter: PrototypeComprehensiveStateObjectives,
    state: PrototypeComprehensiveObjectivesState,
) -> PrototypeComprehensiveTargetReceipt:
    return adapter.make_target_receipt(
        state,
        cumulant=jnp.asarray(0.2, dtype=jnp.float32),
        gvf_continuation=jnp.asarray(0.85, dtype=jnp.float32),
        control_value_target=jnp.asarray(0.4, dtype=jnp.float32),
        selected_action_advantage_target=jnp.asarray(-0.15, dtype=jnp.float32),
        source_revision_words=jnp.asarray([3, 7], dtype=jnp.uint32),
        provenance_words=jnp.asarray([11, 13, 17, 19], dtype=jnp.uint32),
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            dtype, jax.dtypes.prng_key
        ):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def test_config_is_strict_l0_explicit_and_round_trips() -> None:
    adapter = _adapter()
    config = adapter.to_config()
    assert config["evidence_level"] == PROTOTYPE_COMPREHENSIVE_OBJECTIVES_EVIDENCE_LEVEL
    assert config["evidence_level"] == "L0"
    assert config["outcome_status"] == PROTOTYPE_COMPREHENSIVE_OBJECTIVES_OUTCOME_STATUS
    assert config["outcome_status"] == "not_assessed"
    assert config["activation"] == "explicit-adapter-only"
    assert "rtu_generate_and_test_config" not in config
    assert "no-privileged-regime-labels-or-inferred-control-targets" in config["limitations"]
    restored = PrototypeComprehensiveStateObjectives.from_config(config)
    assert restored.to_config() == config

    missing = dict(config)
    missing.pop("ownership")
    with pytest.raises(ValueError, match="manifest is not exact"):
        PrototypeComprehensiveStateObjectives.from_config(missing)
    invented_lifecycle = dict(config)
    invented_lifecycle["rtu_generate_and_test_config"] = None
    with pytest.raises(TypeError, match="exact dict"):
        PrototypeComprehensiveStateObjectives.from_config(invented_lifecycle)


def test_adapter_construction_is_inert_for_the_ordinary_prototype_path() -> None:
    prototype = _prototype()
    observation = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
    before = prototype.start(prototype.init(jr.key(1)), observation)
    PrototypeComprehensiveStateObjectives(prototype, _objectives())
    after = prototype.start(prototype.init(jr.key(1)), observation)
    chex.assert_trees_all_equal(_materialize_keys(before), _materialize_keys(after))


def test_constructor_rejects_dimension_and_concurrent_learning_mismatches() -> None:
    with pytest.raises(ValueError, match="representation_dim"):
        PrototypeComprehensiveStateObjectives(
            _prototype(),
            _objectives(representation_dim=FEATURE_DIM + 1),
        )
    with pytest.raises(ValueError, match="observation_target_dim"):
        PrototypeComprehensiveStateObjectives(
            _prototype(),
            _objectives(observation_target_dim=RAW_DIM + 1),
        )
    with pytest.raises(ValueError, match="representation gradient mixing disabled"):
        PrototypeComprehensiveStateObjectives(_prototype(learn_from_model=True), _objectives())


def test_init_typed_rng_and_resource_partition_are_exact() -> None:
    adapter = _adapter()
    with pytest.raises(TypeError, match="typed Threefry"):
        adapter.init(jr.PRNGKey(2))
    state = adapter.init(jr.key(2), lifecycle_id=jnp.asarray([7, 9], dtype=jnp.uint32))
    assert bool(adapter.state_valid(state))
    assert state.rtu_generate_and_test_state is None
    assert not bool(state.pending_valid)
    budget = adapter.resource_budget(state)
    assert budget.total_state_nbytes == measure_prototype_comprehensive_objectives_state_nbytes(
        state
    )
    assert budget.max_objective_parameter_head_updates_per_transition == 8
    assert budget.max_causal_deletion_units_scored_per_transition == 0
    assert budget.max_causal_deletion_frozen_head_evaluations_per_transition == 0
    assert budget.max_builder_proposals_per_transition == 2
    assert budget.max_builder_commits_per_transition == 1
    assert budget.rtu_generate_and_test_state_nbytes == 0
    assert budget.max_rtu_generate_and_test_proposals_per_transition == 0
    assert budget.max_rtu_generate_and_test_commits_per_transition == 0
    assert budget.max_target_receipts_consumed_per_transition == 1
    assert budget.max_accepted_transitions == PROTOTYPE_COMPREHENSIVE_OBJECTIVES_MAX_TRANSITIONS
    assert budget.max_accepted_transitions == 2**64 - 2


def test_start_binds_exact_decision_representation_and_online_owner() -> None:
    adapter = _adapter()
    started = adapter.start(
        adapter.init(jr.key(3)),
        jnp.asarray([0.25, -0.5], dtype=jnp.float32),
    )
    assert bool(started.start_applied)
    state = started.state
    assert bool(adapter.state_valid(state))
    np.testing.assert_array_equal(
        state.pending_prototype_decision_id,
        state.prototype_state.current_decision_id,
    )
    np.testing.assert_array_equal(
        state.objectives_state.pending_representation,
        state.prototype_state.current_representation,
    )
    np.testing.assert_array_equal(
        state.objectives_state.pending_representation_revision_words,
        state.prototype_state.observation_event_words,
    )
    builder_state = state.prototype_state.state_builder_state
    np.testing.assert_array_equal(state.pending_builder_step_words, builder_state.step_words)
    np.testing.assert_array_equal(
        state.pending_builder_update_words,
        builder_state.update_words,
    )


def test_valid_transition_commits_every_owner_and_builder_exactly_once() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(4)),
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
    ).state
    before_builder = state.prototype_state.state_builder_state.parameters
    before_heads = state.objectives_state.observation_weights
    target_receipt = _targets(adapter, state)
    result = adapter.update_transition(
        state,
        _transition(state, jnp.asarray([0.4, 0.3], dtype=jnp.float32)),
        target_receipt,
    )
    assert bool(result.update_applied)
    assert bool(result.target_receipt_committed)
    assert bool(result.target_owner_matches)
    assert bool(result.target_payload_matches)
    assert bool(result.target_content_tag_matches)
    assert bool(result.target_provenance_valid)
    assert bool(result.target_source_revision_valid)
    assert bool(result.prototype_transaction_applied)
    assert bool(result.objective_transaction_applied)
    assert bool(result.builder_transaction_applied)
    assert bool(result.builder_sources_match)
    assert bool(result.builder_destination_matches)
    assert bool(adapter.state_valid(result.state))
    np.testing.assert_array_equal(result.state.transaction_words, [0, 1])
    np.testing.assert_array_equal(result.state.target_receipt_words, [0, 1])
    np.testing.assert_array_equal(result.state.objectives_state.update_words, [0, 1])
    np.testing.assert_array_equal(result.state.prototype_state.step_words, [0, 1])
    np.testing.assert_array_equal(
        result.state.objectives_state.head_revision_words,
        np.broadcast_to(np.asarray([0, 1], dtype=np.uint32), (8, 2)),
    )
    np.testing.assert_array_equal(result.builder_learning.pre_update_words, [0, 0])
    np.testing.assert_array_equal(result.builder_learning.post_update_words, [0, 1])
    np.testing.assert_array_equal(
        result.state.pending_builder_update_words,
        result.builder_learning.pre_update_words,
    )
    np.testing.assert_array_equal(
        result.state.prototype_state.state_builder_state.update_words,
        result.builder_learning.post_update_words,
    )
    np.testing.assert_array_equal(
        result.state.last_target_source_revision_words,
        target_receipt.source_revision_words,
    )
    np.testing.assert_array_equal(
        result.state.last_target_provenance_words,
        target_receipt.provenance_words,
    )
    assert not np.array_equal(
        np.asarray(result.state.prototype_state.state_builder_state.parameters),
        np.asarray(before_builder),
    )
    assert not np.array_equal(
        np.asarray(result.state.objectives_state.observation_weights),
        np.asarray(before_heads),
    )
    tampered_record = dataclasses.replace(  # type: ignore[type-var]
        result.state,
        last_target_payload_words=result.state.last_target_payload_words.at[0].add(
            jnp.asarray(1, dtype=jnp.uint32)
        ),
    )
    assert not bool(adapter.state_valid(tampered_record))


def test_target_value_or_source_tampering_fails_closed_and_remains_retryable() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(5)),
        jnp.asarray([-0.3, 0.7], dtype=jnp.float32),
    ).state
    transition = _transition(state, jnp.asarray([0.1, -0.2], dtype=jnp.float32))
    receipt = _targets(adapter, state)

    value_tampered = dataclasses.replace(  # type: ignore[type-var]
        receipt,
        cumulant=receipt.cumulant + jnp.asarray(0.25, dtype=jnp.float32),
    )
    rejected_value = adapter.update_transition(state, transition, value_tampered)
    assert not bool(rejected_value.target_payload_matches)
    assert not bool(rejected_value.update_applied)
    chex.assert_trees_all_equal(
        _materialize_keys(rejected_value.state),
        _materialize_keys(state),
    )

    source_tampered = dataclasses.replace(  # type: ignore[type-var]
        receipt,
        source_revision_words=receipt.source_revision_words.at[1].add(
            jnp.asarray(1, dtype=jnp.uint32)
        ),
    )
    rejected_source = adapter.update_transition(state, transition, source_tampered)
    assert not bool(rejected_source.target_content_tag_matches)
    assert not bool(rejected_source.update_applied)
    chex.assert_trees_all_equal(
        _materialize_keys(rejected_source.state),
        _materialize_keys(state),
    )

    accepted = adapter.update_transition(state, transition, receipt)
    assert bool(accepted.update_applied)


def test_canonical_receipt_from_an_earlier_target_source_revision_is_stale() -> None:
    adapter = _adapter()
    initial = adapter.start(
        adapter.init(jr.key(9)),
        jnp.asarray([0.4, -0.2], dtype=jnp.float32),
    ).state
    first_receipt = _targets(adapter, initial)
    first = adapter.update_transition(
        initial,
        _transition(initial, jnp.asarray([0.1, 0.5], dtype=jnp.float32)),
        first_receipt,
    )
    assert bool(first.update_applied)
    state = first.state
    next_transition = _transition(state, jnp.asarray([-0.3, 0.6], dtype=jnp.float32))
    stale_owner = adapter.update_transition(state, next_transition, first_receipt)
    assert not bool(stale_owner.target_owner_matches)
    assert not bool(stale_owner.update_applied)
    chex.assert_trees_all_equal(
        _materialize_keys(stale_owner.state),
        _materialize_keys(state),
    )

    stale = adapter.make_target_receipt(
        state,
        cumulant=jnp.asarray(0.2, dtype=jnp.float32),
        gvf_continuation=jnp.asarray(0.85, dtype=jnp.float32),
        control_value_target=jnp.asarray(0.4, dtype=jnp.float32),
        selected_action_advantage_target=jnp.asarray(-0.15, dtype=jnp.float32),
        source_revision_words=jnp.asarray([3, 6], dtype=jnp.uint32),
        provenance_words=jnp.asarray([11, 13, 17, 19], dtype=jnp.uint32),
    )
    result = adapter.update_transition(
        state,
        next_transition,
        stale,
    )
    assert bool(result.target_content_tag_matches)
    assert not bool(result.target_source_revision_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(_materialize_keys(result.state), _materialize_keys(state))


def test_stale_decision_and_owner_revision_fail_closed() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(6)),
        jnp.asarray([0.5, 0.2], dtype=jnp.float32),
    ).state
    receipt = _targets(adapter, state)
    stale_id = state.prototype_state.current_decision_id.at[3].add(jnp.asarray(1, dtype=jnp.uint32))
    rejected = adapter.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([-0.1, 0.3], dtype=jnp.float32),
            decision_id=stale_id,
        ),
        receipt,
    )
    assert not bool(rejected.transition_identity_matches)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(_materialize_keys(rejected.state), _materialize_keys(state))

    accepted = adapter.update_transition(
        state,
        _transition(state, jnp.asarray([-0.1, 0.3], dtype=jnp.float32)),
        receipt,
    )
    assert bool(accepted.update_applied)
    invalid_owner = dataclasses.replace(  # type: ignore[type-var]
        accepted.state,
        pending_builder_update_words=accepted.state.prototype_state.state_builder_state.update_words,
    )
    assert not bool(adapter.state_valid(invalid_owner))


def test_empty_provenance_is_rejected_without_consuming_receipt() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(7)),
        jnp.asarray([0.1, 0.2], dtype=jnp.float32),
    ).state
    receipt = adapter.make_target_receipt(
        state,
        cumulant=jnp.asarray(0.0, dtype=jnp.float32),
        gvf_continuation=jnp.asarray(0.9, dtype=jnp.float32),
        control_value_target=jnp.asarray(0.0, dtype=jnp.float32),
        selected_action_advantage_target=jnp.asarray(0.0, dtype=jnp.float32),
        source_revision_words=jnp.asarray([0, 0], dtype=jnp.uint32),
        provenance_words=jnp.zeros((4,), dtype=jnp.uint32),
    )
    result = adapter.update_transition(
        state,
        _transition(state, jnp.asarray([0.2, 0.4], dtype=jnp.float32)),
        receipt,
    )
    assert not bool(result.target_provenance_valid)
    assert not bool(result.target_receipt_committed)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(_materialize_keys(result.state), _materialize_keys(state))


def test_exhausted_target_identity_fails_stop_without_partial_mutation() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(8)),
        jnp.asarray([0.2, -0.4], dtype=jnp.float32),
    ).state
    exhausted = dataclasses.replace(  # type: ignore[type-var]
        state,
        target_receipt_words=jnp.asarray(
            [np.iinfo(np.uint32).max, np.iinfo(np.uint32).max],
            dtype=jnp.uint32,
        ),
    )
    receipt = _targets(adapter, exhausted)
    result = adapter.update_transition(
        exhausted,
        _transition(exhausted, jnp.asarray([0.1, 0.3], dtype=jnp.float32)),
        receipt,
    )
    assert not bool(result.target_identity_capacity_available)
    assert not bool(result.target_receipt_committed)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(
        _materialize_keys(result.state),
        _materialize_keys(exhausted),
    )
