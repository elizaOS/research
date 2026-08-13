"""Unit contracts for the opt-in Prototype balanced-objective adapter."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.balanced_state_objectives import (
    BalancedStateObjectives,
    BalancedStateObjectivesConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.prototype_balanced_state_objectives import (
    PROTOTYPE_BALANCED_OBJECTIVES_EVIDENCE_LEVEL,
    PROTOTYPE_BALANCED_OBJECTIVES_MAX_TRANSITIONS,
    PROTOTYPE_BALANCED_OBJECTIVES_OUTCOME_STATUS,
    PrototypeBalancedObjectivesState,
    PrototypeBalancedStateObjectives,
    measure_prototype_balanced_objectives_state_nbytes,
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
    oak = OaKConfig(
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
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=oak,
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


def _objectives(*, representation_dim: int = FEATURE_DIM) -> BalancedStateObjectives:
    return BalancedStateObjectives(
        BalancedStateObjectivesConfig(
            representation_dim=representation_dim,
            n_actions=N_ACTIONS,
            gvf_discounts=(0.25, 0.75, 0.95),
            gvf_step_size=0.03,
            inverse_step_size=0.04,
            initialization_scale=0.08,
            representation_gradient_clip=10.0,
        )
    )


def _adapter() -> PrototypeBalancedStateObjectives:
    return PrototypeBalancedStateObjectives(_prototype(), _objectives())


def _transition(
    state: PrototypeBalancedObjectivesState,
    next_observation: jax.Array,
    *,
    reward: float = 0.3,
    discount: float = 0.9,
    decision_id: jax.Array | None = None,
) -> PrototypeTransition:
    prototype = state.prototype_state
    return PrototypeTransition(
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


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def test_composition_config_is_strict_l0_and_round_trips() -> None:
    adapter = _adapter()
    config = adapter.to_config()
    assert config["evidence_level"] == PROTOTYPE_BALANCED_OBJECTIVES_EVIDENCE_LEVEL == "L0"
    assert (
        config["outcome_status"]
        == PROTOTYPE_BALANCED_OBJECTIVES_OUTCOME_STATUS
        == "not_assessed"
    )
    assert config["limitations"]
    restored = PrototypeBalancedStateObjectives.from_config(config)
    assert restored.to_config() == config

    missing = dict(config)
    missing.pop("ownership")
    with pytest.raises(ValueError, match="manifest is not exact"):
        PrototypeBalancedStateObjectives.from_config(missing)


def test_constructor_rejects_incompatible_or_concurrently_learned_builder() -> None:
    with pytest.raises(ValueError, match="representation_dim"):
        PrototypeBalancedStateObjectives(
            _prototype(),
            _objectives(representation_dim=FEATURE_DIM + 1),
        )
    with pytest.raises(ValueError, match="representation gradient mixing disabled"):
        PrototypeBalancedStateObjectives(
            _prototype(learn_from_model=True),
            _objectives(),
        )


def test_init_requires_typed_threefry_and_resource_partition_is_exact() -> None:
    adapter = _adapter()
    with pytest.raises(TypeError, match="typed Threefry"):
        adapter.init(jr.PRNGKey(3))
    state = adapter.init(jr.key(3), lifecycle_id=jnp.asarray([7, 9], dtype=jnp.uint32))
    assert bool(adapter.state_valid(state))
    assert not bool(state.pending_valid)
    np.testing.assert_array_equal(state.prototype_state.current_decision_id[:2], [7, 9])

    budget = adapter.resource_budget(state)
    assert budget.total_state_nbytes == measure_prototype_balanced_objectives_state_nbytes(
        state
    )
    assert budget.max_prototype_updates_per_transition == 1
    assert budget.max_objective_head_updates_per_transition == 1
    assert budget.max_builder_proposals_per_transition == 2
    assert budget.max_builder_commits_per_transition == 1
    assert budget.max_accepted_transitions == PROTOTYPE_BALANCED_OBJECTIVES_MAX_TRANSITIONS
    assert budget.max_accepted_transitions == 2**64 - 2
    assert "all-JAX-array-leaves" in budget.persistent_bytes_scope
    assert "excluded" in budget.diagnostic_bytes_scope
    assert "not-a-measured-device-peak" in budget.temporary_bytes_scope


def test_start_binds_exact_prototype_action_representation_and_revisions() -> None:
    adapter = _adapter()
    initial = adapter.init(jr.key(4))
    observation = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
    started = adapter.start(initial, observation)
    assert bool(started.start_applied)
    assert bool(started.candidate_state_valid)
    state = started.state
    assert bool(adapter.state_valid(state))
    assert bool(state.pending_valid)
    np.testing.assert_array_equal(
        state.pending_prototype_decision_id,
        state.prototype_state.current_decision_id,
    )
    np.testing.assert_array_equal(
        state.objectives_state.pending_representation,
        state.prototype_state.current_representation,
    )
    assert int(state.objectives_state.pending_action) == int(
        state.prototype_state.current_action
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


def test_valid_transition_updates_heads_builder_and_next_exact_owner() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(5)),
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
    ).state
    before_parameters = state.prototype_state.state_builder_state.parameters
    before_gvf = state.objectives_state.gvf_weights
    transition = _transition(
        state,
        jnp.asarray([0.4, 0.3], dtype=jnp.float32),
        reward=0.6,
        discount=0.8,
    )
    result = adapter.update_transition(state, transition)
    assert bool(result.update_applied)
    assert bool(result.prototype_transaction_applied)
    assert bool(result.objective_transaction_applied)
    assert bool(result.builder_transaction_applied)
    assert bool(result.transition_identity_matches)
    assert bool(result.builder_sources_match)
    assert bool(result.builder_destination_matches)
    assert bool(result.next_cache_valid)
    assert bool(adapter.state_valid(result.state))
    np.testing.assert_array_equal(result.state.transaction_words, [0, 1])
    np.testing.assert_array_equal(result.state.objectives_state.update_words, [0, 1])
    np.testing.assert_array_equal(result.state.prototype_state.step_words, [0, 1])
    np.testing.assert_array_equal(
        result.state.pending_builder_update_words,
        result.builder_learning.pre_update_words,
    )
    np.testing.assert_array_equal(result.state.pending_builder_update_words, [0, 0])
    np.testing.assert_array_equal(
        result.state.prototype_state.state_builder_state.update_words,
        result.builder_learning.post_update_words,
    )
    np.testing.assert_array_equal(
        result.state.prototype_state.state_builder_state.update_words,
        [0, 1],
    )
    np.testing.assert_array_equal(
        result.objective_update.next_representation_revision_words,
        [0, 2],
    )
    assert not np.array_equal(
        np.asarray(result.state.prototype_state.state_builder_state.parameters),
        np.asarray(before_parameters),
    )
    assert not np.array_equal(
        np.asarray(result.state.objectives_state.gvf_weights),
        np.asarray(before_gvf),
    )
    np.testing.assert_array_equal(
        result.state.pending_prototype_decision_id,
        result.state.prototype_state.current_decision_id,
    )
    np.testing.assert_array_equal(
        result.state.objectives_state.pending_representation,
        result.state.prototype_state.current_representation,
    )


def test_rejected_identity_is_an_exact_noop_and_the_receipt_remains_retryable() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(6)),
        jnp.asarray([-0.3, 0.7], dtype=jnp.float32),
    ).state
    next_observation = jnp.asarray([0.1, -0.2], dtype=jnp.float32)
    stale_id = state.prototype_state.current_decision_id.at[3].add(
        jnp.asarray(1, dtype=jnp.uint32)
    )
    rejected = adapter.update_transition(
        state,
        _transition(state, next_observation, decision_id=stale_id),
    )
    assert not bool(rejected.transition_identity_matches)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(
        _materialize_keys(rejected.state),
        _materialize_keys(state),
    )

    accepted = adapter.update_transition(state, _transition(state, next_observation))
    assert bool(accepted.update_applied)
    np.testing.assert_array_equal(accepted.state.transaction_words, [0, 1])


def test_decision_time_builder_revision_is_pre_commit_and_tampering_fails() -> None:
    adapter = _adapter()
    started = adapter.start(
        adapter.init(jr.key(7)),
        jnp.asarray([0.5, 0.2], dtype=jnp.float32),
    ).state
    result = adapter.update_transition(
        started,
        _transition(
            started,
            jnp.asarray([-0.1, 0.3], dtype=jnp.float32),
        ),
    )
    assert bool(result.update_applied)
    state = result.state
    np.testing.assert_array_equal(state.pending_builder_update_words, [0, 0])
    np.testing.assert_array_equal(
        state.prototype_state.state_builder_state.update_words,
        [0, 1],
    )
    np.testing.assert_array_equal(
        state.pending_builder_update_words,
        result.builder_learning.pre_update_words,
    )
    stale = dataclasses.replace(
        state,
        pending_builder_update_words=jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    assert not bool(adapter.state_valid(stale))

    retried = adapter.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([0.4, -0.6], dtype=jnp.float32),
        ),
    )
    assert bool(retried.update_applied)
    np.testing.assert_array_equal(retried.state.pending_builder_update_words, [0, 1])
    np.testing.assert_array_equal(
        retried.state.prototype_state.state_builder_state.update_words,
        [0, 2],
    )
