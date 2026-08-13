# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def,override"
"""Atomic PrototypeAgent integration for pair features and episodic memory."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.experiential_memory import (
    ExperientialMemoryConfig,
    ExperientialMemoryState,
)
from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeExperientialMemoryInput,
    PrototypeFeatureOaKState,
    PrototypeFeatureRepresentationState,
    PrototypeMemoryInteractionState,
    PrototypeTransition,
    load_prototype_checkpoint,
    measure_prototype_agent_state_resources,
    save_prototype_checkpoint,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_feature_memory import (
    PrototypeFeatureMemory,
    PrototypeFeatureMemoryRebindResult,
    PrototypeFeatureMemoryState,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
)

pytestmark = pytest.mark.integration

BASE_DIM = 4
ACTIVE_PAIR_SLOTS = 2
TOTAL_DIM = BASE_DIM + ACTIVE_PAIR_SLOTS
N_ACTIONS = 2
MEMORY_CAPACITY = 3


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest):
    if request.node.name == "test_feature_memory_eager_and_jit_curation_have_parity":
        yield
    else:
        with jax.disable_jit():
            yield


def _feature_config(*, replacement_interval: int = 0) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=ACTIVE_PAIR_SLOTS,
        candidate_pair_slots=6,
        n_tasks=1,
        n_options=2,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=(0, 1),
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=100,
    )


def _oak_config() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, threshold=1.0e6, max_option_steps=8),
                SubtaskSpec(feature_index=1, threshold=1.0e6, max_option_steps=8),
            ),
            observation_dim=TOTAL_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            base_step_size=0.01,
            option_step_size=0.01,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _memory_config() -> ExperientialMemoryConfig:
    return ExperientialMemoryConfig(
        capacity=MEMORY_CAPACITY,
        observation_dim=TOTAL_DIM,
        key_dim=TOTAL_DIM,
        action_dim=N_ACTIONS,
        outcome_dim=TOTAL_DIM + 1,
        top_k=2,
        min_neighbors=1,
        distance_scale=1.0,
        min_similarity=0.0,
        min_effective_reliability=0.01,
        max_uncertainty=1.0,
        max_safety_cost=1.0,
        max_age=100,
        staleness_scale=100.0,
        utility_decay=1.0,
        eviction_utility_weight=1.0,
        eviction_recency_weight=1.0,
        recency_scale=10.0,
    )


def _config(*, replacement_interval: int = 0) -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=_oak_config(),
        state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
        prototype_feature_lifecycle=_feature_config(
            replacement_interval=replacement_interval
        ),
        experiential_memory=_memory_config(),
    )


def _agent(*, replacement_interval: int = 0) -> PrototypeAgent:
    return PrototypeAgent(_config(replacement_interval=replacement_interval))


def _feature_wrapper(state: PrototypeAgentState) -> PrototypeFeatureRepresentationState:
    assert type(state.state_builder_state) is PrototypeFeatureRepresentationState
    return state.state_builder_state


def _bound_oak(state: PrototypeAgentState) -> PrototypeFeatureOaKState:
    assert type(state.oak_state) is PrototypeFeatureOaKState
    return state.oak_state


def _oak(state: PrototypeAgentState) -> OaKState:
    return _bound_oak(state).oak_state


def _binding(state: PrototypeAgentState) -> PrototypeFeatureConsumerBinding:
    return _bound_oak(state).consumer_binding


def _feature_memory_wrapper(state: PrototypeAgentState) -> PrototypeFeatureMemoryState:
    outer = cast(PrototypeMemoryInteractionState, state.ia_state)
    assert type(outer) is PrototypeMemoryInteractionState
    wrapper = outer.experiential_memory_state
    assert type(wrapper) is PrototypeFeatureMemoryState
    return wrapper


def _raw_memory(state: PrototypeAgentState) -> ExperientialMemoryState:
    return _feature_memory_wrapper(state).memory_state


def _start_idle(agent: PrototypeAgent, observation: jax.Array) -> PrototypeAgentState:
    for seed in range(32):
        state = agent.start(agent.init(jr.key(seed)), observation)
        if int(_oak(state).stomp_state.executing_option) == -1:
            return state
    raise AssertionError("could not obtain a deterministic idle initial decision")


def _force_next_extended_action(
    state: PrototypeAgentState,
    extended_action: int,
) -> PrototypeAgentState:
    bound = _bound_oak(state)
    stomp = bound.oak_state.stomp_state
    learner = stomp.base_learner_state
    weights = tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights)
    biases = tuple(
        jnp.full_like(bias, 100.0 if index == extended_action else -100.0)
        for index, bias in enumerate(learner.head_params.biases)
    )
    learner = learner.replace(
        head_params=learner.head_params.replace(weights=weights, biases=biases)
    )
    return cast(
        PrototypeAgentState,
        state.replace(
            oak_state=bound.replace(
                oak_state=bound.oak_state.replace(
                    stomp_state=stomp.replace(base_learner_state=learner)
                )
            )
        ),
    )


def _force_promotion(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
) -> PrototypeAgentState:
    lifecycle = agent.prototype_feature_lifecycle
    assert lifecycle is not None
    wrapper = _feature_wrapper(state)
    feature_state = wrapper.feature_lifecycle_state
    learner = feature_state.learner_state
    active = set(
        zip(
            np.asarray(learner.feature_left).tolist(),
            np.asarray(learner.feature_right).tolist(),
            strict=True,
        )
    )
    candidates = list(
        zip(
            np.asarray(learner.candidate_left).tolist(),
            np.asarray(learner.candidate_right).tolist(),
            strict=True,
        )
    )
    candidate_index = next(
        index for index, pair in enumerate(candidates) if pair not in active
    )
    candidate_utilities = jnp.zeros_like(learner.candidate_utilities)
    candidate_utilities = candidate_utilities.at[candidate_index].set(0.9)
    learner = learner.replace(
        utilities=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
        candidate_utilities=candidate_utilities,
    )
    feature_state = feature_state.replace(learner_state=learner)
    assert bool(lifecycle.state_valid(feature_state))
    return cast(
        PrototypeAgentState,
        state.replace(
            state_builder_state=wrapper.replace(
                feature_lifecycle_state=feature_state
            )
        ),
    )


def _next_decision_id(state: PrototypeAgentState) -> jax.Array:
    return state.current_decision_id.at[3].set(
        state.current_decision_id[3] + jnp.asarray(1, dtype=jnp.uint32)
    )


def _memory_input(
    state: PrototypeAgentState,
    provenance_id: int,
) -> PrototypeExperientialMemoryInput:
    source_version = _binding(state).semantic_generation
    return PrototypeExperientialMemoryInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        current_prototype_decision_id=state.current_decision_id,
        next_prototype_decision_id=_next_decision_id(state),
        query_representation_version=source_version,
        entry_representation_version=source_version,
        query_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        entry_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(1.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
        source_id=jnp.asarray(7, dtype=jnp.int32),
        next_action_safety_mask=jnp.ones((N_ACTIONS,), dtype=jnp.bool_),
    )


def _transition(
    state: PrototypeAgentState,
    next_observation: jax.Array,
    *,
    reward: float = 0.5,
) -> PrototypeTransition:
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
    )


def _encode(
    base: jax.Array,
    binding: PrototypeFeatureConsumerBinding,
) -> jax.Array:
    descriptors = binding.descriptors
    pair_values = base[descriptors[:, 0]] * base[descriptors[:, 1]]
    return jnp.concatenate((base, pair_values)).astype(jnp.float32)


def _materialize_keys(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)
        if type(value) is float:
            return jnp.asarray(value, dtype=jnp.float32)
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_tree_close(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if np.issubdtype(left_array.dtype, np.inexact):
            np.testing.assert_allclose(
                left_array,
                right_array,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        else:
            np.testing.assert_array_equal(left_array, right_array)


def _assert_nonrepresentation_memory_exact(
    before: ExperientialMemoryState,
    after: ExperientialMemoryState,
) -> None:
    for field in dataclasses.fields(cast(Any, before.entries)):
        if field.name in {
            "observations",
            "keys",
            "outcomes",
            "representation_versions",
        }:
            continue
        np.testing.assert_array_equal(
            np.asarray(getattr(before.entries, field.name)),
            np.asarray(getattr(after.entries, field.name)),
        )
    for field in dataclasses.fields(cast(Any, before)):
        if field.name == "entries":
            continue
        np.testing.assert_array_equal(
            np.asarray(getattr(before, field.name)),
            np.asarray(getattr(after, field.name)),
        )


def _seed_one_row(
    agent: PrototypeAgent,
    *,
    provenance_id: int = 101,
) -> tuple[PrototypeAgentState, jax.Array]:
    initial_observation = jnp.asarray(
        [1.0, -2.0, 0.5, 3.0],
        dtype=jnp.float32,
    )
    state = _start_idle(agent, initial_observation)
    state = _force_next_extended_action(state, 0)
    next_observation = jnp.asarray(
        [-0.25, 0.75, 1.5, -0.5],
        dtype=jnp.float32,
    )
    result = agent.update_transition(
        state,
        _transition(state, next_observation),
        experiential_memory_input=_memory_input(state, provenance_id),
    )
    diagnostics = result.experiential_memory_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert bool(diagnostics.transaction_applied)
    assert int(_raw_memory(result.state).active_count) == 1
    return result.state, initial_observation


def _provenance_slot(memory: ExperientialMemoryState, provenance_id: int) -> int:
    matches = np.flatnonzero(
        np.asarray(memory.entries.valid)
        & (np.asarray(memory.entries.provenance_ids) == provenance_id)
    )
    assert matches.shape == (1,)
    return int(matches[0])


class _RejectingFeatureMemory(PrototypeFeatureMemory):
    """Inject one failed bank migration after the lifecycle has routed."""

    def rebind(
        self,
        state: PrototypeFeatureMemoryState,
        source_binding: PrototypeFeatureConsumerBinding,
        destination_binding: PrototypeFeatureConsumerBinding,
    ) -> PrototypeFeatureMemoryRebindResult:
        result = super().rebind(state, source_binding, destination_binding)
        return cast(
            PrototypeFeatureMemoryRebindResult,
            result.replace(
                state=state,
                diagnostics=result.diagnostics.replace(
                    candidate_state_valid=jnp.asarray(False, dtype=jnp.bool_),
                    transaction_applied=jnp.asarray(False, dtype=jnp.bool_),
                    transaction_noop=jnp.asarray(False, dtype=jnp.bool_),
                ),
            ),
        )


def test_config_requires_exact_identity_builder_and_round_trips() -> None:
    class IdentitySubclass(IdentityStateBuilderConfig):
        pass

    class FeatureLifecycleSubclass(PrototypeFeatureLifecycleConfig):
        pass

    class ExperientialMemorySubclass(ExperientialMemoryConfig):
        pass

    config = _config()
    assert PrototypeAgentConfig.from_config(config.to_config()) == config
    assert _agent().prototype_feature_memory is not None
    assert alberta.PrototypeFeatureMemory is PrototypeFeatureMemory
    assert core.PrototypeFeatureMemory is PrototypeFeatureMemory
    assert alberta.PrototypeFeatureMemoryState is PrototypeFeatureMemoryState
    assert core.PrototypeFeatureMemoryState is PrototypeFeatureMemoryState
    assert (
        alberta.PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA
        == PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA
    )
    assert (
        core.PROTOTYPE_FEATURE_MEMORY_CONFIG_SCHEMA
        == alberta.PROTOTYPE_FEATURE_MEMORY_CONFIG_SCHEMA
    )

    with pytest.raises(ValueError, match="exact Identity"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=OnlineGatedStateBuilderConfig(
                observation_dim=2,
                n_actions=N_ACTIONS,
                hidden_dim=2,
                include_raw_observation=True,
                step_size=0.1,
                gradient_clip=100.0,
            ),
            prototype_feature_lifecycle=_feature_config(),
            experiential_memory=_memory_config(),
        )
    with pytest.raises(ValueError, match="Identity or OnlineGated"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentitySubclass(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=_feature_config(),
            experiential_memory=_memory_config(),
        )
    with pytest.raises(ValueError, match="exact PrototypeFeatureLifecycleConfig"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=FeatureLifecycleSubclass(
                **dataclasses.asdict(_feature_config())
            ),
            experiential_memory=_memory_config(),
        )
    with pytest.raises(ValueError, match="exact ExperientialMemoryConfig"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
            prototype_feature_lifecycle=_feature_config(),
            experiential_memory=ExperientialMemorySubclass(
                **dataclasses.asdict(_memory_config())
            ),
        )


def test_no_curation_update_writes_one_destination_bound_exemplar() -> None:
    agent = _agent()
    initial = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32)
    state = _force_next_extended_action(_start_idle(agent, initial), 0)
    next_observation = jnp.asarray([-0.2, 0.7, 1.25, -0.4], dtype=jnp.float32)

    result = agent.update_transition(
        state,
        _transition(state, next_observation, reward=0.75),
        experiential_memory_input=_memory_input(state, 11),
    )

    feature_diagnostics = result.prototype_feature_memory_diagnostics
    memory_diagnostics = result.experiential_memory_diagnostics
    lifecycle_diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert feature_diagnostics is not None
    assert memory_diagnostics is not None
    assert lifecycle_diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert bool(lifecycle_diagnostics.outer_transaction_committed)
    assert bool(feature_diagnostics.rebind.transaction_noop)
    assert not bool(feature_diagnostics.rebind.transaction_applied)
    assert bool(feature_diagnostics.source_versions_match)
    assert bool(feature_diagnostics.current_destination_encoding_valid)
    assert bool(feature_diagnostics.bootstrap_destination_encoding_valid)
    assert bool(feature_diagnostics.decision_destination_encoding_valid)
    assert bool(feature_diagnostics.post_memory_state_valid)
    assert bool(feature_diagnostics.outer_transaction_committed)
    assert bool(memory_diagnostics.query_before_write)
    assert bool(memory_diagnostics.transaction_applied)

    binding = _binding(result.state)
    wrapper = _feature_memory_wrapper(result.state)
    adapter = agent.prototype_feature_memory
    assert adapter is not None
    assert bool(adapter.state_valid(wrapper, binding))
    memory = wrapper.memory_state
    slot = int(memory_diagnostics.slot)
    np.testing.assert_array_equal(memory.entries.observations[slot], state.current_representation)
    np.testing.assert_array_equal(memory.entries.keys[slot], state.current_representation)
    np.testing.assert_array_equal(
        memory.entries.outcomes[slot],
        jnp.concatenate((_encode(next_observation, binding), jnp.asarray([0.75]))),
    )
    assert int(memory.entries.representation_versions[slot]) == int(
        binding.semantic_generation
    )


def test_curation_without_sidecar_atomically_reencodes_existing_rows() -> None:
    agent = _agent(replacement_interval=1)
    seeded_state, initial_observation = _seed_one_row(agent)
    source_binding = _binding(seeded_state)
    before = _raw_memory(seeded_state)
    source_slot = _provenance_slot(before, 101)
    state = _force_promotion(
        agent,
        _force_next_extended_action(seeded_state, 0),
    )

    result = agent.update_transition(
        state,
        _transition(state, initial_observation),
    )

    lifecycle_diagnostics = result.prototype_feature_lifecycle_diagnostics
    feature_diagnostics = result.prototype_feature_memory_diagnostics
    memory_diagnostics = result.experiential_memory_diagnostics
    assert lifecycle_diagnostics is not None
    assert feature_diagnostics is not None
    assert memory_diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert bool(lifecycle_diagnostics.lifecycle.curation_committed)
    assert bool(feature_diagnostics.rebind.destination_descriptors_changed)
    assert bool(feature_diagnostics.rebind.destination_generation_changed)
    assert bool(feature_diagnostics.rebind.generation_is_successor)
    assert bool(feature_diagnostics.rebind.transaction_applied)
    assert int(feature_diagnostics.rebind.valid_rows_reencoded) == 1
    assert not bool(memory_diagnostics.transaction_required)
    assert not bool(memory_diagnostics.transaction_applied)

    destination_binding = _binding(result.state)
    assert not bool(jnp.array_equal(source_binding.descriptors, destination_binding.descriptors))
    after = _raw_memory(result.state)
    destination_slot = _provenance_slot(after, 101)
    assert destination_slot == source_slot
    _assert_nonrepresentation_memory_exact(before, after)
    expected_observation = _encode(
        before.entries.observations[source_slot, :BASE_DIM],
        destination_binding,
    )
    expected_outcome = jnp.concatenate(
        (
            _encode(before.entries.outcomes[source_slot, :BASE_DIM], destination_binding),
            before.entries.rewards[source_slot, None],
        )
    )
    np.testing.assert_array_equal(after.entries.observations[source_slot], expected_observation)
    np.testing.assert_array_equal(after.entries.keys[source_slot], expected_observation)
    np.testing.assert_array_equal(after.entries.outcomes[source_slot], expected_outcome)
    assert int(after.entries.representation_versions[source_slot]) == int(
        destination_binding.semantic_generation
    )


def test_curation_rebinds_before_same_transition_query_and_write() -> None:
    agent = _agent(replacement_interval=1)
    seeded_state, _ = _seed_one_row(agent)
    before = _raw_memory(seeded_state)
    old_slot = _provenance_slot(before, 101)
    query_base = before.entries.observations[old_slot, :BASE_DIM]
    state = _force_promotion(
        agent,
        _force_next_extended_action(seeded_state, 0),
    )
    source_version = int(_binding(state).semantic_generation)

    result = agent.update_transition(
        state,
        _transition(state, query_base, reward=-0.25),
        experiential_memory_input=_memory_input(state, 202),
    )

    feature_diagnostics = result.prototype_feature_memory_diagnostics
    memory_diagnostics = result.experiential_memory_diagnostics
    assert feature_diagnostics is not None
    assert memory_diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert bool(feature_diagnostics.rebind.transaction_applied)
    assert bool(feature_diagnostics.source_versions_match)
    assert bool(memory_diagnostics.query_before_write)
    assert bool(memory_diagnostics.proposal.retrieval.has_neighbors)
    assert int(memory_diagnostics.proposal.retrieval.neighbor_provenance_ids[0]) == 101
    assert bool(memory_diagnostics.transaction_applied)

    destination_binding = _binding(result.state)
    assert int(destination_binding.semantic_generation) == source_version + 1
    memory = _raw_memory(result.state)
    old_slot = _provenance_slot(memory, 101)
    new_slot = _provenance_slot(memory, 202)
    expected_query = _encode(query_base, destination_binding)
    np.testing.assert_array_equal(memory.entries.keys[old_slot], expected_query)
    np.testing.assert_array_equal(
        memory.entries.observations[new_slot],
        _encode(state.current_representation[:BASE_DIM], destination_binding),
    )
    assert int(memory.entries.representation_versions[old_slot]) == int(
        destination_binding.semantic_generation
    )
    assert int(memory.entries.representation_versions[new_slot]) == int(
        destination_binding.semantic_generation
    )


@pytest.mark.parametrize("tamper", ["schema-digest", "stored-key"])
def test_tampered_feature_memory_wrapper_fails_closed(tamper: str) -> None:
    agent = _agent()
    state, _ = _seed_one_row(agent)
    outer = cast(PrototypeMemoryInteractionState, state.ia_state)
    wrapper = _feature_memory_wrapper(state)
    if tamper == "schema-digest":
        wrapper = cast(
            PrototypeFeatureMemoryState,
            wrapper.replace(
                schema_digest=wrapper.schema_digest.at[0].set(
                    wrapper.schema_digest[0] ^ jnp.asarray(1, dtype=jnp.uint8)
                )
            ),
        )
    else:
        slot = _provenance_slot(wrapper.memory_state, 101)
        entries = wrapper.memory_state.entries
        wrapper = cast(
            PrototypeFeatureMemoryState,
            wrapper.replace(
                memory_state=wrapper.memory_state.replace(
                    entries=entries.replace(
                        keys=entries.keys.at[slot, 0].add(
                            jnp.asarray(0.25, dtype=jnp.float32)
                        )
                    )
                )
            ),
        )
    tampered = cast(
        PrototypeAgentState,
        state.replace(
            ia_state=outer.replace(experiential_memory_state=wrapper)
        ),
    )
    adapter = agent.prototype_feature_memory
    assert adapter is not None
    assert not bool(adapter.state_valid(wrapper, _binding(tampered)))
    assert not bool(agent._checkpoint_state_valid(tampered))

    result = agent.update_transition(
        tampered,
        _transition(tampered, tampered.current_raw_observation),
    )
    feature_diagnostics = result.prototype_feature_memory_diagnostics
    assert feature_diagnostics is not None
    assert not bool(result.transition_diagnostics.valid)
    assert bool(result.transition_diagnostics.rejected)
    assert not bool(feature_diagnostics.outer_transaction_committed)
    _assert_tree_exact(result.state, tampered)


def test_failed_feature_memory_rebind_rolls_back_the_whole_transition() -> None:
    agent = _agent(replacement_interval=1)
    seeded_state, initial_observation = _seed_one_row(agent)
    state = _force_promotion(
        agent,
        _force_next_extended_action(seeded_state, 0),
    )
    adapter = agent.prototype_feature_memory
    assert adapter is not None
    agent._prototype_feature_memory = _RejectingFeatureMemory(adapter.config)

    result = agent.update_transition(
        state,
        _transition(state, initial_observation),
    )

    feature_diagnostics = result.prototype_feature_memory_diagnostics
    lifecycle_diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert feature_diagnostics is not None
    assert lifecycle_diagnostics is not None
    assert not bool(result.transition_diagnostics.valid)
    assert bool(result.transition_diagnostics.rejected)
    assert bool(lifecycle_diagnostics.lifecycle.curation_committed)
    assert not bool(lifecycle_diagnostics.outer_transaction_committed)
    assert not bool(feature_diagnostics.rebind.transaction_applied)
    assert not bool(feature_diagnostics.outer_transaction_committed)
    _assert_tree_exact(result.state, state)


def test_feature_memory_eager_and_jit_curation_have_parity() -> None:
    agent = _agent(replacement_interval=1)
    seeded_state, initial_observation = _seed_one_row(agent)
    state = _force_promotion(
        agent,
        _force_next_extended_action(seeded_state, 0),
    )
    transition = _transition(state, initial_observation)
    sidecar = _memory_input(state, 303)

    eager = agent.update_transition(
        state,
        transition,
        experiential_memory_input=sidecar,
    )
    compiled = jax.jit(agent.update_transition)(
        state,
        transition,
        experiential_memory_input=sidecar,
    )

    feature_diagnostics = eager.prototype_feature_memory_diagnostics
    assert feature_diagnostics is not None
    assert bool(feature_diagnostics.rebind.transaction_applied)
    _assert_tree_close(eager, compiled)


def test_v16_checkpoint_and_resource_accounting_are_exact(tmp_path: Path) -> None:
    agent = _agent()
    state, _ = _seed_one_row(agent)
    adapter = agent.prototype_feature_memory
    memory = agent.experiential_memory
    resources = agent.prototype_feature_memory_resource_budget
    memory_resources = agent.experiential_memory_resource_declaration
    assert adapter is not None
    assert memory is not None
    assert resources is not None
    assert memory_resources is not None

    wrapper = _feature_memory_wrapper(state)
    wrapper_nbytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(wrapper)
    )
    measured = measure_prototype_agent_state_resources(state)
    assert resources.memory_state_nbytes == memory.persistent_bytes
    assert resources.wrapper_state_nbytes == wrapper_nbytes
    assert resources.wrapper_state_nbytes == (
        resources.memory_state_nbytes + resources.wrapper_metadata_nbytes
    )
    assert resources.max_valid_rows_reencoded == MEMORY_CAPACITY
    assert resources.max_pair_products_per_rebind == (
        2 * ACTIVE_PAIR_SLOTS * MEMORY_CAPACITY
    )
    assert resources.memory_clock_advances_per_rebind == 0
    assert resources.rng_draws_per_rebind == 0
    assert measured.interaction_memory_bundle_nbytes == wrapper_nbytes
    assert memory_resources.persistent_state_bytes == resources.memory_state_nbytes

    checkpoint = tmp_path / "prototype-feature-memory"
    save_prototype_checkpoint(agent, state, checkpoint)
    metadata = load_checkpoint_metadata(checkpoint)
    assert metadata["schema"] == PROTOTYPE_FEATURE_MEMORY_CHECKPOINT_SCHEMA
    assert metadata["feature_memory_schema_sha256"] == adapter.schema_digest_hex
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_exact(restored_state, state)
    restored_adapter = restored_agent.prototype_feature_memory
    assert restored_adapter is not None
    assert bool(
        restored_adapter.state_valid(
            _feature_memory_wrapper(restored_state),
            _binding(restored_state),
        )
    )

    bad_digest_metadata = dict(metadata)
    bad_digest_metadata["feature_memory_schema_sha256"] = "00" * 32
    bad_digest_checkpoint = tmp_path / "bad-feature-memory-digest"
    save_checkpoint(
        state,
        bad_digest_checkpoint,
        metadata=bad_digest_metadata,
    )
    with pytest.raises(ValueError, match="schema digest does not match"):
        load_prototype_checkpoint(bad_digest_checkpoint)

    relabeled_metadata = dict(metadata)
    relabeled_metadata["schema"] = PROTOTYPE_CHECKPOINT_SCHEMA
    relabeled_checkpoint = tmp_path / "relabeled-feature-memory"
    save_checkpoint(
        state,
        relabeled_checkpoint,
        metadata=relabeled_metadata,
    )
    with pytest.raises(ValueError, match="requires a v16"):
        load_prototype_checkpoint(relabeled_checkpoint)

    corrupted_wrapper = cast(
        PrototypeFeatureMemoryState,
        wrapper.replace(
            schema_digest=wrapper.schema_digest.at[0].set(
                wrapper.schema_digest[0] ^ jnp.asarray(1, dtype=jnp.uint8)
            )
        ),
    )
    outer = cast(PrototypeMemoryInteractionState, state.ia_state)
    corrupted_state = cast(
        PrototypeAgentState,
        state.replace(
            ia_state=outer.replace(
                experiential_memory_state=corrupted_wrapper
            )
        ),
    )
    with pytest.raises(ValueError, match="inconsistent PrototypeAgent state"):
        save_prototype_checkpoint(
            agent,
            corrupted_state,
            tmp_path / "corrupted-feature-memory",
        )
