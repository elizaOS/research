"""Exact standalone Prototype feature-memory adapter contracts."""

from __future__ import annotations

import dataclasses
import json
from typing import cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.experiential_memory import (
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    ExperientialMemoryState,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_feature_memory import (
    PROTOTYPE_FEATURE_MEMORY_CONFIG_SCHEMA,
    PrototypeFeatureMemory,
    PrototypeFeatureMemoryConfig,
    PrototypeFeatureMemoryState,
)
from alberta_framework.core.state_builder import IdentityStateBuilderConfig

pytestmark = pytest.mark.unit

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295


def _feature_config(
    *,
    base_feature_dim: int = 3,
    active_pair_slots: int = 2,
) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=base_feature_dim,
        active_pair_slots=active_pair_slots,
        candidate_pair_slots=1,
        n_tasks=1,
        n_options=1,
        n_primitive_actions=2,
        option_subtask_feature_indices=(0,),
        step_size_output=0.03,
        utility_decay=0.9,
        replacement_interval=1,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=100,
    )


def _memory_config(
    feature: PrototypeFeatureLifecycleConfig,
    *,
    capacity: int = 3,
) -> ExperientialMemoryConfig:
    return ExperientialMemoryConfig(
        capacity=capacity,
        observation_dim=feature.total_feature_dim,
        key_dim=feature.total_feature_dim,
        action_dim=2,
        outcome_dim=feature.total_feature_dim + 1,
        top_k=min(2, capacity),
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


def _config(
    *,
    base_feature_dim: int = 3,
    active_pair_slots: int = 2,
    capacity: int = 3,
) -> PrototypeFeatureMemoryConfig:
    feature = _feature_config(
        base_feature_dim=base_feature_dim,
        active_pair_slots=active_pair_slots,
    )
    return PrototypeFeatureMemoryConfig(
        feature_lifecycle=feature,
        experiential_memory=_memory_config(feature, capacity=capacity),
        base_state_builder=IdentityStateBuilderConfig(
            observation_dim=base_feature_dim
        ),
    )


def _telemetry(words: tuple[int, int]) -> int:
    high, low = words
    return low if high == 0 and low < _INT32_MAX else _INT32_MAX


def _binding(
    descriptors: tuple[tuple[int, int], ...],
    *,
    words: tuple[int, int] = (0, 0),
) -> PrototypeFeatureConsumerBinding:
    return PrototypeFeatureConsumerBinding(
        semantic_generation=jnp.asarray(_telemetry(words), dtype=jnp.int32),
        semantic_generation_words=jnp.asarray(words, dtype=jnp.uint32),
        descriptors=jnp.asarray(descriptors, dtype=jnp.int32),
    )


def _encode(base: jax.Array, binding: PrototypeFeatureConsumerBinding) -> jax.Array:
    descriptors = binding.descriptors
    pairs = base[descriptors[:, 0]] * base[descriptors[:, 1]]
    return jnp.concatenate((base, pairs)).astype(jnp.float32)


def _entry(
    binding: PrototypeFeatureConsumerBinding,
    *,
    observation_base: tuple[float, ...],
    outcome_base: tuple[float, ...],
    reward: float,
    provenance_id: int,
) -> ExperientialMemoryEntry:
    observation = _encode(
        jnp.asarray(observation_base, dtype=jnp.float32),
        binding,
    )
    outcome_representation = _encode(
        jnp.asarray(outcome_base, dtype=jnp.float32),
        binding,
    )
    reward_array = jnp.asarray(reward, dtype=jnp.float32)
    return ExperientialMemoryEntry(
        observation=observation,
        key=observation,
        action=jnp.asarray((0.25, -0.75), dtype=jnp.float32),
        outcome=jnp.concatenate((outcome_representation, reward_array[None])),
        reward=reward_array,
        uncertainty=jnp.asarray(0.2, dtype=jnp.float32),
        uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.1, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(0.8, dtype=jnp.float32),
        utility=jnp.asarray(0.6, dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        representation_version=binding.semantic_generation,
        valid=jnp.asarray(True, dtype=jnp.bool_),
        age=jnp.asarray(0, dtype=jnp.int32),
        provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
        source_id=jnp.asarray(9, dtype=jnp.int32),
    )


def _populated(
    adapter: PrototypeFeatureMemory,
    binding: PrototypeFeatureConsumerBinding,
) -> PrototypeFeatureMemoryState:
    memory_state = adapter.memory.init()
    examples = (
        _entry(
            binding,
            observation_base=(1.5, -2.0, 0.25),
            outcome_base=(-0.5, 3.0, 2.0),
            reward=-0.0,
            provenance_id=11,
        ),
        _entry(
            binding,
            observation_base=(-4.0, 0.5, 2.0),
            outcome_base=(1.25, -0.75, 0.5),
            reward=0.7,
            provenance_id=12,
        ),
    )
    for entry in examples:
        written = adapter.memory.write(memory_state, entry)
        assert bool(written.wrote)
        memory_state = written.state
    return adapter.init(binding, memory_state)


def _assert_exact(left: object, right: object) -> None:
    chex.assert_trees_all_equal_structs(left, right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_nonrepresentation_memory_exact(
    before: ExperientialMemoryState,
    after: ExperientialMemoryState,
) -> None:
    for field in dataclasses.fields(before.entries):
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
    for field in dataclasses.fields(before):
        if field.name == "entries":
            continue
        np.testing.assert_array_equal(
            np.asarray(getattr(before, field.name)),
            np.asarray(getattr(after, field.name)),
        )


def test_config_is_strict_round_trips_and_binds_semantics_in_digest() -> None:
    config = _config()
    payload = json.loads(json.dumps(config.to_config()))
    restored = PrototypeFeatureMemoryConfig.from_config(payload)

    assert restored == config
    assert payload["schema"] == PROTOTYPE_FEATURE_MEMORY_CONFIG_SCHEMA
    assert len(config.schema_digest) == 32
    assert len(config.schema_digest_hex) == 64
    adapter = PrototypeFeatureMemory(config)
    np.testing.assert_array_equal(
        np.asarray(adapter.schema_digest),
        np.frombuffer(config.schema_digest, dtype=np.uint8),
    )
    assert adapter.schema_digest_hex == config.schema_digest_hex
    assert PrototypeFeatureMemory.from_config(payload).config == config

    tampered = dict(payload)
    tampered["migration_semantics"] = "rewrite-all-rows"
    with pytest.raises(ValueError, match="fixed semantics"):
        PrototypeFeatureMemoryConfig.from_config(tampered)
    digest_tampered = dict(payload)
    digest_tampered["schema_digest_sha256"] = "00" * 32
    with pytest.raises(ValueError, match="digest differs"):
        PrototypeFeatureMemoryConfig.from_config(digest_tampered)

    with pytest.raises(ValueError, match="pair-product work"):
        PrototypeFeatureMemoryConfig(
            feature_lifecycle=_feature_config(),
            experiential_memory=dataclasses.replace(
                _memory_config(_feature_config()),
                capacity=_INT32_MAX,
            ),
            base_state_builder=IdentityStateBuilderConfig(observation_dim=3),
        )


def test_config_rejects_nonidentity_subclasses_and_dimension_drift() -> None:
    class IdentitySubclass(IdentityStateBuilderConfig):
        pass

    config = _config()
    with pytest.raises(TypeError, match="exact Identity"):
        PrototypeFeatureMemoryConfig(
            feature_lifecycle=config.feature_lifecycle,
            experiential_memory=config.experiential_memory,
            base_state_builder=IdentitySubclass(observation_dim=3),
        )
    with pytest.raises(ValueError, match="observation_dim"):
        PrototypeFeatureMemoryConfig(
            feature_lifecycle=config.feature_lifecycle,
            experiential_memory=dataclasses.replace(
                config.experiential_memory,
                observation_dim=4,
            ),
            base_state_builder=config.base_state_builder,
        )
    with pytest.raises(ValueError, match="outcome_dim"):
        PrototypeFeatureMemoryConfig(
            feature_lifecycle=config.feature_lifecycle,
            experiential_memory=dataclasses.replace(
                config.experiential_memory,
                outcome_dim=5,
            ),
            base_state_builder=config.base_state_builder,
        )


def test_init_state_and_resource_budget_are_exact() -> None:
    adapter = PrototypeFeatureMemory(_config())
    binding = _binding(((0, 1), (1, 2)))
    state = adapter.init(binding)
    budget = adapter.resource_budget()

    assert bool(adapter.state_valid(state, binding))
    assert budget.memory_state_nbytes == adapter.memory.persistent_bytes
    assert budget.consumer_binding_generation_nbytes == 12
    assert budget.consumer_binding_descriptor_nbytes == 16
    assert budget.consumer_binding_nbytes == 28
    assert budget.schema_digest_nbytes == 32
    assert budget.wrapper_metadata_nbytes == 60
    assert budget.wrapper_state_nbytes == sum(
        int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree.leaves(state)
    )
    assert budget.max_valid_rows_reencoded == 3
    assert budget.max_pair_products_per_rebind == 12
    assert budget.memory_clock_advances_per_rebind == 0
    assert budget.memory_operation_counter_advances_per_rebind == 0
    assert budget.rng_draws_per_rebind == 0


def test_rebind_reencodes_valid_rows_and_preserves_every_other_bit() -> None:
    adapter = PrototypeFeatureMemory(_config())
    source = _binding(((0, 1), (1, 2)))
    destination = _binding(((0, 2), (1, 2)), words=(0, 1))
    state = _populated(adapter, source)
    before = state.memory_state

    result = adapter.rebind(state, source, destination)

    assert bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.transaction_noop)
    assert int(result.diagnostics.valid_rows_reencoded) == 2
    assert int(result.diagnostics.pair_products_evaluated) == 12
    assert int(result.diagnostics.memory_clock_advance_count) == 0
    assert int(result.diagnostics.rng_draw_count) == 0
    assert bool(adapter.state_valid(result.state, destination))
    _assert_nonrepresentation_memory_exact(before, result.state.memory_state)
    np.testing.assert_array_equal(
        result.diagnostics.memory_step_words_before,
        result.diagnostics.memory_step_words_after,
    )
    valid = np.asarray(before.entries.valid)
    for index in np.flatnonzero(valid):
        expected_observation = _encode(
            before.entries.observations[index, :3],
            destination,
        )
        expected_outcome = jnp.concatenate(
            (
                _encode(before.entries.outcomes[index, :3], destination),
                before.entries.rewards[index, None],
            )
        )
        np.testing.assert_array_equal(
            result.state.memory_state.entries.observations[index],
            expected_observation,
        )
        np.testing.assert_array_equal(
            result.state.memory_state.entries.keys[index],
            expected_observation,
        )
        np.testing.assert_array_equal(
            result.state.memory_state.entries.outcomes[index],
            expected_outcome,
        )
        assert (
            int(result.state.memory_state.entries.representation_versions[index])
            == 1
        )
    reward_bits = np.asarray(before.entries.rewards).view(np.uint32)
    outcome_reward_bits = np.asarray(
        result.state.memory_state.entries.outcomes[:, -1]
    ).view(np.uint32)
    np.testing.assert_array_equal(outcome_reward_bits[valid], reward_bits[valid])


def test_eager_jit_noop_and_repeated_rebind_have_exact_parity() -> None:
    adapter = PrototypeFeatureMemory(_config())
    source = _binding(((0, 1), (1, 2)))
    destination = _binding(((0, 2), (1, 2)), words=(0, 1))
    state = _populated(adapter, source)

    eager = adapter.rebind(state, source, destination)
    compiled = jax.jit(adapter.rebind)(state, source, destination)
    _assert_exact(eager, compiled)

    noop = adapter.rebind(eager.state, destination, destination)
    assert bool(noop.diagnostics.transaction_noop)
    assert not bool(noop.diagnostics.transaction_applied)
    _assert_exact(noop.state, eager.state)

    returned_binding = _binding(((0, 1), (1, 2)), words=(0, 2))
    returned = adapter.rebind(eager.state, destination, returned_binding)
    assert bool(returned.diagnostics.transaction_applied)
    initial_valid = np.asarray(state.memory_state.entries.valid)
    np.testing.assert_array_equal(
        np.asarray(returned.state.memory_state.entries.observations)[initial_valid],
        np.asarray(state.memory_state.entries.observations)[initial_valid],
    )
    np.testing.assert_array_equal(
        np.asarray(returned.state.memory_state.entries.keys)[initial_valid],
        np.asarray(state.memory_state.entries.keys)[initial_valid],
    )
    # Outcome representations return exactly; only their authenticated
    # representation-version telemetry advances monotonically.
    np.testing.assert_array_equal(
        np.asarray(returned.state.memory_state.entries.outcomes)[initial_valid],
        np.asarray(state.memory_state.entries.outcomes)[initial_valid],
    )


def test_stale_source_invalid_destination_and_state_tamper_are_atomic_noops() -> None:
    adapter = PrototypeFeatureMemory(_config())
    source = _binding(((0, 1), (1, 2)))
    destination = _binding(((0, 2), (1, 2)), words=(0, 1))
    state = _populated(adapter, source)

    stale_source = _binding(((0, 2), (1, 2)))
    stale = adapter.rebind(state, stale_source, destination)
    assert not bool(stale.diagnostics.source_binding_matches)
    assert not bool(stale.diagnostics.transaction_applied)
    _assert_exact(stale.state, state)

    duplicate_destination = _binding(((0, 2), (0, 2)), words=(0, 1))
    invalid = adapter.rebind(state, source, duplicate_destination)
    assert not bool(invalid.diagnostics.destination_binding_valid)
    _assert_exact(invalid.state, state)

    tampered_state = cast(
        PrototypeFeatureMemoryState,
        state.replace(
            schema_digest=state.schema_digest.at[0].set(
                state.schema_digest[0] ^ jnp.uint8(1)
            )
        ),
    )
    assert not bool(adapter.state_valid(tampered_state))
    rejected = adapter.rebind(tampered_state, source, destination)
    assert not bool(rejected.diagnostics.source_state_valid)
    _assert_exact(rejected.state, tampered_state)


def test_signed_zero_key_and_reward_bit_tamper_fail_state_validation() -> None:
    adapter = PrototypeFeatureMemory(_config())
    binding = _binding(((0, 1), (1, 2)))
    state = _populated(adapter, binding)
    entries = state.memory_state.entries

    key_tamper = entries.keys.at[0, 0].set(jnp.float32(-0.0))
    observation = entries.observations.at[0, 0].set(jnp.float32(0.0))
    key_state = state.replace(
        memory_state=state.memory_state.replace(
            entries=entries.replace(
                observations=observation,
                keys=key_tamper,
            )
        )
    )
    assert not bool(adapter.state_valid(key_state))

    reward_zero = entries.rewards.at[0].set(jnp.float32(0.0))
    outcome_negative_zero = entries.outcomes.at[0, -1].set(jnp.float32(-0.0))
    reward_state = state.replace(
        memory_state=state.memory_state.replace(
            entries=entries.replace(
                rewards=reward_zero,
                outcomes=outcome_negative_zero,
            )
        )
    )
    assert not bool(adapter.state_valid(reward_state))


def test_nonfinite_destination_pair_product_rejects_candidate_atomically() -> None:
    adapter = PrototypeFeatureMemory(
        _config(active_pair_slots=1, capacity=1)
    )
    source = _binding(((0, 1),))
    destination = _binding(((0, 2),), words=(0, 1))
    memory_state = adapter.memory.init()
    huge_base = (2.0e19, 1.0e-20, 2.0e19)
    entry = _entry(
        source,
        observation_base=huge_base,
        outcome_base=huge_base,
        reward=0.5,
        provenance_id=31,
    )
    write = adapter.memory.write(memory_state, entry)
    assert bool(write.wrote)
    state = adapter.init(source, write.state)

    result = adapter.rebind(state, source, destination)

    assert bool(result.diagnostics.reencode_attempted)
    assert not bool(result.diagnostics.candidate_values_finite)
    assert not bool(result.diagnostics.candidate_state_valid)
    assert not bool(result.diagnostics.transaction_applied)
    _assert_exact(result.state, state)


def test_exact_generation_words_survive_telemetry_saturation_and_high_word_rollover() -> None:
    adapter = PrototypeFeatureMemory(_config())
    source = _binding(
        ((0, 1), (1, 2)),
        words=(0, _UINT32_MAX),
    )
    destination = _binding(
        ((0, 2), (1, 2)),
        words=(1, 0),
    )
    state = _populated(adapter, source)
    assert int(source.semantic_generation) == _INT32_MAX
    assert int(destination.semantic_generation) == _INT32_MAX

    result = adapter.rebind(state, source, destination)

    assert bool(result.diagnostics.generation_is_successor)
    assert bool(result.diagnostics.transaction_applied)
    assert int(result.state.consumer_binding.semantic_generation) == _INT32_MAX
    np.testing.assert_array_equal(
        result.state.consumer_binding.semantic_generation_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    valid = result.state.memory_state.entries.valid
    assert bool(
        jnp.all(
            result.state.memory_state.entries.representation_versions[valid]
            == _INT32_MAX
        )
    )
    assert bool(adapter.state_valid(result.state, destination))


def test_static_shape_and_dtype_contracts_reject_without_coercion() -> None:
    adapter = PrototypeFeatureMemory(_config())
    binding = _binding(((0, 1), (1, 2)))
    state = adapter.init(binding)
    wrong_dtype = binding.replace(descriptors=binding.descriptors.astype(jnp.float32))
    with pytest.raises(TypeError, match="descriptors"):
        adapter.rebind(state, wrong_dtype, binding)
    wrong_digest_shape = state.replace(
        schema_digest=jnp.zeros((16,), dtype=jnp.uint8)
    )
    with pytest.raises(ValueError, match="schema_digest"):
        adapter.state_valid(wrong_digest_shape)


def test_state_valid_has_eager_and_compiled_parity() -> None:
    adapter = PrototypeFeatureMemory(_config())
    binding = _binding(((0, 1), (1, 2)))
    state = _populated(adapter, binding)

    eager = adapter.state_valid(state, binding)
    compiled = jax.jit(adapter.state_valid)(state, binding)
    assert bool(eager)
    np.testing.assert_array_equal(eager, compiled)
