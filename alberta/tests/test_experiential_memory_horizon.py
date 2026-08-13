# mypy: disable-error-code="arg-type,attr-defined,call-arg"
"""Exact finite-horizon contracts for bounded experiential memory."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.experiential_memory import (
    EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA,
    EXPERIENTIAL_MEMORY_CONFIG_SCHEMA,
    EXPERIENTIAL_MEMORY_STATE_SCHEMA,
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    ExperientialMemoryState,
    load_experiential_memory_checkpoint,
    migrate_legacy_experiential_memory_config,
    migrate_legacy_experiential_memory_state,
    save_experiential_memory_checkpoint,
)

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295


def _config(**overrides: Any) -> ExperientialMemoryConfig:
    values: dict[str, Any] = {
        "capacity": 2,
        "observation_dim": 2,
        "key_dim": 2,
        "action_dim": 2,
        "outcome_dim": 1,
        "top_k": 1,
        "min_neighbors": 1,
        "distance_scale": 0.25,
        "min_similarity": 0.5,
        "min_effective_reliability": 0.05,
        "max_uncertainty": 1.0,
        "max_safety_cost": 1.0,
        "max_age": 100,
        "staleness_scale": 100.0,
        "utility_decay": 1.0,
        "eviction_utility_weight": 0.0,
        "eviction_recency_weight": 1.0,
        "recency_scale": 10.0,
    }
    values.update(overrides)
    return ExperientialMemoryConfig(**values)


def _entry(provenance_id: int, *, age: int = 0) -> ExperientialMemoryEntry:
    return ExperientialMemoryEntry(
        observation=jnp.zeros((2,), dtype=jnp.float32),
        key=jnp.zeros((2,), dtype=jnp.float32),
        action=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        outcome=jnp.ones((1,), dtype=jnp.float32),
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(1.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        representation_version=jnp.asarray(1, dtype=jnp.int32),
        valid=jnp.asarray(True, dtype=jnp.bool_),
        age=jnp.asarray(age, dtype=jnp.int32),
        provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
        source_id=jnp.asarray(7, dtype=jnp.int32),
    )


def _assert_trees_equal(left: object, right: object) -> None:
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_v2_schemas_are_strict_and_public() -> None:
    assert alberta.EXPERIENTIAL_MEMORY_CONFIG_SCHEMA == EXPERIENTIAL_MEMORY_CONFIG_SCHEMA
    assert core.EXPERIENTIAL_MEMORY_STATE_SCHEMA == EXPERIENTIAL_MEMORY_STATE_SCHEMA
    assert alberta.EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA == EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA

    config = _config()
    payload = config.to_config()
    assert payload["schema"] == EXPERIENTIAL_MEMORY_CONFIG_SCHEMA
    assert ExperientialMemoryConfig.from_config(payload) == config

    missing = dict(payload)
    missing.pop("schema")
    with pytest.raises(ValueError, match="legacy"):
        ExperientialMemoryConfig.from_config(missing)

    extra = dict(payload)
    extra["unknown"] = 1
    with pytest.raises(ValueError, match="fields"):
        ExperientialMemoryConfig.from_config(extra)

    memory_payload = ExperientialMemory(config).to_config()
    assert memory_payload["state_schema"] == EXPERIENTIAL_MEMORY_STATE_SCHEMA
    assert ExperientialMemory.from_config(memory_payload).config == config

    with pytest.raises(ValueError, match="float32"):
        ExperientialMemory(_config(recency_scale=1.0e300))
    with pytest.raises(ValueError, match="float32"):
        ExperientialMemory(_config(staleness_scale=1.0e-300))


def test_legacy_config_migration_is_explicit() -> None:
    current = _config().to_config()
    legacy = dict(current)
    legacy.pop("schema")
    migrated = migrate_legacy_experiential_memory_config(legacy)
    assert migrated == _config()

    legacy["unknown"] = 1
    with pytest.raises(ValueError, match="fields"):
        migrate_legacy_experiential_memory_config(legacy)


def test_low_word_carry_preserves_exact_write_identity() -> None:
    memory = ExperientialMemory(_config(capacity=1))
    near_carry = memory.init().replace(
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    assert bool(memory.state_valid(near_carry))

    result = memory.write(near_carry, _entry(1, age=7))

    assert bool(result.wrote)
    np.testing.assert_array_equal(
        result.state.step_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    np.testing.assert_array_equal(
        result.state.entries.insertion_step_words[0],
        result.state.step_words,
    )
    np.testing.assert_array_equal(
        result.state.entries.last_access_step_words[0],
        result.state.step_words,
    )
    assert int(result.state.entries.insertion_age_offsets[0]) == 7
    assert int(result.state.entries.last_access_age_offsets[0]) == 7
    assert int(result.state.entries.ages[0]) == 7
    assert bool(memory.state_valid(result.state))


def test_exact_recency_beyond_int32_controls_eviction() -> None:
    memory = ExperientialMemory(_config())
    first = memory.write(memory.init(), _entry(10)).state
    seeded = memory.write(first, _entry(20)).state
    entries = seeded.entries.replace(
        insertion_step_words=jnp.asarray(((0, 0), (0, 10_000)), dtype=jnp.uint32),
        last_access_step_words=jnp.asarray(((0, 0), (0, 10_000)), dtype=jnp.uint32),
        insertion_age_offsets=jnp.zeros((2,), dtype=jnp.int32),
        last_access_age_offsets=jnp.zeros((2,), dtype=jnp.int32),
        ages=jnp.full((2,), _INT32_MAX, dtype=jnp.int32),
        recency_ages=jnp.full((2,), _INT32_MAX, dtype=jnp.int32),
    )
    high_horizon = seeded.replace(
        entries=entries,
        step_words=jnp.asarray((1, 20_000), dtype=jnp.uint32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    assert bool(memory.state_valid(high_horizon))

    result = memory.write(high_horizon, _entry(30))

    assert bool(result.evicted)
    assert int(result.evicted_provenance_id) == 10
    assert set(np.asarray(result.state.entries.provenance_ids).tolist()) == {20, 30}


def test_exact_freshness_does_not_alias_saturated_age_telemetry() -> None:
    memory = ExperientialMemory(_config(capacity=1, max_age=_INT32_MAX))
    seeded = memory.write(memory.init(), _entry(10)).state
    entries = seeded.entries.replace(
        insertion_step_words=jnp.asarray(((0, 0),), dtype=jnp.uint32),
        last_access_step_words=jnp.asarray(((1, 0),), dtype=jnp.uint32),
        insertion_age_offsets=jnp.zeros((1,), dtype=jnp.int32),
        last_access_age_offsets=jnp.zeros((1,), dtype=jnp.int32),
        ages=jnp.asarray((_INT32_MAX,), dtype=jnp.int32),
        recency_ages=jnp.asarray((5,), dtype=jnp.int32),
    )
    high_horizon = seeded.replace(
        entries=entries,
        step_words=jnp.asarray((1, 5), dtype=jnp.uint32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    assert bool(memory.state_valid(high_horizon))

    retrieval = memory.query(
        high_horizon,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
    )

    assert not bool(retrieval.freshness_ok)
    assert not bool(retrieval.accepted)
    assert int(retrieval.neighbor_ages[0]) == _INT32_MAX


def test_terminal_and_corrupt_states_fail_closed_atomically() -> None:
    memory = ExperientialMemory(_config(capacity=1))
    terminal = memory.init().replace(
        step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    assert bool(memory.state_valid(terminal))

    write = memory.write(terminal, _entry(1))
    step = memory.step(
        terminal,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        _entry(1),
    )
    assert not bool(write.wrote)
    assert not bool(step.wrote)
    _assert_trees_equal(write.state, terminal)
    _assert_trees_equal(step.state, terminal)

    seeded = memory.write(memory.init(), _entry(2)).state
    corrupt = seeded.replace(
        entries=seeded.entries.replace(
            insertion_step_words=jnp.asarray(((0, 2),), dtype=jnp.uint32)
        )
    )
    assert not bool(memory.state_valid(corrupt))
    rejected = memory.write(corrupt, _entry(3))
    assert not bool(rejected.wrote)
    _assert_trees_equal(rejected.state, corrupt)


def test_unsaturated_legacy_state_migration_and_saturation_rejection() -> None:
    memory = ExperientialMemory(_config(capacity=1))
    current = memory.write(memory.init(), _entry(1, age=4)).state
    exact_fields = {
        "step_words",
        "insertion_step_words",
        "last_access_step_words",
        "insertion_age_offsets",
        "last_access_age_offsets",
    }
    legacy_state = {
        field.name: getattr(current, field.name)
        for field in dataclasses.fields(current)
        if field.name != "step_words"
    }
    legacy_state["persistent_bytes"] = jnp.asarray(
        int(current.persistent_bytes) - 24 * memory.config.capacity - 8,
        dtype=jnp.uint32,
    )
    legacy_entries = {
        field.name: getattr(current.entries, field.name)
        for field in dataclasses.fields(current.entries)
        if field.name not in exact_fields
    }
    legacy_state["entries"] = legacy_entries

    migrated = migrate_legacy_experiential_memory_state(memory, legacy_state)
    assert bool(memory.state_valid(migrated))
    assert int(migrated.entries.ages[0]) == 4
    np.testing.assert_array_equal(migrated.step_words, jnp.asarray((0, 1), dtype=jnp.uint32))

    saturated = dict(legacy_state)
    saturated["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    with pytest.raises(ValueError, match="saturated"):
        migrate_legacy_experiential_memory_state(memory, saturated)


def test_strict_checkpoint_and_exact_resource_delta(tmp_path: Any) -> None:
    memory = ExperientialMemory(_config())
    state = memory.write(memory.init(), _entry(1)).state
    budget = memory.resource_budget(state)
    assert budget.exact_global_step_identity_bytes == 8
    assert budget.exact_slot_temporal_identity_bytes == 24 * memory.config.capacity
    assert budget.persistent_state_bytes == sum(
        int(leaf.nbytes) for leaf in jax.tree.leaves(state)
    )

    path = tmp_path / "memory-v2"
    save_experiential_memory_checkpoint(memory, state, path)
    restored_memory, restored_state = load_experiential_memory_checkpoint(path)
    assert restored_memory.config == memory.config
    _assert_trees_equal(restored_state, state)

    assert memory.to_config()["state_schema"] == (
        EXPERIENTIAL_MEMORY_STATE_SCHEMA
    )

    invalid = state.replace(
        step_words=jnp.asarray((0, 0), dtype=jnp.uint32),
    )
    with pytest.raises(ValueError, match="state is invalid"):
        save_experiential_memory_checkpoint(memory, invalid, tmp_path / "invalid")

    mismatched_budget = memory.resource_budget(state).to_config()
    mismatched_budget["persistent_state_bytes"] = (
        int(mismatched_budget["persistent_state_bytes"]) + 1
    )
    bad_path = tmp_path / "bad-resource"
    save_checkpoint(
        state,
        bad_path,
        metadata={
            "schema": EXPERIENTIAL_MEMORY_CHECKPOINT_SCHEMA,
            "state_schema": EXPERIENTIAL_MEMORY_STATE_SCHEMA,
            "mechanism_status": "development_mechanism_only",
            "scientific_promotion_allowed": False,
            "memory": memory.to_config(),
            "resource_budget": mismatched_budget,
        },
    )
    with pytest.raises(ValueError, match="resource contract"):
        load_experiential_memory_checkpoint(bad_path)


def test_scan_matches_eager_across_carry() -> None:
    memory = ExperientialMemory(_config(capacity=1))
    initial = memory.init().replace(
        step_words=jnp.asarray((0, _UINT32_MAX - 1), dtype=jnp.uint32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    ids = jnp.asarray((1, 2, 3), dtype=jnp.int32)

    def run(state: ExperientialMemoryState) -> ExperientialMemoryState:
        def body(
            carry: ExperientialMemoryState,
            provenance_id: jax.Array,
        ) -> tuple[ExperientialMemoryState, jax.Array]:
            result = memory.write(carry, _entry(0).replace(provenance_id=provenance_id))
            return result.state, result.wrote

        return jax.lax.scan(body, state, ids)[0]

    eager = run(initial)
    compiled = jax.jit(run)(initial)
    _assert_trees_equal(eager, compiled)
    np.testing.assert_array_equal(eager.step_words, jnp.asarray((1, 1), dtype=jnp.uint32))
    assert bool(memory.state_valid(eager))
